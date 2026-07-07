#!/usr/bin/env bash
# run-flash-2node-bench.sh — MiMo-V2-Flash 2-pod PD 1P1D + Non-PD bench
#
# Runs two sequential GKE jobs against the same NFS model VM:
#   1. PD 1P1D disaggregation (prefill pod + decode pod, dp=1 each, JAX transfer)
#   2. Non-PD serve-level DP  (2 pods × dp=2, round-robin proxy)
#
# Results:
#   gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-pd1p1d/
#   gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-nonpd/
#
# Dependencies: gcloud, kubectl, gsutil
# Notes:
#   - Requires 4 TPU v7x 4-chip slots (DWS queued-provisioning); may queue.
#   - PD and non-PD jobs run sequentially; NFS VM serves both.
#   - PD test uses dp=1 (stage2 PD disaggregation limitation; PDF used dp=2).

set -euo pipefail

PROJECT="tpu-launchpad-playground"
VM_NAME="nfs-flash-2node"
NONPD_JOB="mimo-v2-flash-2node-nonpd"
PD_JOB="mimo-v2-flash-2node-pd1p1d"
READY_FLAG="gs://jingnw-mimo-v2-flash-us-central1/nfs-flash-2node-ready"
NFS_IP_FLAG="gs://jingnw-mimo-v2-flash-us-central1/nfs-flash-2node-ip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

# ── Step 0: Recreate firewall rules ──────────────────────────────────────────

log "=== Step 0: Firewall rules ==="
for RULE in allow-iap-ssh allow-direct-ssh; do
  gcloud compute firewall-rules delete "${RULE}" --project="${PROJECT}" --quiet 2>/dev/null || true
done
gcloud compute firewall-rules create allow-iap-ssh \
  --project="${PROJECT}" --network=default --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --description="Allow SSH via IAP tunneling" --quiet
gcloud compute firewall-rules create allow-direct-ssh \
  --project="${PROJECT}" --network=default --allow=tcp:22 \
  --source-ranges=0.0.0.0/0 \
  --description="Allow direct SSH from anywhere (recreated each run)" --quiet

# ── Step 1: Cleanup stale GCS flags ─────────────────────────────────────────

log "=== Step 1: Clearing GCS coordination flags ==="
for FLAG in \
    "${READY_FLAG}" "${NFS_IP_FLAG}" \
    "gs://jingnw-mimo-v2-flash-us-central1/nonpd-pod0-ip" \
    "gs://jingnw-mimo-v2-flash-us-central1/nonpd-pod1-done" \
    "gs://jingnw-mimo-v2-flash-us-central1/pd1p1d-pod0-ip" \
    "gs://jingnw-mimo-v2-flash-us-central1/pd1p1d-pod1-done"; do
  gsutil rm "${FLAG}" 2>/dev/null || true
done

# ── Step 2: Create NFS VM (background) + submit PD 1P1D job ─────────────────

log "=== Step 2: Creating NFS VM in background and submitting PD 1P1D job ==="

create_nfs_vm() {
  local STARTUP
  STARTUP=$(cat << 'STARTUP_SCRIPT'
#!/bin/bash
exec > /var/log/nfs-setup.log 2>&1
set -euxo pipefail
echo "[startup] Installing NFS server..."
apt-get update -qq
apt-get install -y -qq nfs-kernel-server rpcbind
echo "[startup] Creating 315 GiB tmpfs..."
mkdir -p /export/flash
mount -t tmpfs -o size=315g tmpfs /export/flash
echo "[startup] Configuring NFS export..."
echo '/export/flash *(ro,no_root_squash,sync,no_subtree_check,no_wdelay)' > /etc/exports
systemctl enable --now rpcbind nfs-server
exportfs -ra
showmount -e localhost
echo "[startup] Copying Flash weights from GCS (~5 min)..."
gsutil -m cp 'gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-flash-hf-weights/*' /export/flash/
COUNT=$(ls /export/flash/*.safetensors 2>/dev/null | wc -l)
echo "[startup] Flash weights loaded: ${COUNT} files"
INTERNAL_IP=$(curl -sf "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip" \
  -H "Metadata-Flavor: Google")
echo "${INTERNAL_IP}" | gsutil cp - gs://jingnw-mimo-v2-flash-us-central1/nfs-flash-2node-ip
echo "ready at $(date -u) ip=${INTERNAL_IP}" | gsutil cp - \
  gs://jingnw-mimo-v2-flash-us-central1/nfs-flash-2node-ready
echo "[startup] NFS setup complete."
STARTUP_SCRIPT
)

  local CREATED_ZONE=""
  for ZONE in us-central1-c us-central1-f us-central1-ai1a; do
    log "[nfs-vm] Trying zone ${ZONE}..."
    echo "${STARTUP}" > /tmp/nfs-2node-startup.sh
    if gcloud compute instances create "${VM_NAME}" \
      --project="${PROJECT}" \
      --zone="${ZONE}" \
      --machine-type=n2-highmem-48 \
      --boot-disk-size=50GB \
      --boot-disk-type=pd-standard \
      --image-family=debian-12 \
      --image-project=debian-cloud \
      --scopes=storage-rw,logging-write,monitoring-write \
      --network=default --subnet=default \
      --metadata-from-file=startup-script=/tmp/nfs-2node-startup.sh \
      --quiet 2>&1; then
      CREATED_ZONE="${ZONE}"
      log "[nfs-vm] VM created in ${ZONE}."
      break
    else
      log "[nfs-vm] Zone ${ZONE} full, trying next..."
    fi
  done
  [ -z "${CREATED_ZONE}" ] && { log "[nfs-vm] ERROR: all zones exhausted"; return 1; }
}

if gsutil ls "${READY_FLAG}" >/dev/null 2>&1; then
  log "[nfs-vm] NFS VM already ready (flag exists), skipping creation."
  NFS_VM_PID=""
else
  create_nfs_vm &
  NFS_VM_PID=$!
fi

kubectl apply -f "${SCRIPT_DIR}/${PD_JOB}.yaml"
log "PD 1P1D job submitted. DWS may take minutes to hours to provision."

# ── Step 3: Wait for NFS VM ──────────────────────────────────────────────────

log "=== Step 3: Waiting for NFS VM ==="
log "Waiting for NFS VM to be ready..."
[ -n "${NFS_VM_PID}" ] && wait "${NFS_VM_PID}" && log "NFS VM create call returned"
log "Polling GCS for NFS ready flag (up to 30 min)..."
for i in $(seq 1 360); do
  if gsutil ls "${READY_FLAG}" >/dev/null 2>&1; then
    CONTENT=$(gsutil cat "${READY_FLAG}" 2>/dev/null || echo "")
    log "NFS VM ready: ${CONTENT}"; break
  fi
  [ $((i % 12)) -eq 0 ] && log "NFS VM still setting up... $((i*5))s"
  sleep 5
done

# ── Step 4: Wait for PD 1P1D job ─────────────────────────────────────────────

log "=== Step 4: Waiting for PD 1P1D job completion (up to 4 h) ==="
kubectl wait --for=condition=complete "job/${PD_JOB}" --timeout=14400s \
  || { log "PD job timed out or failed"; kubectl get pods -l "job-name=${PD_JOB}" -o wide; }
log "PD job final: $(kubectl get job ${PD_JOB} \
  -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)"

log "PD results: gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-pd1p1d/"
for bsz in 64 128; do
  log "  bs${bsz}/bench.log:"
  gsutil cat "gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-pd1p1d/bs${bsz}/bench.log" \
    2>/dev/null | grep -E "Output token throughput|Total token throughput|Mean ITL|Mean TTFT" || true
done

# ── Step 5: Submit Non-PD job ─────────────────────────────────────────────────

log "=== Step 5: Submitting Non-PD job ==="
gsutil rm "gs://jingnw-mimo-v2-flash-us-central1/nonpd-pod0-ip" 2>/dev/null || true
gsutil rm "gs://jingnw-mimo-v2-flash-us-central1/nonpd-pod1-done" 2>/dev/null || true

kubectl apply -f "${SCRIPT_DIR}/${NONPD_JOB}.yaml"
log "Non-PD job submitted."

# ── Step 6: Wait for Non-PD job ───────────────────────────────────────────────

log "=== Step 6: Waiting for Non-PD job completion (up to 4 h) ==="
kubectl wait --for=condition=complete "job/${NONPD_JOB}" --timeout=14400s \
  || { log "Non-PD job timed out or failed"; kubectl get pods -l "job-name=${NONPD_JOB}" -o wide; }
log "Non-PD job final: $(kubectl get job ${NONPD_JOB} \
  -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null)"

log "Non-PD results: gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-nonpd/"
for bsz in 64 128; do
  log "  bs${bsz}/bench.log:"
  gsutil cat "gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-nonpd/bs${bsz}/bench.log" \
    2>/dev/null | grep -E "Output token throughput|Total token throughput|Mean ITL|Mean TTFT" || true
done

# ── Step 7: Destroy NFS VM ────────────────────────────────────────────────────

log "=== Step 7: Destroying NFS VM ==="
NFS_ZONE=$(gcloud compute instances list \
  --project="${PROJECT}" --filter="name=${VM_NAME}" \
  --format='get(zone)' | sed 's|.*/||' 2>/dev/null || echo "")
if [ -n "${NFS_ZONE}" ]; then
  gcloud compute instances delete "${VM_NAME}" \
    --project="${PROJECT}" --zone="${NFS_ZONE}" --quiet
  log "VM ${VM_NAME} deleted from ${NFS_ZONE}"
else
  log "VM ${VM_NAME} not found (may have already been deleted)"
fi
gsutil rm "${READY_FLAG}" "${NFS_IP_FLAG}" 2>/dev/null || true

# ── Summary ───────────────────────────────────────────────────────────────────

log ""
log "=== Flash 2-node benchmark complete ==="
log "  PD 1P1D results: gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-pd1p1d/"
log "  Non-PD results:  gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-nonpd/"
