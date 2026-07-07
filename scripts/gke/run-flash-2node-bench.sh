#!/usr/bin/env bash
# run-flash-2node-bench.sh — MiMo-V2-Flash 2-pod PD 1P1D + Non-PD bench
#
# Pool: jingnw-dws-tpu7-8ch (2x2x2 multi-host, gang size=2).
# One DWS PR (count=1) provisions the full 2-VM slice atomically.
# Each test uses a 2-pod IndexedJob sharing that single PR.
#
# Order:
#   1. PD 1P1D:  1 PR + 1 IndexedJob (pod0=prefill+bootstrap, pod1=decode+bench)
#   2. Non-PD:   1 PR + 1 IndexedJob (pod0=server, pod1=server+proxy+bench)
#   3. Tear down NFS VM
#
# NFS VM:
#   - Created in background during DWS wait for PD job (skipped if already ready)
#   - Serves Flash weights to all pods via NFS
#
# Results:
#   gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-pd1p1d/
#   gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-nonpd/

set -euo pipefail

PROJECT="tpu-launchpad-playground"
VM_NAME="nfs-flash-2node"
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
  --description="Allow direct SSH from anywhere" --quiet

# ── Step 1: Clear pod coordination flags (preserve NFS VM flags) ──────────────

log "=== Step 1: Clearing pod coordination flags ==="
for FLAG in \
    "gs://jingnw-mimo-v2-flash-us-central1/nonpd-pod0-ip" \
    "gs://jingnw-mimo-v2-flash-us-central1/nonpd-pod1-done" \
    "gs://jingnw-mimo-v2-flash-us-central1/pd1p1d-pod0-ip" \
    "gs://jingnw-mimo-v2-flash-us-central1/pd1p1d-pod1-done"; do
  gsutil rm "${FLAG}" 2>/dev/null || true
done

# ── Step 2: Create NFS VM (if not already ready) + submit PD jobs ────────────

log "=== Step 2: NFS VM check + PD jobs ==="

create_nfs_vm() {
  local STARTUP
  STARTUP=$(cat << 'STARTUP_SCRIPT'
#!/bin/bash
exec > /var/log/nfs-setup.log 2>&1
set -euxo pipefail
apt-get update -qq
apt-get install -y -qq nfs-kernel-server rpcbind
mkdir -p /export/flash
mount -t tmpfs -o size=315g tmpfs /export/flash
echo '/export/flash *(ro,no_root_squash,sync,no_subtree_check,no_wdelay)' > /etc/exports
systemctl enable --now rpcbind nfs-server
exportfs -ra
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
      --project="${PROJECT}" --zone="${ZONE}" \
      --machine-type=n2-highmem-48 --boot-disk-size=50GB \
      --boot-disk-type=pd-standard --image-family=debian-12 \
      --image-project=debian-cloud \
      --scopes=storage-rw,logging-write,monitoring-write \
      --network=default --subnet=default \
      --metadata-from-file=startup-script=/tmp/nfs-2node-startup.sh \
      --quiet 2>&1; then
      CREATED_ZONE="${ZONE}"
      log "[nfs-vm] VM created in ${ZONE}."
      break
    else
      log "[nfs-vm] Zone ${ZONE} unavailable, trying next..."
    fi
  done
  [ -z "${CREATED_ZONE}" ] && { log "[nfs-vm] ERROR: all zones exhausted"; return 1; }
}

NFS_VM_PID=""
if gsutil ls "${READY_FLAG}" >/dev/null 2>&1; then
  log "[nfs-vm] NFS VM already ready (flag exists), skipping creation."
else
  create_nfs_vm &
  NFS_VM_PID=$!
fi

kubectl apply -f "${SCRIPT_DIR}/mimo-v2-flash-2node-pd1p1d.yaml"
log "PD 1P1D job submitted (1 PR count=1, 2-pod IndexedJob)."

# ── Step 3: Wait for DWS (PD) + NFS VM ───────────────────────────────────────

wait_dws() {
  local PR_NAME="$1"
  for i in $(seq 1 720); do
    STATUS=$(kubectl get provisioningrequest "${PR_NAME}" \
      -o jsonpath='{.status.conditions[?(@.type=="Provisioned")].status}' 2>/dev/null || echo "")
    [ "${STATUS}" = "True" ] && { log "DWS ${PR_NAME} provisioned after $((i*30))s"; return 0; }
    FAILED=$(kubectl get provisioningrequest "${PR_NAME}" \
      -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
    if [ "${FAILED}" = "True" ]; then
      MSG=$(kubectl get provisioningrequest "${PR_NAME}" \
        -o jsonpath='{.status.conditions[?(@.type=="Failed")].message}' 2>/dev/null || echo "")
      log "ERROR: DWS ${PR_NAME} failed: ${MSG}"
      return 1
    fi
    [ $((i % 4)) -eq 0 ] && log "DWS ${PR_NAME} queuing... $((i*30))s"
    sleep 30
  done
  log "WARNING: DWS ${PR_NAME} timed out after 6h"
}

log "=== Step 3: Waiting for PD DWS provisioning ==="
wait_dws "mimo-v2-flash-2node-pd1p1d" &
WAIT_P=$!

log "Waiting for NFS VM..."
[ -n "${NFS_VM_PID}" ] && wait "${NFS_VM_PID}" && log "NFS VM create returned"
log "Polling GCS for NFS ready flag (up to 30 min)..."
for i in $(seq 1 360); do
  if gsutil ls "${READY_FLAG}" >/dev/null 2>&1; then
    CONTENT=$(gsutil cat "${READY_FLAG}" 2>/dev/null || echo "")
    log "NFS VM ready: ${CONTENT}"; break
  fi
  [ $((i % 12)) -eq 0 ] && log "NFS VM still setting up... $((i*5))s"
  sleep 5
done

wait "${WAIT_P}" || log "WARNING: PD DWS provisioning failed"

# ── Step 4: Wait for PD jobs ──────────────────────────────────────────────────

log "=== Step 4: Waiting for PD 1P1D job ==="
kubectl wait --for=condition=complete "job/mimo-v2-flash-2node-pd1p1d" --timeout=14400s \
  || { log "PD job failed/timed out"; kubectl get pods -l "job-name=mimo-v2-flash-2node-pd1p1d" -o wide; }

log "PD results:"
for bsz in 64 128; do
  gsutil cat "gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-pd1p1d/bs${bsz}/bench.log" \
    2>/dev/null | grep -E "Output token throughput|Total token throughput|Mean ITL|Mean TTFT" || true
done

# ── Step 5: Submit Non-PD jobs ────────────────────────────────────────────────

log "=== Step 5: Submitting Non-PD job ==="
gsutil rm "gs://jingnw-mimo-v2-flash-us-central1/nonpd-pod0-ip" 2>/dev/null || true
gsutil rm "gs://jingnw-mimo-v2-flash-us-central1/nonpd-pod1-done" 2>/dev/null || true

kubectl apply -f "${SCRIPT_DIR}/mimo-v2-flash-2node-nonpd.yaml"
log "Non-PD job submitted (1 PR count=1, 2-pod IndexedJob)."

log "=== Step 5b: Waiting for Non-PD DWS provisioning ==="
wait_dws "mimo-v2-flash-2node-nonpd" &
WAIT_N=$!
wait "${WAIT_N}" || log "WARNING: nonpd DWS failed"

# ── Step 6: Wait for Non-PD jobs ─────────────────────────────────────────────

log "=== Step 6: Waiting for Non-PD bench job ==="
kubectl wait --for=condition=complete "job/mimo-v2-flash-2node-nonpd" --timeout=14400s \
  || { log "Non-PD job failed/timed out"; kubectl get pods -l "job-name=mimo-v2-flash-2node-nonpd" -o wide; }

log "Non-PD results:"
for bsz in 64 128; do
  gsutil cat "gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-nonpd/bs${bsz}/bench.log" \
    2>/dev/null | grep -E "Output token throughput|Total token throughput|Mean ITL|Mean TTFT" || true
done

# ── Step 7: Destroy NFS VM ────────────────────────────────────────────────────

log "=== Step 7: Destroying NFS VM ==="
while IFS= read -r ZONE_PATH; do
  ZONE="${ZONE_PATH##*/}"
  [ -n "${ZONE}" ] || continue
  gcloud compute instances delete "${VM_NAME}" \
    --project="${PROJECT}" --zone="${ZONE}" --quiet 2>/dev/null \
    && log "VM deleted from ${ZONE}" || true
done < <(gcloud compute instances list \
  --project="${PROJECT}" --filter="name=${VM_NAME}" --format='get(zone)')
gsutil rm "${READY_FLAG}" "${NFS_IP_FLAG}" 2>/dev/null || true

log ""
log "=== Flash 2-node benchmark complete ==="
log "  PD 1P1D: gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-pd1p1d/"
log "  Non-PD:  gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-2node-nonpd/"
