#!/usr/bin/env bash
# run-flash-nfs-bench-dvfs7-mtp2.sh — MiMo-V2-Flash No-MTP+MTP retest with DVFS P-state 7
#
# Identical to run-flash-nfs-bench-dvfs7.sh except:
#   - Job/VM/flag names use the -dvfs7-mtp2 suffix
#   - No-MTP runs first (warms XLA cache), MTP runs second
#   - Results go to: gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-nfs-tpu7-dvfs7-mtp2/
#
# Purpose: retest MTP throughput (especially bsz=32) with warm XLA cache to verify gap vs §1.3.

set -euo pipefail

PROJECT="tpu-launchpad-playground"
VM_NAME="nfs-flash-dvfs7-mtp2"
JOB_NAME="mimo-v2-flash-1node-nfs-tpu7-dvfs7-mtp2"
READY_FLAG="gs://jingnw-mimo-v2-flash-us-central1/nfs-flash-dvfs7-mtp2-ready"
NFS_IP_FLAG="gs://jingnw-mimo-v2-flash-us-central1/nfs-flash-dvfs7-mtp2-ip"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

# ── Step 0: Recreate SSH firewall rules ───────────────────────────────────────

log "=== Step 0: (Re)creating SSH firewall rules ==="
gcloud compute firewall-rules delete allow-iap-ssh \
  --project="${PROJECT}" --quiet 2>/dev/null || true
gcloud compute firewall-rules create allow-iap-ssh \
  --project="${PROJECT}" \
  --network=default \
  --allow=tcp:22 \
  --source-ranges=35.235.240.0/20 \
  --description="Allow SSH via IAP tunneling" \
  --quiet
gcloud compute firewall-rules delete allow-direct-ssh \
  --project="${PROJECT}" --quiet 2>/dev/null || true
gcloud compute firewall-rules create allow-direct-ssh \
  --project="${PROJECT}" \
  --network=default \
  --allow=tcp:22 \
  --source-ranges=0.0.0.0/0 \
  --description="Allow direct SSH from anywhere (recreated each run)" \
  --quiet

# ── Phase 1: Submit DWS GKE job ────────────────────────────────────────────────

log "=== Phase 1: Submitting GKE DWS job: ${JOB_NAME} ==="
kubectl apply -f "${SCRIPT_DIR}/mimo-v2-flash-1node-nfs-tpu7-dvfs7-mtp2.yaml"
log "Job submitted. DWS may take minutes to hours to provision TPU capacity."

# ── Phase 2: Create NFS VM in parallel with DWS wait ──────────────────────────

log "=== Phase 2: Creating NFS VM (runs in background while DWS queues) ==="

gsutil rm "${READY_FLAG}" 2>/dev/null || true
gsutil rm "${NFS_IP_FLAG}" 2>/dev/null || true

create_nfs_vm() {
  local STARTUP
  STARTUP=$(cat << 'STARTUP_SCRIPT'
#!/bin/bash
exec > /var/log/nfs-setup.log 2>&1
set -euxo pipefail

echo "[startup] Installing NFS server..."
apt-get update -qq
apt-get install -y -qq nfs-kernel-server rpcbind

echo "[startup] Creating 315 GiB tmpfs as the NFS export root..."
mkdir -p /export/flash
mount -t tmpfs -o size=315g tmpfs /export/flash

echo "[startup] Configuring NFS export for /export/flash..."
echo '/export/flash *(ro,no_root_squash,sync,no_subtree_check,no_wdelay)' > /etc/exports
systemctl enable --now rpcbind nfs-server
exportfs -ra
showmount -e localhost

echo "[startup] Copying Flash weights from GCS into RAM (~5 min)..."
gsutil -m cp 'gs://jingnw-mimo-v2-5-pro-us-central1/mimo-v2-flash-hf-weights/*' \
  /export/flash/

COUNT=$(ls /export/flash/*.safetensors 2>/dev/null | wc -l)
echo "[startup] Flash weights loaded: ${COUNT} safetensors files"

INTERNAL_IP=$(curl -sf \
  "http://metadata.google.internal/computeMetadata/v1/instance/network-interfaces/0/ip" \
  -H "Metadata-Flavor: Google")
echo "[startup] Internal IP: ${INTERNAL_IP}"
echo "${INTERNAL_IP}" | gsutil cp - \
  gs://jingnw-mimo-v2-flash-us-central1/nfs-flash-dvfs7-mtp2-ip

echo "ready at $(date -u) ip=${INTERNAL_IP}" | gsutil cp - \
  gs://jingnw-mimo-v2-flash-us-central1/nfs-flash-dvfs7-mtp2-ready
echo "[startup] NFS setup complete. Ready flag written to GCS."
STARTUP_SCRIPT
)

  local CREATED_ZONE=""
  for ZONE in us-central1-c us-central1-f us-central1-ai1a; do
    log "[nfs-vm] Trying zone ${ZONE} for n2-highmem-48..."
    echo "${STARTUP}" > /tmp/nfs-flash-dvfs7-mtp2-startup.sh
    if gcloud compute instances create "${VM_NAME}" \
      --project="${PROJECT}" \
      --zone="${ZONE}" \
      --machine-type=n2-highmem-48 \
      --boot-disk-size=50GB \
      --boot-disk-type=pd-standard \
      --image-family=debian-12 \
      --image-project=debian-cloud \
      --scopes=storage-rw,logging-write,monitoring-write \
      --network=default \
      --subnet=default \
      --metadata-from-file=startup-script=/tmp/nfs-flash-dvfs7-mtp2-startup.sh \
      --quiet 2>&1; then
      CREATED_ZONE="${ZONE}"
      log "[nfs-vm] VM created in ${ZONE}."
      break
    else
      log "[nfs-vm] Zone ${ZONE} has no capacity, trying next..."
    fi
  done

  if [ -z "${CREATED_ZONE}" ]; then
    log "[nfs-vm] ERROR: All zones exhausted. Cannot create NFS VM."
    return 1
  fi

  VM_IP=$(gcloud compute instances describe "${VM_NAME}" \
    --project="${PROJECT}" --zone="${CREATED_ZONE}" \
    --format='get(networkInterfaces[0].networkIP)')
  log "[nfs-vm] VM created in ${CREATED_ZONE}. Internal IP: ${VM_IP}"
}

create_nfs_vm &
NFS_VM_PID=$!

# ── Phase 3: Wait for DWS to provision ────────────────────────────────────────

log "=== Phase 3: Waiting for DWS provisioning ==="
for i in $(seq 1 360); do
  STATUS=$(kubectl get provisioningrequest "${JOB_NAME}" \
    -o jsonpath='{.status.conditions[?(@.type=="Provisioned")].status}' 2>/dev/null || echo "Unknown")
  if [ "${STATUS}" = "True" ]; then
    log "DWS provisioned after $((i * 30))s"
    break
  fi
  [ $((i % 12)) -eq 0 ] && log "DWS still queuing... $((i * 30))s elapsed (status: ${STATUS})"
  sleep 30
done

# ── Phase 4: Wait for NFS VM setup to complete ────────────────────────────────

log "=== Phase 4: Waiting for NFS VM create to return ==="
wait "${NFS_VM_PID}" && log "VM create call returned"

log "=== Phase 4b: Polling GCS for NFS ready flag (up to 30 min) ==="
for i in $(seq 1 360); do
  if gsutil ls "${READY_FLAG}" >/dev/null 2>&1; then
    CONTENT=$(gsutil cat "${READY_FLAG}" 2>/dev/null || echo "")
    log "NFS VM ready: ${CONTENT}"
    break
  fi
  [ $((i % 12)) -eq 0 ] && log "NFS VM still setting up... $((i * 5))s elapsed"
  sleep 5
done

# ── Phase 5: Wait for GKE job to complete ─────────────────────────────────────

log "=== Phase 5: Waiting for GKE job completion (up to 3 h) ==="
kubectl wait --for=condition=complete \
  "job/${JOB_NAME}" \
  --timeout=10800s \
  || {
    log "Job timed out or failed. Checking pod status..."
    kubectl get pods -l "job-name=${JOB_NAME}" -o wide
    kubectl logs -l "job-name=${JOB_NAME}" --tail=50
  }

JOB_STATUS=$(kubectl get job "${JOB_NAME}" \
  -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "Unknown")
log "Job final status: ${JOB_STATUS}"

# ── Phase 6: Destroy NFS VM ───────────────────────────────────────────────────

log "=== Phase 6: Destroying NFS VM ==="
NFS_ZONE=$(gcloud compute instances list \
  --project="${PROJECT}" --filter="name=${VM_NAME}" \
  --format='get(zone)' | sed 's|.*/||' 2>/dev/null || echo "")
if [ -n "${NFS_ZONE}" ]; then
  gcloud compute instances delete "${VM_NAME}" \
    --project="${PROJECT}" --zone="${NFS_ZONE}" \
    --quiet
  log "VM ${VM_NAME} deleted from ${NFS_ZONE}"
else
  log "VM ${VM_NAME} not found (may have already been deleted)"
fi
gsutil rm "${READY_FLAG}" 2>/dev/null || true
gsutil rm "${NFS_IP_FLAG}" 2>/dev/null || true

# ── Summary ───────────────────────────────────────────────────────────────────

log ""
log "=== Flash NFS dvfs7-mtp2 benchmark complete ==="
log "Results: gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-nfs-tpu7-dvfs7-mtp2/"
log "  No-MTP bench: .../nomtp/bs{32,64,128}/bench.log"
log "  MTP bench:    .../mtp/bs{32,64,128}/bench.log"
log "  Server logs:  .../nomtp/server.log, .../mtp/server.log"
