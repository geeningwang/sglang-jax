#!/usr/bin/env bash
# run-pro-bench-dvfs7-sharegpt.sh — MiMo-V2.5-Pro MTP benchmark, ShareGPT dataset, DVFS P-state 7
#
# Identical to the Pro dvfs7 workflow but uses ShareGPT instead of random dataset,
# to measure realistic MTP acceptance rate and prefill/decode throughput.
# Pro model uses gcsfuse (not NFS) so no NFS VM lifecycle is needed.
#
# Results: gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/pro-4host-tpu7-dvfs7-sharegpt/

set -euo pipefail

JOB_NAME="mimo-v2-pro-4host-tpu7-dvfs7-sharegpt"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }

# ── Phase 1: Submit DWS GKE job ───────────────────────────────────────────────

log "=== Phase 1: Submitting GKE DWS job: ${JOB_NAME} ==="
kubectl apply -f "${SCRIPT_DIR}/mimo-v2-pro-4host-tpu7-dvfs7-sharegpt.yaml"
log "Job submitted. DWS may take minutes to hours to provision 4-host TPU capacity."

# ── Phase 2: Wait for DWS to provision ────────────────────────────────────────

log "=== Phase 2: Waiting for DWS provisioning ==="
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

# ── Phase 3: Wait for GKE job to complete ─────────────────────────────────────

log "=== Phase 3: Waiting for GKE job completion (up to 4 h) ==="
kubectl wait --for=condition=complete \
  "job/${JOB_NAME}" \
  --timeout=14400s \
  || {
    log "Job timed out or failed. Checking pod status..."
    kubectl get pods -l "job-name=${JOB_NAME}" -o wide
    kubectl logs -l "job-name=${JOB_NAME}" --tail=50
  }

JOB_STATUS=$(kubectl get job "${JOB_NAME}" \
  -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo "Unknown")
log "Job final status: ${JOB_STATUS}"

log ""
log "=== Pro dvfs7 ShareGPT benchmark complete ==="
log "Results: gs://jingnw-mimo-v2-5-pro-us-central1/perf-results/pro-4host-tpu7-dvfs7-sharegpt/"
log "  MTP bench: .../mtp/bs{32,64,128,192}/bench.log"
log "  Server log: .../mtp/server_rank0.log"
