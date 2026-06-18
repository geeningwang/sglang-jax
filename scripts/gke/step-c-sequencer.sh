#!/usr/bin/env bash
# step-c-sequencer.sh — Runs from system crontab every 5 min.
# Manages sequential resubmission of cp4096 → ps128 after ps128cp4096 completes.
# Stops itself (removes crontab entry) once all 3 jobs are done.
#
# Log: /tmp/step-c-sequencer.log

set -euo pipefail
LOG="/tmp/step-c-sequencer.log"
REPO="/home/jingnw_google_com/sglang-jax"
PROJECT="tpu-launchpad-playground"

log() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*" | tee -a "${LOG}"; }

job_status() {
  kubectl get job "$1" -o jsonpath='{.status.conditions[?(@.type=="Complete")].status}' 2>/dev/null || echo ""
}

job_exists() {
  kubectl get job "$1" --no-headers 2>/dev/null | grep -q . && echo "yes" || echo "no"
}

dws_failed() {
  local FAILED
  FAILED=$(kubectl get provisioningrequest "$1" \
    -o jsonpath='{.status.conditions[?(@.type=="Failed")].status}' 2>/dev/null || echo "")
  [ "${FAILED}" = "True" ] && echo "yes" || echo "no"
}

cleanup_variant() {
  local VARIANT=$1
  local JOB="mimo-v2-flash-1node-nfs-tpu7-${VARIANT}"
  local VM="nfs-flash-${VARIANT}"
  log "Cleaning up failed ${VARIANT}..."
  pgrep -f "run-flash-nfs-bench-${VARIANT}.sh" | xargs kill 2>/dev/null || true
  kubectl delete job "${JOB}" --ignore-not-found 2>/dev/null || true
  kubectl delete provisioningrequest "${JOB}" --ignore-not-found 2>/dev/null || true
  kubectl delete podtemplate "${JOB}-template" --ignore-not-found 2>/dev/null || true
  ZONE=$(gcloud compute instances list --project="${PROJECT}" --filter="name=${VM}" \
    --format='get(zone)' | sed 's|.*/||' 2>/dev/null || echo "")
  if [ -n "${ZONE}" ]; then
    gcloud compute instances delete "${VM}" --project="${PROJECT}" --zone="${ZONE}" --quiet 2>/dev/null || true
    log "Deleted VM ${VM} in ${ZONE}"
  fi
  gsutil rm "gs://jingnw-mimo-v2-flash-us-central1/${VM}-ready" \
             "gs://jingnw-mimo-v2-flash-us-central1/${VM}-ip" 2>/dev/null || true
}

remove_self_from_crontab() {
  crontab -l 2>/dev/null | grep -v "step-c-sequencer" | crontab - || true
  log "Removed self from crontab — all done."
}

log "=== step-c-sequencer tick ==="

PS128CP4096_STATUS=$(job_status "mimo-v2-flash-1node-nfs-tpu7-ps128cp4096")
CP4096_EXISTS=$(job_exists "mimo-v2-flash-1node-nfs-tpu7-cp4096")
PS128_EXISTS=$(job_exists "mimo-v2-flash-1node-nfs-tpu7-ps128")
CP4096_STATUS=$(job_status "mimo-v2-flash-1node-nfs-tpu7-cp4096")
PS128_STATUS=$(job_status "mimo-v2-flash-1node-nfs-tpu7-ps128")

log "ps128cp4096=${PS128CP4096_STATUS:-running} cp4096_exists=${CP4096_EXISTS} cp4096=${CP4096_STATUS:-pending} ps128_exists=${PS128_EXISTS} ps128=${PS128_STATUS:-pending}"

# ── Handle DWS failures for cp4096 and ps128 ──────────────────────────────────
if [ "${CP4096_EXISTS}" = "yes" ] && [ "$(dws_failed mimo-v2-flash-1node-nfs-tpu7-cp4096)" = "yes" ]; then
  log "cp4096 DWS failed — cleaning up, will retry next tick"
  cleanup_variant "cp4096"
  exit 0
fi

if [ "${PS128_EXISTS}" = "yes" ] && [ "$(dws_failed mimo-v2-flash-1node-nfs-tpu7-ps128)" = "yes" ]; then
  log "ps128 DWS failed — cleaning up, will retry next tick"
  cleanup_variant "ps128"
  exit 0
fi

# ── Case D: All 3 complete ────────────────────────────────────────────────────
if [ "${PS128CP4096_STATUS}" = "True" ] && \
   [ "${CP4096_STATUS}" = "True" ] && \
   [ "${PS128_STATUS}" = "True" ]; then
  log "All 3 jobs complete! Collecting results..."
  for variant in cp4096 ps128 ps128cp4096; do
    for mode in mtp nomtp; do
      for bs in 32 64 128; do
        RESULT=$(gsutil cat \
          "gs://jingnw-mimo-v2-flash-us-central1/perf-results/flash-1node-nfs-tpu7-${variant}/${mode}/bs${bs}/bench.log" \
          2>/dev/null | grep "Output token throughput" || echo "(not ready)")
        log "  ${variant}/${mode}/bs${bs}: ${RESULT}"
      done
    done
  done
  remove_self_from_crontab
  exit 0
fi

# ── Case C: cp4096 complete, ps128 not yet started ───────────────────────────
if [ "${CP4096_STATUS}" = "True" ] && [ "${PS128_EXISTS}" = "no" ]; then
  log "cp4096 complete. Starting ps128..."
  nohup bash "${REPO}/scripts/gke/run-flash-nfs-bench-ps128.sh" \
    >> /tmp/bench-ps128.log 2>&1 &
  log "ps128 launched PID $!"
  exit 0
fi

# ── Case B: ps128cp4096 complete, cp4096 not yet started ─────────────────────
if [ "${PS128CP4096_STATUS}" = "True" ] && [ "${CP4096_EXISTS}" = "no" ]; then
  log "ps128cp4096 complete. Starting cp4096..."
  nohup bash "${REPO}/scripts/gke/run-flash-nfs-bench-cp4096.sh" \
    >> /tmp/bench-cp4096.log 2>&1 &
  log "cp4096 launched PID $!"
  exit 0
fi

# ── Case A: Still waiting ─────────────────────────────────────────────────────
log "Waiting — no action needed this tick."
