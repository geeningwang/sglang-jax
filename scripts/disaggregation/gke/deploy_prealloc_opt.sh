#!/usr/bin/env bash
# Deploy PD single-host eval/benchmark for prealloc overhead profiling on GKE.
#
# Usage:
#   ./scripts/disaggregation/gke/deploy_prealloc_opt.sh [--delete-only]
#
# Optional overrides:
#   JOB_NAME=... GH_ORG=... BRANCH=... ./scripts/disaggregation/gke/deploy_prealloc_opt.sh
#
# Prerequisites:
#   - gcloud + kubectl configured, USE_GKE_GCLOUD_AUTH_PLUGIN=True
#   - Cluster a6e-wangez in us-east5 with node pool tpu-nodepool-wangez
#   - SSH key at ~/.ssh/id_ed25519 for repo clone
#   - GCS bucket gs://0-wangez accessible from the cluster
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
JOB_NAME="${JOB_NAME:-pd-prealloc-opt}"
GH_ORG="${GH_ORG:-geeningwang}"
BRANCH="${BRANCH:-mimo-tpu7-stage3}"
TEMPLATE="${SCRIPT_DIR}/pd_prealloc_opt_job.yaml"
GENERATED="/tmp/${JOB_NAME}.yaml"

export USE_GKE_GCLOUD_AUTH_PLUGIN=True

# Ensure we have the right cluster context
gcloud container clusters get-credentials a6e-wangez \
  --region us-east5 --project gpu-launchpad-playground 2>/dev/null

# Clean up any previous run
echo "Cleaning up previous job (if any)..."
kubectl delete job "$JOB_NAME" --ignore-not-found 2>/dev/null || true
kubectl delete svc "${JOB_NAME}-headless-svc" --ignore-not-found 2>/dev/null || true

if [[ "${1:-}" == "--delete-only" ]]; then
  echo "Delete-only mode. Done."
  exit 0
fi

# Inject SSH key
KEY_B64=$(base64 < ~/.ssh/id_ed25519 | tr -d '\n')

sed \
  -e "s|<JOB_NAME>|${JOB_NAME}|g" \
  -e "s|<SSH_KEY_B64>|${KEY_B64}|g" \
  -e "s|<GH_ORG>|${GH_ORG}|g" \
  -e "s|<BRANCH>|${BRANCH}|g" \
  "$TEMPLATE" > "$GENERATED"

echo "Applying job..."
kubectl apply -f "$GENERATED" --validate=false

echo ""
echo "Job submitted. Monitor with:"
echo "  kubectl get pods -l job-name=${JOB_NAME} -w"
echo "  kubectl logs ${JOB_NAME}-0 -f   # prefill pod"
echo "  kubectl logs ${JOB_NAME}-1 -f   # decode + test pod"
echo ""
echo "Waiting for pods to start..."
kubectl wait --for=condition=Ready pod -l "job-name=${JOB_NAME}" --timeout=120s 2>/dev/null || true
kubectl get pods -l "job-name=${JOB_NAME}"
