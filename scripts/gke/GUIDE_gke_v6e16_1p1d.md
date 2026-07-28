# MiMo-V2.5 1P1D benchmark on GKE (v6e-16 × 2)

Profiling harness for [issue 323](https://github.com/primatrix/sglang-jax/issues/323) —
PD-disaggregated scheduler overlap. Replaces the standalone-TPU-VM procedure in
`../tpu-vm/GUIDE_v6e16_1p1d_benchmark.md`, whose VMs and results bucket no longer exist.

> **The §4 numbers in the tpu-vm guide are not a valid baseline for this work.** They were
> measured on MiMo-V2-Flash over NFS on two standalone tpu-vms. Before/after for issue 323
> must both be measured here.

## Topology

| | Job | Node pool | Role |
|---|---|---|---|
| Prefill | `mimo-p` | `tpu-v6e-slice-0` | 4 pods × 4 chips, TP=16. Pod 0 also runs the bootstrap server on `:8998`. |
| Decode | `mimo-d` | `tpu-v6e-slice-1` | 4 pods × 4 chips, TP=16. Pod 0 also runs the PD router on `:30000` and drives `bench_serving`. |

Both jobs are Indexed Jobs behind a headless Service, so pod `N` is reachable at
`mimo-{p,d}-N.mimo-{p,d}-svc` — that is what `--dist-init-addr`, `TPU_PROCESS_ADDRESSES`,
and the cross-job bootstrap URL all resolve through. See
[`../../docs/cookbook/deployment/gke-indexed-job.md`](../../docs/cookbook/deployment/gke-indexed-job.md)
for why Indexed Job rather than Deployment.

Coordination between the two jobs is a done-flag file on the shared GCS mount
(`/gcs-out/coord/gke-v6e16-1p1d-done`) plus an HTTP poll of the bootstrap `/health`
endpoint. Decode blocks until prefill's bootstrap answers; prefill holds its slice until
decode writes the flag.

## Environment

| | |
|---|---|
| Cluster | `mimo-tpu-cluster`, `us-east5-a`, project `gpu-launchpad-playground` |
| Namespace / identity | `mimo` / `mimo-sa` (already bound to the bucket via Workload Identity) |
| Reservation | `reservation-frankie` — both node pools are already bound to it |
| Model | `gs://0-mimo-25/MiMo-V2.5`, mounted read-only at `/gcs-model` via gcsfuse CSI |
| Results | `gs://0-mimo-25/perf-results/gke-v6e16-1p1d/<RUN_TAG>/`, mounted rw at `/gcs-out` |

MiMo-V2.5 is 48 layers (9 full-attention, 39 SWA), FP8 e4m3, 256 experts. It builds a
`SWAKVPool`, so the PD path needs the SWA compat landed in `8677518` — do **not** run
`gs://0-mimo-25/swa_pd_patch.py` against this branch. That script is stale (its
`multihost_sync` step targets pre-`56bfad0` code), it has no rollback, and it partially
applies: it injects a `_room_to_i32` import into `decode.py` and then dies before defining
the symbol, leaving a server that boots and then fails at the first KV-transfer reap.

## Run

Both manifests default to `BRANCH=perf/issue-323-scheduler-overlap` and `RUN_TAG=baseline`.

```bash
kubectl apply -f scripts/gke/1p1d-prefill.yaml
kubectl apply -f scripts/gke/1p1d-decode.yaml     # safe to apply immediately; decode waits
```

**`RUN_TAG` must differ between runs** — results go to a tag-scoped GCS prefix, and reusing
a tag overwrites the previous run. Use `baseline` for the pre-optimization capture and
`phase3` (or similar) after. Override both jobs consistently:

```bash
kubectl set env job/mimo-p RUN_TAG=phase3 BRANCH=perf/issue-323-scheduler-overlap -n mimo
kubectl set env job/mimo-d RUN_TAG=phase3 BRANCH=perf/issue-323-scheduler-overlap -n mimo
```

`kubectl set env` on an existing Job is rejected — delete and re-apply with the edited
manifest instead, or use `kubectl create -f - <<<"$(sed ...)"`.

### Watch

```bash
kubectl get pods -n mimo -w
kubectl logs -n mimo job/mimo-d -c decode --tail=100 -f      # pod 0 drives the sweep
kubectl logs -n mimo job/mimo-p -c prefill --tail=100 -f
```

Expect roughly: ~3 min image pull + pip install, weight load over gcsfuse, then XLA
precompile, then the bs 32/64/128 sweep. The first run pays full XLA compilation;
`JAX_COMPILATION_CACHE_DIR` points at `/gcs-out/jit-cache/{prefill,decode}` so later runs
with identical shape flags reuse it.

### Teardown

```bash
kubectl delete job mimo-p mimo-d -n mimo
kubectl delete svc mimo-p-svc mimo-d-svc -n mimo
gcloud storage rm gs://0-mimo-25/coord/gke-v6e16-1p1d-done    # else the next run exits early
```

Deleting the done flag is required between runs — a stale flag makes both jobs shut down
as soon as their servers come up.

## What to collect

Written automatically to `gs://0-mimo-25/perf-results/gke-v6e16-1p1d/<RUN_TAG>/`:

| File | Contents |
|---|---|
| `loop-profile.log` | `PD-DECODE-LOOP-PROFILE` lines — per-phase decode event-loop segment cost. **The primary issue-323 signal.** |
| `time-stats-decode.log` | `PD-TIME-STATS` — per-request phase breakdown (`prealloc_wait`, `kv_wait`, `metadata_wait`) |
| `bs{32,64,128}/result.jsonl` | bench_serving throughput / TTFT / ITL |
| `bs{32,64,128}/bench.log` | bench_serving stdout |
| `server-{prefill,decode}-w*.log` | full server logs, all 8 pods |
| `router.log`, `bootstrap.log` | PD router and bootstrap server |

Reading the profile line: phases are sorted by share of the window, so the serialized
segment sorts first. `beats=` is total phase transitions (iterations × distinct phases),
**not** an iteration count — per-phase `n=` is the iteration count.

The segments that matter for issue 323 are `admit_prealloc`, `reap_allgather`, and
`reap_writeback`. `reap_allgather` and `reap_writeback` are the two genuine SPMD
collectives and must stay inside the forward-thread-drained window; `admit_prealloc` is
collective-free and is what Phase 3-A moves back out of it.

## Safety signal

The regression to watch for is `E0200` / SPMD program-order mismatch in any decode pod log.
Commit `3c30125` serialized the decode loop to eliminate exactly that race, and issue 323
is about restoring overlap without bringing it back. Its absence across a full
bs 32/64/128 sweep is the pass condition. A hang surfaces as
`PD-DECODE-WATCHDOG stall detected` (enable with
`--disaggregation-decode-watchdog-seconds`, independent of the profiler).

## Notes

- `backoffLimit: 0` on both jobs, deliberately. A silent retry costs a full weight load and
  can leave the peer job waiting on a bootstrap that moved. Failures should be visible.
- gcsfuse sidecar limits are set to `"0"` (unlimited); the defaults throttle a
  293 GiB weight load badly.
- The bench sweep uses `--random-output-len 4096`. That is well past the 512-token
  `disaggregation_num_reserved_decode_tokens` floor, which is the regime the admission
  reserve fix in `1533b88` addresses — without it this sweep can deadlock the decode loop.
