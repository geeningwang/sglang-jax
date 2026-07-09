# moe.py — Comprehensive Explanation

This document walks through every class, method, and significant line of
`python/sgl_jax/srt/layers/moe.py`, explaining **what** each piece does,
**why** it is written that way, and **how** the pieces fit together in the
SGLang-JAX serving stack.

`moe.py` provides the **GMM-based (non-fused) Expert-Parallel MoE backend**
and the **weight-mapping utility** shared by all MoE backends. The production
backend for MiMo-V2-Flash is `FusedEPMoEV2` (in `fused_moe.py`); `EPMoE` here
is the fallback path and the reference implementation.

---

## Table of Contents

1. [Role in the MoE Backend Hierarchy](#1-role-in-the-moe-backend-hierarchy)
2. [Imports](#2-imports)
3. [EPMoE.__init__ — Weight Init and Mesh Setup](#3-epmoe__init--weight-init-and-mesh-setup)
4. [_detect_device_capabilities — Platform Probe](#4-_detect_device_capabilities--platform-probe)
5. [_normalize_scale_for_gmm — Scale Layout Normalization](#5-_normalize_scale_for_gmm--scale-layout-normalization)
6. [quantize_weights — Online and Static Quantization](#6-quantize_weights--online-and-static-quantization)
7. [__call__ — Expert-Parallel Dispatch via shard_map](#7-__call--expert-parallel-dispatch-via-shard_map)
8. [_forward — Per-Expert-Shard Computation](#8-_forward--per-expert-shard-computation)
9. [_gmm_compute — Three-GEMM SwiGLU via megablox gmm](#9-_gmm_compute--three-gemm-swiglu-via-megablox-gmm)
10. [_dispatch — Expert Offset within a Shard](#10-_dispatch--expert-offset-within-a-shard)
11. [_permute — Token Sorting by Expert Assignment](#11-_permute--token-sorting-by-expert-assignment)
12. [_unpermute — Weighted Aggregation of Expert Outputs](#12-_unpermute--weighted-aggregation-of-expert-outputs)
13. [_combine — Expert-Axis All-Reduce](#13-_combine--expert-axis-all-reduce)
14. [create_moe_weights_mapping — HF → JAX Weight Mapping](#14-create_moe_weights_mapping--hf--jax-weight-mapping)
15. [Complete Tensor Inventory](#15-complete-tensor-inventory)
16. [Summary and Key Design Decisions](#16-summary-and-key-design-decisions)

---

## 1. Role in the MoE Backend Hierarchy

`moe.py` defines three things:

| Export | Role |
|---|---|
| `EPMoE` | Non-fused GMM-based EP MoE layer (this file's core) |
| `FusedEPMoE`, `FusedEPMoEV2` | Re-exported from `fused_moe.py` for backward compatibility |
| `GateLogit`, `TopK` | Re-exported from `gate.py` for backward compatibility |
| `create_moe_weights_mapping` | Utility: generates HF → JAX weight mappings for any MoE backend |

Three MoE backends exist in the codebase:

| Backend | Class | Kernel | When used |
|---|---|---|---|
| `epmoe` | `EPMoE` (this file) | megablox `gmm` via `shard_map` | Fallback; CPU/GPU-compatible |
| `fused` | `FusedEPMoE` | Pallas fused kernel v1 | Older fused variant |
| `fused_v2` | `FusedEPMoEV2` | Pallas fused kernel v2 (Strix double-buffer) | MiMo production path |

`EPMoE` uses JAX's `shard_map` primitive to execute each device's expert
slice independently, then reduces across devices. This is conceptually clean
and portable, but requires more explicit control flow than the Pallas fused
backends.

---

## 2. Imports

```python
"""GMM-based Expert-Parallel MoE layer and weight mapping utilities."""

import math
from functools import partial

import jax
from flax import nnx
from jax import numpy as jnp
from jax import shard_map
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from sgl_jax.srt.eplb.expert_location import get_global_expert_location_metadata
from sgl_jax.srt.kernels.gmm.megablox_gmm_backend import gmm

# Re-export for backward compatibility: external code imports from this module.
from sgl_jax.srt.layers.fused_moe import FusedEPMoE, FusedEPMoEV2  # noqa: F401
from sgl_jax.srt.layers.gate import GateLogit, TopK  # noqa: F401
from sgl_jax.srt.utils.profiling_utils import named_scope
from sgl_jax.srt.utils.quantization.quantization_utils import (
    quantize_tensor,
    quantize_tensor_simple,
)
from sgl_jax.srt.utils.weight_utils import WeightMapping
```

**Key imports explained:**

- **`shard_map`** — JAX's SPMD primitive for writing per-device programs. Inside
  a `shard_map` body, each device runs the same Python function but sees only
  its local slice of the sharded inputs. Collective ops (`psum`, `psum_scatter`)
  communicate across devices. `EPMoE` uses `shard_map` over the `expert` axis
  so each device computes only its local expert slice.

- **`get_global_expert_location_metadata`** — EPLB (Expert Load Balancing) hook.
  When redundant experts are enabled (e.g., popular experts are replicated across
  devices), this returns a metadata object carrying `num_physical_experts` (which
  may be greater than `num_experts` if some experts are duplicated). `EPMoE` uses
  this to size its weight tensors correctly.

- **`gmm`** — The megablox Grouped Matrix Multiplication kernel. Takes a
  left-hand-side token matrix, a right-hand-side weight tensor, and per-group
  sizes, and computes a block-sparse batched GEMM where each group (expert) has
  its own weight block. This is more efficient than iterating over experts with
  separate dense GEMMs.

- **`FusedEPMoE, FusedEPMoEV2` (re-exports)** — Historical: before the backend
  split, all MoE code lived in `moe.py`. External code still does
  `from sgl_jax.srt.layers.moe import FusedEPMoEV2`. The `# noqa: F401`
  suppresses the "imported but unused" linter warning — these are intentional
  re-exports, not dead code.

- **`GateLogit, TopK` (re-exports)** — Same reason. `GateLogit` computes the
  router logits (`hidden @ gate_kernel`); `TopK` selects the top-K experts.
  Both live in `gate.py` but are surfaced here for backward compatibility.

- **`named_scope`** — A profiling decorator. Wraps the `__call__` method in a
  named XLA op scope, making it visible by name in profiler traces (e.g.
  TensorBoard, Perfetto). Has no effect on correctness.

- **`quantize_tensor` / `quantize_tensor_simple`** — Quantize a float32/bf16
  tensor to a lower dtype (e.g. FP8, INT8) and return the quantized value plus
  a scale factor. `quantize_tensor` supports block quantization (per-group
  scales); `quantize_tensor_simple` is a single-pass per-channel quantizer
  used for fast activation quantization in `_gmm_compute`.

- **`WeightMapping`** — Dataclass describing one HuggingFace checkpoint tensor →
  one JAX parameter mapping: target path, sharding spec, whether to transpose,
  and optional expert-concatenation axis. Used by `create_moe_weights_mapping`.

---

## 3. EPMoE.__init__ — Weight Init and Mesh Setup

```python
class EPMoE(nnx.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        ep_size: int,
        mesh: Mesh,
        intermediate_dim: int = 2048,
        weight_dtype: jnp.dtype = jnp.bfloat16,
        dtype: jnp.dtype = jnp.bfloat16,
        activation: str = "silu",
        layer_id: int = 0,
        quantization_config=None,
        physical_to_logical_map: "jax.Array | None" = None,
        pre_gather_quant_dtype=None,
    ):
```

**Parameters:**

| Parameter | Meaning |
|---|---|
| `hidden_size` | Hidden dimension (e.g. 4096 for Flash, 6144 for Pro) |
| `num_experts` | Number of logical experts (e.g. 256 for Flash, 384 for Pro) |
| `num_experts_per_tok` | Top-K experts selected per token (e.g. 8) |
| `ep_size` | Expert parallelism degree (e.g. 8 means 8 devices each hold `num_experts/8` experts) |
| `mesh` | The global JAX device mesh (axes: `"data"`, `"tensor"`) |
| `intermediate_dim` | Expert FFN hidden size (e.g. 2048) |
| `weight_dtype` | Dtype for stored weights (bf16 normally; overridden to FP8 by `quantize_weights`) |
| `dtype` | Runtime computation dtype (always bf16 for inference) |
| `activation` | Non-linearity name: `"silu"` (SwiGLU gate) or `"gelu"` |
| `layer_id` | Decoder layer index; used to look up EPLB metadata |
| `quantization_config` | Optional config providing weight/activation quantization dtypes |
| `physical_to_logical_map` | EPLB mapping: physical expert index → logical expert index |
| `pre_gather_quant_dtype` | If set, activations are quantized before the gather in `_gmm_compute` |

### Lines 44–62: Instance attribute initialization

```python
self.num_experts_per_tok = num_experts_per_tok
self.physical_to_logical_map = physical_to_logical_map
self.pre_gather_quant_dtype = pre_gather_quant_dtype

metadata = get_global_expert_location_metadata()
if metadata is not None and layer_id is not None:
    self.num_experts = metadata.num_physical_experts
else:
    self.num_experts = num_experts
```

**EPLB physical expert count.** When EPLB is active, popular experts are
replicated to additional devices. `num_physical_experts` ≥ `num_experts`. The
weight tensors are sized to `num_physical_experts` so each physical slot has its
own weight copy. If EPLB is off (metadata is None), `num_experts` is used as-is.

```python
self.intermediate_dim = intermediate_dim
self.weight_dtype = weight_dtype
self.dtype = dtype
self.layer_id = layer_id
self.ep_size = ep_size
self.original_mesh = mesh
self.mesh = mesh
self.activation = activation
self.hidden_size = hidden_size
```

`self.original_mesh` and `self.mesh` are initially the same. Keeping both
allows future code to distinguish the inference-time mesh (which might be
modified by sequence-parallel or other reconfigurations) from the mesh that
was passed in at construction.

### Lines 64–73: Quantization config extraction

```python
self.quantized_dtype = (
    quantization_config.get_moe_weight_dtype() if quantization_config else None
)
self.activation_quantized_dtype = (
    quantization_config.get_moe_activation_dtype() if quantization_config else None
)
self.weight_block_size = (
    getattr(quantization_config, "weight_block_size", None) if quantization_config else None
)
```

Three independent quantization knobs:

- **`quantized_dtype`**: dtype for weight quantization (e.g. `jnp.float8_e4m3fn`
  for FP8 weights). If None, weights stay in `weight_dtype` (bf16).
- **`activation_quantized_dtype`**: dtype for activation quantization. If set,
  activations are quantized to this dtype before the GEMM, enabling
  lower-precision matmuls.
- **`weight_block_size`**: `[block_n, block_k]` for block-wise quantization.
  Each `block_k × block_n` tile of a weight matrix has its own scale factor.
  If None, per-channel (row-wise) scales are used.

### Lines 75–88: EP divisibility check and derived sizes

```python
if self.num_experts % self.ep_size != 0:
    raise ValueError(...)
world_size = math.prod(self.mesh.shape.values())
self.tp_size = world_size // self.ep_size
self.experts_per_device = self.num_experts // self.ep_size
```

**Constraint:** experts must divide evenly across EP devices. If `num_experts=256`
and `ep_size=8`, each device owns 32 experts.

`world_size` = total devices in the mesh (e.g., `tp=8` × `dp=2` = 16).
`tp_size` = devices per expert group = `world_size / ep_size`. This is the
tensor-parallel degree within each EP shard: tokens at one expert shard are
further split across `tp_size` devices.

### Lines 83–93: MoE mesh construction

```python
devices = self.mesh.devices.flatten()
self.moe_mesh = jax.sharding.Mesh(
    devices.reshape(self.ep_size, self.tp_size),
    axis_names=("expert", "tensor"),
    axis_types=(jax.sharding.AxisType.Explicit, jax.sharding.AxisType.Explicit),
)

abstract_mesh = self.mesh.abstract_mesh
self.updated_mesh = abstract_mesh.update(
    axis_sizes=(self.ep_size, self.tp_size), axis_names=("expert", "tensor")
)
```

**Why a second mesh?** The incoming `mesh` has axes `("data", "tensor")` —
matching the serving stack's data-parallel and tensor-parallel layout. Expert
parallelism needs a different axis called `"expert"` so that `shard_map`'s
`in_specs` can address the expert dimension explicitly.

`self.moe_mesh` reshapes the flat device list into `(ep_size, tp_size)` and
renames the axes `"expert"` and `"tensor"`. The physical devices are the same
objects — just addressed differently.

`self.updated_mesh` is an **abstract mesh** version of the same idea, used with
`jax.sharding.use_abstract_mesh(...)` context manager. Abstract meshes allow
resharding operations to be expressed without attaching to concrete devices,
which is required by JAX's newer sharding APIs.

**Important:** The axis named `"tensor"` is shared between the original mesh and
the moe_mesh. This is what allows `EPMoE` to correctly propagate tensor-parallel
sharding through the `shard_map` boundary.

### Lines 95–128: Weight parameter initialization

```python
with jax.sharding.use_abstract_mesh(self.updated_mesh):
    self.wi_0 = nnx.Param(
        jax.random.normal(
            jax.random.PRNGKey(0),
            (self.num_experts, hidden_size, intermediate_dim),
            dtype=weight_dtype,
            out_sharding=P("expert", None, "tensor"),
        )
    )
    self.wi_1 = nnx.Param(...)  # same shape as wi_0
    self.wo = nnx.Param(
        jax.random.normal(
            jax.random.PRNGKey(0),
            (self.num_experts, intermediate_dim, hidden_size),
            dtype=weight_dtype,
            out_sharding=P("expert", "tensor", None),
        )
    )
    self.wi_0_scale = None
    self.wi_1_scale = None
    self.wo_scale = None
```

**Weight names:**

| Attribute | Role | Shape | HF equivalent |
|---|---|---|---|
| `wi_0` | Gate projection (SwiGLU gate branch) | `(E, hidden, intermediate)` | `gate_proj.weight` (transposed) |
| `wi_1` | Up projection (SwiGLU up branch) | `(E, hidden, intermediate)` | `up_proj.weight` (transposed) |
| `wo` | Down projection | `(E, intermediate, hidden)` | `down_proj.weight` (transposed) |

The naming `wi_0/wi_1/wo` is megablox convention: `wi` = "weight input" (gate
and up), `wo` = "weight output" (down).

**Weight layout `[E, k, n]`:** Each weight is `(num_experts, in_features, out_features)`.
This is the **transpose** of the HuggingFace convention `(out_features, in_features)`.
The `k` dimension (contraction axis) comes before `n` (output axis), matching the
megablox `gmm` kernel's expected layout.

**Sharding `P("expert", None, "tensor")`** for `wi_0`/`wi_1`:
- `"expert"` axis: splits across EP devices — each device holds `E/ep_size` experts.
- `None` axis: `hidden_size` is unsharded (replicated on the tensor axis).
- `"tensor"` axis: `intermediate_dim` is split across TP devices within each EP group.

**Sharding `P("expert", "tensor", None)`** for `wo`:
- The contraction axis (`intermediate_dim`) is sharded across `"tensor"`.
- The output axis (`hidden_size`) is unsharded.

This is the row-parallel convention for `wo`: each TP device computes a partial
sum over `intermediate_dim/tp_size` input channels, requiring an all-reduce to
get the full output (handled later by `psum` in `_forward`).

**Scale initialization:** Scales are `None` at init time. They are allocated either
by `quantize_weights(is_static=True)` (loading a pre-quantized checkpoint) or
by `quantize_weights(is_static=False)` (online quantization of loaded bf16 weights).
The `None` default allows the `gmm` kernel to skip the scale multiply when no
quantization is configured.

**Random init with `PRNGKey(0)`:** These weights are immediately overwritten by
`load_weights`. The `normal()` call allocates device memory with the correct shape
and sharding; the values don't matter.

---

## 4. _detect_device_capabilities — Platform Probe

```python
def _detect_device_capabilities(self):
    try:
        devices = jax.devices()
        is_cpu_only = all(device.platform == "cpu" for device in devices)
        can_use_ragged = not is_cpu_only and hasattr(jax.lax, "ragged_all_to_all")

        device_types = [device.platform for device in devices]
        primary_device = device_types[0] if device_types else "unknown"

        return can_use_ragged, primary_device
    except Exception as _:
        return False, "cpu"
```

**Purpose:** Checks whether the runtime has TPU-specific primitives available.
`jax.lax.ragged_all_to_all` is a TPU-only JAX op for variable-size all-to-all
collectives. If it exists (TPU environment) and devices are not CPU-only, the
more efficient ragged path can be used.

**Note:** This method is defined but not called anywhere in the current `EPMoE`
implementation. It is a dead-code remnant from an earlier EP design that used
ragged all-to-all. The production `EPMoE` uses `shard_map` + `gmm` instead.

---

## 5. _normalize_scale_for_gmm — Scale Layout Normalization

```python
def _normalize_scale_for_gmm(
    self,
    scale: jax.Array | None,
    weight: jax.Array,
    *,
    scale_name: str,
) -> jax.Array | None:
    """Normalize offline/runtime scale tensors to GMM's 4D layout."""
```

The `gmm` kernel expects scales in a specific 4D layout:

```
[E, k_blocks, 1, out_dim]
 ^      ^      ^    ^
 |      |      |    output channels (n)
 |      |      singleton broadcast dim
 |      number of quantization blocks along k-axis
 expert dimension
```

But checkpoint files and different quantization schemes produce scales in many
different layouts. This method normalizes all of them to the GMM contract.

### Accepted input layouts

**Line 169–196: 4D input (already GMM-ready or nearly so)**

```python
if scale.ndim == 4:
    if scale.shape[0] != num_experts or scale.shape[2] != 1 or scale.shape[3] != out_dim:
        raise ValueError(...)
    ...
    return scale
```

If the scale already has 4 dims with the right structure, return it unchanged.
The validation checks ensure the expert count, singleton dim, and output dim match
the weight. A FIXME comment notes a sharding annotation issue that will surface
with stricter JAX versions (jax 0.10.x) when `ep_size == world_size`.

**Lines 198–199: 2D per-channel scale `[E, out_dim]`**

```python
if scale.ndim == 2 and scale.shape == (num_experts, out_dim):
    return scale[:, None, None, :]
```

The simplest case: one scale per expert per output channel. Insert two singleton
dims at positions 1 and 2 to become `[E, 1, 1, out_dim]`, satisfying the GMM
4D contract with `k_blocks=1` (per-channel, not per-block).

**Lines 201–245: 3D scale — three sub-cases**

```python
if scale.ndim == 3:
    if scale.shape == (num_experts, 1, out_dim):
        return scale[:, :, None, :]
```

`[E, 1, out_dim]` → insert singleton at axis 2 → `[E, 1, 1, out_dim]`.

```python
    if scale.shape == (num_experts, out_dim, expected_k_blocks):
        scale_gmm = jnp.transpose(scale, (0, 2, 1))[:, :, None, :]
        return jax.sharding.reshard(scale_gmm, final_scale_sharding)
```

**Offline block-quantization format:** HuggingFace stores block-quant scales as
`[E, out_blocks, in_blocks]` (output-major). The GMM kernel expects
`[E, k_blocks, 1, out_dim]` (input-major, expanded per output channel). The
`transpose(0,2,1)` swaps the block axes, `[:, :, None, :]` inserts the singleton,
and `reshard` places the result on the correct device axis.

```python
    if scale.shape == (num_experts, expected_out_blocks, expected_k_blocks):
        out_block_ids = jnp.arange(out_dim, dtype=jnp.int32) // block_size_out
        scale_per_out = scale.at[:, out_block_ids, :].get(...)
        scale_gmm = jnp.transpose(scale_per_out, (0, 2, 1))[:, :, None, :]
        return jax.sharding.reshard(scale_gmm, final_scale_sharding)
```

**Coarser block-quantization:** Scale has one entry per `block_out × block_k`
tile, not per output channel. The `scale.at[:, out_block_ids, :].get(...)` call
expands the coarse block scale to a per-channel layout using gather, then the
same transpose+reshape path applies. `out_block_ids[c]` = which output block
channel `c` belongs to, so every channel in the same block gets the same scale.

```python
    if scale.shape == (num_experts, expected_k_blocks, out_dim):
        return scale[:, :, None, :]
```

Already in `[E, k_blocks, out_dim]` order; just insert the singleton dim.

---

## 6. quantize_weights — Online and Static Quantization

```python
def quantize_weights(self, is_static: bool = False):
    """Quantize MoE weights in-place or initialize params for static loading."""
    if self.quantized_dtype is None:
        return
```

**Early return** if no quantization is configured. Most bf16 inference uses no
weight quantization and this method is a no-op.

### Helper: `_get_block_size_k`

```python
    def _get_block_size_k(*, hidden_size, intermediate_dim, weight_block_size) -> int | None:
```

Extracts the K-dimension block size from `weight_block_size = [block_n, block_k]`.
EPMoE quantizes along axis 1 (the contraction/K dimension in `[E, k, n]` layout),
so only `block_k` is relevant here. Validates that `hidden_size` and
`intermediate_dim` are both divisible by `block_k`.

### Static path (`is_static=True`)

```python
    with jax.set_mesh(self.moe_mesh):
        if is_static:
            # Allocate zero-filled placeholder scale tensors
            self.wi_0_scale = nnx.Param(
                jnp.zeros((num_experts, k_blocks_wi, 1, intermediate_dim), ...),
                out_sharding=wi_scale_sharding,
            )
            ...
            return
```

**When used:** A pre-quantized checkpoint already has FP8 weights and explicit
scale tensors on disk. `quantize_weights(is_static=True)` creates the scale
`nnx.Param` slots with the correct shapes and shardings so the weight loader can
fill them. No quantization math is done here — the loader writes the real scales
directly.

**Scale sharding:**

| Scale | Sharding | Reason |
|---|---|---|
| `wi_0_scale`, `wi_1_scale` | `P("expert", None, None, "tensor")` | Output dim (`n`) is TP-sharded |
| `wo_scale` | `P("expert", None, None, None)` | Output dim (`hidden_size`) is fully replicated |

`del self.wi_0_scale` before creating the new one: NNX tracks parameters as
instance attributes. Doing `self.foo = new_value` when `foo` already exists can
confuse NNX's graph traversal because the old parameter object still exists in
Python's memory. The explicit delete removes it from NNX's view before re-binding.

### Dynamic path (`is_static=False`)

```python
        # Quantize weights along k-dim (axis=1 in [g, k, n] layout)
        w0_value, w0_scale = quantize_tensor(self.quantized_dtype, self.wi_0.value, axis=1, block_size=block_size_k)
        w1_value, w1_scale = quantize_tensor(self.quantized_dtype, self.wi_1.value, axis=1, ...)
        wo_value, wo_scale = quantize_tensor(self.quantized_dtype, self.wo.value, axis=1, ...)

        self.wi_0 = nnx.Param(w0_value, out_sharding=P("expert", None, "tensor"))
        self.wi_1 = nnx.Param(w1_value, out_sharding=P("expert", None, "tensor"))
        self.wo  = nnx.Param(wo_value,  out_sharding=P("expert", "tensor", None))
```

**When used:** Weights were loaded from a bf16 checkpoint. Online quantization
converts them to `quantized_dtype` (e.g. FP8) in-place, freeing HBM. The entire
quantization runs in JAX (no Python loops over experts), so XLA can fuse and
pipeline the quantization ops.

`axis=1` in `[E, k, n]` quantizes along the K (hidden) dimension. Scales have
shape `[E, k_blocks, n]` (block quant) or `[E, n]` (per-channel, when
`block_size_k is None`).

**Scale reshape to GMM contract:**

```python
        if block_size_k is not None:
            w0_scale = w0_scale[:, :, None, :]    # [E, k_blocks, n] → [E, k_blocks, 1, n]
        else:
            w0_scale = w0_scale.reshape(w0_scale.shape[0], 1, 1, w0_scale.shape[1])
            # [E, n] → [E, 1, 1, n]
```

Both paths produce the GMM 4D layout `[E, k_blocks, 1, n]`. For per-channel
quantization, `k_blocks=1`.

---

## 7. __call__ — Expert-Parallel Dispatch via shard_map

```python
@named_scope
def __call__(
    self,
    hidden_states,
    topk_weights,
    topk_ids,
    *,
    out_sharding: jax.sharding.NamedSharding | None = None,
) -> jax.Array:
```

`@named_scope` wraps the method body in an XLA profiling scope named `"EPMoE"`.

**Inputs:**

| Tensor | Shape | Meaning |
|---|---|---|
| `hidden_states` | `(T, H)` | Token hidden states; `T` = tokens on this DP rank, `H` = `hidden_size` |
| `topk_weights` | `(T, K)` | Softmax-normalized routing weights for the top-K experts |
| `topk_ids` | `(T, K)` | Expert indices for the top-K selected experts; values in `[0, num_experts)` |

**Lines 421–432: Output sharding**

```python
    if out_sharding is None:
        out_sharding = jax.sharding.NamedSharding(self.mesh, P(*([None] * hidden_states.ndim)))
    out_specs = P(
        *[
            "tensor" if (s == "tensor" or (isinstance(s, tuple) and "tensor" in s)) else None
            for s in out_sharding.spec
        ]
    )
    scatter_on_tensor = "tensor" in out_specs
```

The caller (e.g. `MiMoV2Moe`) passes `out_sharding` specifying how the output
should be sharded on `self.mesh` (axes: `data`, `tensor`). But `shard_map` runs
on `self.moe_mesh` (axes: `expert`, `tensor`). The two meshes share the `"tensor"`
axis name, so the translation just extracts which dimensions need `"tensor"`
sharding and ignores `"data"` (which has no meaning inside `shard_map`).

`scatter_on_tensor = True` means the output should be scatter-reduced (RS
pattern) along the `"tensor"` axis instead of all-reduced. This is the
Sequence Parallel path where each device accumulates its share of the output.

**Lines 436–439: Reshard inputs to moe_mesh**

```python
    with jax.sharding.use_abstract_mesh(self.updated_mesh):
        hidden_states_reshard = jax.sharding.reshard(hidden_states, P(None))
        topk_weights_reshard  = jax.sharding.reshard(topk_weights,  P(None))
        topk_ids_reshard      = jax.sharding.reshard(topk_ids,      P(None))
```

The `use_abstract_mesh` context tells JAX to interpret sharding specs relative to
`self.updated_mesh` (axes: `expert`, `tensor`). `P(None)` = fully replicated on
all axes. This ensures `hidden_states`, `topk_weights`, and `topk_ids` are
replicated across the `expert` axis before entering `shard_map`, so every expert
device sees all tokens and can route them locally.

**Lines 441–456: Scale normalization**

```python
        w0_scale = self._normalize_scale_for_gmm(self.wi_0_scale.value if ..., ...)
        w1_scale = self._normalize_scale_for_gmm(self.wi_1_scale.value if ..., ...)
        wo_scale = self._normalize_scale_for_gmm(self.wo_scale.value if ..., ...)
```

Scales from disk may be in various layouts. `_normalize_scale_for_gmm` converts
each to `[E, k_blocks, 1, out_dim]` before passing to `shard_map`. Inside
`shard_map` each device sees a local slice `[experts_per_device, k_blocks, 1, out_dim]`.

**Lines 458–493: shard_map call**

```python
        result = shard_map(
            partial(self._forward, scatter_on_tensor=scatter_on_tensor),
            mesh=self.moe_mesh,
            in_specs=(
                P(None),              # hidden_states: replicated
                P(None),              # topk_weights: replicated
                P(None),              # topk_ids: replicated
                P("expert", None, "tensor"),   # wi_0
                P("expert", None, "tensor"),   # wi_1
                P("expert", "tensor", None),   # wo
                P("expert", None, None, "tensor"),  # w0_scale
                P("expert", None, None, "tensor"),  # w1_scale
                P("expert", None, None, None),      # wo_scale
                P("expert", None, "tensor"),   # bias0 (None)
                P("expert", None, "tensor"),   # bias1 (None)
                P("expert", None, None),       # biasO (None)
            ),
            out_specs=out_specs,
            check_vma=False,
        )(hidden_states_reshard, topk_weights_reshard, topk_ids_reshard,
          self.wi_0.value, self.wi_1.value, self.wo.value,
          w0_scale, w1_scale, wo_scale,
          None, None, None)
```

**What `shard_map` does:**
1. Slices each input along its specified `"expert"` axis: each device receives
   `wi_0[local_expert_start:local_expert_end, ...]`, i.e. its `experts_per_device`
   weight rows.
2. Inputs marked `P(None)` (tokens) are replicated on every device — each device
   receives all tokens but only its local expert weights.
3. Calls `self._forward(...)` once per device in parallel.
4. Collects the outputs and places them according to `out_specs`.

**`check_vma=False`:** Disables verification of virtual mesh alignment. The
moe_mesh and the abstract_mesh are constructed to be consistent, but the
explicit layout makes the vma check unnecessary and it can be slow.

**Three `None` biases:** The `gmm` kernel accepts optional per-expert biases
for each matmul. They are not used in the MiMo architecture and are passed as
`None`. The `in_specs` entries for them (`P("expert", None, "tensor")` etc.)
are still required for `shard_map` to know the in-spec even when the values
are `None`.

**Lines 495–498: Reshard output back to original mesh**

```python
    return jax.sharding.reshard(result, out_sharding)
```

The `shard_map` result is on `self.moe_mesh` (axes: `expert`, `tensor`).
`reshard` converts it to the caller's expected layout on `self.mesh`
(axes: `data`, `tensor`). Downstream ops (residual addition, layernorm) run
on `self.mesh`, so this reshard is required for correctness.

---

## 8. _forward — Per-Expert-Shard Computation

```python
def _forward(
    self,
    hidden_states, topk_weights, topk_ids,
    w0_weights, w1_weights, wo_weights,
    w0_kernel_scale=None, w1_kernel_scale=None, wo_kernel_scale=None,
    w0_kernel_bias=None, w1_kernel_bias=None, wo_kernel_bias=None,
    *, scatter_on_tensor: bool = False,
):
```

This function runs **inside `shard_map`** — on each device independently. The
device sees a local slice of the weight tensors but all tokens.

**Lines 517–523: Tensor shape normalization**

```python
    expert_shard_id = jax.lax.axis_index("expert")
    if hidden_states.ndim == 2:
        total_tokens = hidden_states.shape[0]
        batch_size, seq_len = 1, total_tokens
    else:
        batch_size, seq_len = hidden_states.shape[0], hidden_states.shape[1]
        total_tokens = batch_size * seq_len
```

`axis_index("expert")` returns this device's position along the `"expert"` axis
(0, 1, …, ep_size-1). This is the device's expert shard index — which contiguous
block of experts it holds.

`EPMoE` supports both 2D `(tokens, hidden)` and 3D `(batch, seq, hidden)` inputs,
normalizing to `batch_size` and `seq_len` for shape tracking.

**Lines 525–531: Permute → dispatch → compute → unpermute**

```python
    inputs_2d, token_indices, sorted_selected_experts, weights, group_sizes = self._permute(
        hidden_states, topk_ids, topk_weights
    )
    group_sizes = group_sizes.astype(jnp.int32)
    group_offset = self._dispatch(group_sizes, expert_shard_id)
    intermediate_output = self._gmm_compute(inputs_2d, token_indices, group_sizes, ...)
    output = self._unpermute(intermediate_output, sorted_selected_experts, weights, ...)
```

The overall computation follows a four-phase pipeline:

```
tokens → [_permute] → sorted token-expert pairs
       → [_gmm_compute] → expert outputs (sorted order)
       → [_unpermute] → weighted sum over experts, back to token order
       → [psum/psum_scatter] → all-reduce across TP devices
       → [_combine] → all-reduce across EP devices
```

**Lines 557–567: TP reduce and EP combine**

```python
    if self.tp_size > 1:
        if scatter_on_tensor:
            output = jax.lax.psum_scatter(output, "tensor", scatter_dimension=0, tiled=True)
        else:
            output = jax.lax.psum(output, "tensor")
    if self.ep_size > 1:
        output = self._combine(output)
    return output
```

After computing the weighted expert output, two reductions are needed:

1. **TP reduce (`"tensor"` axis):** `wo`'s output features were split across
   `tp_size` devices (row-parallel projection). Each device computed a partial
   dot-product that must be summed. `psum` is an all-reduce (AR); `psum_scatter`
   is a reduce-scatter (RS) for Sequence Parallel — each device keeps only
   `1/tp_size` of the reduced output, sharded along the token dimension.

2. **EP combine (`"expert"` axis):** Each expert device computed output for
   tokens that were routed to its experts. Tokens routed to other devices'
   experts contribute zero output on this device. `_combine` does a `psum`
   over the `"expert"` axis so every device gets the sum of all expert outputs
   for all its tokens.

**Why EP psum is correct:** After `_permute` and `_gmm_compute`, a device holds
non-zero values only for the tokens that were assigned to its experts. For tokens
not assigned to this device's experts, the output is zero (initialized as such in
`_gmm_compute`). The `psum` across `"expert"` then accumulates all devices'
contributions, producing the correct total output for every token.

---

## 9. _gmm_compute — Three-GEMM SwiGLU via megablox gmm

```python
def _gmm_compute(
    self, inputs_2d, token_indices, group_sizes,
    w0_kernel, w1_kernel, wo_kernel, group_offset,
    w0_kernel_scale=None, w1_kernel_scale=None, wo_kernel_scale=None,
    w0_kernel_bias=None, w1_kernel_bias=None, wo_kernel_bias=None,
):
```

**Lines 586–588: Empty batch guard**

```python
    if token_indices.shape[0] == 0:
        return jnp.zeros((0, wo_kernel.shape[-1]), dtype=inputs_2d.dtype)
```

If no tokens were routed to this device's experts (possible with imbalanced
routing), return a zero tensor immediately. Without this guard, `gmm` would
receive an empty input and may behave unexpectedly.

**Lines 590–599: Optional activation quantization before gather**

```python
    pre_gather_q = getattr(self, "pre_gather_quant_dtype", None)
    if pre_gather_q is not None:
        x_q, x_scale = quantize_tensor_simple(inputs_2d, pre_gather_q, dim=-1)
        x = x_q[token_indices]
        x_scale = x_scale[token_indices]
        x = (x.astype(jnp.float32) * x_scale).astype(self.dtype)
    else:
        x = inputs_2d[token_indices].astype(self.dtype)
```

`token_indices` maps each sorted position back to the original token. The gather
`inputs_2d[token_indices]` reorders tokens to match their assigned expert order.

**Indexed GMM pattern:** Instead of materializing a full `[M*top_k, D]` sorted
tensor in `_permute` and then feeding it to `gmm`, the gather is deferred here.
XLA can fuse the gather with the GEMM, avoiding a temporary tensor that would
cost `M * top_k * D * dtype_bytes` of HBM (for top_k=8, D=4096, M=1024 at bf16:
~67 MB just for the intermediate).

If `pre_gather_quant_dtype` is set, activations are quantized to that dtype
before the gather, then immediately dequantized after. This can enable faster
data movement and GEMM on quantization-aware hardware.

**Lines 610–638: Two-GEMM gate+up (SwiGLU)**

```python
    gmm_kwargs = dict(
        group_sizes=group_sizes,
        preferred_element_type=self.dtype,
        group_offset=group_offset,
        maybe_quantize_lhs=act_q_dtype is not None,
        acc_dtype=jnp.float32,
    )

    layer_w0 = gmm(lhs=x, rhs=w0_kernel, rhs_scale=w0_kernel_scale, ..., **gmm_kwargs)
    layer_w1 = gmm(lhs=x, rhs=w1_kernel, rhs_scale=w1_kernel_scale, ..., **gmm_kwargs)
```

`group_sizes` tells the GMM kernel how many tokens go to each expert in this shard.
`group_offset` is the starting expert index on this device (e.g. device 0 has
experts 0–31, device 1 has 32–63 → `group_offset=32`).

`gmm` runs a batched GEMM: for each expert `e`, compute
`output_e = x_group_e @ w0_kernel[e]`. The `group_sizes` array tells the kernel
where one expert's token block ends and the next begins in the sorted `x`.

`preferred_element_type=self.dtype` (bf16) sets the output element type.
`acc_dtype=jnp.float32` uses float32 accumulation for numerical stability, then
rounds down to bf16 for the output.

`maybe_quantize_lhs=True` enables in-kernel activation quantization when
`act_q_dtype` is set — the LHS (`x`) is quantized to `act_q_dtype` inside the
`gmm` kernel before the matmul.

**Lines 641–648: SwiGLU activation**

```python
    if self.activation == "silu":
        layer_act = jax.nn.silu(layer_w0)
    elif self.activation == "gelu":
        layer_act = jax.nn.gelu(layer_w0)
    intermediate_layer = jnp.multiply(layer_act, layer_w1)
```

SwiGLU: `output = silu(gate) * up`. `layer_w0` = gate branch output, `layer_w1` =
up branch output. The element-wise multiply produces the gated activation.
Shape: `[M*top_k, intermediate_dim]` (sorted token order, local experts only).

**Lines 650–658: Down projection**

```python
    return gmm(
        lhs=intermediate_layer,
        rhs=wo_kernel,
        rhs_scale=wo_kernel_scale,
        zero_initialize=True,
        ...
    )
```

`intermediate_layer @ wo_kernel` maps `intermediate_dim → hidden_size`.
`zero_initialize=True` initializes the output accumulator to zero — needed
because tokens routed to other experts produce no contribution here, so their
output slots should be zero (not random memory).

---

## 10. _dispatch — Expert Offset within a Shard

```python
def _dispatch(self, group_sizes, expert_shard_id):
    if self.ep_size <= 1:
        return jnp.array(0, dtype=jnp.int32)
    group_offset = jnp.array(expert_shard_id * self.experts_per_device, dtype=jnp.int32)
    return group_offset
```

**Purpose:** Tells the `gmm` kernel what the global expert index offset is for
this device. If device 2 holds experts 64–95 (with `experts_per_device=32`),
`group_offset = 2 * 32 = 64`. The `gmm` kernel uses this to correctly look up
`group_sizes[group_offset:group_offset+experts_per_device]` for the local experts.

With `ep_size=1` (no EP), all experts are on one device and the offset is 0.

---

## 11. _permute — Token Sorting by Expert Assignment

```python
def _permute(self, inputs, top_k_indices, top_k_weights):
```

**Purpose:** Sorts tokens into contiguous groups by their assigned expert,
producing the layout that `gmm` expects.

**Lines 692–697: Flatten to 2D**

```python
    if len(inputs_shape) == 2:
        inputs_2d = inputs
        bsz_times_seq_len = inputs_shape[0]
    else:
        bsz_times_seq_len = inputs_shape[0] * inputs_shape[1]
        inputs_2d = jnp.reshape(inputs, (bsz_times_seq_len, inputs_shape[-1]))
```

Normalize 3D `(batch, seq, hidden)` input to 2D `(tokens, hidden)`.

**Lines 700–707: Sort tokens by expert**

```python
    flatten_selected_experts = jnp.ravel(top_k_indices)
    sorted_selected_experts = jnp.argsort(flatten_selected_experts, stable=True)
    token_indices = sorted_selected_experts // self.num_experts_per_tok
    group_sizes = jnp.bincount(flatten_selected_experts, length=self.num_experts)
```

`top_k_indices` has shape `(T, K)` where `T=tokens, K=top_k`. Flattening gives
`(T*K,)` expert IDs, one per token-expert slot. `argsort` produces indices that
sort this flat array in ascending expert order.

**`token_indices`:** Because `top_k_indices` is laid out as `[tok_0_exp_0, tok_0_exp_1, ..., tok_T_exp_K]`,
dividing the sorted position by `K` gives the original token index. This lets
`_gmm_compute` gather the correct input rows without materializing the full
sorted hidden state tensor.

**`group_sizes`:** `bincount` counts how many of the `T*K` token-expert assignments
went to each expert. For `num_experts=256`, this produces a length-256 array where
`group_sizes[e]` = number of token-expert slots assigned to expert `e`. The `gmm`
kernel uses this to know where each expert's computation block begins and ends in
the sorted token list.

**Returns:**

| Value | Shape | Meaning |
|---|---|---|
| `inputs_2d` | `(T, H)` | Original hidden states, 2D |
| `token_indices` | `(T*K,)` | Which original token each sorted slot maps to |
| `sorted_selected_experts` | `(T*K,)` | The argsort indices (used to unsort in `_unpermute`) |
| `top_k_weights` | `(T, K)` | Routing weights (passed through, not sorted) |
| `group_sizes` | `(E,)` | Token count per expert |

---

## 12. _unpermute — Weighted Aggregation of Expert Outputs

```python
def _unpermute(self, intermediate, sorted_selected_experts, weights, batch_size, seq_len):
```

**Purpose:** Reverses the permutation and computes the weighted sum of
expert outputs for each token.

**Lines 718–727: Length correction**

```python
    if actual_tokens != expected_tokens:
        if actual_tokens > expected_tokens:
            intermediate = intermediate[:expected_tokens]
        else:
            padding_size = expected_tokens - actual_tokens
            padding = jnp.zeros((padding_size, intermediate.shape[1]), ...)
            intermediate = jnp.concatenate([intermediate, padding], axis=0)
```

The `gmm` kernel may return a slightly different token count than `sorted_selected_experts`
due to internal alignment padding (the megablox backend pads the LHS to its
required alignment and may return extra rows). This guard trims or pads to ensure
the unsort step has matching shapes.

**Lines 729–730: Unsort**

```python
    argsort_indices = jnp.argsort(sorted_selected_experts, stable=True)
    unsort_intermediate = jnp.take(intermediate, indices=argsort_indices, axis=0)
```

`argsort(sorted_selected_experts)` gives the inverse permutation: position `i` in
the sorted list corresponds to which position in the original layout. Applying
this index to `intermediate` restores tokens to their original order.

**Lines 732–748: Weighted expert combination**

```python
    reshaped_weights = jnp.reshape(weights, (total_tokens, self.num_experts_per_tok))
    reshaped_intermediate = jnp.reshape(
        unsort_intermediate,
        (total_tokens, self.num_experts_per_tok, -1),
    )
    intermediate_fp32 = reshaped_intermediate.astype(jnp.float32)
    weights_fp32 = reshaped_weights.astype(jnp.float32)

    output = jnp.einsum("BKE,BK -> BE", intermediate_fp32, weights_fp32)
```

After unsorting, `unsort_intermediate` has shape `(T*K, H)`. Reshaping to
`(T, K, H)` groups the K expert outputs for each token together.

`einsum("BKE,BK -> BE")`: for each token `B`, sum the `K` expert outputs
`BKE` weighted by `BK`, producing the final `H`-dimensional output `BE`.
This is the MoE weighted combination: `output[t] = Σ_k weight[t,k] * expert_output[t,k]`.

**fp32 accumulation:** Both `weights` and `intermediate` are cast to float32 for
the einsum. The weighted sum is a numerically sensitive reduction; accumulating in
bf16 would lose precision, especially for top-K=8 where 8 values are summed.
The result is cast back to `self.dtype` (bf16) at the end.

---

## 13. _combine — Expert-Axis All-Reduce

```python
def _combine(self, data):
    return jax.lax.psum(data, "expert")
```

After `_unpermute`, each device holds the weighted sum of outputs from its own
expert shard only. Tokens routed to other devices' experts contribute zero. This
`psum` across the `"expert"` axis accumulates all devices' contributions,
giving each device the correct total output for all of its tokens.

**Why not `psum_scatter`?** Unlike the TP reduce, the EP reduce is a true
all-reduce (AR): every device needs the complete result for all tokens, because
different tokens may be served by experts on different devices, and the host device
(which owns the KV cache state for a token) needs the complete output to proceed.

---

## 14. create_moe_weights_mapping — HF → JAX Weight Mapping

```python
def create_moe_weights_mapping(
    prefix: str,
    target_prefix: str,
    num_experts: int,
    expert_type_names: tuple[str, str, str] = ("gate_proj", "up_proj", "down_proj"),
    expert_concat_axis_map: dict[str, int] = None,
    moe_backend: str = "epmoe",
    moe_path: str = "mlp",
    source_expert_pattern: str = "experts.{i}",
    physical_to_logical_map=None,
) -> dict:
    """Generate a unified mapping dictionary for MoE layer expert weights."""
```

**Purpose:** HuggingFace checkpoints store expert weights as `num_experts`
separate tensors, one per expert. JAX `EPMoE` and `FusedEPMoE` store them as
a single stacked tensor per projection type. This function generates the
`WeightMapping` entries that instruct the weight loader to gather and stack
those individual tensors.

**Parameters:**

| Parameter | Example | Meaning |
|---|---|---|
| `prefix` | `"model.layers.5"` | HF key prefix for this layer |
| `target_prefix` | `"model.layers[5]"` | JAX parameter path prefix |
| `num_experts` | `256` | Number of logical experts |
| `expert_type_names` | `("gate_proj", "up_proj", "down_proj")` | HF source weight names |
| `expert_concat_axis_map` | `{"gate_proj": 0}` | If an expert weight needs concatenation along a specific axis |
| `moe_backend` | `"fused_v2"` | Determines target attribute names and sharding |
| `moe_path` | `"mlp"` | Sub-path within each layer |
| `source_expert_pattern` | `"experts.{i}"` | Pattern for each expert's HF key (`.format(i=i)`) |
| `physical_to_logical_map` | `np.array([0,1,2,...])` | EPLB mapping |

### Lines 776–788: Backend-specific target attribute names

```python
    if moe_backend == "epmoe":
        expert_type_map = {
            expert_type_names[0]: "wi_0",
            expert_type_names[1]: "wi_1",
            expert_type_names[2]: "wo",
        }
    elif moe_backend in ("fused", "fused_v2"):
        expert_type_map = {
            expert_type_names[0]: "w1",
            expert_type_names[1]: "w3",
            expert_type_names[2]: "w2",
        }
```

`EPMoE` uses `wi_0/wi_1/wo` (megablox naming). `FusedEPMoE`/`FusedEPMoEV2` use
`w1/w3/w2`. This mapping translates the HF source names (e.g. `gate_proj`) to the
correct JAX parameter attribute for each backend.

**Why `w1/w3/w2` not `w1/w2/w3`?** In the fused backends, the convention is:
`w1` = gate (SwiGLU gate branch), `w3` = up (SwiGLU up branch), `w2` = down.
This naming originates from the Pallas kernel design where w1 and w3 are computed
together in GEMM1 and w2 is GEMM2.

### Lines 795–831: Build WeightMapping entries

```python
    for source_name, target_name in expert_type_map.items():
        target_path_base = f"{target_prefix}.{moe_path}.{target_name}"
        expert_keys = [
            f"{prefix}.{moe_path}.{source_expert_pattern.format(i=i)}.{source_name}.weight"
            for i in range(num_experts)
        ]
```

For each projection type (`gate_proj/wi_0`, `up_proj/wi_1`, `down_proj/wo`):
- `target_path_base` = the stacked JAX parameter (e.g. `"model.layers[5].mlp.wi_0"`)
- `expert_keys` = list of `num_experts` HF tensor keys, one per expert
  (e.g. `"model.layers.5.mlp.experts.0.gate_proj.weight"`, ..., `"experts.255.gate_proj.weight"`)

```python
        if moe_backend == "epmoe":
            sharding = ("expert", "tensor", None) if target_name == "wo" else ("expert", None, "tensor")
            transpose = True
        elif moe_backend in ("fused", "fused_v2"):
            sharding = (("data", "tensor"), None, None)
            transpose = True
```

**Sharding for `epmoe`:**
- `wi_0/wi_1`: `P("expert", None, "tensor")` — experts split on `"expert"` axis, output
  features split on `"tensor"`.
- `wo`: `P("expert", "tensor", None)` — experts split on `"expert"`, input features
  split on `"tensor"` (row-parallel).

**Sharding for `fused/fused_v2`:** `(("data", "tensor"), None, None)` — the expert
dimension is sharded across the full EP mesh (product of `"data"` and `"tensor"`
axes), the other dims are unsharded. The fused kernels use their own internal
all-to-all and don't need the `"expert"`/`"tensor"` split used by EPMoE.

**`transpose=True`** for both: HF stores weights as `(out, in)`, but EPMoE's
`[E, k, n]` layout expects `(in, out)`. The weight loader transposes each
`(out, in)` expert weight before stacking into `[E, in, out]`.

```python
        mappings[f"__MOE_EXPERTS__{target_path_base}"] = WeightMapping(
            target_path=[target_path_base] + expert_keys,
            sharding=sharding,
            transpose=transpose,
            concat_axis=concat_axis,
            physical_to_logical_map=physical_to_logical_map,
        )
```

**`__MOE_EXPERTS__` prefix:** A sentinel that tells the weight loader to use the
expert-stacking code path rather than the normal tensor-assignment path. When the
loader sees `target_path` as a list, it reads each of the `expert_keys` tensors
individually, transposes them, concatenates along the new expert axis 0, and
writes the resulting stacked tensor to `target_path_base`.

**`physical_to_logical_map`:** When EPLB is active, some physical expert slots
hold copies of popular logical experts. This map defines which logical expert
fills each physical slot. The weight loader uses it to replicate the correct
source tensors into the extra physical slots.

---

## 15. Complete Tensor Inventory

For a model with `E` experts, `H` hidden size, `I` intermediate size, `ep_size` EP degree.

### Weight Parameters

| Attribute | Shape | Sharding (moe_mesh) | Description |
|---|---|---|---|
| `wi_0` | `(E, H, I)` | `P("expert", None, "tensor")` | Gate projection weights |
| `wi_1` | `(E, H, I)` | `P("expert", None, "tensor")` | Up projection weights |
| `wo` | `(E, I, H)` | `P("expert", "tensor", None)` | Down projection weights |
| `wi_0_scale` | `(E, k_b, 1, I)` or None | `P("expert", None, None, "tensor")` | Gate weight quant scales |
| `wi_1_scale` | `(E, k_b, 1, I)` or None | `P("expert", None, None, "tensor")` | Up weight quant scales |
| `wo_scale` | `(E, k_b, 1, H)` or None | `P("expert", None, None, None)` | Down weight quant scales |

`k_b` = number of quantization blocks along the K dimension = `H // block_size_k`
(or 1 for per-channel quantization).

Each device holds `E / ep_size` experts (or more with EPLB redundancy).

### Intermediate Tensors (inside `_forward`, per `shard_map` invocation)

| Tensor | Shape | Description |
|---|---|---|
| `flatten_selected_experts` | `(T*K,)` | All expert assignments, flat |
| `sorted_selected_experts` | `(T*K,)` | Argsort indices for expert-order sort |
| `token_indices` | `(T*K,)` | Original token index for each sorted slot |
| `group_sizes` | `(E,)` | Token count per expert across all tokens |
| `x` (gathered) | `(T*K, H)` | Token hidden states in sorted-expert order |
| `layer_w0` | `(T*K, I)` | Gate branch output |
| `layer_w1` | `(T*K, I)` | Up branch output |
| `intermediate_layer` | `(T*K, I)` | SwiGLU gated activation |
| `intermediate_output` | `(T*K, H)` | Down projection output (expert order) |
| `unsort_intermediate` | `(T*K, H)` | Re-sorted to original token-expert order |
| `output` | `(T, H)` | Weighted expert sum, per token |

---

## 16. Summary and Key Design Decisions

### EPMoE vs FusedEPMoEV2

`EPMoE` implements Expert Parallelism at the JAX collective level using
`shard_map` and the megablox `gmm` kernel. Each step — permute, GMM, unpermute,
all-reduce — is a separate JAX operation. XLA schedules them as separate kernels.

`FusedEPMoEV2` (Pallas) does all of this in a single kernel invocation: the EP
dispatch, expert FFN computation, and result aggregation happen in VMEM with
DMA pipelining and MXU computation overlapped. There are no intermediate HBM
round-trips for sorted tokens or expert outputs.

`EPMoE` is easier to understand, debug, and port to new hardware. `FusedEPMoEV2`
is faster on TPU v7x but is TPU-specific.

### shard_map enables EP without explicit collective design

`shard_map` lets the EPMoE code be written as if running on a single device
over a local expert slice. The EP all-to-all is implicit: because all devices
receive all tokens (replicated `P(None)` input) and each device computes only
its local experts, no explicit token redistribution (all-to-all scatter/gather)
is needed. The `psum` at the end combines results. This avoids the complexity
of a ragged all-to-all while still achieving EP behavior.

### Indexed GMM defers the gather

The full token-expert expansion `[T*K, H]` is never materialized in HBM.
Instead, `_permute` returns `token_indices` — the per-slot token mapping — and
`_gmm_compute` performs the gather inside the `gmm` call via indexed access.
For 1024 tokens, top_k=8, H=4096, bf16: this saves `1024 * 8 * 4096 * 2 ≈ 67 MB`
of HBM compared to materializing the sorted tensor.

### `create_moe_weights_mapping` is backend-agnostic

The function generates weight mappings for all three backends (`epmoe`,
`fused`, `fused_v2`) from a single call, differing only in the target attribute
names and sharding specs. This means model code (`mimo_v2_flash.py`) can pass
`moe_backend=config.moe_backend` and get correct mappings without any
if/else in the model itself.

### Scale layout normalization is defensive

Checkpoints from different quantization pipelines produce scales in different
layouts. Rather than requiring every checkpoint to produce GMM-compatible
scales, `_normalize_scale_for_gmm` handles all known layouts (2D, 3D, 4D,
block-major, channel-major, per-channel) and converts them to the GMM contract
`[E, k_blocks, 1, out_dim]` at load time. This makes `EPMoE` compatible with
any quantization toolchain without modifying the kernel.

### Re-exports maintain backward compatibility

`FusedEPMoE`, `FusedEPMoEV2`, `GateLogit`, and `TopK` are re-exported from
this module even though they are defined elsewhere. Before the fused backends
were split into separate files, all MoE code lived here. Existing model files
and tests that import `from sgl_jax.srt.layers.moe import FusedEPMoEV2` continue
to work without modification.
