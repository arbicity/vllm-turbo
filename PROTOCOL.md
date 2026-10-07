# KV Backend Capability Protocol

This fork carries the small seam an out-of-tree, compressed-KV attention
backend needs on top of upstream vLLM. Everything a backend can express
through upstream's own extension points lives in the plugin; this fork adds
only what upstream has no hook for.

The seam is **strictly additive** and **default-preserving**: every hook has
a default that reproduces upstream behaviour, and nothing activates unless a
plugin registers a backend or a KV-cache dtype. The reference consumer is the
Turbo Attention plugin (`tkv`, `--kv-cache-dtype tkv`); the seam itself holds
no TKV-specific knowledge beyond the `TURBO_ATTN` enum slot and the
`_tq_layer_idx` injection.

Base: upstream **v0.31.0**.

---

## What upstream already provides (and the plugin uses directly)

| Need | Upstream extension point |
|---|---|
| Register the backend class | `register_backend(AttentionBackendEnum.TURBO_ATTN, ...)` |
| Load the plugin in every process | `vllm.general_plugins` entry point |
| Pack the KV page (one head slot of the codec's slot bytes) | `AttentionBackend.customize_spec`, applied by both model runners to every layer's spec and by the hybrid block-size alignment |
| Page geometry the allocator reads | `AttentionSpec.num_heads` / `state_content_size_bytes` (`[B, H, N, C]` views, vllm#51718) |
| KV layout the kernels consume | `AttentionBackend.supported_kv_cache_layouts` |
| Single-type manager for a custom spec | `KVCacheSpecRegistry` (MRO lookup: a `FullAttentionSpec` subclass gets the full-attention manager) |

## The hooks this fork adds

### Plugin KV-cache dtypes — `vllm/config/cache.py`

`register_cache_dtype(name, torch_dtype)`, `validate_cache_dtype`,
`cache_dtype_choices`, `is_plugin_cache_dtype`. `CacheConfig.cache_dtype` is
typed `str` and gated by the runtime validator: pydantic compiles a `Literal`
annotation into the class validator at class-creation time, before any plugin
has registered, so a `Literal` field makes every plugin dtype
unconstructible. `CacheDType` survives as the static alias for the builtins.

Consumers: `--kv-cache-dtype` choices/type (`engine/arg_utils.py`) and the
dtype assertion in `get_attn_backend` (`v1/attention/selector.py`).

### The `TURBO_ATTN` backend slot

- `AttentionBackendEnum.TURBO_ATTN = None` (`v1/attention/backends/registry.py`),
  filled by the plugin's `register_backend`; `CUSTOM` stays free.
- `--attention-backend turbo-attn` parses (`config/attention.py` maps `-` to `_`).
- A plugin-registered `--kv-cache-dtype` with no `--attention-backend`
  auto-selects `TURBO_ATTN` (`engine/arg_utils.py`), because the worker's
  load-time hooks dispatch before the attention groups exist.
- `TURBO_ATTN` is a CUDA candidate once registered (`platforms/cuda.py`), so
  auto-selection paths that do not go through `EngineArgs` (e.g. a draft
  model's own selection) find it; it self-rejects every other dtype.

### Lifecycle hooks — `vllm/v1/attention/backend.py`, `vllm/v1/worker/gpu_worker.py`

Classmethods with no-op defaults. The worker dispatches each to every
backend in use (`_backends_in_use`: the attention groups' backends, falling
back to the user-selected one before the groups exist), in order, and aborts
on the first that raises (`_call_backend_hook`).

| Hook | Called from | Used for |
|---|---|---|
| `on_model_loaded(worker, model)` | `Worker.load_model` | pre-flight checks, one-time o_proj fold, memory booking before profiling |
| `on_draft_model_loaded(worker, draft_model)` | `Worker.load_model`, when a model-based drafter exists | fold the drafter's own layers |
| `adjust_kv_budget(profiled_bytes, vllm_config)` | `Worker.determine_available_memory` | replace a non-positive profiled budget |
| `on_kv_cache_initialized(worker)` | `Worker.compile_or_warm_up_model`, before capture | bind composite regions, prefill prewarm, decode autotune |

`AttentionBackend.resolve_user_selected_backend(vllm_config)` resolves the
fallback backend class.

### MLA wrapping — `vllm/v1/attention/selector.py`

`AttentionBackend.wraps_mla_backend(base_mla_backend_cls)`. When the selected
(or, with no `--attention-backend`, the dtype-claiming plugin) backend
overrides it on an MLA model, the selector picks the MLA candidate with the
dtype gate lifted and returns the wrapper the hook builds around it. The
wrapper declares the plugin dtypes in its `supported_kv_cache_dtypes` and
packs its page through `customize_spec`.

### MLA chunked-context gather — `vllm/model_executor/layers/attention/mla_attention.py`

`MLACommonBaseImpl._get_gather_op()` returns the op the chunked-context
**prefill** uses to gather (and dequantize) cached latents; the default is
`ops.gather_and_maybe_dequant_cache` itself. A wrapper impl returns its own
op with the same signature to read a packed format.

### Per-layer index — `vllm/model_executor/layers/attention/attention.py`

For `kv_cache_dtype == "tkv"`, `Attention.__init__` passes the layer ordinal
from the prefix as `_tq_layer_idx` so the impl resolves its per-layer bit
widths before the slot layout is frozen.

### Fused pages — `vllm/v1/kv_cache_interface.py`, `vllm/v1/core/kv_cache_utils.py`

`KVCacheSpec.aggregated_layer_count` (default 1). A spec whose
`page_size_bytes` already sums N layers' per-layer pages (the TKV composite
spec for per-layer bit widths) returns N; the planner then reserves one page
per N layers in each block (`get_tensor_slots`) and gives every fused layer a
view of that same page. Without it the summed page is charged once per layer
and usable capacity drops N-fold.

---

## Sync workflow when moving to a new upstream release

1. Squash the carry (`git diff <old-tag> origin/main`) into one commit on the
   old tag and cherry-pick it onto the new tag (a 3-way merge against the
   real base).
2. Resolve conflicts with upstream's structure winning; re-apply the hook on
   top. Check each hunk against the table above: when upstream adds an
   extension point that covers a hook, move the capability into the plugin
   and drop the hook.
3. `pytest tests/v1/core/test_kv_seam_invariants.py
   tests/v1/core/test_aggregated_layer_count.py
   tests/v1/attention/test_mla_wrapper_selection.py
   tests/v1/worker/test_gpu_worker.py -k hook
   tests/engine/test_arg_utils.py -k plugin_kv_cache_dtype` (CPU).
4. Bump the engine image base and COPY list in turbo-attn (`docker/Dockerfile`,
   `docker/PATCHES.md`; `scripts/ci/overlay_drift_guard.sh` checks the list),
   then let its engine-image pipeline build and GPU-validate the image.
