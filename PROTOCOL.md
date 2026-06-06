# KV Backend Capability Protocol

This fork adds a small set of method overrides to `AttentionBackend` so
plugin-supplied attention backends can declare their KV layout,
lifecycle, and MLA-wrapping needs through the backend object itself —
instead of monkey-patching downstream vLLM code.

The protocol is **strictly additive** and **default-preserving**: every new
method has a default that reproduces upstream behavior. Existing
backends (FlashAttention, Triton, ROCm, CPU, ...) are unchanged. The
protocol only takes effect when a plugin-registered CUSTOM backend opts in.

This document is the architectural reference for upstream review. The
reference consumer of these hooks is the TurboQuant KV-cache plugin
(`tqkv`); the protocol contains no TurboQuant-specific knowledge.

---

## Why a protocol

Plugin attention backends today must do most of their work by **monkey-
patching** vLLM internals from a `register_*` entry point. The TurboQuant
plugin grew to **15 separate runtime patches** spanning `KVCacheSpec`
dispatch, page sizing, the MLA selector, profiler accounting, the KV-cache
manager lifecycle, and the o-projection fold.

The patches all answered variants of the same question: *"the backend
knows X about its own KV layout / lifecycle / MLA needs — how do I get X
into the upstream code path that needs to consult it?"*

The fix is to let the backend **declare** X on itself, and have the
upstream code path **consult the backend object** at the point where it
already lives. That single discipline collapses every patch.

---

## General approach

For each capability that a non-stock backend needs to influence:

1. **Add an override** on `AttentionBackend` (classmethod or static, never
   stateful). Default returns `None` / preserves current behavior.
2. **Replace the upstream hardcoded branch** with: *if the backend's
   override is non-default, use it; otherwise fall through to the existing
   path.* No upstream behavior change in the default case.
3. **No new types in the public surface.** All overrides return existing
   vLLM types (`KVCacheSpec`, `SingleTypeKVCacheManager`, callables, ...)
   or simple primitives.

This is the same shape as vLLM's existing platform-detect / scheduler-
hook / CustomOp registries — small, opt-in, additive.

---

## The hooks

The 7 protocol hooks are organized into three layers. Default behavior
(when the backend doesn't override) is identical to current upstream.

### Layer 1 — KV layout

These describe how the backend wants its KV pages laid out and managed.

#### `get_supported_kv_cache_dtypes() -> Iterable[str] | None`

Classmethod. Lets a backend declare which `--kv-cache-dtype` values it
accepts (e.g. a compressed backend declares `"tqkv"`). Default `None`
means "accept whatever upstream already accepts."

Consumed by the dtype validator in `Attention.__init__`, immediately
after `--kv-cache-dtype` has been parsed.

#### `get_kv_cache_spec_class(spec_kind: SpecKind) -> type[KVCacheSpec] | None`

Lets a backend swap in a custom `KVCacheSpec` subclass for a given spec
kind (`"full_attention"`, `"sliding_window"`, `"mla"`, `"chunked_local"`,
...). Default `None` means "use the upstream spec class for that kind."

Consumed by `Attention.get_kv_cache_spec` and `MLAAttention.get_kv_cache_spec`,
which previously hardcoded the upstream spec classes. Removes 3 of the 15
prior monkey patches.

#### `KVCacheSpec.get_manager_class() -> type[SingleTypeKVCacheManager] | None`

Method on `KVCacheSpec` (defined in `vllm/v1/kv_cache_interface.py`). Lets
a custom spec choose its own single-type manager. Default `None` falls
through to the existing dispatch table in
`vllm/v1/core/single_type_kv_cache_manager.py:get_manager_for_kv_cache_spec`.

Allows compressed-page-aware managers (e.g. one that knows compressed
pages are a different size than uncompressed) without editing the
dispatch table.

#### `_get_gather_op() -> Callable | None`

Method on the per-step `MLACommonImpl`. Lets a backend wrapping MLA
substitute its own gather operation (used by MLA's decode path to pull
KV pages). Default `None` means "use the stock `kv_cache.gather()`."

This is the only hook whose owner is the per-step impl, not the backend
class — it's parameterized by the live request batch, so the backend
class can't know it ahead of time. Backends that wrap MLA implement it
via a small shim returned from a backend-class factory.

### Layer 2 — Lifecycle

These let a backend hook into worker-level events. They're called
unconditionally by the worker; the default body is empty so non-opt-in
backends are unaffected.

#### `on_model_loaded(worker, model) -> None`

Classmethod. Called from `gpu_worker.load_model` after the model is on
device, before profiling. Lets the backend run one-time post-load
mutations against the materialized model — e.g. a compressed-KV plugin
that needs to fold a residual rotation into the output projection
weights.

The classmethod signature receives the worker so the backend can stash
state (model ref, config) for later hooks.

#### `adjust_kv_budget(profiled_bytes: int, vllm_config) -> int | None`

Classmethod. Called from `gpu_worker.determine_available_memory` after
the profiler reports its KV budget. The backend may return a
substituted budget or `None` to accept the profiler's number. Used by
backends whose memory accounting genuinely diverges from the profiler's
heuristic — e.g. some Mamba/GDN paths over-count non-KV memory and
report ≤0 bytes; the backend can fall back to actual free memory.

Default `None` is identity. **Critically additive**: a backend that
doesn't override this gets exactly today's profiler-driven budget.

#### `on_kv_manager_created(mgr) -> None`

Classmethod. Called from `KVCacheManager.__init__` after the per-group
managers are wired up. Lets a backend register pre-step / pre-allocate
callbacks on the manager — e.g. a cold-tier plugin draining a write-back
buffer before the scheduler picks the next batch.

### Layer 3 — MLA wrapping

This is the hook that lets a plugin backend **wrap** MLA, instead of
**replacing** it.

#### `wraps_mla_backend(base_mla_backend_cls) -> type[AttentionBackend] | None`

Classmethod. Called from `_cached_get_attn_backend` when the user
selected a CUSTOM backend on a model that vLLM would otherwise dispatch
to MLA. The CUSTOM backend may return a wrapper subclass that delegates
to the stock MLA backend with extra steps (e.g. compress / decompress
on the K/V tensors). Default `None` means "I don't wrap MLA; treat me
as a standalone backend."

When non-None, the selector resolves the MLA candidate as it normally
would, with the CUSTOM backend's dtype gate applied, and then calls the
returned wrapper class — a tiny composition that the plugin owns end
to end. Removes the two largest prior monkey patches (the MLA-dispatch
patch and the dtype-extension patch), which together accounted for
hundreds of LOC.

---

## Per-problem mapping (TQKV reference consumer)

The TurboQuant plugin previously needed 15 monkey patches. Each one is
consumed by exactly one of the 7 hooks above (some hooks subsume
multiple patches because they generalized the underlying need):

| Old monkey patch                              | New protocol hook                  |
|-----------------------------------------------|------------------------------------|
| Add `"tqkv"` to allowed cache-dtype list      | `get_supported_kv_cache_dtypes`    |
| Inject `TQKVAttentionSpec` for full-attn      | `get_kv_cache_spec_class("full_attention")` |
| Inject `TQKVAttentionSpec` for sliding-window | `get_kv_cache_spec_class("sliding_window")` |
| Inject `TQMLAAttentionSpec` for MLA           | `get_kv_cache_spec_class("mla")`   |
| Page-size override for compressed pages       | `KVCacheSpec.get_manager_class`    |
| Manager-dispatch override                     | `KVCacheSpec.get_manager_class`    |
| MLA selector dtype gate                       | `wraps_mla_backend`                |
| MLA backend dispatch                          | `wraps_mla_backend`                |
| MLA per-step gather override                  | `MLACommonImpl._get_gather_op`     |
| Worker post-load o-proj fold                  | `on_model_loaded`                  |
| Worker post-load model-ref stash              | `on_model_loaded`                  |
| KV budget profiler fallback                   | `adjust_kv_budget`                 |
| Cold-drain pre-step callback registration     | `on_kv_manager_created`            |
| OOM-snapshot diagnostic install               | (out of scope — moved to plugin's `tqkv.debug`) |
| Auto-config defaults                          | (out of scope — users now set flags explicitly) |

The bottom two were moved out of upstream-touching code entirely; the
top thirteen are dissolved into the seven hooks.

---

## Files modified

```
vllm/_custom_ops.py
vllm/platforms/interface.py
vllm/model_executor/layers/attention/attention.py
vllm/model_executor/layers/attention/mla_attention.py
vllm/model_executor/layers/rotary_embedding/common.py
vllm/v1/attention/backend.py
vllm/v1/attention/selector.py
vllm/v1/kv_cache_interface.py
vllm/v1/core/kv_cache_manager.py
vllm/v1/core/single_type_kv_cache_manager.py
vllm/v1/worker/gpu_worker.py
```

`rotary_embedding/common.py` is not part of the protocol itself — it is a
targeted bug fix on the same branch. The previous code probed for
`flash_attn` via `find_spec` (which falsely succeeds under FA4, whose
package only ships `flash_attn.cute` as a namespace), then assumed
`flash_attn.ops.triton.rotary` exists and crashed. The fix replaces the
package probe with a small registry of fast-path factories that
feature-detect by **symbol import**, not by package name. Any future
fast paths (FA4 cute rotary, fused custom kernel, ...) plug in via
`register_rotary_fast_path` from anywhere.

---

## Sync workflow when rebasing on upstream

1. `git fetch upstream && git rebase upstream/main`.
2. Conflicts will fall on the touched files in the list above. Most are
   single-method additions or single-callsite dispatch substitutions —
   resolve by re-applying the protocol logic on top of upstream's new
   code.
3. Run `pytest tests/v1/core/test_kv_drain_hook.py` to confirm the
   lifecycle hook still fires.
4. Run a TQKV serve smoke (`vllm serve <model> --kv-cache-dtype tqkv
   --attention-backend custom`) — exercises every hook.
5. Bump the base SHA pin in the consuming Docker overlay to the new
   upstream tip.

---

## Behavior preservation guarantee

The diff is `~321 insertions / ~305 deletions` across 11 files. The
deletions are mostly within bodies of refactored methods (replacing a
hardcoded branch with a backend-consultation), not removed call sites.
Run the existing vLLM test suite — the protocol's default impls
preserve every pre-existing code path. The protocol only activates
when a plugin backend overrides a hook and is selected via
`--attention-backend custom`.
