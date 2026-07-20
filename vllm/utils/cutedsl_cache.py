# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persistent CuTeDSL compile cache — thin defer to the canonical tkv home.

The mechanism (cache-dir rooting + the signature-guarded compile-only
``no_cache`` override) originally lived here (#9) but is engine-agnostic:
every tkv host engine pays the same per-boot re-JIT cost.  It now lives
in ``tkv.kernels.cute_dsl_cache`` (turbo-attn), auto-armed on
``tkv.kernels.cute`` import; this module re-exports it so the fork's
call sites (worker init, TURBO_ATTN warmup) keep working unchanged.

Guards (no silent dual implementations):

- turbo-attn installed but predating the canonical module: hard
  ``ImportError`` — an outdated tkv is a rebuild-the-image error, not a
  compatibility case to adapt to.
- turbo-attn not installed at all: no-op stubs.  Without tkv there are
  no TURBO_ATTN CuTeDSL kernels to cache (the ``enable_*`` call site is
  already gated on the TURBO_ATTN backend being registered), and stock
  vLLM deployments must not be forced to carry the dependency.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)

try:
    from tkv.kernels import cute_dsl_cache as _tkv_impl
except ModuleNotFoundError as e:
    if e.name in ("tkv", "tkv.kernels"):
        _tkv_impl = None
    else:
        raise ImportError(
            "The installed turbo-attn (tkv) package predates "
            "tkv.kernels.cute_dsl_cache, the canonical home of the "
            "persistent CuTeDSL compile cache. Update turbo-attn / rebuild "
            "the image; this fork no longer carries its own copy."
        ) from e

CUTEDSL_CACHE_DIR_ENV = "CUTE_DSL_CACHE_DIR"

if _tkv_impl is not None:
    assert _tkv_impl.CUTEDSL_CACHE_DIR_ENV == CUTEDSL_CACHE_DIR_ENV
    cutedsl_cache_artifact_count = _tkv_impl.cutedsl_cache_artifact_count
    ensure_persistent_cutedsl_cache_dir = _tkv_impl.ensure_persistent_cutedsl_cache_dir
    enable_cutedsl_compile_only_cache = _tkv_impl.enable_cutedsl_compile_only_cache
else:

    def cutedsl_cache_artifact_count() -> int:
        """No tkv installed: no persistent CuTeDSL cache to count."""
        return 0

    def ensure_persistent_cutedsl_cache_dir() -> None:
        """No tkv installed: nothing to persist; leave the DSL default."""
        logger.debug_once(
            "turbo-attn (tkv) is not installed; persistent CuTeDSL "
            "compile cache not armed."
        )

    def enable_cutedsl_compile_only_cache() -> None:
        """No tkv installed: no compile-only override to install."""
        logger.debug_once(
            "turbo-attn (tkv) is not installed; persistent CuTeDSL "
            "compile cache not armed."
        )
