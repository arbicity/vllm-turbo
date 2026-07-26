# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Persist the CuTeDSL JIT compile cache across engine boots.

nvidia-cutlass-dsl ships a content-addressed on-disk compile cache: the
artifact name is a hash of the traced MLIR module bytecode, the DSL
environment (including GPU arch and compile options) and the DSL version,
and payloads are CRC32-checked on load (``cutlass.base_dsl.dsl
.compile_and_cache`` / ``cache_helpers``).  A stale artifact therefore
cannot replay across a kernel-code, arch, or toolchain change — a changed
input simply hashes to a different key and misses.

The cache is enabled by default, but the DSL roots it under
``tempfile.gettempdir()`` unless ``CUTE_DSL_CACHE_DIR`` is set.  Inside a
serve container /tmp is ephemeral, so every boot re-JITs every CuTeDSL
kernel from scratch — for the TURBO_ATTN lane that is the
``ArbiAttentionForward`` prefill op stalling the first prefill wave for
several seconds after every restart (arbicity/arbi-serve#977).

This module roots the cache under ``~/.cache/tkv`` (respecting
``XDG_CACHE_HOME``), which the serve containers already bind-mount from
the persistent ``/cache/tkv`` host volume, so compiled kernels survive
container restarts.  An operator-set ``CUTE_DSL_CACHE_DIR`` always wins —
this only fills in the code default.

The DSL re-reads ``CUTE_DSL_CACHE_DIR`` on every cache access
(``get_default_generated_ir_path``), so setting it here is effective even
if ``cutlass`` was already imported.
"""

import functools
import glob
import inspect
import os

from vllm.logger import init_logger

logger = init_logger(__name__)

CUTEDSL_CACHE_DIR_ENV = "CUTE_DSL_CACHE_DIR"

_compile_only_cache_patched = False


def _default_cutedsl_cache_dir() -> str:
    cache_home = os.environ.get("XDG_CACHE_HOME") or os.path.join(
        os.path.expanduser("~"), ".cache"
    )
    return os.path.join(cache_home, "tkv", "cute_dsl_cache")


def cutedsl_cache_artifact_count() -> int:
    """Number of compiled-kernel artifacts currently in the cache dir."""
    cache_dir = os.environ.get(CUTEDSL_CACHE_DIR_ENV)
    if not cache_dir or not os.path.isdir(cache_dir):
        return 0
    return len(glob.glob(os.path.join(cache_dir, "*.mlir")))


def ensure_persistent_cutedsl_cache_dir() -> None:
    """Point the CuTeDSL file cache at a boot-persistent directory.

    Call once per worker process, any time before the first
    ``cute.compile``.  Idempotent; an operator-set ``CUTE_DSL_CACHE_DIR``
    is left untouched.
    """
    existing = os.environ.get(CUTEDSL_CACHE_DIR_ENV)
    if existing:
        logger.info_once(
            "CuTeDSL compile cache dir (operator-set): %s (%d cached kernels)",
            existing,
            cutedsl_cache_artifact_count(),
        )
        return

    cache_dir = _default_cutedsl_cache_dir()
    try:
        os.makedirs(cache_dir, exist_ok=True)
    except OSError as e:
        # Fail loud, run degraded: the DSL falls back to its tmpdir
        # default, i.e. per-boot recompilation.
        logger.warning(
            "Could not create persistent CuTeDSL cache dir %s (%s); "
            "CuTeDSL kernels will be re-JIT-compiled every boot.",
            cache_dir,
            e,
        )
        return

    os.environ[CUTEDSL_CACHE_DIR_ENV] = cache_dir
    logger.info_once(
        "CuTeDSL compile cache dir: %s (%d cached kernels present)",
        cache_dir,
        cutedsl_cache_artifact_count(),
    )


# Parameters generate_mlir must expose, in order, for the compile-only
# cache override below to be applicable.  If the installed cutlass DSL
# diverges, we refuse to patch (fail loud, run stock) instead of
# guessing.
_GENERATE_MLIR_EXPECTED_PARAMS = (
    "self",
    "funcBody",
    "function_name",
    "gpu_module_attrs",
    "args",
    "kwonlyargs",
    "sig",
    "pipeline",
    "no_cache",
    "no_jit_engine",
    "compile_only",
)


def enable_cutedsl_compile_only_cache() -> None:
    """Let explicit ``cute.compile`` calls use the DSL's own file cache.

    ``cute.compile`` (``CompileCallable._compile``) hard-sets
    ``compile_only=True, no_cache=True``, so the DSL's content-addressed
    on-disk compile cache never applies to explicitly compiled kernels —
    every boot re-runs the full MLIR pipeline + cubin generation for ops
    like the TURBO_ATTN ``ArbiAttentionForward`` prefill kernel.

    The underlying machinery (``BaseDSL.generate_mlir`` ->
    ``compile_and_cache``) fully supports caching compile-only results:
    on a hit it loads the lowered module (embedded cubin) from disk,
    CRC32-checked and keyed by IR-content + DSL envars (GPU arch) + DSL
    version, and only rebuilds the host-side execution engine.  This
    wrapper re-enables that path by flipping the forced ``no_cache`` for
    compile-only invocations.  Signature-guarded against DSL drift:
    on mismatch we log and leave stock behavior (per-boot recompile).

    Idempotent; call before the first ``cute.compile``.
    """
    global _compile_only_cache_patched
    if _compile_only_cache_patched:
        return

    try:
        from cutlass.base_dsl.dsl import BaseDSL
    except Exception:
        logger.debug("cutlass DSL not importable; no compile cache to enable.")
        return

    original = BaseDSL.generate_mlir
    try:
        sig = inspect.signature(original)
        params = tuple(sig.parameters)
    except (TypeError, ValueError):
        params = ()
    if params[: len(_GENERATE_MLIR_EXPECTED_PARAMS)] != _GENERATE_MLIR_EXPECTED_PARAMS:
        logger.warning(
            "cutlass BaseDSL.generate_mlir signature changed (%s); NOT "
            "enabling the compile-only CuTeDSL cache — explicit "
            "cute.compile kernels will re-JIT every boot.",
            params,
        )
        return

    @functools.wraps(original)
    def generate_mlir_with_persistent_cache(self, *args, **kwargs):
        bound = sig.bind(self, *args, **kwargs)
        if bound.arguments.get("compile_only") and bound.arguments.get("no_cache"):
            bound.arguments["no_cache"] = False
        return original(*bound.args, **bound.kwargs)

    BaseDSL.generate_mlir = generate_mlir_with_persistent_cache
    _compile_only_cache_patched = True
    logger.info_once(
        "Enabled the CuTeDSL on-disk compile cache for explicit "
        "cute.compile calls (compile-only no_cache override)."
    )
