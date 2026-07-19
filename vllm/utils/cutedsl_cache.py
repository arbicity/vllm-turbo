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

import glob
import os

from vllm.logger import init_logger

logger = init_logger(__name__)

CUTEDSL_CACHE_DIR_ENV = "CUTE_DSL_CACHE_DIR"


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
