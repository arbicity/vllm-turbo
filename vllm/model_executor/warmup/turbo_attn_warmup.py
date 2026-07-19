# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boot-time warmup for the TURBO_ATTN (tkv) attention backend.

The TURBO_ATTN prefill op (``ArbiAttentionForward`` and its loader
siblings) is a CuTeDSL kernel that is JIT-compiled per
(loader, geometry, layout) bucket on first use.  Decode buckets are
exercised by CUDA-graph capture, but prefill attention runs eagerly
(piecewise cudagraphs split around attention) and the stock warmup never
builds real prefill attention metadata — so the first real prefill wave
after every boot paid a multi-second JIT stall in-band
(arbicity/arbi-serve#977).

This warmup runs force-attention dummy passes covering the prefill
buckets a serving config can hit, so the JIT happens at boot (and, with
the persistent CuTeDSL compile cache, only on the first boot after a
code/arch change):

1. a pure prefill wave (fresh prompts: ``seq_len == query_len``),
2. a mixed decode+prefill batch (prefill with cache history),
3. a chunked-prefill continuation at max context
   (``seq_len == max_model_len > query_len``).
"""

import time
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.utils.cutedsl_cache import (
    cutedsl_cache_artifact_count,
    ensure_persistent_cutedsl_cache_dir,
)

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _turbo_attn_backend_class():
    """The plugin-registered TURBO_ATTN backend class, or None."""
    from vllm.v1.attention.backends.registry import AttentionBackendEnum

    if not AttentionBackendEnum.TURBO_ATTN.is_overridden():
        return None
    try:
        return AttentionBackendEnum.TURBO_ATTN.get_class()
    except Exception:
        return None


def turbo_attn_prefill_warmup(worker: "Worker") -> None:
    """Pre-compile TURBO_ATTN prefill attention kernels at boot.

    Must run after KV-cache initialization and before CUDA-graph capture
    (the eager window ``kernel_warmup`` executes in).
    """
    backend_cls = _turbo_attn_backend_class()
    if backend_cls is None:
        return

    runner = worker.model_runner
    if runner.is_pooling_model or not runner.attn_groups:
        return
    if not any(
        group.backend is backend_cls
        for groups in runner.attn_groups
        for group in groups
    ):
        return

    # Belt-and-braces: make sure the compiles below land in the
    # persistent cache even if the worker-init call was skipped.
    ensure_persistent_cutedsl_cache_dir()

    max_tokens = min(
        worker.scheduler_config.max_num_batched_tokens, runner.max_num_tokens
    )
    max_model_len = worker.model_config.max_model_len
    cached_before = cutedsl_cache_artifact_count()
    logger.info(
        "Warming up TURBO_ATTN prefill attention "
        "(num_tokens=%d, max_model_len=%d, %d kernels in compile cache).",
        max_tokens,
        max_model_len,
        cached_before,
    )
    start = time.perf_counter()

    # Bucket 1: pure prefill wave — every request on its first chunk
    # (seq_len == query_len, no cache history).  _dummy_run splits
    # num_tokens evenly across max_num_seqs requests; pass the per-req
    # query length as profile_seq_lens so seq_len == query_len exactly.
    num_reqs = min(max_tokens, worker.scheduler_config.max_num_seqs)
    fresh_tokens = max(num_reqs, max_tokens - max_tokens % num_reqs)
    runner._dummy_run(
        num_tokens=fresh_tokens,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        profile_seq_lens=fresh_tokens // num_reqs,
    )

    # Bucket 2: mixed decode + prefill batch (decodes at seq_len 1,
    # one prefill with a single token of history).
    runner._dummy_run(
        num_tokens=min(max_tokens, 256),
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        create_mixed_batch=True,
    )

    # Bucket 3: chunked-prefill continuation — queries attending a full
    # cache history (seq_len == max_model_len > query_len).
    runner._dummy_run(
        num_tokens=max_tokens,
        skip_eplb=True,
        is_profile=True,
        force_attention=True,
        profile_seq_lens=max_model_len,
    )

    logger.info(
        "TURBO_ATTN prefill attention warmup took %.2f s "
        "(%d kernels in compile cache).",
        time.perf_counter() - start,
        cutedsl_cache_artifact_count(),
    )
