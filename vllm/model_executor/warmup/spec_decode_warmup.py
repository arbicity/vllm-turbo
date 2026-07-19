# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Boot-time warmup for speculative-decoding input-prep Triton kernels.

The padded EAGLE/MTP proposer path launches a handful of small Triton
kernels (``eagle_prepare_next_token_padded_kernel``,
``eagle_prepare_inputs_padded_kernel``,
``eagle_step_slot_mapping_metadata_kernel``) plus the rejection
sampler's ``expand_kernel``.  None of these run during the stock dummy
runs or CUDA-graph capture, so they JIT-compiled on the first real
request after every boot (flagged by the JIT monitor as in-inference
compilations; arbicity/arbi-serve#977).

This warmup launches each kernel on throwaway tensors that mirror the
dtypes, constexpr values, and Triton value-specializations
(``batch_size == 1`` vs ``> 1``) of the real call sites, so the compiles
happen at boot instead.
"""

from typing import TYPE_CHECKING

import torch

from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_worker import Worker

logger = init_logger(__name__)


def _warm_batch_sizes(max_num_seqs: int) -> tuple[int, ...]:
    # Triton value-specializes integer arguments on == 1, so a kernel
    # compiled at batch 2 recompiles at batch 1 (and vice versa).
    return (1, 2) if max_num_seqs > 1 else (1,)


def _warm_eagle_step_kernel(worker: "Worker", drafter) -> None:
    from vllm.v1.spec_decode.utils import (
        eagle_step_update_slot_mapping_and_metadata,
    )

    device = worker.device
    runner = worker.model_runner
    # n_blocks_per_req is a constexpr taken from the block table width;
    # use a real block-table tensor so the specialization matches the
    # serving call exactly.
    max_num_seqs = worker.scheduler_config.max_num_seqs
    block_table = runner.input_batch.block_table[0].get_device_tensor(max_num_seqs)
    for bs in _warm_batch_sizes(max_num_seqs):
        positions = torch.zeros(bs, dtype=torch.int64, device=device)
        seq_lens = torch.ones(bs, dtype=torch.int32, device=device)
        out_positions = torch.empty(bs, dtype=torch.int64, device=device)
        out_slot_mapping = torch.empty(bs, dtype=torch.int64, device=device)
        eagle_step_update_slot_mapping_and_metadata(
            positions_1d=positions,
            block_table_tensor=block_table[:bs],
            seq_lens=seq_lens,
            block_size=worker.cache_config.block_size,
            max_model_len=drafter.max_model_len,
            out_clamped_positions=out_positions,
            out_slot_mapping=out_slot_mapping,
            input_batch_size=bs,
        )


def _warm_eagle_padded_prep_kernels(worker: "Worker", num_spec_tokens: int) -> None:
    from vllm.v1.spec_decode.utils import (
        eagle_prepare_inputs_padded_kernel,
        eagle_prepare_next_token_padded_kernel,
        next_power_of_2,
    )

    device = worker.device
    vocab_size = worker.model_config.get_vocab_size()
    for bs in _warm_batch_sizes(worker.scheduler_config.max_num_seqs):
        # prepare_next_token_ids_padded.  The first decode step after a
        # prefill samples a single token (no draft yet), so
        # sampled_token_ids is [bs, 1]; spec steps use [bs, K + 1].
        # Both are distinct Triton specializations — warm each.
        valid_counts = torch.empty(bs, dtype=torch.int32, device=device)
        for num_sampled in (1, num_spec_tokens + 1):
            sampled_token_ids = torch.zeros(
                (bs, num_sampled), dtype=torch.int32, device=device
            )
            discard_mask = torch.zeros(bs, dtype=torch.bool, device=device)
            backup_tokens = torch.zeros(bs, dtype=torch.int32, device=device)
            next_token_ids = torch.empty(bs, dtype=torch.int32, device=device)
            eagle_prepare_next_token_padded_kernel[(bs,)](
                sampled_token_ids,
                discard_mask,
                backup_tokens,
                next_token_ids,
                valid_counts,
                vocab_size,
                num_sampled,
                bs,
                sampled_token_ids.stride(0),
                BLOCK_SIZE_TOKENS=next_power_of_2(num_sampled),
            )

        # prepare_inputs_padded
        cu_num_draft = (
            torch.arange(1, bs + 1, dtype=torch.int32, device=device) * num_spec_tokens
        )
        query_start_loc = torch.arange(0, bs + 1, dtype=torch.int32, device=device) * (
            num_spec_tokens + 1
        )
        token_indices_to_sample = torch.empty(bs, dtype=torch.int32, device=device)
        num_rejected = torch.empty(bs, dtype=torch.int32, device=device)
        eagle_prepare_inputs_padded_kernel[(bs,)](
            cu_num_draft,
            valid_counts,
            query_start_loc,
            token_indices_to_sample,
            num_rejected,
            bs,
        )


def _warm_rejection_expand_kernel(worker: "Worker", num_spec_tokens: int) -> None:
    from vllm.v1.sample.rejection_sampler import expand_batch_to_tokens

    device = worker.device
    bs = min(2, worker.scheduler_config.max_num_seqs)
    cu_num_tokens = (
        torch.arange(1, bs + 1, dtype=torch.int32, device=device) * num_spec_tokens
    )
    num_tokens = bs * num_spec_tokens
    # The real call sites: temperature (fp32, replace 0 -> 1), top_k
    # (int32, no replace), top_p (fp32, no replace).  replace_to == 1 is
    # value-specialized by Triton, so warm both variants.
    temperature = torch.ones(bs, dtype=torch.float32, device=device)
    expand_batch_to_tokens(
        temperature, cu_num_tokens, num_tokens, replace_from=0, replace_to=1
    )
    expand_batch_to_tokens(temperature, cu_num_tokens, num_tokens)
    top_k = torch.ones(bs, dtype=torch.int32, device=device)
    expand_batch_to_tokens(top_k, cu_num_tokens, num_tokens)


def spec_decode_prep_kernel_warmup(worker: "Worker") -> None:
    """Warm spec-decode input-prep Triton kernels at boot.

    No-op unless a drafter with the padded EAGLE/MTP prep path is
    active.  Best-effort: a failure here only costs first-request JIT
    latency, so it is logged loudly but does not abort boot.
    """
    runner = worker.model_runner
    spec_config = worker.vllm_config.speculative_config
    drafter = getattr(runner, "drafter", None)
    if spec_config is None or drafter is None:
        return
    # Only the llm-base (EAGLE/MTP) proposer family uses these kernels.
    if not hasattr(drafter, "_slot_mapping_buffer"):
        return

    num_spec_tokens = spec_config.num_speculative_tokens
    try:
        logger.info("Warming up spec-decode input-prep Triton kernels.")
        _warm_eagle_step_kernel(worker, drafter)
        _warm_eagle_padded_prep_kernels(worker, num_spec_tokens)
        _warm_rejection_expand_kernel(worker, num_spec_tokens)
        torch.cuda.synchronize()
    except Exception as e:
        logger.warning(
            "Spec-decode input-prep kernel warmup failed (%s); first "
            "request will pay the JIT cost.",
            e,
        )
