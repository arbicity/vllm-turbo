# SPDX-License-Identifier: Apache-2.0
"""A spec may fuse several layers into one shared KV page.

`KVCacheSpec.aggregated_layer_count` is how a spec declares that its
`page_size_bytes` already SUMS N layers' per-layer pages (the layers occupy
disjoint byte slices of the shared page). The planner must then allocate ONE
tensor per N layers, not one per layer — otherwise it charges the summed page
once per layer and usable KV capacity drops by N.

This is the layout turbo-attn's mixed-bit-width ("composite") tkv spec uses:
each full-attention layer keeps its own K/V bit widths, so the layers have
different slot sizes and are packed into one shared page.

The specs below are real: `page_size_bytes` and `aggregated_layer_count` are
both derived from a genuine per-layer byte-offset map, and the production
planner is what computes the budget. Nothing about the fusing is mocked — a
fixture that manufactured those attributes could pass against a planner that
never consults them.
"""

from dataclasses import dataclass

import pytest
import torch

from vllm.config import ModelConfig, SchedulerConfig, VllmConfig
from vllm.v1.core.kv_cache_utils import get_kv_cache_configs
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheSpec,
    MambaSpec,
)

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
NUM_KV_HEADS = 2
HEAD_SIZE = 256

# Qwen3.5-0.8B: 24 layers, every 4th is full attention -> 6 attention layers.
NUM_LAYERS = 24
ATTN_INTERVAL = 4

# Per-layer slot bytes for the 4.0-bpe mixed-width allocation and for the
# uniform allocation that costs exactly the same bytes per token. Equal
# totals are the point: the two arms must report the same token capacity.
MIXED_SLOT_BYTES = (544, 544, 544, 480, 544, 608)
UNIFORM_SLOT_BYTES = (544, 544, 544, 544, 544, 544)
assert sum(MIXED_SLOT_BYTES) == sum(UNIFORM_SLOT_BYTES) == 3264


@dataclass(frozen=True)
class FusedAttentionSpec(FullAttentionSpec):
    """Full attention over layers packed into one shared page.

    `layer_slot_bytes` is the real per-layer byte-slice map: layer i owns
    bytes [sum(slots[:i]), sum(slots[:i+1])) of every slot. Both
    `page_size_bytes` and `aggregated_layer_count` are computed from it.
    """

    layer_slot_bytes: tuple[int, ...] = ()

    @property
    def page_size_bytes(self) -> int:
        return self.block_size * sum(self.layer_slot_bytes)

    @property
    def aggregated_layer_count(self) -> int:
        return len(self.layer_slot_bytes)

    @classmethod
    def merge(cls, specs):
        # Already fused; merging the group is a no-op.
        return specs[0]


def _make_specs(slot_bytes_per_layer, with_mamba):
    """One fused attention spec shared by all attention layers.

    `with_mamba=True` reproduces the hybrid GDN+attention topology (which
    takes the split O(1)/O(n) allocation path); `False` takes the plain
    shared-tensor path. Both must honour the fusing.
    """
    attn_spec = FusedAttentionSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=NUM_KV_HEADS,
        head_size=HEAD_SIZE,
        dtype=torch.uint8,
        layer_slot_bytes=tuple(slot_bytes_per_layer),
    )
    mamba_spec = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((3, 8192), (32, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="none",
    )
    specs = {}
    for i in range(NUM_LAYERS):
        is_attn = (i + 1) % ATTN_INTERVAL == 0
        if is_attn:
            specs[f"layer_{i}"] = attn_spec
        elif with_mamba:
            specs[f"layer_{i}"] = mamba_spec
    return specs, attn_spec


def _config(max_model_len=1024):
    model_config = ModelConfig(max_model_len=max_model_len)
    return VllmConfig(
        model_config=model_config,
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=max_model_len,
            max_num_seqs=8,
            enable_chunked_prefill=True,
            max_model_len=max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
    )


def _plan(slot_bytes_per_layer, with_mamba, available_memory):
    specs, attn_spec = _make_specs(slot_bytes_per_layer, with_mamba)
    config = get_kv_cache_configs(_config(), [specs], [available_memory])[0]
    attn_layers = {n for n, s in specs.items() if isinstance(s, FusedAttentionSpec)}
    attn_tensors = [
        t for t in config.kv_cache_tensors if attn_layers.intersection(t.shared_by)
    ]
    return config, attn_spec, attn_tensors


def test_default_specs_declare_no_fusing():
    """Unfused specs keep one tensor per layer — the default must not move."""
    assert (
        FullAttentionSpec(
            block_size=BLOCK_SIZE,
            num_kv_heads=NUM_KV_HEADS,
            head_size=HEAD_SIZE,
            dtype=torch.bfloat16,
        ).aggregated_layer_count
        == 1
    )
    assert (
        MambaSpec(
            block_size=BLOCK_SIZE,
            shapes=((3, 8192),),
            dtypes=(torch.bfloat16,),
        ).aggregated_layer_count
        == 1
    )
    assert KVCacheSpec(block_size=BLOCK_SIZE).aggregated_layer_count == 1


@pytest.mark.parametrize("with_mamba", [True, False])
def test_fused_group_gets_one_tensor(with_mamba):
    """N fused layers share ONE tensor, not one each.

    Without the fix the planner emits one composite-sized tensor per layer,
    so both the tensor count and the block budget are off by N.
    """
    available = 2 * 1024**3
    config, attn_spec, attn_tensors = _plan(MIXED_SLOT_BYTES, with_mamba, available)

    assert attn_spec.aggregated_layer_count == len(MIXED_SLOT_BYTES)
    assert len(attn_tensors) == 1, (
        f"expected 1 shared tensor for the {len(MIXED_SLOT_BYTES)} fused layers, "
        f"got {len(attn_tensors)} — the planner sized the group per layer"
    )
    assert len(attn_tensors[0].shared_by) == len(MIXED_SLOT_BYTES)


@pytest.mark.parametrize("with_mamba", [True, False])
def test_fused_capacity_matches_equal_byte_uniform_control(with_mamba):
    """The fused arm must report the capacity its bytes actually buy.

    The control is the uniform allocation the mixed one is meant to replace:
    544 B/slot on every layer, one layer per page, so vLLM allocates it the
    ordinary way (6 tensors, `aggregated_layer_count == 1`). Both arms cost
    3264 B/token, so any difference in reported tokens is pure accounting.
    """
    available = 2 * 1024**3
    mixed, mixed_spec, _ = _plan(MIXED_SLOT_BYTES, with_mamba, available)
    control, control_spec, control_tensors = _plan(
        UNIFORM_SLOT_BYTES[:1], with_mamba, available
    )

    # Sanity: the control really is the unfused path.
    assert control_spec.aggregated_layer_count == 1
    assert len(control_tensors) == len(MIXED_SLOT_BYTES)
    assert mixed_spec.page_size_bytes == control_spec.page_size_bytes * len(
        MIXED_SLOT_BYTES
    ), "arms must cost the same bytes per token for the comparison to mean anything"

    assert mixed.num_blocks == control.num_blocks, (
        f"equal-byte arms disagree on capacity: mixed={mixed.num_blocks} blocks "
        f"vs uniform control={control.num_blocks} — the fused page was charged "
        f"once per layer"
    )
    assert mixed.num_blocks > 0


@pytest.mark.parametrize("with_mamba", [True, False])
def test_budget_and_allocation_stay_in_lockstep(with_mamba):
    """Never plan more blocks than the allocation can back.

    Budgeting per fused group while still allocating per layer would OOM
    instead of merely under-reporting, so pin both ends.
    """
    available = 2 * 1024**3
    config, attn_spec, attn_tensors = _plan(MIXED_SLOT_BYTES, with_mamba, available)

    total = sum(t.size for t in config.kv_cache_tensors)
    assert total <= available, (
        f"planner allocated {total} bytes from a {available}-byte budget"
    )
    assert attn_tensors[0].size == attn_spec.page_size_bytes * config.num_blocks, (
        "the shared tensor must be sized for the blocks the scheduler will hand out"
    )
