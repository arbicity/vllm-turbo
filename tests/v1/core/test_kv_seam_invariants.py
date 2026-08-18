# SPDX-License-Identifier: Apache-2.0
"""Upstream invariants the tkv/turbo-attn seam depends on."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.cache import _BUILTIN_CACHE_DTYPES
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
)
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

pytestmark = pytest.mark.cpu_test

BACKEND_MANAGED_DTYPES = sorted(
    d for d in _BUILTIN_CACHE_DTYPES if d.startswith("turboquant")
)


@pytest.mark.parametrize("cache_dtype", BACKEND_MANAGED_DTYPES)
def test_backend_managed_dtype_maps_to_a_quant_mode(cache_dtype):
    # The skip-layer branches in gpu_model_runner and gpu/attn_utils read
    # KVQuantMode.NONE as "this layer is unquantized" and hand the backend
    # "auto" in place of the dtype string. A dtype whose page layout only the
    # attention backend understands must reach it verbatim, so it must never
    # map to NONE.
    assert get_kv_quant_mode(cache_dtype) != KVQuantMode.NONE


class _BlockDimZeroBackend:
    @classmethod
    def get_kv_cache_block_dim(
        cls,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str,
    ) -> int:
        return 0


def _zeroer(num_blocks: int) -> KVBlockZeroer:
    spec = FullAttentionSpec(
        block_size=16,
        num_kv_heads=2,
        head_size=64,
        dtype=torch.bfloat16,
    )
    kv_cache = torch.zeros((num_blocks, 2, 16, 2, 64), dtype=torch.bfloat16)
    group = AttentionGroup(
        backend=_BlockDimZeroBackend,
        layer_names=["model.layers.0.self_attn.attn"],
        kv_cache_spec=spec,
        kv_cache_group_id=0,
    )
    return KVBlockZeroer(
        device=torch.device("cpu"),
        attn_groups_iter=[group],
        kernel_block_sizes=[16],
        cache_dtype="auto",
        static_forward_context={
            "model.layers.0.self_attn.attn": SimpleNamespace(kv_cache=kv_cache)
        },
    )


def test_zeroer_reads_the_real_block_count_off_the_tensors():
    assert _zeroer(4).real_num_blocks == 4


def test_zeroer_warmup_skips_a_group_with_no_blocks(monkeypatch):
    # On a hybrid model the attention group can profile to zero blocks while
    # the aggregate `kv_cache_config.num_blocks` its caller passes still counts
    # the Mamba groups. Zeroing block id 0 would then write past the tensor.
    zeroer = _zeroer(0)
    requested: list[list[int]] = []
    monkeypatch.setattr(zeroer, "zero_block_ids", requested.append)
    zeroer.warmup(1024)
    assert requested == []


def test_zeroer_warmup_runs_when_blocks_exist(monkeypatch):
    zeroer = _zeroer(4)
    requested: list[list[int]] = []
    monkeypatch.setattr(zeroer, "zero_block_ids", requested.append)
    zeroer.warmup(1024)
    assert requested == [[0]]
