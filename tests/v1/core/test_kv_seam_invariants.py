# SPDX-License-Identifier: Apache-2.0
"""Upstream invariants the tkv/turbo-attn seam depends on."""

from types import SimpleNamespace

import pytest
import torch

from vllm.config.cache import (
    _BUILTIN_CACHE_DTYPES,
    _PLUGIN_CACHE_DTYPES,
    register_cache_dtype,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVQuantMode,
    get_kv_quant_mode,
    kv_cache_dtype_is_backend_managed,
)
from vllm.v1.worker.utils import AttentionGroup, KVBlockZeroer

pytestmark = pytest.mark.cpu_test

TURBOQUANT_DTYPES = sorted(
    d for d in _BUILTIN_CACHE_DTYPES if d.startswith("turboquant")
)


@pytest.mark.parametrize("cache_dtype", TURBOQUANT_DTYPES)
def test_turboquant_dtype_maps_to_a_quant_mode(cache_dtype):
    # The skip-layer branches in gpu_model_runner and gpu/attn_utils read
    # KVQuantMode.NONE as "this layer is unquantized" and hand the backend
    # "auto" in place of the dtype string. A dtype KVQuantMode does model must
    # not reach them as NONE, or every layer using it loses its packed layout.
    assert get_kv_quant_mode(cache_dtype) != KVQuantMode.NONE


@pytest.mark.parametrize("cache_dtype", TURBOQUANT_DTYPES)
def test_modelled_dtype_is_not_backend_managed(cache_dtype):
    # A layer skipped by --kv-cache-dtype-skip-layers under such a dtype keeps
    # KVQuantMode.NONE and must still be sized unquantized, so the predicate
    # must not claim the dtype back for the backend.
    assert not kv_cache_dtype_is_backend_managed(cache_dtype)


def test_plugin_dtype_is_backend_managed():
    name = "test_backend_managed_kv_dtype"
    register_cache_dtype(name, torch.uint8)
    try:
        assert get_kv_quant_mode(name) == KVQuantMode.NONE
        assert kv_cache_dtype_is_backend_managed(name)
    finally:
        _PLUGIN_CACHE_DTYPES.discard(name)


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "fp8", "nvfp4"])
def test_stock_dtype_is_not_backend_managed(cache_dtype):
    assert not kv_cache_dtype_is_backend_managed(cache_dtype)


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
