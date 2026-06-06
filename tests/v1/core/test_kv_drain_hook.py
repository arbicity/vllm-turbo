# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for KVCacheManager.drain_request_blocks + pre-step callback.

Verifies the scheduler-cooperative drain hook used by compressed-KV
plugin backends (tqkv cold-tier). The safety contract: (1) req_to_blocks
slots are nulled BEFORE block_pool.free_blocks fires, so a subsequent
allocate_slots cannot hand out the block while it is still referenced
by the request's block_table; (2) freed blocks are immediately
reusable by the same or another request this step; (3) ref-counted
prefix-cache sharing is preserved.
"""
from __future__ import annotations

import pytest
import torch

from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import FullAttentionSpec

pytestmark = pytest.mark.cpu_test


class _FakeCoordinator:
    """Minimal coordinator stub matching the bits drain_request_blocks
    and register_pre_step_callback exercise."""

    def __init__(self, mgr):
        self.single_type_managers = [mgr]
        self.new_step_count = 0
        self.block_pool = mgr.block_pool

    def new_step_starts(self):
        self.new_step_count += 1


class _FakeManager:
    """Wraps a real FullAttentionManager but exposes only the surface
    KVCacheManager itself touches. Avoids the full kv_cache_config
    machinery needed by get_kv_cache_coordinator()."""

    def __init__(self, single_type_mgr):
        self.coordinator = _FakeCoordinator(single_type_mgr)
        self._pre_step_callbacks: list = []

    # Inherit real implementations by attaching KVCacheManager methods.
    from vllm.v1.core.kv_cache_manager import KVCacheManager as _KVCM
    new_step_starts = _KVCM.new_step_starts
    register_pre_step_callback = _KVCM.register_pre_step_callback
    drain_request_blocks = _KVCM.drain_request_blocks


def _make_mgr(num_gpu_blocks=16, block_size=4):
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=1,
        head_size=1,
        dtype=torch.float32,
    )
    pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks,
        enable_caching=True,
        hash_block_size=block_size,
    )
    single_type_mgr = FullAttentionManager(
        spec, block_pool=pool, enable_caching=True, kv_cache_group_id=0,
    )
    return _FakeManager(single_type_mgr), single_type_mgr, pool


def _allocate_request(single_type_mgr, pool, request_id, num_blocks):
    # get_new_blocks already sets ref_cnt=1 (matches allocate_new_blocks).
    blocks = pool.get_new_blocks(num_blocks)
    single_type_mgr.req_to_blocks[request_id] = list(blocks)
    return [b.block_id for b in blocks]


def test_drain_nulls_slots_and_frees_to_pool():
    wrapper, stm, pool = _make_mgr(num_gpu_blocks=16)
    block_ids = _allocate_request(stm, pool, "req-0", num_blocks=4)
    free_before = pool.get_num_free_blocks()

    # Drain the middle two; head/tail stay.
    n = wrapper.drain_request_blocks("req-0", [block_ids[1], block_ids[2]])
    assert n == 2

    req_blocks = stm.req_to_blocks["req-0"]
    assert req_blocks[0].block_id == block_ids[0]
    assert req_blocks[1] is stm._null_block
    assert req_blocks[2] is stm._null_block
    assert req_blocks[3].block_id == block_ids[3]

    # Blocks came back to the pool (ref_cnt dropped to 0).
    assert pool.get_num_free_blocks() == free_before + 2


def test_drain_noop_on_unknown_request():
    wrapper, stm, pool = _make_mgr()
    assert wrapper.drain_request_blocks("ghost", [5, 6]) == 0


def test_drain_skips_unknown_block_ids():
    wrapper, stm, pool = _make_mgr()
    block_ids = _allocate_request(stm, pool, "req-0", num_blocks=2)
    # Ask to drain a block_id the request doesn't actually own.
    unknown = 99
    assert unknown not in block_ids
    n = wrapper.drain_request_blocks("req-0", [unknown])
    assert n == 0


def test_drain_idempotent_second_call_nop():
    wrapper, stm, pool = _make_mgr()
    block_ids = _allocate_request(stm, pool, "req-0", num_blocks=2)
    assert wrapper.drain_request_blocks("req-0", block_ids) == 2
    # Already nulled — second drain is zero.
    assert wrapper.drain_request_blocks("req-0", block_ids) == 0


def test_drain_preserves_prefix_cache_sharing():
    """A block whose ref_cnt>1 (another request holds it too) stays out
    of the free queue after drain — matches BlockPool.free_blocks."""
    wrapper, stm, pool = _make_mgr()
    ids0 = _allocate_request(stm, pool, "req-0", num_blocks=2)
    # Simulate req-1 sharing the same block by bumping its ref.
    shared = pool.blocks[ids0[0]]
    shared.ref_cnt += 1
    free_before = pool.get_num_free_blocks()

    wrapper.drain_request_blocks("req-0", [ids0[0]])
    assert shared.ref_cnt == 1           # decremented but still held
    assert pool.get_num_free_blocks() == free_before  # still allocated


def test_freed_block_reusable_this_step():
    """After drain the block must be handed to a subsequent allocate."""
    wrapper, stm, pool = _make_mgr(num_gpu_blocks=8)
    ids = _allocate_request(stm, pool, "req-0", num_blocks=4)
    wrapper.drain_request_blocks("req-0", ids[:2])
    # Now allocate 2 new blocks — pool should be able to satisfy.
    new = pool.get_new_blocks(2)
    assert len(new) == 2


def test_pre_step_callback_fires_before_coordinator():
    """Registered callback runs BEFORE coordinator.new_step_starts each step."""
    wrapper, _stm, _pool = _make_mgr()
    order: list[str] = []
    def cb(mgr):
        order.append("cb")
        # Snapshot coordinator counter at call time.
        order.append(f"coord_before={mgr.coordinator.new_step_count}")
    wrapper.register_pre_step_callback(cb)
    wrapper.new_step_starts()
    assert order == ["cb", "coord_before=0"]
    assert wrapper.coordinator.new_step_count == 1

    # Second step — callback fires again, coordinator bumped to 2.
    order.clear()
    wrapper.new_step_starts()
    assert order == ["cb", "coord_before=1"]


def test_pre_step_callback_idempotent_registration():
    wrapper, _stm, _pool = _make_mgr()
    calls = []
    def cb(mgr):
        calls.append(1)
    wrapper.register_pre_step_callback(cb)
    wrapper.register_pre_step_callback(cb)  # duplicate — should not dup-register
    wrapper.new_step_starts()
    assert len(calls) == 1


def test_pre_step_callback_exception_does_not_break_scheduling():
    """A raising callback is logged but must not abort new_step_starts
    or prevent coordinator.new_step_starts from running."""
    wrapper, _stm, _pool = _make_mgr()
    def bad(mgr):
        raise RuntimeError("boom")
    wrapper.register_pre_step_callback(bad)
    wrapper.new_step_starts()  # must not raise
    assert wrapper.coordinator.new_step_count == 1


def test_drain_via_pre_step_callback_end_to_end():
    """Exercises the full production pattern: eviction queue -> pre-step
    callback -> drain_request_blocks -> block returned to pool."""
    wrapper, stm, pool = _make_mgr(num_gpu_blocks=16)
    block_ids = _allocate_request(stm, pool, "req-0", num_blocks=3)
    pending: dict[str, list[int]] = {"req-0": [block_ids[0], block_ids[2]]}
    free_before = pool.get_num_free_blocks()

    def drain_cb(mgr):
        for rid, bids in list(pending.items()):
            mgr.drain_request_blocks(rid, bids)
            pending.pop(rid)

    wrapper.register_pre_step_callback(drain_cb)
    wrapper.new_step_starts()

    assert pool.get_num_free_blocks() == free_before + 2
    assert stm.req_to_blocks["req-0"][0] is stm._null_block
    assert stm.req_to_blocks["req-0"][1].block_id == block_ids[1]
    assert stm.req_to_blocks["req-0"][2] is stm._null_block
    assert pending == {}
