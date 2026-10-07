# SPDX-License-Identifier: Apache-2.0
"""A hybrid model's O(1) state layers draw from their own block pool.

Mamba/GDN layers hold a fixed number of blocks per running request, never one
per token. Sharing the attention pool gives every recurrent layer a tensor of
``num_blocks`` state pages and unifies its page with the attention page, which
spends the budget on state no request can use (on Qwen3.5-0.8B at 10 GB that
was 5,728 tokens against 355,696 with separate pools). These tests pin the
split layout: the planner, the scheduler's pools, and the worker's copies.
"""

from dataclasses import dataclass

import pytest
import torch

from vllm.config import DeviceConfig, ModelConfig, SchedulerConfig, VllmConfig
from vllm.sampling_params import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import (
    KVCacheBlockCopy,
    get_kv_cache_configs,
    get_request_block_hasher,
    init_none_hash,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request
from vllm.v1.worker.utils import copy_kv_cache_blocks_inplace

pytestmark = pytest.mark.cpu_test

BLOCK_SIZE = 16
MAX_MODEL_LEN = 1024
MAX_NUM_SEQS = 8
NUM_LAYERS = 24  # every 4th is attention: 6 attention + 18 GDN layers


@pytest.fixture(autouse=True)
def _fixed_sampler_reserve(monkeypatch):
    monkeypatch.setenv("VLLM_SAMPLER_RESERVE_MIB", "64")


def _attention_spec():
    return FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=2, head_size=256, dtype=torch.bfloat16
    )


def _mamba_spec():
    # A GDN state: far larger than one attention page.
    return MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((3, 8192), (16, 128, 128)),
        dtypes=(torch.bfloat16, torch.bfloat16),
        mamba_cache_mode="none",
    )


def _specs(attention_spec=None):
    attention_spec = attention_spec or _attention_spec()
    return {
        f"layer_{i}": attention_spec if (i + 1) % 4 == 0 else _mamba_spec()
        for i in range(NUM_LAYERS)
    }


def _config(max_model_len=4096):
    model_config = ModelConfig(max_model_len=max_model_len)
    config = VllmConfig(
        model_config=model_config,
        device_config=DeviceConfig(device="cpu"),
        scheduler_config=SchedulerConfig(
            max_num_batched_tokens=max_model_len,
            max_num_seqs=MAX_NUM_SEQS,
            enable_chunked_prefill=True,
            max_model_len=max_model_len,
            is_encoder_decoder=model_config.is_encoder_decoder,
        ),
    )
    config.cache_config.kv_cache_layout = "LBNHC"
    config.cache_config.mamba_cache_mode = "none"
    config.scheduler_config.disable_hybrid_kv_cache_manager = False
    return config


def _plan(specs, available=4 * 1024**3, vllm_config=None):
    return get_kv_cache_configs(vllm_config or _config(), [specs], [available])[0]


def _mamba_groups(config):
    return [
        g
        for g in range(len(config.kv_cache_groups))
        if isinstance(config.kv_cache_groups[g].kv_cache_spec, MambaSpec)
    ]


def test_state_groups_get_a_pool_sized_for_running_requests():
    config = _plan(_specs())
    mamba_groups = _mamba_groups(config)

    assert len(mamba_groups) == 3
    # One state block per running request per group ("none" mode), + null.
    assert config.mamba_pool_num_blocks == MAX_NUM_SEQS * len(mamba_groups) + 1
    assert all(config.uses_mamba_pool(g) for g in mamba_groups)
    for g in mamba_groups:
        assert config.group_num_blocks(g) == config.mamba_pool_num_blocks
    assert config.group_num_blocks(0 if 0 not in mamba_groups else 3) == (
        config.num_blocks
    )


def test_split_pool_buys_the_attention_capacity_the_shared_pool_wasted():
    split = _plan(_specs())
    shared_config = _config()
    shared_config.cache_config.mamba_cache_mode = "all"
    shared_specs = {
        k: (
            MambaSpec(
                block_size=v.block_size,
                shapes=v.shapes,
                dtypes=v.dtypes,
                mamba_cache_mode="all",
            )
            if isinstance(v, MambaSpec)
            else v
        )
        for k, v in _specs().items()
    }
    shared = _plan(shared_specs, vllm_config=shared_config)

    assert shared.mamba_pool_num_blocks is None
    # Same bytes; 16.9x the attention blocks here with bf16 attention (more
    # with a compressed attention page, whose block a state page dwarfs).
    assert split.num_blocks > 15 * shared.num_blocks


def test_pools_never_alias_and_fit_the_allocation():
    config = _plan(_specs())
    (size,) = {t.size for t in config.kv_cache_tensors}
    main_end = 0
    for t in config.kv_cache_tensors:
        spec = next(
            g.kv_cache_spec
            for g in config.kv_cache_groups
            if t.layers[0] in g.layer_names
        )
        blocks = config.num_blocks_of(t)
        end = t.offset + (len(t.layers) - 1) * t.layer_stride + (
            blocks - 1
        ) * t.block_stride + spec.page_size_bytes
        assert end <= size
        if t.num_blocks is None:
            main_end = max(main_end, end)
            assert t.offset < config.mamba_pool_offset
        else:
            assert t.num_blocks == config.mamba_pool_num_blocks
            assert t.offset >= config.mamba_pool_offset
    assert main_end <= config.mamba_pool_offset


@pytest.mark.parametrize(
    "feature",
    ["kv_transfer_config", "decode_context_parallel_size", "mamba_cache_mode_all"],
)
def test_shared_pool_kept_where_ids_cross_groups(feature):
    """Features that address blocks across groups by one id space keep the
    shared pool."""
    from vllm.v1.core.kv_cache_utils import _mamba_pool_eligible

    vllm_config = _config()
    specs = list(_specs().values())
    assert _mamba_pool_eligible(vllm_config, specs)
    if feature == "kv_transfer_config":
        object.__setattr__(vllm_config, "kv_transfer_config", object())
    elif feature == "decode_context_parallel_size":
        object.__setattr__(vllm_config.parallel_config, feature, 2)
    else:
        vllm_config.cache_config.mamba_cache_mode = "all"
    assert not _mamba_pool_eligible(vllm_config, specs)


# ── Scheduler side ───────────────────────────────────────────────────────────


def _manager_config(num_blocks=64, mamba_blocks=5):
    attention = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float32
    )
    # "none" mode: the state block spans the whole sequence (upstream sets
    # mamba_block_size = max_model_len), one block per request per group.
    mamba = MambaSpec(block_size=MAX_MODEL_LEN, shapes=((1, 1),), dtypes=(torch.float32,))
    return KVCacheConfig(
        num_blocks=num_blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["gdn_a"], mamba),
            KVCacheGroupSpec(["gdn_b"], mamba),
        ],
        mamba_pool_num_blocks=mamba_blocks,
        mamba_pool_offset=0,
    )


def _request(request_id, num_tokens):
    from vllm.utils.hashing import sha256

    init_none_hash(sha256)
    params = SamplingParams(max_tokens=4)
    params.update_from_generation_config({}, eos_token_id=100)
    return Request(
        request_id=request_id,
        prompt_token_ids=list(range(num_tokens)),
        sampling_params=params,
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK_SIZE, sha256),
    )


def _manager(config):
    return KVCacheManager(
        config,
        max_model_len=MAX_MODEL_LEN,
        scheduler_block_size=MAX_MODEL_LEN,
        hash_block_size=BLOCK_SIZE,
        enable_caching=False,
    )


def test_state_groups_allocate_from_their_own_pool():
    manager = _manager(_manager_config())
    coordinator = manager.coordinator
    attn_mgr, gdn_a, gdn_b = coordinator.single_type_managers

    assert attn_mgr.block_pool is coordinator.block_pool
    assert gdn_a.block_pool is gdn_b.block_pool is coordinator.mamba_block_pool
    assert coordinator.block_pool.num_gpu_blocks == 64
    assert coordinator.mamba_block_pool.num_gpu_blocks == 5

    blocks = manager.allocate_slots(_request("r0", 3 * BLOCK_SIZE), 3 * BLOCK_SIZE)
    assert blocks is not None
    attn_blocks, a_blocks, b_blocks = blocks.blocks
    assert len(attn_blocks) == 3
    assert all(b.pool is coordinator.block_pool for b in attn_blocks)
    assert all(b.pool is coordinator.mamba_block_pool for b in a_blocks + b_blocks)


def test_admission_checks_each_pool_against_its_own_blocks():
    # 4 usable state blocks: two requests fill them (one per group each).
    manager = _manager(_manager_config(mamba_blocks=5))
    assert manager.allocate_slots(_request("r0", BLOCK_SIZE), BLOCK_SIZE)
    assert manager.allocate_slots(_request("r1", BLOCK_SIZE), BLOCK_SIZE)
    assert manager.coordinator.block_pool.get_num_free_blocks() > 50
    # Plenty of attention blocks, no state blocks: refused, not over-allocated.
    assert manager.allocate_slots(_request("r2", BLOCK_SIZE), BLOCK_SIZE) is None


def test_deferred_frees_return_blocks_to_their_own_pools():
    manager = _manager(_manager_config())
    coordinator = manager.coordinator
    main_free = coordinator.block_pool.get_num_free_blocks()
    mamba_free = coordinator.mamba_block_pool.get_num_free_blocks()
    request = _request("r0", 2 * BLOCK_SIZE)
    assert manager.allocate_slots(request, 2 * BLOCK_SIZE)

    # The scheduler's deferred path: pop every group's blocks, free them
    # through the main pool in one call.
    blocks = manager.pop_blocks_for_free(request)
    manager.block_pool.free_blocks(reversed(blocks))

    assert coordinator.block_pool.get_num_free_blocks() == main_free
    assert coordinator.mamba_block_pool.get_num_free_blocks() == mamba_free


def test_usage_reports_the_fuller_pool():
    manager = _manager(_manager_config(mamba_blocks=5))
    manager.allocate_slots(_request("r0", BLOCK_SIZE), BLOCK_SIZE)
    assert manager.usage == pytest.approx(2 / 4)


# ── Worker side ──────────────────────────────────────────────────────────────


def test_block_copies_reach_only_their_own_pool(monkeypatch):
    import vllm.v1.worker.utils as worker_utils

    monkeypatch.setattr(
        worker_utils,
        "async_tensor_h2d",
        lambda data, device=None, dtype=None: torch.tensor(data, device=device),
    )
    num_main, num_mamba, page = 6, 3, 4
    buf = torch.zeros(num_main * page + num_mamba * page, dtype=torch.int8)
    main = buf[: num_main * page].view(num_main, page)
    mamba = buf[num_main * page :].view(num_mamba, page)
    main.copy_(torch.arange(num_main).repeat_interleave(page).view(num_main, page))
    mamba.copy_(
        (10 + torch.arange(num_mamba)).repeat_interleave(page).view(num_mamba, page)
    )

    copies = [KVCacheBlockCopy(1, 4), KVCacheBlockCopy(num_main + 2, num_main + 1)]
    copy_kv_cache_blocks_inplace(
        [main, mamba], num_main, copies, mamba_pool=(num_main * page, num_mamba)
    )

    assert main[:, 0].tolist() == [0, 1, 2, 3, 1, 5]
    assert mamba[:, 0].tolist() == [10, 12, 12]


# ── Composite (fused) attention pages on the split layout ────────────────────


@dataclass(frozen=True, kw_only=True)
class _FusedAttentionSpec(FullAttentionSpec):
    """Six attention layers packed into one shared page (smart-mix)."""

    layer_slot_bytes: tuple[int, ...] = (544, 544, 544, 480, 544, 608)

    @property
    def num_heads(self) -> int:
        return 1

    @property
    def state_content_size_bytes(self) -> int:
        return sum(self.layer_slot_bytes)

    @property
    def aggregated_layer_count(self) -> int:
        return len(self.layer_slot_bytes)

    @classmethod
    def merge(cls, specs):
        return specs[0]


def test_composite_attention_shares_one_dense_page_beside_the_state_pool():
    fused = _FusedAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=2, head_size=256, dtype=torch.uint8
    )
    config = _plan(_specs(fused))
    attention = [
        t
        for t in config.kv_cache_tensors
        if t.num_blocks is None and t.layers[0] in {f"layer_{i}" for i in range(3, 24, 4)}
    ]

    assert config.mamba_pool_num_blocks is not None
    assert len(attention) == 6
    # Every fused layer views the same page region, laid out block-dense so
    # the plugin can carve its layer-major regions out of it.
    assert len({(t.offset, t.block_stride) for t in attention}) == 1
    assert attention[0].block_stride == fused.page_size_bytes
    assert attention[0].offset + config.num_blocks * fused.page_size_bytes <= (
        config.mamba_pool_offset
    )
