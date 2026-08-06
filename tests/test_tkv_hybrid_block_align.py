# SPDX-License-Identifier: Apache-2.0
"""Every tkv-family cache dtype must skip hybrid block-size alignment.

The alignment raises attention ``block_size`` to cover the mamba page size.
TKV sizes attention and mamba from separate per-group pools, so applying it
collapses attention KV capacity by ~100x. The guard was an equality test on
``"tkv"``, which let ``"tkv-bypass"`` fall through and serve at block 832
while ``"tkv"`` served at 32.
"""

from types import SimpleNamespace

import pytest

from vllm.platforms.interface import Platform

TKV_DTYPES = ["tkv", "tkv-bypass"]


def _config(cache_dtype: str, block_size: int = 32):
    return SimpleNamespace(
        cache_config=SimpleNamespace(cache_dtype=cache_dtype, block_size=block_size),
        model_config=None,
        parallel_config=None,
    )


@pytest.mark.parametrize("cache_dtype", TKV_DTYPES)
def test_tkv_dtypes_leave_block_size_untouched(cache_dtype):
    cfg = _config(cache_dtype)
    Platform._align_hybrid_block_size(cfg, backend_cls=None)
    assert cfg.cache_config.block_size == 32, (
        f"{cache_dtype} was realigned; it must keep its own per-group page size"
    )


def test_non_tkv_dtype_does_not_return_early():
    cfg = _config("auto")
    with pytest.raises(AttributeError):
        Platform._align_hybrid_block_size(cfg, backend_cls=None)
