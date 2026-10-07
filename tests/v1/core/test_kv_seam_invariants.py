# SPDX-License-Identifier: Apache-2.0
"""Plugin KV-cache dtypes: the registry the tkv/turbo-attn seam depends on."""

import pytest
import torch

from vllm.config.cache import (
    _PLUGIN_CACHE_DTYPES,
    CacheConfig,
    cache_dtype_choices,
    is_plugin_cache_dtype,
    register_cache_dtype,
    validate_cache_dtype,
)
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.kv_cache_interface import KVQuantMode, get_kv_quant_mode

pytestmark = pytest.mark.cpu_test

NAME = "test_seam_plugin_kv_dtype"


@pytest.fixture
def plugin_dtype():
    register_cache_dtype(NAME, torch.uint8)
    try:
        yield NAME
    finally:
        _PLUGIN_CACHE_DTYPES.discard(NAME)
        STR_DTYPE_TO_TORCH_DTYPE.pop(NAME, None)


def test_registered_dtype_is_constructible(plugin_dtype):
    # A Literal annotation on CacheConfig.cache_dtype would be compiled into
    # the pydantic validator before any plugin loads, so this would raise.
    assert CacheConfig(cache_dtype=plugin_dtype).cache_dtype == plugin_dtype


def test_registered_dtype_is_listed_and_mapped(plugin_dtype):
    assert plugin_dtype in cache_dtype_choices()
    assert validate_cache_dtype(plugin_dtype) == plugin_dtype
    assert is_plugin_cache_dtype(plugin_dtype)
    assert STR_DTYPE_TO_TORCH_DTYPE[plugin_dtype] is torch.uint8
    # The backend owns the layout: KVQuantMode does not model it.
    assert get_kv_quant_mode(plugin_dtype) == KVQuantMode.NONE


def test_unregistered_dtype_is_refused():
    with pytest.raises(ValueError, match="Unknown --kv-cache-dtype"):
        validate_cache_dtype("no_such_kv_dtype")
    with pytest.raises(ValueError):
        CacheConfig(cache_dtype="no_such_kv_dtype")


@pytest.mark.parametrize("cache_dtype", ["auto", "bfloat16", "fp8"])
def test_builtin_dtypes_are_not_plugin_dtypes(cache_dtype):
    assert validate_cache_dtype(cache_dtype) == cache_dtype
    assert not is_plugin_cache_dtype(cache_dtype)
