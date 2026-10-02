# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""MLA wrapper resolution for plugin-registered compressed KV dtypes.

An MLA model with a plugin KV-cache dtype has no candidate in the MLA
backend priority list — every true MLA backend rejects the dtype. The
selector must therefore resolve the plugin's wrapper backend from the
dtype alone, without the user naming it via ``--attention-backend``.
"""

import pytest

from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.v1.attention.selector import _mla_wrapper_cls, _mla_wrapper_for_dtype


class WrapperBackend(AttentionBackend):
    """Stands in for a plugin backend that wraps an MLA backend."""

    @classmethod
    def get_supported_kv_cache_dtypes(cls):
        return ["plugin_kv"]

    @classmethod
    def wraps_mla_backend(cls, base_mla_backend_cls):
        return base_mla_backend_cls


class PlainBackend(AttentionBackend):
    """A plugin backend that does not wrap MLA."""

    @classmethod
    def get_supported_kv_cache_dtypes(cls):
        return ["plugin_kv"]


@pytest.fixture
def registered():
    """Register the stand-ins as CUSTOM/TURBO_ATTN and clean up after."""
    register_backend(
        AttentionBackendEnum.TURBO_ATTN,
        f"{WrapperBackend.__module__}.{WrapperBackend.__qualname__}",
    )
    yield
    AttentionBackendEnum.TURBO_ATTN.clear_override()


def test_wrapper_resolved_from_dtype_without_explicit_backend(registered):
    assert _mla_wrapper_for_dtype("plugin_kv") is WrapperBackend


def test_unclaimed_dtype_resolves_to_nothing(registered):
    assert _mla_wrapper_for_dtype("fp8") is None


@pytest.mark.parametrize("dtype", [None, "auto"])
def test_uncompressed_dtype_never_resolves_a_wrapper(registered, dtype):
    assert _mla_wrapper_for_dtype(dtype) is None


def test_no_plugin_registered_leaves_mla_selection_unchanged():
    assert _mla_wrapper_for_dtype("plugin_kv") is None


def test_non_wrapping_backend_is_not_treated_as_a_wrapper():
    register_backend(
        AttentionBackendEnum.TURBO_ATTN,
        f"{PlainBackend.__module__}.{PlainBackend.__qualname__}",
    )
    try:
        assert _mla_wrapper_cls(AttentionBackendEnum.TURBO_ATTN) is None
        assert _mla_wrapper_for_dtype("plugin_kv") is None
    finally:
        AttentionBackendEnum.TURBO_ATTN.clear_override()


def test_stock_backend_is_not_a_wrapper():
    assert _mla_wrapper_cls(AttentionBackendEnum.TRITON_MLA) is None
