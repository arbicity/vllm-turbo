# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from functools import cache
from typing import NamedTuple, cast, get_args

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.utils.import_utils import resolve_obj_by_qualname
from vllm.v1.attention.backend import AttentionBackend, AttentionType
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
)

logger = init_logger(__name__)


class AttentionSelectorConfig(NamedTuple):
    head_size: int
    dtype: torch.dtype
    kv_cache_dtype: CacheDType | None
    block_size: int | None
    use_mla: bool = False
    has_sink: bool = False
    use_sparse: bool = False
    use_mm_prefix: bool = False
    use_per_head_quant_scales: bool = False
    attn_type: str = AttentionType.DECODER
    use_non_causal: bool = False
    use_batch_invariant: bool = False
    use_kv_connector: bool = False

    def __repr__(self):
        return (
            f"AttentionSelectorConfig(head_size={self.head_size}, "
            f"dtype={self.dtype}, "
            f"kv_cache_dtype={self.kv_cache_dtype}, "
            f"block_size={self.block_size}, "
            f"use_mla={self.use_mla}, "
            f"has_sink={self.has_sink}, "
            f"use_sparse={self.use_sparse}, "
            f"use_mm_prefix={self.use_mm_prefix}, "
            f"use_per_head_quant_scales={self.use_per_head_quant_scales}, "
            f"attn_type={self.attn_type}, "
            f"use_non_causal={self.use_non_causal}, "
            f"use_batch_invariant={self.use_batch_invariant}, "
            f"use_kv_connector={self.use_kv_connector})"
        )


def get_attn_backend(
    head_size: int,
    dtype: torch.dtype,
    kv_cache_dtype: str | None,
    use_mla: bool = False,
    has_sink: bool = False,
    use_sparse: bool = False,
    use_mm_prefix: bool = False,
    use_per_head_quant_scales: bool = False,
    attn_type: str | None = None,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    """Selects which attention backend to use and lazily imports it."""

    if kv_cache_dtype is not None:
        from vllm.config.cache import validate_cache_dtype
        # Raises ValueError with a clear message if the dtype isn't
        # builtin or plugin-registered.
        validate_cache_dtype(kv_cache_dtype)

    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()

    cache_config = vllm_config.cache_config
    if cache_config is not None and cache_config.user_specified_block_size:
        block_size = cache_config.block_size
    else:
        block_size = None

    kv_transfer_config = vllm_config.kv_transfer_config
    use_kv_connector = (
        kv_transfer_config is not None and kv_transfer_config.is_kv_transfer_instance
    )

    attn_selector_config = AttentionSelectorConfig(
        head_size=head_size,
        dtype=dtype,
        kv_cache_dtype=cast(CacheDType | None, kv_cache_dtype),
        block_size=block_size,
        use_mla=use_mla,
        has_sink=has_sink,
        use_sparse=use_sparse,
        use_mm_prefix=use_mm_prefix,
        use_per_head_quant_scales=use_per_head_quant_scales,
        attn_type=attn_type or AttentionType.DECODER,
        use_non_causal=vllm_config.attention_config.use_non_causal,
        use_batch_invariant=envs.VLLM_BATCH_INVARIANT,
        use_kv_connector=use_kv_connector,
    )

    return _cached_get_attn_backend(
        backend=vllm_config.attention_config.backend,
        attn_selector_config=attn_selector_config,
        num_heads=num_heads,
    )


@cache
def _cached_get_attn_backend(
    backend,
    attn_selector_config: AttentionSelectorConfig,
    num_heads: int | None = None,
) -> type[AttentionBackend]:
    from vllm.platforms import current_platform

    # MLA wrapping: when the user selected an attention backend that
    # declares wraps_mla_backend (i.e. it isn't an MLA backend itself,
    # but knows how to wrap one), we fall through to standard MLA
    # candidate selection — ignoring the user's --attention-backend
    # for the MLA layer's primary pick — and apply the wrapper after.
    # The dtype gate is also lifted (kv_cache_dtype="auto") because the
    # wrapper class is what supports the user's compressed dtype, not
    # the picked base MLA backend.
    user_selected_backend_cls = None
    if backend is not None and attn_selector_config.use_mla:
        try:
            _maybe_user_cls = backend.get_class()
        except (ValueError, ImportError):
            _maybe_user_cls = None
        if _maybe_user_cls is not None:
            _user_wraps_fn = getattr(
                _maybe_user_cls, "wraps_mla_backend", None,
            )
            _base_wraps_fn = getattr(AttentionBackend, "wraps_mla_backend", None)
            # Only treat as wrapper-capable if the backend overrode the
            # default no-op (avoids matching every backend).
            if (
                _user_wraps_fn is not None
                and _base_wraps_fn is not None
                and getattr(_user_wraps_fn, "__func__", _user_wraps_fn)
                is not getattr(_base_wraps_fn, "__func__", _base_wraps_fn)
            ):
                user_selected_backend_cls = _maybe_user_cls

    if user_selected_backend_cls is not None:
        # Fall through to standard MLA selection without the user's
        # backend constraint AND with dtype gate lifted (kv_cache_dtype=
        # "auto"). The wrapper class will be applied below.
        from typing import cast

        relaxed_config = attn_selector_config._replace(
            kv_cache_dtype=cast(CacheDType | None, "auto"),
        )
        attention_cls = current_platform.get_attn_backend_cls(
            None,
            attn_selector_config=relaxed_config,
            num_heads=num_heads,
        )
    else:
        attention_cls = current_platform.get_attn_backend_cls(
            backend,
            attn_selector_config=attn_selector_config,
            num_heads=num_heads,
        )
    if not attention_cls:
        raise ValueError(
            f"Invalid attention backend for {current_platform.device_name}"
        )
    backend = resolve_obj_by_qualname(attention_cls)

    # Apply the MLA wrapper if applicable. Wrapper's
    # supported_kv_cache_dtypes is what advertises the user's dtype;
    # the base backend's spec/impl is wrapped to compress KV.
    if user_selected_backend_cls is not None:
        wrapper = user_selected_backend_cls.wraps_mla_backend(backend)
        if wrapper is not None and wrapper is not backend:
            logger.info(
                "Wrapping MLA backend %s with %s.",
                backend.__name__,
                getattr(wrapper, "__name__", repr(wrapper)),
            )
            backend = wrapper

    # Adjust kv cache layout if the selected backend requires a specific one
    required_layout = backend.get_required_kv_cache_layout()
    if required_layout is not None:
        from vllm.v1.attention.backends.utils import set_kv_cache_layout

        set_kv_cache_layout(required_layout)
        logger.info(
            "Using %s KV cache layout for %s backend.",
            required_layout,
            backend.get_name(),
        )

    return backend


def get_mamba_attn_backend(
    mamba_type: MambaAttentionBackendEnum,
) -> type[AttentionBackend]:
    """Select which mamba attention backend to use and lazily import it."""
    return _cached_get_mamba_attn_backend(mamba_type)


@cache
def _cached_get_mamba_attn_backend(
    mamba_type: MambaAttentionBackendEnum,
) -> type[AttentionBackend]:
    assert mamba_type and isinstance(mamba_type, MambaAttentionBackendEnum)

    mamba_attn_backend = mamba_type.get_class()
    if envs.VLLM_BATCH_INVARIANT and not mamba_attn_backend.supports_batch_invariance():
        raise RuntimeError(
            "VLLM batch_invariant mode is not supported for "
            f"{mamba_attn_backend.get_name()}."
        )
    return mamba_attn_backend
