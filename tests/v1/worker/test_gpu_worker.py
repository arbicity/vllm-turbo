# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from vllm.utils.mem_constants import GiB_bytes
from vllm.v1.worker import startup_plan
from vllm.v1.worker.startup_plan import (
    maybe_apply_startup_plan,
    maybe_save_startup_plan,
)

# Startup-plan persistence (vllm/v1/worker/startup_plan.py), applied and
# saved by Worker.determine_available_memory / compile_or_warm_up_model.


def _plan_worker(config_hash="abc123", free_memory=78 * GiB_bytes, kv_bytes=None):
    """The minimal Worker surface the startup-plan entry points touch."""
    return SimpleNamespace(
        vllm_config=SimpleNamespace(compute_hash=lambda: config_hash),
        rank=0,
        parallel_config=SimpleNamespace(world_size=1),
        init_snapshot=SimpleNamespace(free_memory=free_memory),
        cache_config=SimpleNamespace(kv_cache_memory_bytes=kv_bytes),
    )


def _plan_platform(name="NVIDIA H100 PCIe"):
    return SimpleNamespace(
        get_device_name=lambda device_id=0: name,
        get_device_total_memory=lambda device_id=0: 80 * GiB_bytes,
        get_device_capability=lambda device_id=0: (9, 0),
    )


@pytest.fixture
def plan_env(monkeypatch: pytest.MonkeyPatch, tmp_path):
    """Enable the startup plan, isolated under a tmp cache root."""
    monkeypatch.setenv("VLLM_ENABLE_STARTUP_PLAN", "1")
    monkeypatch.setenv("VLLM_CACHE_ROOT", str(tmp_path))
    with patch.object(startup_plan, "current_platform", _plan_platform()):
        yield


def test_startup_plan_fingerprint_sensitivity(plan_env):
    """The fingerprint is the OOM-safety key: stable for identical inputs,
    different for anything the profiled value depends on."""
    fp = startup_plan.compute_plan_fingerprint
    base = fp(_plan_worker().vllm_config, 0, 1)
    assert base == fp(_plan_worker().vllm_config, 0, 1)
    assert base != fp(_plan_worker("other").vllm_config, 0, 1)
    assert base != fp(_plan_worker().vllm_config, 1, 2)
    with patch.object(startup_plan, "current_platform", _plan_platform("NVIDIA A100")):
        assert base != fp(_plan_worker().vllm_config, 0, 1)
    with patch("vllm.__version__", "0.0.0+plan-test"):
        assert base != fp(_plan_worker().vllm_config, 0, 1)


def test_startup_plan_apply_gate(plan_env):
    """Only a fingerprint-matching, memory-safe plan is ever applied."""
    maybe_save_startup_plan(_plan_worker(), 50 * GiB_bytes)

    applied = _plan_worker()
    maybe_apply_startup_plan(applied)
    assert applied.cache_config.kv_cache_memory_bytes == 50 * GiB_bytes

    less_memory = _plan_worker(free_memory=60 * GiB_bytes)
    other_config = _plan_worker(config_hash="zzz999")
    for refused in (less_memory, other_config):
        maybe_apply_startup_plan(refused)
        assert refused.cache_config.kv_cache_memory_bytes is None

    # An explicit --kv-cache-memory is never overridden.
    explicit = _plan_worker(kv_bytes=7 * GiB_bytes)
    maybe_apply_startup_plan(explicit)
    assert explicit.cache_config.kv_cache_memory_bytes == 7 * GiB_bytes


# Attention-backend lifecycle hooks (vllm/v1/worker/gpu_worker.py). The
# engine dispatches on_model_loaded / on_draft_model_loaded once per
# resolved backend after Worker.load_model. Backends use them to run
# pre-flight validation and one-time weight transforms, so a hook that
# raises must abort the load rather than leave the model half-prepared.


def _hook_backend(name, record, *, raises_on=()):
    """A backend class exposing only the lifecycle hooks the worker calls."""

    def _hook(hook_name):
        @classmethod
        def impl(cls, *args):
            record.append((cls.__qualname__, hook_name))
            if hook_name in raises_on:
                raise ValueError(f"{cls.__qualname__} refuses {hook_name}")

        return impl

    return type(
        name,
        (),
        {
            "on_model_loaded": _hook("on_model_loaded"),
            "on_draft_model_loaded": _hook("on_draft_model_loaded"),
            "on_kv_cache_initialized": _hook("on_kv_cache_initialized"),
        },
    )


def _hook_worker(backends, draft_model=None):
    """The minimal Worker surface Worker.load_model touches."""
    from contextlib import nullcontext

    model_runner = SimpleNamespace(
        model=object(),
        attn_groups=[[SimpleNamespace(backend=b) for b in backends]],
        load_model=lambda **kwargs: None,
    )
    if draft_model is not None:
        model_runner.drafter = SimpleNamespace(model=draft_model)
    return SimpleNamespace(
        vllm_config=SimpleNamespace(weight_transfer_config=None),
        model_runner=model_runner,
        _maybe_get_memory_pool_context=lambda tag: nullcontext(),
        _scoped_allocator_max_split=lambda max_split_size_mb: nullcontext(),
    )


@pytest.fixture
def load_model_call(monkeypatch: pytest.MonkeyPatch):
    """Invoke the real Worker.load_model against a stub worker."""
    from contextlib import nullcontext

    from vllm.v1.worker import gpu_worker

    monkeypatch.setattr(
        gpu_worker, "set_current_vllm_config", lambda cfg: nullcontext()
    )

    def call(worker):
        gpu_worker.Worker.load_model(worker)

    return call


def test_hooks_dispatch_to_every_backend_in_order(load_model_call):
    calls: list[tuple[str, str]] = []
    backends = [
        _hook_backend("BackendA", calls),
        _hook_backend("BackendB", calls),
    ]
    load_model_call(_hook_worker(backends))
    assert calls == [
        ("BackendA", "on_model_loaded"),
        ("BackendB", "on_model_loaded"),
    ]


def test_draft_hook_fires_per_backend_when_a_drafter_is_present(load_model_call):
    calls: list[tuple[str, str]] = []
    backends = [
        _hook_backend("BackendA", calls),
        _hook_backend("BackendB", calls),
    ]
    load_model_call(_hook_worker(backends, draft_model=object()))
    assert calls == [
        ("BackendA", "on_model_loaded"),
        ("BackendA", "on_draft_model_loaded"),
        ("BackendB", "on_model_loaded"),
        ("BackendB", "on_draft_model_loaded"),
    ]


def test_failing_hook_aborts_model_load_naming_the_backend(load_model_call):
    # Backends fold weights and run pre-flight validation in this hook. A
    # swallowed failure serves a model that skipped both, so the load dies.
    calls: list[tuple[str, str]] = []
    backends = [
        _hook_backend("BackendA", calls, raises_on=("on_model_loaded",)),
        _hook_backend("BackendB", calls),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        load_model_call(_hook_worker(backends))

    assert "BackendA" in str(excinfo.value)
    assert "on_model_loaded" in str(excinfo.value)
    assert isinstance(excinfo.value.__cause__, ValueError)
    assert "BackendA refuses on_model_loaded" in str(excinfo.value.__cause__)


def test_abort_is_ordered_and_says_which_backends_were_skipped(load_model_call):
    # Fail-fast, not best-effort: a half-applied weight transform must not
    # run a second backend's transform on top of it. The error states the
    # dispatch order so the skip is explicit rather than silent.
    calls: list[tuple[str, str]] = []
    backends = [
        _hook_backend("BackendA", calls, raises_on=("on_model_loaded",)),
        _hook_backend("BackendB", calls),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        load_model_call(_hook_worker(backends))

    assert calls == [("BackendA", "on_model_loaded")]
    message = str(excinfo.value)
    assert "BackendA, BackendB" in message
    assert "not dispatched" in message


def test_a_later_backend_failing_still_aborts_after_earlier_ones_ran(
    load_model_call,
):
    calls: list[tuple[str, str]] = []
    backends = [
        _hook_backend("BackendA", calls),
        _hook_backend("BackendB", calls, raises_on=("on_model_loaded",)),
    ]
    with pytest.raises(RuntimeError, match="BackendB"):
        load_model_call(_hook_worker(backends))

    assert calls == [
        ("BackendA", "on_model_loaded"),
        ("BackendB", "on_model_loaded"),
    ]


def test_failing_draft_hook_aborts_model_load(load_model_call):
    calls: list[tuple[str, str]] = []
    backends = [
        _hook_backend("BackendA", calls, raises_on=("on_draft_model_loaded",)),
        _hook_backend("BackendB", calls),
    ]
    with pytest.raises(RuntimeError) as excinfo:
        load_model_call(_hook_worker(backends, draft_model=object()))

    assert "on_draft_model_loaded" in str(excinfo.value)
    assert calls == [
        ("BackendA", "on_model_loaded"),
        ("BackendA", "on_draft_model_loaded"),
    ]


def test_call_backend_hook_returns_the_hook_value():
    # determine_available_memory reads adjust_kv_budget's return value
    # through the same dispatcher.
    from vllm.v1.worker.gpu_worker import _call_backend_hook

    class Backend:
        @classmethod
        def adjust_kv_budget(cls, profiled_bytes, vllm_config):
            return profiled_bytes * 2

    assert _call_backend_hook(Backend, "adjust_kv_budget", [Backend], 16, None) == 32
