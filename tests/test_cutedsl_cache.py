# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import os
from pathlib import Path

from vllm.utils import cutedsl_cache


def test_operator_set_cache_dir_wins(monkeypatch, tmp_path):
    operator_dir = str(tmp_path / "operator-cache")
    monkeypatch.setenv(cutedsl_cache.CUTEDSL_CACHE_DIR_ENV, operator_dir)

    cutedsl_cache.ensure_persistent_cutedsl_cache_dir()

    assert os.environ[cutedsl_cache.CUTEDSL_CACHE_DIR_ENV] == operator_dir
    # The operator's setting is respected verbatim — not even created.
    assert not os.path.exists(operator_dir)


def test_default_cache_dir_created_and_exported(monkeypatch, tmp_path):
    monkeypatch.delenv(cutedsl_cache.CUTEDSL_CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    cutedsl_cache.ensure_persistent_cutedsl_cache_dir()

    expected = str(tmp_path / "tkv" / "cute_dsl_cache")
    assert os.environ[cutedsl_cache.CUTEDSL_CACHE_DIR_ENV] == expected
    assert os.path.isdir(expected)


def test_idempotent(monkeypatch, tmp_path):
    monkeypatch.delenv(cutedsl_cache.CUTEDSL_CACHE_DIR_ENV, raising=False)
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))

    cutedsl_cache.ensure_persistent_cutedsl_cache_dir()
    first = os.environ[cutedsl_cache.CUTEDSL_CACHE_DIR_ENV]
    cutedsl_cache.ensure_persistent_cutedsl_cache_dir()

    assert os.environ[cutedsl_cache.CUTEDSL_CACHE_DIR_ENV] == first


def test_artifact_count(monkeypatch, tmp_path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    monkeypatch.setenv(cutedsl_cache.CUTEDSL_CACHE_DIR_ENV, str(cache_dir))

    assert cutedsl_cache.cutedsl_cache_artifact_count() == 0
    Path(cache_dir / "cute_dsl_abc123.mlir").touch()
    Path(cache_dir / "cute_dsl_def456.mlir").touch()
    Path(cache_dir / "not-an-artifact.txt").touch()
    assert cutedsl_cache.cutedsl_cache_artifact_count() == 2


def test_artifact_count_missing_dir(monkeypatch, tmp_path):
    monkeypatch.setenv(
        cutedsl_cache.CUTEDSL_CACHE_DIR_ENV, str(tmp_path / "does-not-exist")
    )
    assert cutedsl_cache.cutedsl_cache_artifact_count() == 0

    monkeypatch.delenv(cutedsl_cache.CUTEDSL_CACHE_DIR_ENV, raising=False)
    assert cutedsl_cache.cutedsl_cache_artifact_count() == 0


# ------------------------------------------------------------------
# enable_cutedsl_compile_only_cache — patched against a fake cutlass DSL
# ------------------------------------------------------------------


def _install_fake_cutlass(monkeypatch, generate_mlir):
    import sys
    from types import ModuleType

    cutlass = ModuleType("cutlass")
    base_dsl = ModuleType("cutlass.base_dsl")
    dsl_mod = ModuleType("cutlass.base_dsl.dsl")

    class BaseDSL:
        pass

    BaseDSL.generate_mlir = generate_mlir
    dsl_mod.BaseDSL = BaseDSL
    base_dsl.dsl = dsl_mod
    cutlass.base_dsl = base_dsl
    monkeypatch.setitem(sys.modules, "cutlass", cutlass)
    monkeypatch.setitem(sys.modules, "cutlass.base_dsl", base_dsl)
    monkeypatch.setitem(sys.modules, "cutlass.base_dsl.dsl", dsl_mod)
    return BaseDSL


def _reset_patch_state(monkeypatch):
    monkeypatch.setattr(cutedsl_cache, "_compile_only_cache_patched", False)


def test_compile_only_cache_flips_no_cache(monkeypatch):
    _reset_patch_state(monkeypatch)
    seen = {}

    def generate_mlir(
        self,
        funcBody,
        function_name,
        gpu_module_attrs,
        args,
        kwonlyargs,
        sig,
        pipeline,
        no_cache,
        no_jit_engine,
        compile_only,
        location=None,
    ):
        seen["no_cache"] = no_cache
        return "ok"

    BaseDSL = _install_fake_cutlass(monkeypatch, generate_mlir)
    cutedsl_cache.enable_cutedsl_compile_only_cache()
    assert BaseDSL.generate_mlir is not generate_mlir

    # cute.compile style call: compile_only=True, no_cache=True — the
    # wrapper must flip no_cache back to False.
    dsl = BaseDSL()
    out = BaseDSL.generate_mlir(
        dsl, None, "fn", {}, (), {}, None, None, True, False, True
    )
    assert out == "ok"
    assert seen["no_cache"] is False

    # Plain call path (compile_only=False) stays untouched.
    BaseDSL.generate_mlir(dsl, None, "fn", {}, (), {}, None, None, True, False, False)
    assert seen["no_cache"] is True


def test_compile_only_cache_refuses_unknown_signature(monkeypatch):
    _reset_patch_state(monkeypatch)

    def generate_mlir(self, something_else):
        return "ok"

    BaseDSL = _install_fake_cutlass(monkeypatch, generate_mlir)
    cutedsl_cache.enable_cutedsl_compile_only_cache()
    # Refused: original left in place.
    assert BaseDSL.generate_mlir is generate_mlir


def test_compile_only_cache_idempotent(monkeypatch):
    _reset_patch_state(monkeypatch)

    def generate_mlir(
        self,
        funcBody,
        function_name,
        gpu_module_attrs,
        args,
        kwonlyargs,
        sig,
        pipeline,
        no_cache,
        no_jit_engine,
        compile_only,
        location=None,
    ):
        return "ok"

    BaseDSL = _install_fake_cutlass(monkeypatch, generate_mlir)
    cutedsl_cache.enable_cutedsl_compile_only_cache()
    patched = BaseDSL.generate_mlir
    cutedsl_cache.enable_cutedsl_compile_only_cache()
    assert BaseDSL.generate_mlir is patched
