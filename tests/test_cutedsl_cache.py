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
