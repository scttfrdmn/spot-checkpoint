"""Tests for the spot-checkpoint CLI."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np
import pytest
from typer.testing import CliRunner

from spot_checkpoint.cli import app
from spot_checkpoint.storage import LocalStore

runner = CliRunner()


def _save_sync(store: LocalStore, ckpt_id: str, ts_offset: float = 0.0) -> None:
    """Synchronously save a minimal checkpoint with a patched timestamp."""

    async def _inner() -> None:
        tensor = np.zeros((4,), dtype=np.float64)
        await store.save_checkpoint(ckpt_id, {"state": tensor}, {"method": "fake"})
        manifest_path = store._ckpt_dir / ckpt_id / "_manifest.json"
        data = json.loads(manifest_path.read_text())
        data["timestamp"] = 1000.0 + ts_offset
        manifest_path.write_text(json.dumps(data))

    asyncio.run(_inner())


@pytest.fixture
def populated_store(tmp_path: Path) -> tuple[Path, str]:
    """A LocalStore with 3 checkpoints at controlled timestamps (-200s, -100s, 0s)."""
    job_id = "test-job"
    store = LocalStore(base_dir=tmp_path, job_id=job_id)
    _save_sync(store, "ckpt-001", -200.0)
    _save_sync(store, "ckpt-002", -100.0)
    _save_sync(store, "ckpt-003", 0.0)
    return tmp_path, job_id


class TestHelp:
    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "info" in result.output
        assert "gc" in result.output
        assert "bench" in result.output

    def test_list_help(self) -> None:
        result = runner.invoke(app, ["list", "--help"])
        assert result.exit_code == 0

    def test_gc_help(self) -> None:
        result = runner.invoke(app, ["gc", "--help"])
        assert result.exit_code == 0


class TestListCommand:
    def test_list_empty(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["list", str(tmp_path), "no-such-job"])
        assert result.exit_code == 0
        assert "No checkpoints found" in result.output

    def test_list_populated(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["list", str(path), job_id])
        assert result.exit_code == 0
        assert "ckpt-001" in result.output
        assert "ckpt-002" in result.output
        assert "ckpt-003" in result.output

    def test_list_json(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["list", "--json", str(path), job_id])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert isinstance(data, list)
        assert len(data) == 3

    def test_list_json_fields(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["list", "--json", str(path), job_id])
        data = json.loads(result.output)
        assert "checkpoint_id" in data[0]
        assert "method" in data[0]
        assert "timestamp" in data[0]
        assert "total_bytes" in data[0]


class TestInfoCommand:
    def test_info_latest(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["info", str(path), job_id])
        assert result.exit_code == 0
        # ckpt-003 has the highest timestamp (1000.0)
        assert "ckpt-003" in result.output

    def test_info_specific(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["info", str(path), job_id, "ckpt-001"])
        assert result.exit_code == 0
        assert "ckpt-001" in result.output

    def test_info_json(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["info", "--json", str(path), job_id])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "checkpoint_id" in data
        assert data["checkpoint_id"] == "ckpt-003"

    def test_info_not_found(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["info", str(path), job_id, "missing-id"])
        assert result.exit_code == 1

    def test_info_tensor_table(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["info", str(path), job_id])
        assert result.exit_code == 0
        assert "state" in result.output


class TestGcCommand:
    def test_gc_no_keep(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["gc", str(path), job_id])
        assert result.exit_code == 0
        assert "0" in result.output  # 0 deleted

    def test_gc_keep_one(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["gc", "--keep", "1", str(path), job_id])
        assert result.exit_code == 0
        assert "2" in result.output  # 2 deleted

    def test_gc_dry_run(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["gc", "--keep", "1", "--dry-run", str(path), job_id])
        assert result.exit_code == 0
        # Store should be unchanged
        store = LocalStore(base_dir=path, job_id=job_id)

        async def _count() -> int:
            return len(await store.list_checkpoints(""))

        assert asyncio.run(_count()) == 3

    def test_gc_json(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["gc", "--json", "--keep", "2", str(path), job_id])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "total" in data
        assert "kept" in data
        assert "deleted" in data


class TestStatusCommand:
    def test_status_latest(self, populated_store: tuple[Path, str]) -> None:
        """status shows the latest checkpoint's info table, exits 0."""
        path, job_id = populated_store
        result = runner.invoke(app, ["status", str(path), job_id])
        assert result.exit_code == 0
        # ckpt-003 has highest timestamp
        assert "ckpt-003" in result.output

    def test_status_json(self, populated_store: tuple[Path, str]) -> None:
        """status --json emits a dict with required keys, exits 0."""
        path, job_id = populated_store
        result = runner.invoke(app, ["status", "--json", str(path), job_id])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "checkpoint_id" in data
        assert "method" in data
        assert "timestamp" in data
        assert "total_bytes" in data

    def test_status_empty_store(self, tmp_path: Path) -> None:
        """status exits with code 1 when no checkpoints exist."""
        result = runner.invoke(app, ["status", str(tmp_path), "no-such-job"])
        assert result.exit_code == 1


class TestValidateCommand:
    def test_validate_latest(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["validate", str(path), job_id])
        assert result.exit_code == 0
        assert "ckpt-003" in result.output
        assert "OK" in result.output

    def test_validate_specific(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["validate", str(path), job_id, "ckpt-001"])
        assert result.exit_code == 0
        assert "ckpt-001" in result.output

    def test_validate_json(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["validate", "--json", str(path), job_id])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["ok"] is True
        assert "tensors" in data
        assert data["tensors"][0]["ok"] is True

    def test_validate_empty_store(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["validate", str(tmp_path), "no-such-job"])
        assert result.exit_code == 1

    def test_validate_not_found(self, populated_store: tuple[Path, str]) -> None:
        path, job_id = populated_store
        result = runner.invoke(app, ["validate", str(path), job_id, "nonexistent"])
        assert result.exit_code == 1


class TestBenchCommand:
    def test_bench_local(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["bench", str(tmp_path), "--size-mb", "1"])
        assert result.exit_code == 0
        assert "MB/s" in result.output

    def test_bench_json(self, tmp_path: Path) -> None:
        result = runner.invoke(app, ["bench", "--json", str(tmp_path), "--size-mb", "1"])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert "size_mb" in data
        assert "write_mbps" in data
        assert "read_mbps" in data
        assert "write_elapsed_s" in data
        assert "read_elapsed_s" in data
        assert "concurrency" in data


class TestMakeStore:
    def test_make_store_empty_bucket_raises(self) -> None:
        """_make_store('s3://', 'job') raises ValueError for empty bucket."""
        from spot_checkpoint.cli import _make_store

        with pytest.raises(ValueError, match="bucket name"):
            _make_store("s3://", "job")

    def test_env_var_invalid_int_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Invalid SPOT_CHECKPOINT_SHARD_SIZE falls back to default, no exception."""
        from spot_checkpoint.cli import _make_store

        monkeypatch.setenv("SPOT_CHECKPOINT_SHARD_SIZE", "abc")
        # Should not raise; returns an S3ShardedStore with default shard_size
        # We patch S3ShardedStore to avoid real AWS calls
        from unittest.mock import MagicMock, patch

        mock_store = MagicMock()
        with patch("spot_checkpoint.cli.S3ShardedStore", return_value=mock_store) as mock_cls:
            store = _make_store("s3://my-bucket", "job")
        # Verify it was called with an integer shard_size (the default)
        _, kwargs = mock_cls.call_args
        assert isinstance(kwargs["shard_size"], int)


class TestRestoreCommandOverwrite:
    def test_restore_warns_on_overwrite(
        self, populated_store: tuple[Path, str], tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """restore command logs a warning when an .npy file already exists."""
        import logging

        path, job_id = populated_store
        output_dir = tmp_path / "output"
        output_dir.mkdir()

        # Pre-create the output file to trigger overwrite warning
        (output_dir / "state.npy").write_bytes(b"old")

        with caplog.at_level(logging.WARNING, logger="spot_checkpoint.cli"):
            result = runner.invoke(
                app,
                ["restore", str(path), job_id, "--output", str(output_dir)],
            )

        assert result.exit_code == 0
        assert any("Overwriting" in r.message for r in caplog.records)
