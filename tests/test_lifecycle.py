"""Tests for lifecycle backends and SpotLifecycleManager."""

import json
import os
import time
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from spot_checkpoint.lifecycle import (
    DirectEC2Backend,
    InterruptEvent,
    InterruptReason,
    LifecycleBackend,
    SlurmLifecycleBackend,
    SporeLifecycleBackend,
    SpotLifecycleManager,
    _detect_adapter_class,
    _status_from_store,
    detect_backend,
    spot_restore_async,
    spot_safe,
    spot_status,
    spot_status_async,
)
from spot_checkpoint.storage import LocalStore


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


class _NoOpBackend(LifecycleBackend):
    """Backend that does nothing — no signals, no threads."""

    def start(self, on_interrupt: Any) -> None:
        pass

    def stop(self) -> None:
        pass


class TestSporeLifecycleBackend:
    def test_detects_signal_file(self, tmp_path: Path):
        """Backend fires callback when spored signal file appears."""
        backend = SporeLifecycleBackend(poll_interval=0.1, interrupt_headroom=10)
        backend.SPOT_SIGNAL_FILE = str(tmp_path / "spot-signal.json")

        events: list[InterruptEvent] = []
        backend.start(on_interrupt=events.append)

        # Write signal file
        signal_data = {
            "event": "spot-interruption",
            "action": "terminate",
            "time": "2026-01-01T12:02:00Z",
            "detected_at": "2026-01-01T12:00:00Z",
        }
        Path(backend.SPOT_SIGNAL_FILE).write_text(json.dumps(signal_data))

        time.sleep(0.3)
        backend.stop()

        assert len(events) == 1
        assert events[0].reason == InterruptReason.SPOT_RECLAIM

    def test_no_false_positive(self, tmp_path: Path):
        """Backend does not fire when no signal file exists."""
        backend = SporeLifecycleBackend(poll_interval=0.1)
        backend.SPOT_SIGNAL_FILE = str(tmp_path / "nonexistent.json")

        events: list[InterruptEvent] = []
        backend.start(on_interrupt=events.append)

        time.sleep(0.3)
        backend.stop()

        assert len(events) == 0


class TestSlurmLifecycleBackend:
    def test_detect_slurm_env(self):
        """detect_backend picks Slurm when SLURM_JOB_ID is set."""
        with patch.dict(os.environ, {"SLURM_JOB_ID": "12345"}):
            backend = detect_backend()
            assert isinstance(backend, SlurmLifecycleBackend)


class TestDetectBackend:
    def test_fallback_to_direct_ec2(self):
        """Without Slurm or spored, falls back to DirectEC2Backend."""
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("spot_checkpoint.lifecycle._spored_is_running", return_value=False),
        ):
            backend = detect_backend()
            assert isinstance(backend, DirectEC2Backend)

    def test_spored_detected(self):
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("spot_checkpoint.lifecycle._spored_is_running", return_value=True),
        ):
            backend = detect_backend()
            assert isinstance(backend, SporeLifecycleBackend)


class TestKPointDetection:
    @pytest.mark.parametrize("cls_name", ["KRHF", "KUHF", "KRKS", "KUKS"])
    def test_kpoint_detected_as_scf(self, cls_name: str) -> None:
        """k-point solver class names in MRO map to SCFCheckpointAdapter."""
        from spot_checkpoint.lifecycle import _detect_adapter_class

        # Create a mock solver whose class name appears in its MRO
        MockKClass = type(cls_name, (), {})
        mock_solver = MockKClass()

        # _detect_adapter_class inspects type(solver).__mro__ class names
        # For mock objects the MRO is [MockKClass, object], so cls_name is present
        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter
        adapter_cls = _detect_adapter_class(mock_solver)
        assert adapter_cls is SCFCheckpointAdapter


class TestSpotStatus:
    async def test_spot_status_empty_store(self, tmp_path: Path) -> None:
        """_status_from_store returns None when no checkpoints exist."""
        store = LocalStore(base_dir=tmp_path, job_id="empty-job")
        result = await _status_from_store(store, "")
        assert result is None

    async def test_spot_status_returns_latest_metadata(self, tmp_path: Path) -> None:
        """_status_from_store returns the latest checkpoint's metadata."""
        store = LocalStore(base_dir=tmp_path, job_id="status-job")
        tensor = np.zeros((4,), dtype=np.float64)

        await store.save_checkpoint(
            "ckpt-old",
            {"state": tensor},
            {"method": "fake", "iteration": 1},
        )
        # Patch timestamp so ckpt-new is definitively latest
        import json as _json
        manifest_path = store._ckpt_dir / "ckpt-old" / "_manifest.json"
        data = _json.loads(manifest_path.read_text())
        data["timestamp"] = 1000.0
        manifest_path.write_text(_json.dumps(data))

        await store.save_checkpoint(
            "ckpt-new",
            {"state": tensor},
            {"method": "fake", "iteration": 5},
        )
        manifest_path2 = store._ckpt_dir / "ckpt-new" / "_manifest.json"
        data2 = _json.loads(manifest_path2.read_text())
        data2["timestamp"] = 2000.0
        manifest_path2.write_text(_json.dumps(data2))

        result = await _status_from_store(store, "")
        assert result is not None
        assert result["checkpoint_id"] == "ckpt-new"
        assert result["method"] == "fake"

    async def test_spot_status_merges_metadata_fields(self, tmp_path: Path) -> None:
        """_status_from_store merges user metadata keys into the top-level dict."""
        store = LocalStore(base_dir=tmp_path, job_id="meta-job")
        tensor = np.zeros((4,), dtype=np.float64)

        await store.save_checkpoint(
            "ckpt-001",
            {"state": tensor},
            {
                "method": "fake",
                "iteration": 42,
                "e_tot": -1.117,
                "converged": True,
            },
        )

        result = await _status_from_store(store, "")
        assert result is not None
        assert result["iteration"] == 42
        assert result["e_tot"] == pytest.approx(-1.117)
        assert result["converged"] is True
        assert "checkpoint_id" in result
        assert "total_bytes" in result


# ---------------------------------------------------------------------------
# _detect_adapter_class — all branches, no PySCF needed
# ---------------------------------------------------------------------------


class TestDetectAdapterClass:
    def test_ccsd_detected(self) -> None:
        from spot_checkpoint.adapters.pyscf import CCSDCheckpointAdapter

        MockCCSD = type("CCSD", (), {})
        assert _detect_adapter_class(MockCCSD()) is CCSDCheckpointAdapter

    def test_casscf_detected(self) -> None:
        from spot_checkpoint.adapters.pyscf import CASSCFCheckpointAdapter

        MockCAS = type("CASSCF", (), {})
        assert _detect_adapter_class(MockCAS()) is CASSCFCheckpointAdapter

    def test_scf_rhf_detected(self) -> None:
        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter

        MockRHF = type("RHF", (), {})
        assert _detect_adapter_class(MockRHF()) is SCFCheckpointAdapter

    def test_unknown_raises_value_error(self) -> None:
        MockUnknown = type("UnknownSolver", (), {})
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            _detect_adapter_class(MockUnknown())


# ---------------------------------------------------------------------------
# SpotLifecycleManager — make_callback, check, context manager
# ---------------------------------------------------------------------------


class TestSpotLifecycleManagerCallbacks:
    def _make_mgr(
        self, tmp_path: Path, fake_adapter: Any, periodic_interval: float = 0.0
    ) -> tuple[SpotLifecycleManager, LocalStore]:
        store = LocalStore(base_dir=tmp_path, job_id="callback-test")
        mgr = SpotLifecycleManager(
            store=store,
            adapter=fake_adapter,
            backend=_NoOpBackend(),
            periodic_interval=periodic_interval,
        )
        return mgr, store

    async def test_make_callback_fires_periodic(
        self, tmp_path: Path, fake_adapter: Any
    ) -> None:
        """make_callback() returns a callable that triggers periodic checkpoints."""
        mgr, store = self._make_mgr(tmp_path, fake_adapter, periodic_interval=0.0)
        callback = mgr.make_callback()  # also calls start()
        time.sleep(0.002)  # ensure time.time() - _last > 0
        callback({"cycle": 5})
        mgr.stop()

        checkpoints = await store.list_checkpoints("")
        assert len(checkpoints) >= 1

    async def test_check_fires_periodic(
        self, tmp_path: Path, fake_adapter: Any
    ) -> None:
        """check() triggers a periodic checkpoint when interval has elapsed."""
        mgr, store = self._make_mgr(tmp_path, fake_adapter, periodic_interval=0.0)
        mgr.start()
        time.sleep(0.002)
        mgr.check(0)
        mgr.stop()

        checkpoints = await store.list_checkpoints("")
        assert len(checkpoints) >= 1

    def test_context_manager(self, tmp_path: Path, fake_adapter: Any) -> None:
        """The context manager protocol starts and stops the manager."""
        store = LocalStore(base_dir=tmp_path, job_id="ctx-test")
        with SpotLifecycleManager(
            store=store,
            adapter=fake_adapter,
            backend=_NoOpBackend(),
        ):
            pass  # __enter__ calls start(), __exit__ calls stop()

    async def test_make_callback_emergency_raises_system_exit(
        self, tmp_path: Path, fake_adapter: Any
    ) -> None:
        """Callback fires emergency checkpoint when interrupt is set, then exits."""
        import asyncio

        mgr, store = self._make_mgr(tmp_path, fake_adapter, periodic_interval=300.0)
        callback = mgr.make_callback()

        # Register an interrupt so callback takes the emergency path
        event = InterruptEvent(
            reason=InterruptReason.SPOT_RECLAIM,
            deadline=time.time() + 300,
        )
        mgr._on_interrupt(event)

        with pytest.raises(SystemExit) as exc_info:
            callback({"cycle": 1})
        assert exc_info.value.code == 0

        mgr.stop()

        # Verify emergency checkpoint was written
        checkpoints = await store.list_checkpoints("")
        assert len(checkpoints) >= 1


# ---------------------------------------------------------------------------
# Top-level API — ValueError paths (no S3 needed)
# ---------------------------------------------------------------------------


class TestTopLevelAPIErrors:
    def test_spot_safe_raises_without_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SPOT_CHECKPOINT_BUCKET", raising=False)
        with pytest.raises(ValueError, match="bucket is required"):
            spot_safe(MagicMock())

    async def test_spot_restore_async_raises_without_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SPOT_CHECKPOINT_BUCKET", raising=False)
        with pytest.raises(ValueError, match="bucket is required"):
            await spot_restore_async(MagicMock())

    def test_spot_status_raises_without_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SPOT_CHECKPOINT_BUCKET", raising=False)
        with pytest.raises(ValueError, match="bucket is required"):
            spot_status()

    async def test_spot_status_async_raises_without_bucket(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("SPOT_CHECKPOINT_BUCKET", raising=False)
        with pytest.raises(ValueError, match="bucket is required"):
            await spot_status_async()


# ---------------------------------------------------------------------------
# Top-level API — full-body coverage using LocalStore via mock
# ---------------------------------------------------------------------------


class TestTopLevelAPIWithLocalStore:
    """Cover spot_restore, spot_restore_async, spot_status_async bodies by
    patching S3ShardedStore to return a LocalStore-backed store."""

    def _mock_adapter(self) -> Any:
        adapter = MagicMock()
        adapter.restore_state = MagicMock()
        return adapter

    def test_spot_restore_returns_false_on_empty_store(
        self, tmp_path: Path
    ) -> None:
        """spot_restore() returns False when no checkpoints exist."""
        from spot_checkpoint.lifecycle import spot_restore

        local = LocalStore(base_dir=tmp_path, job_id="restore-sync")
        mock_adapter_class = MagicMock(return_value=self._mock_adapter())

        with patch("spot_checkpoint.storage.S3ShardedStore", return_value=local):
            result = spot_restore(
                MagicMock(),
                bucket="test-bucket",
                adapter_class=mock_adapter_class,
            )
        assert result is False

    async def test_spot_restore_async_returns_false_on_empty_store(
        self, tmp_path: Path
    ) -> None:
        """spot_restore_async() returns False when no checkpoints exist."""
        local = LocalStore(base_dir=tmp_path, job_id="restore-async")
        mock_adapter_class = MagicMock(return_value=self._mock_adapter())

        with patch("spot_checkpoint.storage.S3ShardedStore", return_value=local):
            result = await spot_restore_async(
                MagicMock(),
                bucket="test-bucket",
                adapter_class=mock_adapter_class,
            )
        assert result is False

    async def test_spot_status_async_returns_none_on_empty_store(
        self, tmp_path: Path
    ) -> None:
        """spot_status_async() returns None when no checkpoints exist."""
        local = LocalStore(base_dir=tmp_path, job_id="status-async")

        with patch("spot_checkpoint.storage.S3ShardedStore", return_value=local):
            result = await spot_status_async(bucket="test-bucket")
        assert result is None

    def test_spot_status_returns_none_on_empty_store(self, tmp_path: Path) -> None:
        """spot_status() returns None when no checkpoints exist."""
        local = LocalStore(base_dir=tmp_path, job_id="status-sync")

        with patch("spot_checkpoint.storage.S3ShardedStore", return_value=local):
            result = spot_status(bucket="test-bucket")
        assert result is None

    def test_spot_safe_returns_callable_with_mock_s3(
        self, tmp_path: Path, fake_adapter: Any
    ) -> None:
        """spot_safe() runs end-to-end and returns a callable callback."""
        local = LocalStore(base_dir=tmp_path, job_id="safe-test")
        mock_adapter_class = MagicMock(return_value=fake_adapter)

        with patch("spot_checkpoint.storage.S3ShardedStore", return_value=local):
            cb = spot_safe(
                MagicMock(),
                bucket="test-bucket",
                adapter_class=mock_adapter_class,
            )
        assert callable(cb)


# ---------------------------------------------------------------------------
# New v0.9.0 tests
# ---------------------------------------------------------------------------


class TestRunAsyncRaisesBeforeStart:
    def test_run_async_raises_before_start(self, tmp_path: Path, fake_adapter: Any) -> None:
        """_run_async raises RuntimeError when called before start()."""
        store = LocalStore(base_dir=tmp_path, job_id="run-async-test")
        mgr = SpotLifecycleManager(
            store=store,
            adapter=fake_adapter,
            backend=_NoOpBackend(),
        )

        async def _noop() -> None:
            pass

        coro = _noop()
        try:
            with pytest.raises(RuntimeError, match="_run_async called before start"):
                mgr._run_async(coro)
        finally:
            coro.close()  # prevent "coroutine was never awaited" warning


class TestJobIdFallbackIncludesHostname:
    def test_job_id_fallback_includes_hostname(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When no SLURM_JOB_ID/SPAWN_INSTANCE_ID, job_id includes hostname."""
        import socket

        monkeypatch.delenv("SLURM_JOB_ID", raising=False)
        monkeypatch.delenv("SPAWN_INSTANCE_ID", raising=False)

        captured: dict[str, str] = {}

        mock_store = MagicMock()

        def _capture_store(**kwargs: Any) -> MagicMock:
            captured["job_id"] = kwargs.get("job_id", "")
            return mock_store

        mock_adapter_class = MagicMock()
        mock_adapter_class.return_value = MagicMock()

        with patch("spot_checkpoint.storage.S3ShardedStore", side_effect=_capture_store):
            spot_safe(
                MagicMock(),
                bucket="test-bucket",
                adapter_class=mock_adapter_class,
            )

        assert socket.gethostname() in captured["job_id"]
