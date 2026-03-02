"""
End-to-end lifecycle scenario tests.

Two full scenarios using FakeCheckpointAdapter + LocalStore (no S3, no PySCF):

  Scenario 1 — Happy path with cleanup:
    Start → checkpoint × 5 → clean exit → checkpoints deleted

  Scenario 2 — Interrupt + restore + cleanup:
    Phase 1: Start → checkpoint → spot interruption → emergency ckpt → exit
    Phase 2: Restore → continue → clean exit → checkpoints deleted
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from spot_checkpoint.lifecycle import (
    InterruptEvent,
    InterruptReason,
    LifecycleBackend,
    SpotLifecycleManager,
)
from spot_checkpoint.storage import LocalStore

from conftest import FakeCheckpointAdapter, FakeSolver


class _NoOpBackend(LifecycleBackend):
    """Backend that does nothing — no signals, no threads."""

    def start(self, on_interrupt: object) -> None:
        pass

    def stop(self) -> None:
        pass


class TestHappyPathWithCleanup:
    async def test_happy_path_cleanup_on_complete(self, tmp_path: Path) -> None:
        """Normal exit with cleanup_on_complete=0 deletes all checkpoints."""
        solver = FakeSolver()
        adapter = FakeCheckpointAdapter(solver)
        store = LocalStore(base_dir=tmp_path, job_id="happy")
        mgr = SpotLifecycleManager(
            store=store,
            adapter=adapter,
            backend=_NoOpBackend(),
            periodic_interval=0.0,
            cleanup_on_complete=0,
        )

        with mgr:
            for i in range(5):
                solver.step()
                time.sleep(0.002)
                mgr.check(i)

        # __exit__ fires complete() → GC → 0 checkpoints
        remaining = await store.list_checkpoints("")
        assert remaining == []
        assert solver.iteration == 5

    async def test_happy_path_keep_latest(self, tmp_path: Path) -> None:
        """cleanup_on_complete=1 keeps only the most recent checkpoint."""
        solver = FakeSolver()
        adapter = FakeCheckpointAdapter(solver)
        store = LocalStore(base_dir=tmp_path, job_id="happy-keep")
        mgr = SpotLifecycleManager(
            store=store,
            adapter=adapter,
            backend=_NoOpBackend(),
            periodic_interval=0.0,
            cleanup_on_complete=1,
        )

        with mgr:
            for i in range(4):
                solver.step()
                time.sleep(0.002)
                mgr.check(i)

        remaining = await store.list_checkpoints("")
        assert len(remaining) == 1
        assert solver.iteration == 4

    async def test_no_cleanup_without_param(self, tmp_path: Path) -> None:
        """Without cleanup_on_complete, checkpoints persist after normal exit."""
        solver = FakeSolver()
        adapter = FakeCheckpointAdapter(solver)
        store = LocalStore(base_dir=tmp_path, job_id="no-cleanup")
        mgr = SpotLifecycleManager(
            store=store,
            adapter=adapter,
            backend=_NoOpBackend(),
            periodic_interval=0.0,
            # cleanup_on_complete not set
        )

        with mgr:
            for i in range(3):
                solver.step()
                time.sleep(0.002)
                mgr.check(i)

        remaining = await store.list_checkpoints("")
        assert len(remaining) == 3


class TestInterruptRestoreComplete:
    async def test_interrupt_restore_complete(self, tmp_path: Path) -> None:
        """Full interrupt + restore + complete cycle."""
        store = LocalStore(base_dir=tmp_path, job_id="interrupt")

        # --- Phase 1: first "instance" interrupted at iteration 3 ---
        solver1 = FakeSolver()
        adapter1 = FakeCheckpointAdapter(solver1)
        mgr1 = SpotLifecycleManager(
            store=store,
            adapter=adapter1,
            backend=_NoOpBackend(),
            periodic_interval=0.0,
            # no cleanup_on_complete — exception path should preserve checkpoints
        )

        with pytest.raises(SystemExit) as exc_info:
            with mgr1:
                for i in range(5):
                    solver1.step()
                    time.sleep(0.002)
                    mgr1.check(i)
                    if i == 2:
                        # Simulate spot interruption
                        event = InterruptEvent(
                            reason=InterruptReason.SPOT_RECLAIM,
                            deadline=time.time() + 300,
                        )
                        mgr1._on_interrupt(event)

        assert exc_info.value.code == 0

        # Emergency checkpoint must have been written
        checkpoints = await store.list_checkpoints("")
        assert len(checkpoints) >= 1

        # --- Phase 2: second "instance" restores, completes, cleans up ---
        solver2 = FakeSolver()
        adapter2 = FakeCheckpointAdapter(solver2)
        mgr2 = SpotLifecycleManager(
            store=store,
            adapter=adapter2,
            backend=_NoOpBackend(),
            periodic_interval=0.0,
            cleanup_on_complete=0,
        )

        restored = await mgr2.restore_latest()
        assert restored is True
        # Restored from emergency checkpoint written when i=3 (after interrupt
        # was set at i=2).  By that time solver1.step() had run 4 times.
        assert solver2.iteration >= 3

        with mgr2:
            for i in range(solver2.iteration, 5):
                solver2.step()
                time.sleep(0.002)
                mgr2.check(i)

        # __exit__ fires GC with keep=0
        remaining = await store.list_checkpoints("")
        assert remaining == []
        assert solver2.iteration == 5

    async def test_exception_preserves_checkpoints(self, tmp_path: Path) -> None:
        """An unexpected exception leaves checkpoints intact for restart."""
        solver = FakeSolver()
        adapter = FakeCheckpointAdapter(solver)
        store = LocalStore(base_dir=tmp_path, job_id="exc-preserve")
        mgr = SpotLifecycleManager(
            store=store,
            adapter=adapter,
            backend=_NoOpBackend(),
            periodic_interval=0.0,
            cleanup_on_complete=0,
        )

        with pytest.raises(RuntimeError):
            with mgr:
                solver.step()
                time.sleep(0.002)
                mgr.check(0)
                raise RuntimeError("unexpected failure")

        # cleanup_on_complete must NOT have fired — checkpoints preserved
        remaining = await store.list_checkpoints("")
        assert len(remaining) >= 1
