"""
Real PySCF + SpotLifecycleManager integration tests.

Requires PySCF to be installed (skipped otherwise).
Uses H2/sto-3g RHF — minimal basis, converges in < 10 iterations, < 1 second.
"""

from __future__ import annotations

from typing import Any
from pathlib import Path

import pytest

pyscf = pytest.importorskip("pyscf")


class _NoOpBackend:
    """Lifecycle backend that does nothing (no signals, no threads)."""

    def start(self, on_interrupt: Any) -> None:
        pass

    def stop(self) -> None:
        pass


@pytest.mark.integration
class TestSCFLifecycleIntegration:
    def test_scf_emergency_checkpoint_and_restore(self, tmp_path: Path) -> None:
        """Simulate spot interrupt mid-SCF; verify emergency checkpoint written
        and a new manager restores the calculation to convergence."""
        from pyscf import gto, scf

        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter
        from spot_checkpoint.lifecycle import (
            InterruptEvent,
            InterruptReason,
            SpotLifecycleManager,
        )
        from spot_checkpoint.storage import LocalStore
        import time

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)

        # --- Phase 1: run SCF with interrupt triggered before kernel ---
        mf = scf.RHF(mol)
        store = LocalStore(base_dir=tmp_path, job_id="scf-interrupt-test")
        adapter = SCFCheckpointAdapter(mf)
        mgr = SpotLifecycleManager(
            store=store,
            adapter=adapter,
            backend=_NoOpBackend(),
            periodic_interval=1.0,
        )
        callback = mgr.make_callback()
        mf.callback = callback

        # Signal emergency *before* running kernel so the first callback fires emergency
        event = InterruptEvent(
            reason=InterruptReason.SPOT_RECLAIM,
            deadline=time.time() + 300,
        )
        mgr._on_interrupt(event)

        # kernel() will call callback after the first SCF cycle; callback fires
        # _do_emergency_checkpoint which raises SystemExit(0)
        with pytest.raises(SystemExit) as exc_info:
            mf.kernel()
        assert exc_info.value.code == 0

        # Verify at least one checkpoint was written
        import asyncio
        checkpoints = asyncio.run(store.list_checkpoints(""))
        assert len(checkpoints) >= 1

        mgr.stop()

        # --- Phase 2: restore into a new solver and run to convergence ---
        mf2 = scf.RHF(mol)
        store2 = LocalStore(base_dir=tmp_path, job_id="scf-interrupt-test")
        adapter2 = SCFCheckpointAdapter(mf2)
        mgr2 = SpotLifecycleManager(
            store=store2,
            adapter=adapter2,
            backend=_NoOpBackend(),
        )
        mgr2.start()
        restored = asyncio.run(mgr2.restore_latest())
        assert restored is True

        # Run kernel from restored checkpoint; should converge to reference energy
        e_tot = mf2.kernel()
        assert e_tot == pytest.approx(-1.117, abs=0.01)

        mgr2.stop()

    def test_restore_false_when_empty(self, tmp_path: Path) -> None:
        """restore_latest() returns False on empty LocalStore."""
        from pyscf import gto, scf

        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter
        from spot_checkpoint.lifecycle import SpotLifecycleManager
        from spot_checkpoint.storage import LocalStore
        import asyncio

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        store = LocalStore(base_dir=tmp_path, job_id="empty-job")
        adapter = SCFCheckpointAdapter(mf)
        mgr = SpotLifecycleManager(
            store=store,
            adapter=adapter,
            backend=_NoOpBackend(),
        )
        mgr.start()
        result = asyncio.run(mgr.restore_latest())
        assert result is False
        mgr.stop()

    def test_keep_checkpoints_prunes_during_scf(self, tmp_path: Path) -> None:
        """keep_checkpoints=2 with 5 periodic checkpoints leaves ≤ 2 in store."""
        from pyscf import gto, scf

        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter
        from spot_checkpoint.lifecycle import SpotLifecycleManager
        from spot_checkpoint.storage import LocalStore
        import asyncio

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        # Run kernel first so adapter.checkpoint_state() has valid data
        mf.kernel()

        store = LocalStore(base_dir=tmp_path, job_id="keep-test")
        adapter = SCFCheckpointAdapter(mf)
        mgr = SpotLifecycleManager(
            store=store,
            adapter=adapter,
            backend=_NoOpBackend(),
            keep_checkpoints=2,
        )
        mgr.start()
        try:
            for i in range(5):
                mgr._do_periodic_checkpoint(i)

            remaining = asyncio.run(store.list_checkpoints(""))
            assert len(remaining) <= 2
        finally:
            mgr.stop()
