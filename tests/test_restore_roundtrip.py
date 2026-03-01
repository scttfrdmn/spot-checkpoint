"""
Restore round-trip integration tests.

All tests use LocalStore to avoid S3 dependencies.
They exercise the SpotLifecycleManager.restore_latest() path and verify
that state is correctly recovered after a simulated interrupt.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spot_checkpoint.lifecycle import SpotLifecycleManager
from spot_checkpoint.protocol import CheckpointPayload
from spot_checkpoint.storage import LocalStore


# ---------------------------------------------------------------------------
# Minimal fake solver + adapter (inline, no conftest dependency)
# ---------------------------------------------------------------------------

@dataclass
class _Solver:
    state: np.ndarray = field(default_factory=lambda: np.ones((5, 5), dtype=np.float64))
    iteration: int = 0
    energy: float = -100.0


@dataclass
class _Adapter:
    solver: _Solver

    def checkpoint_state(self) -> CheckpointPayload:
        import time
        return CheckpointPayload(
            tensors={"state": self.solver.state.copy()},
            metadata={
                "iteration": self.solver.iteration,
                "energy": self.solver.energy,
                "method": "fake",
            },
            method="fake",
            timestamp=time.time(),
        )

    def restore_state(self, payload: CheckpointPayload) -> None:
        self.solver.state = payload.tensors["state"]
        self.solver.iteration = payload.metadata["iteration"]
        self.solver.energy = payload.metadata["energy"]

    @property
    def checkpoint_size_estimate(self) -> int:
        return self.solver.state.nbytes


# ---------------------------------------------------------------------------
# Helper: build a manager without starting a backend
# ---------------------------------------------------------------------------

def _make_mgr(store: LocalStore, solver: _Solver) -> SpotLifecycleManager:
    adapter = _Adapter(solver=solver)
    from spot_checkpoint.lifecycle import SlurmLifecycleBackend
    # Use Slurm backend but never call start() — we only invoke restore_latest()
    # directly, which doesn't require a running backend.
    mgr = SpotLifecycleManager(
        store=store,
        adapter=adapter,
        backend=SlurmLifecycleBackend(),
        checkpoint_id_prefix="ckpt",
    )
    return mgr


# ---------------------------------------------------------------------------
# Test 1: empty store → restore_latest returns False, state unchanged
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spot_restore_no_checkpoint(tmp_path: Path) -> None:
    """restore_latest() returns False when no checkpoint exists."""
    store = LocalStore(base_dir=tmp_path, job_id="job-empty")
    solver = _Solver()
    original_iteration = solver.iteration

    mgr = _make_mgr(store, solver)
    restored = await mgr.restore_latest()

    assert restored is False
    assert solver.iteration == original_iteration


# ---------------------------------------------------------------------------
# Test 2: two checkpoints present → restore picks the latest
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spot_restore_finds_latest(tmp_path: Path) -> None:
    """restore_latest() picks the most recent checkpoint by timestamp."""
    store = LocalStore(base_dir=tmp_path, job_id="job-two")

    # Save first checkpoint at iteration 3
    solver_write = _Solver()
    adapter_write = _Adapter(solver=solver_write)
    solver_write.iteration = 3
    payload3 = adapter_write.checkpoint_state()
    await store.save_checkpoint("ckpt-fake-iter000003", payload3.tensors, payload3.metadata)

    # Small sleep so timestamps differ
    import time
    time.sleep(0.05)

    # Save second checkpoint at iteration 7
    solver_write.iteration = 7
    payload7 = adapter_write.checkpoint_state()
    await store.save_checkpoint("ckpt-fake-iter000007", payload7.tensors, payload7.metadata)

    # Restore into a fresh solver — should get iteration 7
    solver_read = _Solver()
    mgr = _make_mgr(store, solver_read)
    restored = await mgr.restore_latest()

    assert restored is True
    assert solver_read.iteration == 7


# ---------------------------------------------------------------------------
# Test 3: restore state exactly matches saved state at iteration 5
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_spot_restore_specific_state(tmp_path: Path) -> None:
    """Restored solver state matches the saved state at iteration 5."""
    store = LocalStore(base_dir=tmp_path, job_id="job-specific")

    rng = np.random.default_rng(42)
    original_state = rng.standard_normal((5, 5))

    solver_write = _Solver()
    solver_write.state = original_state.copy()
    solver_write.iteration = 5
    solver_write.energy = -42.5

    adapter_write = _Adapter(solver=solver_write)
    payload = adapter_write.checkpoint_state()
    await store.save_checkpoint("ckpt-fake-iter000005", payload.tensors, payload.metadata)

    solver_read = _Solver()
    mgr = _make_mgr(store, solver_read)
    restored = await mgr.restore_latest()

    assert restored is True
    assert solver_read.iteration == 5
    assert solver_read.energy == pytest.approx(-42.5)
    np.testing.assert_array_equal(solver_read.state, original_state)


# ---------------------------------------------------------------------------
# Test 4: full cycle — simulate interrupt, emergency checkpoint, then restore
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_full_cycle_checkpoint_restore_resume(tmp_path: Path) -> None:
    """
    Simulate a full spot-interrupt cycle:
      1. Run a few iterations with a lifecycle manager.
      2. Trigger _do_periodic_checkpoint at iteration 4.
      3. Restore on a fresh manager.
      4. Verify state + confirm iteration loop can continue.
    """
    store = LocalStore(base_dir=tmp_path, job_id="job-full-cycle")
    solver = _Solver()
    adapter = _Adapter(solver=solver)

    from spot_checkpoint.lifecycle import SlurmLifecycleBackend
    mgr = SpotLifecycleManager(
        store=store,
        adapter=adapter,
        backend=SlurmLifecycleBackend(),
        checkpoint_id_prefix="ckpt",
        periodic_interval=0,  # checkpoint every call
    )
    # Manually start only the event loop (no signal handlers)
    import asyncio
    mgr._loop = asyncio.new_event_loop()
    import threading
    mgr._loop_thread = threading.Thread(
        target=mgr._loop.run_forever, daemon=True, name="test-loop"
    )
    mgr._loop_thread.start()

    # Run 4 solver steps and write a periodic checkpoint at step 4
    for i in range(1, 5):
        solver.state = solver.state + float(i) * 0.01
        solver.iteration = i
        solver.energy = -100.0 + i * 0.5

    mgr._do_periodic_checkpoint(solver.iteration)

    # Verify checkpoint was written
    checkpoints = await store.list_checkpoints("ckpt")
    assert len(checkpoints) == 1

    saved_iteration = solver.iteration
    saved_energy = solver.energy
    saved_state = solver.state.copy()

    # Clean up the manager's loop
    mgr._loop.call_soon_threadsafe(mgr._loop.stop)
    mgr._loop_thread.join(timeout=5)
    mgr._loop.close()
    mgr._loop = None

    # Restore into a brand-new solver
    solver2 = _Solver()
    mgr2 = _make_mgr(store, solver2)
    restored = await mgr2.restore_latest()

    assert restored is True
    assert solver2.iteration == saved_iteration
    assert solver2.energy == pytest.approx(saved_energy)
    np.testing.assert_array_equal(solver2.state, saved_state)

    # Confirm the restored solver can continue iterating
    solver2.iteration += 1
    solver2.energy -= 0.1
    assert solver2.iteration == saved_iteration + 1
