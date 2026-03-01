"""End-to-end integration test: fake solver + LocalStore + lifecycle manager."""

import time
from pathlib import Path

import numpy as np
import pytest

from spot_checkpoint.lifecycle import SpotLifecycleManager
from spot_checkpoint.storage import LocalStore
from tests.conftest import FakeCheckpointAdapter, FakeSolver


@pytest.mark.asyncio
async def test_periodic_checkpoint_via_check(tmp_path: Path):
    """Manager writes periodic checkpoints when check() is called."""
    store = LocalStore(base_dir=tmp_path, job_id="integration-test")
    solver = FakeSolver()
    adapter = FakeCheckpointAdapter(solver=solver)

    mgr = SpotLifecycleManager(
        store=store,
        adapter=adapter,
        periodic_interval=0.1,  # Very short for testing
    )

    with mgr:
        for i in range(5):
            solver.step()
            time.sleep(0.05)
            mgr.check(i)

    # Should have at least one checkpoint
    checkpoints = await store.list_checkpoints("ckpt")
    assert len(checkpoints) >= 1


@pytest.mark.asyncio
async def test_checkpoint_restore_roundtrip(tmp_path: Path):
    """Save state, create new solver, restore, verify state matches."""
    store = LocalStore(base_dir=tmp_path, job_id="roundtrip-test")

    # Run solver and checkpoint
    solver1 = FakeSolver(state=np.array([[1.0, 2.0], [3.0, 4.0]]))
    solver1.iteration = 42
    solver1.energy = -99.5
    adapter1 = FakeCheckpointAdapter(solver=solver1)

    await store.save_checkpoint(
        "ckpt-manual",
        adapter1.checkpoint_state().tensors,
        adapter1.checkpoint_state().metadata,
    )

    # New solver, restore
    solver2 = FakeSolver()
    adapter2 = FakeCheckpointAdapter(solver=solver2)

    tensors, metadata = await store.load_checkpoint("ckpt-manual")
    from spot_checkpoint.protocol import CheckpointPayload
    payload = CheckpointPayload(
        tensors=tensors,
        metadata=metadata,
        method=metadata.get("method", "fake"),
        timestamp=time.time(),
    )
    adapter2.restore_state(payload)

    np.testing.assert_array_equal(solver2.state, solver1.state)
    assert solver2.iteration == 42
    assert solver2.energy == -99.5
