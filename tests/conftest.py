"""Shared fixtures for spot-checkpoint tests."""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spot_checkpoint.protocol import CheckpointPayload
from spot_checkpoint.storage import LocalStore


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Temporary directory for test artifacts."""
    return tmp_path


@pytest.fixture
def local_store(tmp_path: Path) -> LocalStore:
    """LocalStore pointed at a temp directory."""
    return LocalStore(base_dir=tmp_path, job_id="test-job")


@dataclass
class FakeSolver:
    """Fake iterative solver for testing Checkpointable protocol."""

    state: np.ndarray = field(default_factory=lambda: np.random.rand(10, 10))
    iteration: int = 0
    energy: float = -75.0
    converged: bool = False

    def step(self) -> None:
        """Simulate one iteration."""
        self.iteration += 1
        self.energy += np.random.rand() * 0.01
        self.state = self.state + np.random.rand(*self.state.shape) * 0.001


@dataclass
class FakeCheckpointAdapter:
    """Adapter for FakeSolver that implements Checkpointable."""

    solver: FakeSolver

    def checkpoint_state(self) -> CheckpointPayload:
        import time
        return CheckpointPayload(
            tensors={"state": self.solver.state.copy()},
            metadata={
                "iteration": self.solver.iteration,
                "energy": self.solver.energy,
                "converged": self.solver.converged,
                "method": "fake",
            },
            method="fake",
            timestamp=time.time(),
        )

    def restore_state(self, payload: CheckpointPayload) -> None:
        self.solver.state = payload.tensors["state"]
        self.solver.iteration = payload.metadata["iteration"]
        self.solver.energy = payload.metadata["energy"]
        self.solver.converged = payload.metadata["converged"]

    @property
    def checkpoint_size_estimate(self) -> int:
        return self.solver.state.nbytes


@pytest.fixture
def fake_solver() -> FakeSolver:
    return FakeSolver()


@pytest.fixture
def fake_adapter(fake_solver: FakeSolver) -> FakeCheckpointAdapter:
    return FakeCheckpointAdapter(solver=fake_solver)
