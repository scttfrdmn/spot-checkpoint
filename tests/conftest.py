"""Shared fixtures for spot-checkpoint tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from spot_checkpoint.protocol import CheckpointPayload
from spot_checkpoint.storage import LocalStore, S3ShardedStore


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


@pytest.fixture(scope="session", autouse=True)
def aws_credentials() -> None:
    """Set fake AWS credentials so aioboto3 doesn't try to look up real ones."""
    import os

    os.environ.setdefault("AWS_ACCESS_KEY_ID", "testing")
    os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "testing")
    os.environ.setdefault("AWS_SECURITY_TOKEN", "testing")
    os.environ.setdefault("AWS_SESSION_TOKEN", "testing")
    os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")


@pytest.fixture(scope="session")
def moto_server() -> Any:
    """Session-scoped ThreadedMotoServer for S3 tests."""
    from moto.server import ThreadedMotoServer

    server = ThreadedMotoServer(port=5555, verbose=False)
    server.start()
    yield server
    server.stop()


@pytest.fixture
async def s3_store(moto_server: Any) -> Any:
    """S3ShardedStore backed by a ThreadedMotoServer, with 4KB shards."""
    import boto3

    endpoint_url = "http://127.0.0.1:5555"
    # Create a fresh bucket for this test
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=endpoint_url,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    s3.create_bucket(Bucket="test-bucket")
    store = S3ShardedStore(
        bucket="test-bucket",
        job_id="test-job",
        shard_size=4 * 1024,
        region="us-east-1",
        endpoint_url=endpoint_url,
    )
    yield store
    # Clean up bucket after test
    response = s3.list_objects_v2(Bucket="test-bucket")
    objects = [{"Key": obj["Key"]} for obj in response.get("Contents", [])]
    if objects:
        s3.delete_objects(Bucket="test-bucket", Delete={"Objects": objects})
    s3.delete_bucket(Bucket="test-bucket")
