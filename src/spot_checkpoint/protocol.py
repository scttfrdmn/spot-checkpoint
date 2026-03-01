"""
Layer 2: Computation State Protocol.

Defines what "checkpointable state" means for any iterative solver.
Domain-specific adapters implement these protocols.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


# ---------------------------------------------------------------------------
# Exception hierarchy
# ---------------------------------------------------------------------------

class SpotCheckpointError(Exception):
    """Base exception for spot-checkpoint."""


class CheckpointWriteError(SpotCheckpointError):
    """Failed to write a checkpoint."""


class CheckpointReadError(SpotCheckpointError):
    """Failed to read/restore a checkpoint."""


class CheckpointCorruptionError(CheckpointReadError):
    """Checkpoint data failed integrity check."""


class AdapterError(SpotCheckpointError):
    """Adapter could not extract or restore solver state."""


# ---------------------------------------------------------------------------
# Checkpoint payload — the unit of persistence
# ---------------------------------------------------------------------------

@dataclass
class CheckpointPayload:
    """
    Serializable snapshot of an iterative computation's state.

    Attributes:
        tensors: Named numpy arrays representing computation state.
            Keys are logical names (e.g., "mo_coeff", "t2").
        metadata: Scalar metadata — iteration count, convergence metrics,
            method parameters, anything JSON-serializable.
        method: Identifier for the computation method (e.g., "rhf", "ccsd").
        timestamp: Unix timestamp when the snapshot was taken.
    """

    tensors: dict[str, np.ndarray]
    metadata: dict[str, Any]
    method: str
    timestamp: float = field(default_factory=time.time)

    @property
    def total_bytes(self) -> int:
        """Total size of all tensors in bytes."""
        return sum(t.nbytes for t in self.tensors.values())

    @property
    def tensor_summary(self) -> dict[str, str]:
        """Human-readable summary of tensor shapes and sizes."""
        return {
            name: f"{t.shape} {t.dtype} ({t.nbytes / 1e6:.1f} MB)"
            for name, t in self.tensors.items()
        }


# ---------------------------------------------------------------------------
# Protocols — structural typing for adapters and stores
# ---------------------------------------------------------------------------

@runtime_checkable
class Checkpointable(Protocol):
    """
    Any iterative computation that can save and restore state.

    Implement this protocol in domain-specific adapters
    (see adapters/pyscf.py for examples).
    """

    def checkpoint_state(self) -> CheckpointPayload:
        """Extract current iteration state as a checkpoint payload."""
        ...

    def restore_state(self, payload: CheckpointPayload) -> None:
        """Restore computation state from a previously saved payload."""
        ...

    @property
    def checkpoint_size_estimate(self) -> int:
        """
        Estimated bytes for a checkpoint at current state.

        Used by the lifecycle manager to assess whether an emergency
        checkpoint is feasible within the interruption window.
        Does not need to be exact — within 20% is fine.
        """
        ...


@runtime_checkable
class CheckpointStore(Protocol):
    """
    Backend for persisting and retrieving checkpoints.

    See storage.py for S3ShardedStore and LocalStore implementations.
    """

    async def save_checkpoint(
        self,
        checkpoint_id: str,
        tensors: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> CheckpointManifest:
        """
        Persist a checkpoint.

        Args:
            checkpoint_id: Unique identifier for this checkpoint.
            tensors: Named numpy arrays to persist.
            metadata: JSON-serializable metadata dict.

        Returns:
            Manifest describing the written checkpoint.

        Raises:
            CheckpointWriteError: If the write fails.
        """
        ...

    async def load_checkpoint(
        self,
        checkpoint_id: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Load a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to load.

        Returns:
            Tuple of (tensors dict, metadata dict).

        Raises:
            CheckpointReadError: If the checkpoint doesn't exist or can't be read.
            CheckpointCorruptionError: If integrity check fails.
        """
        ...

    async def list_checkpoints(
        self,
        prefix: str,
    ) -> list[dict[str, Any]]:
        """List available checkpoints matching prefix."""
        ...

    async def delete_checkpoint(
        self,
        checkpoint_id: str,
    ) -> None:
        """Delete a checkpoint and all its shards."""
        ...


# ---------------------------------------------------------------------------
# Manifest — describes a written checkpoint
# ---------------------------------------------------------------------------

@dataclass
class CheckpointManifest:
    """
    Describes a completed checkpoint write.

    Written as _manifest.json in the checkpoint directory.
    Its existence is the commit signal — incomplete checkpoints
    have no manifest and are treated as garbage.
    """

    checkpoint_id: str
    method: str
    timestamp: float
    total_bytes: int
    tensor_specs: dict[str, TensorSpec]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "method": self.method,
            "timestamp": self.timestamp,
            "total_bytes": self.total_bytes,
            "tensor_specs": {
                name: spec.to_dict() for name, spec in self.tensor_specs.items()
            },
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CheckpointManifest:
        """Deserialize from JSON-compatible dict."""
        return cls(
            checkpoint_id=data["checkpoint_id"],
            method=data["method"],
            timestamp=data["timestamp"],
            total_bytes=data["total_bytes"],
            tensor_specs={
                name: TensorSpec.from_dict(spec)
                for name, spec in data["tensor_specs"].items()
            },
            metadata=data["metadata"],
        )


@dataclass
class TensorSpec:
    """Describes how a tensor was sharded and stored."""

    shape: tuple[int, ...]
    dtype: str
    nbytes: int
    num_shards: int
    shard_size: int
    checksums: list[str]  # xxhash per shard

    def to_dict(self) -> dict[str, Any]:
        return {
            "shape": list(self.shape),
            "dtype": self.dtype,
            "nbytes": self.nbytes,
            "num_shards": self.num_shards,
            "shard_size": self.shard_size,
            "checksums": self.checksums,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TensorSpec:
        return cls(
            shape=tuple(data["shape"]),
            dtype=data["dtype"],
            nbytes=data["nbytes"],
            num_shards=data["num_shards"],
            shard_size=data["shard_size"],
            checksums=data["checksums"],
        )
