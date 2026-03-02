"""
spot-checkpoint: Fault-tolerant iterative computation on preemptible instances.

Quick start:
    from pyscf import gto, scf
    from spot_checkpoint import spot_safe

    mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='aug-cc-pvtz')
    mf = scf.RHF(mol)
    mf.callback = spot_safe(mf, bucket="my-checkpoints")
    mf.kernel()
"""

from spot_checkpoint.lifecycle import (
    DirectEC2Backend,
    SlurmLifecycleBackend,
    SporeLifecycleBackend,
    SpotLifecycleManager,
    detect_backend,
    spot_complete,
    spot_complete_async,
    spot_restore,
    spot_restore_async,
    spot_safe,
    spot_safe_async,
    spot_status,
    spot_status_async,
)
from spot_checkpoint.protocol import (
    Checkpointable,
    CheckpointCorruptionError,
    CheckpointPayload,
    CheckpointReadError,
    CheckpointStore,
    CheckpointWriteError,
    SpotCheckpointError,
)
from spot_checkpoint.storage import LocalStore, S3ShardedStore

__version__ = "0.11.0"

__all__ = [  # noqa: RUF022  (grouped by category, not alphabetically)
    # Top-level API
    "spot_safe",
    "spot_safe_async",
    "spot_restore",
    "spot_restore_async",
    "spot_complete",
    "spot_complete_async",
    "spot_status",
    "spot_status_async",
    "SpotLifecycleManager",
    # Backends
    "detect_backend",
    "SporeLifecycleBackend",
    "SlurmLifecycleBackend",
    "DirectEC2Backend",
    # Storage
    "S3ShardedStore",
    "LocalStore",
    # Protocol
    "CheckpointPayload",
    "Checkpointable",
    "CheckpointStore",
    # Errors
    "SpotCheckpointError",
    "CheckpointWriteError",
    "CheckpointReadError",
    "CheckpointCorruptionError",
]
