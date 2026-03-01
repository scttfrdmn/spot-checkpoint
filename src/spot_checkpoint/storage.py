"""
Checkpoint Storage Engine — Layer 1.

S3-sharded and local-filesystem implementations of CheckpointStore.

TODO: Implement S3ShardedStore and LocalStore.
See docs/ARCHITECTURE.md for full design specification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

from spot_checkpoint.protocol import (
    CheckpointCorruptionError,
    CheckpointManifest,
    CheckpointReadError,
    TensorSpec,
)

logger = logging.getLogger(__name__)


class LocalStore:
    """
    Filesystem-based checkpoint store for testing and local development.

    Uses directories instead of S3 prefixes. No sharding — local disk
    doesn't benefit from prefix partitioning.

    Layout:
        {base_dir}/{job_id}/ckpt/{checkpoint_id}/{tensor_name}.npy
        {base_dir}/{job_id}/ckpt/{checkpoint_id}/_manifest.json
    """

    def __init__(self, base_dir: str | Path, job_id: str) -> None:
        self.base_dir = Path(base_dir)
        self.job_id = job_id
        self._ckpt_dir = self.base_dir / job_id / "ckpt"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)

    async def save_checkpoint(
        self,
        checkpoint_id: str,
        tensors: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> CheckpointManifest:
        ckpt_path = self._ckpt_dir / checkpoint_id
        ckpt_path.mkdir(parents=True, exist_ok=True)

        import time

        import xxhash

        tensor_specs: dict[str, TensorSpec] = {}
        total_bytes = 0

        for name, tensor in tensors.items():
            tensor_path = ckpt_path / f"{name}.npy"
            np.save(str(tensor_path), tensor)
            total_bytes += tensor.nbytes

            checksum = xxhash.xxh64(tensor.tobytes()).hexdigest()
            tensor_specs[name] = TensorSpec(
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype),
                nbytes=tensor.nbytes,
                num_shards=1,
                shard_size=tensor.nbytes,
                checksums=[checksum],
            )

        manifest = CheckpointManifest(
            checkpoint_id=checkpoint_id,
            method=metadata.get("method", "unknown"),
            timestamp=time.time(),
            total_bytes=total_bytes,
            tensor_specs=tensor_specs,
            metadata=metadata,
        )

        manifest_path = ckpt_path / "_manifest.json"
        manifest_path.write_text(json.dumps(manifest.to_dict(), indent=2))

        logger.info("Saved checkpoint %s (%.1f MB)", checkpoint_id, total_bytes / 1e6)
        return manifest

    async def load_checkpoint(
        self,
        checkpoint_id: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        ckpt_path = self._ckpt_dir / checkpoint_id
        manifest_path = ckpt_path / "_manifest.json"

        if not manifest_path.exists():
            raise CheckpointReadError(f"No manifest found for checkpoint {checkpoint_id}")

        import xxhash

        manifest_data = json.loads(manifest_path.read_text())
        tensors: dict[str, np.ndarray] = {}

        for name, tm in manifest_data["tensor_specs"].items():
            tensor_path = ckpt_path / f"{name}.npy"
            if not tensor_path.exists():
                raise CheckpointCorruptionError(f"Missing tensor file: {tensor_path}")
            tensors[name] = np.load(str(tensor_path))

            # Verify checksum
            actual = xxhash.xxh64(tensors[name].tobytes()).hexdigest()
            expected = tm["checksums"][0]
            if actual != expected:
                raise CheckpointCorruptionError(
                    f"Checksum mismatch for tensor {name}: {actual} != {expected}"
                )

        return tensors, manifest_data["metadata"]

    async def list_checkpoints(self, prefix: str) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        if not self._ckpt_dir.exists():
            return results

        for ckpt_dir in sorted(self._ckpt_dir.iterdir()):
            if not ckpt_dir.is_dir():
                continue
            if not ckpt_dir.name.startswith(prefix):
                continue
            manifest_path = ckpt_dir / "_manifest.json"
            if manifest_path.exists():
                data = json.loads(manifest_path.read_text())
                results.append(data)

        return results

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        import shutil
        ckpt_path = self._ckpt_dir / checkpoint_id
        if ckpt_path.exists():
            shutil.rmtree(ckpt_path)
            logger.info("Deleted checkpoint %s", checkpoint_id)


class S3ShardedStore:
    """
    S3-optimized checkpoint store with parallel sharded writes.

    TODO: Full implementation. See docs/ARCHITECTURE.md for design.

    Key features:
      - Tensors chunked into configurable shards (default 64MB)
      - Parallel PUTs across prefixed keys for horizontal S3 scaling
      - Manifest-last atomic commits
      - xxhash checksums for integrity
    """

    def __init__(
        self,
        bucket: str,
        job_id: str,
        shard_size: int = 64 * 1024 * 1024,
        max_concurrency: int = 32,
        endpoint_strategy: Literal["standard", "accelerate", "vpc"] = "vpc",
        region: str | None = None,
    ) -> None:
        self.bucket = bucket
        self.job_id = job_id
        self.shard_size = shard_size
        self.max_concurrency = max_concurrency
        self.endpoint_strategy = endpoint_strategy
        self.region = region

    async def save_checkpoint(
        self,
        checkpoint_id: str,
        tensors: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> CheckpointManifest:
        raise NotImplementedError("S3ShardedStore.save_checkpoint — see ARCHITECTURE.md")

    async def load_checkpoint(
        self,
        checkpoint_id: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        raise NotImplementedError("S3ShardedStore.load_checkpoint — see ARCHITECTURE.md")

    async def list_checkpoints(self, prefix: str) -> list[dict[str, Any]]:
        raise NotImplementedError("S3ShardedStore.list_checkpoints — see ARCHITECTURE.md")

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        raise NotImplementedError("S3ShardedStore.delete_checkpoint — see ARCHITECTURE.md")
