"""
Checkpoint Storage Engine — Layer 1.

S3-sharded and local-filesystem implementations of CheckpointStore.

See docs/ARCHITECTURE.md for full design specification.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, Literal

import aioboto3  # type: ignore[import-untyped]
import numpy as np
import xxhash
from botocore.config import Config  # type: ignore[import-untyped]
from botocore.exceptions import ClientError  # type: ignore[import-untyped]

from spot_checkpoint.protocol import (
    CheckpointCorruptionError,
    CheckpointManifest,
    CheckpointReadError,
    CheckpointWriteError,
    TensorSpec,
)

logger = logging.getLogger(__name__)


async def _put_with_sem(
    s3: Any,
    sem: asyncio.Semaphore,
    bucket: str,
    key: str,
    body: bytes,
) -> None:
    """Upload a single shard to S3, respecting the concurrency semaphore."""
    async with sem:
        await s3.put_object(Bucket=bucket, Key=key, Body=body)


async def _get_with_sem(
    s3: Any,
    sem: asyncio.Semaphore,
    bucket: str,
    key: str,
) -> bytes:
    """Download a single shard from S3, respecting the concurrency semaphore."""
    async with sem:
        resp = await s3.get_object(Bucket=bucket, Key=key)
        return await resp["Body"].read()  # type: ignore[no-any-return]


class LocalStore:
    """
    Filesystem-based checkpoint store for testing and local development.

    Uses directories instead of S3 prefixes. No sharding — local disk
    doesn't benefit from prefix partitioning.

    Layout:
        {base_dir}/{job_id}/ckpt/{checkpoint_id}/{tensor_name}.npy
        {base_dir}/{job_id}/ckpt/{checkpoint_id}/_manifest.json
    """

    def __init__(self, base_dir: str | Path, job_id: str, compress: bool = False) -> None:
        self.base_dir = Path(base_dir)
        self.job_id = job_id
        self._ckpt_dir = self.base_dir / job_id / "ckpt"
        self._ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._compress = compress

    async def save_checkpoint(
        self,
        checkpoint_id: str,
        tensors: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> CheckpointManifest:
        ckpt_path = self._ckpt_dir / checkpoint_id
        ckpt_path.mkdir(parents=True, exist_ok=True)

        tensor_specs: dict[str, TensorSpec] = {}
        total_bytes = 0

        for name, tensor in tensors.items():
            raw = tensor.tobytes()
            checksum = xxhash.xxh64(raw).hexdigest()
            total_bytes += tensor.nbytes

            if self._compress:
                try:
                    import zstandard as zstd
                except ImportError as exc:
                    raise ImportError(
                        "zstandard is required for compression: "
                        "pip install spot-checkpoint[compress]"
                    ) from exc
                data = zstd.ZstdCompressor(level=3).compress(raw)
                tensor_path = ckpt_path / f"{name}.bin.zst"
            else:
                data = raw
                tensor_path = ckpt_path / f"{name}.npy"
                # Use numpy save format for uncompressed (backward-compatible)
                np.save(str(tensor_path), tensor)
                data = None  # already written

            if data is not None:
                tensor_path.write_bytes(data)

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
            compression="zstd" if self._compress else None,
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

        manifest_data = json.loads(manifest_path.read_text())
        use_decompress = manifest_data.get("compression") == "zstd"
        tensors: dict[str, np.ndarray] = {}

        for name, tm in manifest_data["tensor_specs"].items():
            if use_decompress:
                tensor_path = ckpt_path / f"{name}.bin.zst"
                if not tensor_path.exists():
                    raise CheckpointCorruptionError(f"Missing tensor file: {tensor_path}")
                try:
                    import zstandard as zstd
                except ImportError as exc:
                    raise ImportError(
                        "zstandard is required for decompression: "
                        "pip install spot-checkpoint[compress]"
                    ) from exc
                raw = zstd.ZstdDecompressor().decompress(tensor_path.read_bytes())
                tensors[name] = np.frombuffer(raw, dtype=tm["dtype"]).reshape(tm["shape"])
            else:
                tensor_path = ckpt_path / f"{name}.npy"
                if not tensor_path.exists():
                    raise CheckpointCorruptionError(f"Missing tensor file: {tensor_path}")
                tensors[name] = np.load(str(tensor_path))

            # Verify checksum (always on raw bytes)
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


class S3ShardedStore:  # pragma: no cover
    """
    S3-optimized checkpoint store with parallel sharded writes.

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
        endpoint_url: str | None = None,
        compress: bool = False,
    ) -> None:
        self.bucket = bucket
        self.job_id = job_id
        self.shard_size = shard_size
        self.max_concurrency = max_concurrency
        self.endpoint_strategy = endpoint_strategy
        self.region = region
        self.endpoint_url = endpoint_url
        self._compress = compress

    def _prefix(self, checkpoint_id: str) -> str:
        return f"{self.job_id}/ckpt/{checkpoint_id}"

    def _client_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {}
        if self.region:
            kwargs["region_name"] = self.region
        if self.endpoint_url:
            kwargs["endpoint_url"] = self.endpoint_url
        if self.endpoint_strategy == "accelerate":
            kwargs["config"] = Config(s3={"use_accelerate_endpoint": True})
        return kwargs

    def _shard_bytes(self, data: bytes) -> list[bytes]:
        return [data[i : i + self.shard_size] for i in range(0, len(data), self.shard_size)]

    @staticmethod
    def _compress_shard(data: bytes) -> bytes:
        """Compress shard bytes with zstd level 3."""
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError(
                "zstandard is required for compression: pip install spot-checkpoint[compress]"
            ) from exc
        return zstd.ZstdCompressor(level=3).compress(data)

    @staticmethod
    def _decompress_shard(data: bytes) -> bytes:
        """Decompress zstd-compressed shard bytes."""
        try:
            import zstandard as zstd
        except ImportError as exc:
            raise ImportError(
                "zstandard is required for decompression: pip install spot-checkpoint[compress]"
            ) from exc
        return zstd.ZstdDecompressor().decompress(data)

    async def save_checkpoint(
        self,
        checkpoint_id: str,
        tensors: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> CheckpointManifest:
        """Save tensors to S3 as parallel sharded objects, then write manifest last."""
        session = aioboto3.Session()
        async with session.client("s3", **self._client_kwargs()) as s3:
            sem = asyncio.Semaphore(self.max_concurrency)
            tensor_specs: dict[str, TensorSpec] = {}
            all_puts: list[Any] = []

            for name, tensor in tensors.items():
                raw = tensor.tobytes()
                shards = self._shard_bytes(raw)
                # Checksums are computed on raw (pre-compression) bytes for integrity
                checksums = [xxhash.xxh64(s).hexdigest() for s in shards]
                tensor_specs[name] = TensorSpec(
                    shape=tuple(tensor.shape),
                    dtype=str(tensor.dtype),
                    nbytes=tensor.nbytes,
                    num_shards=len(shards),
                    shard_size=self.shard_size,
                    checksums=checksums,
                )
                for i, shard in enumerate(shards):
                    body = self._compress_shard(shard) if self._compress else shard
                    key = f"{self._prefix(checkpoint_id)}/{name}/shard-{i:04d}"
                    all_puts.append(_put_with_sem(s3, sem, self.bucket, key, body))

            try:
                await asyncio.gather(*all_puts)
            except Exception as exc:
                raise CheckpointWriteError(
                    f"Failed to write shards for checkpoint {checkpoint_id}"
                ) from exc

            manifest = CheckpointManifest(
                checkpoint_id=checkpoint_id,
                method=metadata.get("method", "unknown"),
                timestamp=time.time(),
                total_bytes=sum(t.nbytes for t in tensors.values()),
                tensor_specs=tensor_specs,
                metadata=metadata,
                compression="zstd" if self._compress else None,
            )
            manifest_key = f"{self._prefix(checkpoint_id)}/_manifest.json"
            try:
                await s3.put_object(
                    Bucket=self.bucket,
                    Key=manifest_key,
                    Body=json.dumps(manifest.to_dict()).encode(),
                )
            except Exception as exc:
                raise CheckpointWriteError(
                    f"Failed to write manifest for checkpoint {checkpoint_id}"
                ) from exc

            logger.info(
                "Saved checkpoint %s (%.1f MB, %d tensors)",
                checkpoint_id,
                manifest.total_bytes / 1e6,
                len(tensors),
            )
            return manifest

    async def load_checkpoint(
        self,
        checkpoint_id: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Load tensors from S3, verifying per-shard checksums."""
        session = aioboto3.Session()
        async with session.client("s3", **self._client_kwargs()) as s3:
            manifest_key = f"{self._prefix(checkpoint_id)}/_manifest.json"
            try:
                resp = await s3.get_object(Bucket=self.bucket, Key=manifest_key)
                manifest_data = json.loads(await resp["Body"].read())
            except ClientError as exc:
                if exc.response["Error"]["Code"] == "NoSuchKey":
                    raise CheckpointReadError(
                        f"No manifest found for checkpoint {checkpoint_id}"
                    ) from exc
                raise CheckpointReadError(
                    f"Failed to read manifest for checkpoint {checkpoint_id}"
                ) from exc

            manifest = CheckpointManifest.from_dict(manifest_data)
            sem = asyncio.Semaphore(self.max_concurrency)
            tensors: dict[str, np.ndarray] = {}
            use_decompress = manifest.compression == "zstd"

            for name, spec in manifest.tensor_specs.items():
                keys = [
                    f"{self._prefix(checkpoint_id)}/{name}/shard-{i:04d}"
                    for i in range(spec.num_shards)
                ]
                try:
                    shard_bytes: list[bytes] = list(await asyncio.gather(
                        *[_get_with_sem(s3, sem, self.bucket, k) for k in keys]
                    ))
                except ClientError as exc:
                    raise CheckpointReadError(
                        f"Failed to read shards for tensor {name} in checkpoint {checkpoint_id}"
                    ) from exc

                if use_decompress:
                    shard_bytes = [self._decompress_shard(s) for s in shard_bytes]

                for i, (shard, expected) in enumerate(
                    zip(shard_bytes, spec.checksums, strict=True)
                ):
                    actual = xxhash.xxh64(shard).hexdigest()
                    if actual != expected:
                        raise CheckpointCorruptionError(
                            f"Checksum mismatch for tensor {name} shard {i}: "
                            f"{actual} != {expected}"
                        )

                raw = b"".join(shard_bytes)
                tensors[name] = np.frombuffer(raw, dtype=spec.dtype).reshape(spec.shape)

            return tensors, manifest.metadata

    async def list_checkpoints(self, prefix: str = "") -> list[dict[str, Any]]:
        """Return manifest dicts for all checkpoints matching the prefix, sorted by timestamp."""
        session = aioboto3.Session()
        async with session.client("s3", **self._client_kwargs()) as s3:
            prefix_filter = f"{self.job_id}/ckpt/{prefix}"
            paginator = s3.get_paginator("list_objects_v2")
            manifest_keys: list[str] = []

            async for page in paginator.paginate(Bucket=self.bucket, Prefix=prefix_filter):
                for obj in page.get("Contents", []):
                    if obj["Key"].endswith("/_manifest.json"):
                        manifest_keys.append(obj["Key"])

            results: list[dict[str, Any]] = []
            for key in manifest_keys:
                resp = await s3.get_object(Bucket=self.bucket, Key=key)
                data = json.loads(await resp["Body"].read())
                results.append(data)

            return sorted(results, key=lambda d: d["timestamp"])

    async def delete_checkpoint(self, checkpoint_id: str) -> None:
        """Delete all S3 objects under the checkpoint prefix."""
        session = aioboto3.Session()
        async with session.client("s3", **self._client_kwargs()) as s3:
            paginator = s3.get_paginator("list_objects_v2")
            keys_to_delete: list[str] = []

            async for page in paginator.paginate(
                Bucket=self.bucket, Prefix=f"{self._prefix(checkpoint_id)}/"
            ):
                for obj in page.get("Contents", []):
                    keys_to_delete.append(obj["Key"])

            for i in range(0, len(keys_to_delete), 1000):
                batch = [{"Key": k} for k in keys_to_delete[i : i + 1000]]
                if batch:
                    await s3.delete_objects(Bucket=self.bucket, Delete={"Objects": batch})

            logger.info("Deleted checkpoint %s (%d objects)", checkpoint_id, len(keys_to_delete))
