"""Tests for S3ShardedStore using moto ThreadedMotoServer."""

from __future__ import annotations

from typing import Any

import boto3
import numpy as np
import pytest

from spot_checkpoint.protocol import CheckpointCorruptionError, CheckpointReadError
from spot_checkpoint.storage import S3ShardedStore

pytestmark = pytest.mark.asyncio


def _boto3_client(endpoint_url: str) -> Any:
    return boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=endpoint_url,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )


async def test_save_and_load_roundtrip(s3_store: S3ShardedStore) -> None:
    """Save a small tensor and reload it; verify array equality and metadata passthrough."""
    arr = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float64)
    metadata = {"method": "test", "iteration": 5}

    await s3_store.save_checkpoint("ckpt-001", {"weights": arr}, metadata)
    tensors, meta = await s3_store.load_checkpoint("ckpt-001")

    np.testing.assert_array_equal(tensors["weights"], arr)
    assert meta["iteration"] == 5
    assert meta["method"] == "test"


async def test_sharding_splits_large_tensor(s3_store: S3ShardedStore) -> None:
    """Tensor larger than shard_size (4KB) must produce multiple shard objects in S3."""
    # 10 rows × 100 float64 = 8000 bytes > 4096 → at least 2 shards
    arr = np.ones((10, 100), dtype=np.float64)

    await s3_store.save_checkpoint("ckpt-sharded", {"data": arr}, {})

    s3 = _boto3_client(s3_store.endpoint_url)  # type: ignore[arg-type]
    resp = s3.list_objects_v2(
        Bucket="test-bucket", Prefix="test-job/ckpt/ckpt-sharded/data/"
    )
    shard_keys = [obj["Key"] for obj in resp.get("Contents", [])]
    assert len(shard_keys) >= 2, f"Expected multiple shards, got {shard_keys}"

    # Verify round-trip still works
    tensors, _ = await s3_store.load_checkpoint("ckpt-sharded")
    np.testing.assert_array_equal(tensors["data"], arr)


async def test_checksum_corruption_raises(s3_store: S3ShardedStore) -> None:
    """Overwriting a shard with junk bytes must raise CheckpointCorruptionError on load."""
    arr = np.ones((10, 100), dtype=np.float64)  # 8000 bytes → multiple shards

    await s3_store.save_checkpoint("ckpt-corrupt", {"data": arr}, {})

    # Overwrite the first shard with garbage
    s3 = _boto3_client(s3_store.endpoint_url)  # type: ignore[arg-type]
    s3.put_object(
        Bucket="test-bucket",
        Key="test-job/ckpt/ckpt-corrupt/data/shard-0000",
        Body=b"\x00" * 4096,
    )

    with pytest.raises(CheckpointCorruptionError):
        await s3_store.load_checkpoint("ckpt-corrupt")


async def test_missing_manifest_raises(s3_store: S3ShardedStore) -> None:
    """Loading a non-existent checkpoint id must raise CheckpointReadError."""
    with pytest.raises(CheckpointReadError):
        await s3_store.load_checkpoint("does-not-exist")


async def test_list_checkpoints_prefix_filter(s3_store: S3ShardedStore) -> None:
    """list_checkpoints with a prefix filter returns only matching checkpoints."""
    arr = np.ones((4,), dtype=np.float32)

    await s3_store.save_checkpoint("scf-001", {"x": arr}, {"method": "scf"})
    await s3_store.save_checkpoint("scf-002", {"x": arr}, {"method": "scf"})
    await s3_store.save_checkpoint("ccsd-001", {"x": arr}, {"method": "ccsd"})

    scf_results = await s3_store.list_checkpoints("scf-")
    assert len(scf_results) == 2
    assert all(r["metadata"]["method"] == "scf" for r in scf_results)

    ccsd_results = await s3_store.list_checkpoints("ccsd-")
    assert len(ccsd_results) == 1


async def test_list_excludes_incomplete(s3_store: S3ShardedStore) -> None:
    """A checkpoint with shards but no manifest must not appear in list_checkpoints."""
    s3 = _boto3_client(s3_store.endpoint_url)  # type: ignore[arg-type]
    # Write a shard without a manifest
    s3.put_object(
        Bucket="test-bucket",
        Key="test-job/ckpt/orphan-001/data/shard-0000",
        Body=b"\x00" * 100,
    )

    results = await s3_store.list_checkpoints("orphan-")
    assert results == []


async def test_delete_removes_all_keys(s3_store: S3ShardedStore) -> None:
    """After delete_checkpoint, no S3 keys should remain under that prefix."""
    arr = np.ones((10, 100), dtype=np.float64)

    await s3_store.save_checkpoint("ckpt-del", {"weights": arr}, {})
    await s3_store.delete_checkpoint("ckpt-del")

    s3 = _boto3_client(s3_store.endpoint_url)  # type: ignore[arg-type]
    resp = s3.list_objects_v2(Bucket="test-bucket", Prefix="test-job/ckpt/ckpt-del/")
    assert resp.get("KeyCount", 0) == 0, "Expected no keys after deletion"


async def test_large_sharded_roundtrip(s3_store: S3ShardedStore) -> None:
    """Full roundtrip for a ~720KB tensor producing ~175 shards at 4KB each."""
    # (10, 10, 30, 30) float64 = 720,000 bytes → ~175 shards at 4096 bytes
    arr = np.random.default_rng(42).standard_normal((10, 10, 30, 30)).astype(np.float64)

    manifest = await s3_store.save_checkpoint(
        "ckpt-large", {"mo_coeff": arr}, {"method": "casscf"}
    )
    assert manifest.tensor_specs["mo_coeff"].num_shards > 100

    tensors, meta = await s3_store.load_checkpoint("ckpt-large")
    np.testing.assert_array_equal(tensors["mo_coeff"], arr)
    assert meta["method"] == "casscf"


# ---------------------------------------------------------------------------
# Compression tests (skip if zstandard not installed)
# ---------------------------------------------------------------------------

_needs_zstd = pytest.mark.skipif(
    __import__("importlib").util.find_spec("zstandard") is None,
    reason="zstandard not installed",
)


@pytest.fixture
async def s3_store_compressed(moto_server: Any) -> Any:
    """S3ShardedStore with compress=True, backed by ThreadedMotoServer."""
    endpoint_url = "http://127.0.0.1:5555"
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=endpoint_url,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    bucket = "test-bucket-compressed"
    s3.create_bucket(Bucket=bucket)
    store = S3ShardedStore(
        bucket=bucket,
        job_id="test-job",
        shard_size=4 * 1024,
        region="us-east-1",
        endpoint_url=endpoint_url,
        compress=True,
    )
    yield store
    response = s3.list_objects_v2(Bucket=bucket)
    objects = [{"Key": obj["Key"]} for obj in response.get("Contents", [])]
    if objects:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
    s3.delete_bucket(Bucket=bucket)


@_needs_zstd
async def test_compressed_roundtrip(s3_store_compressed: S3ShardedStore) -> None:
    """Save with compress=True, load, verify tensors and metadata match exactly."""
    arr = np.random.default_rng(0).standard_normal((20, 20)).astype(np.float64)
    metadata = {"method": "ccsd", "iteration": 10}

    manifest = await s3_store_compressed.save_checkpoint(
        "ckpt-compressed-001", {"t2": arr}, metadata
    )
    assert manifest.compression == "zstd"

    tensors, meta = await s3_store_compressed.load_checkpoint("ckpt-compressed-001")
    np.testing.assert_array_equal(tensors["t2"], arr)
    assert meta["iteration"] == 10
    assert meta["method"] == "ccsd"


@_needs_zstd
async def test_compress_reduces_size(s3_store: S3ShardedStore, s3_store_compressed: S3ShardedStore) -> None:
    """Compressed checkpoint total S3 object size must be smaller than uncompressed."""
    # Use a compressible tensor (repeated pattern compresses well)
    arr = np.zeros((100, 100), dtype=np.float64)

    manifest_raw = await s3_store.save_checkpoint("ckpt-raw-size", {"data": arr}, {})
    manifest_cmp = await s3_store_compressed.save_checkpoint("ckpt-cmp-size", {"data": arr}, {})

    # The raw byte count in the manifest is the same (uncompressed logical size)
    assert manifest_raw.total_bytes == manifest_cmp.total_bytes

    # But compressed objects on S3 should be smaller — verify via listing object sizes
    raw_client = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=s3_store.endpoint_url,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    cmp_client = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=s3_store_compressed.endpoint_url,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )

    raw_resp = raw_client.list_objects_v2(
        Bucket=s3_store.bucket, Prefix="test-job/ckpt/ckpt-raw-size/"
    )
    cmp_resp = cmp_client.list_objects_v2(
        Bucket=s3_store_compressed.bucket, Prefix="test-job/ckpt/ckpt-cmp-size/"
    )

    raw_total = sum(obj["Size"] for obj in raw_resp.get("Contents", []))
    cmp_total = sum(obj["Size"] for obj in cmp_resp.get("Contents", []))

    assert cmp_total < raw_total, (
        f"Compressed ({cmp_total}B) should be smaller than raw ({raw_total}B)"
    )


@_needs_zstd
@pytest.mark.parametrize("compress", [False, True])
async def test_roundtrip_parameterized(moto_server: Any, compress: bool) -> None:
    """Parameterized roundtrip: both compressed and uncompressed paths must produce identical tensors."""
    endpoint_url = "http://127.0.0.1:5555"
    bucket = f"test-bucket-param-{int(compress)}"
    s3 = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=endpoint_url,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    s3.create_bucket(Bucket=bucket)

    store = S3ShardedStore(
        bucket=bucket,
        job_id="test-job",
        shard_size=4 * 1024,
        region="us-east-1",
        endpoint_url=endpoint_url,
        compress=compress,
    )

    arr = np.random.default_rng(99).standard_normal((8, 8)).astype(np.float32)
    metadata = {"method": "rhf", "converged": True}

    manifest = await store.save_checkpoint("ckpt-param", {"weights": arr}, metadata)
    assert manifest.compression == ("zstd" if compress else None)

    tensors, meta = await store.load_checkpoint("ckpt-param")
    np.testing.assert_array_equal(tensors["weights"], arr)
    assert meta["converged"] is True

    # Cleanup
    response = s3.list_objects_v2(Bucket=bucket)
    objects = [{"Key": obj["Key"]} for obj in response.get("Contents", [])]
    if objects:
        s3.delete_objects(Bucket=bucket, Delete={"Objects": objects})
    s3.delete_bucket(Bucket=bucket)
