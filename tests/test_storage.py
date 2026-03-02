"""Tests for LocalStore checkpoint storage."""

import importlib

import numpy as np
import pytest

from spot_checkpoint.protocol import CheckpointReadError
from spot_checkpoint.storage import LocalStore

_needs_zstd = pytest.mark.skipif(
    importlib.util.find_spec("zstandard") is None,
    reason="zstandard not installed",
)


@pytest.mark.asyncio
async def test_save_and_load_roundtrip(local_store: LocalStore):
    """Save tensors, load them back, verify equality."""
    tensors = {
        "mo_coeff": np.random.rand(50, 50),
        "mo_occ": np.array([2.0, 2.0, 0.0, 0.0, 0.0]),
    }
    metadata = {"iteration": 5, "e_tot": -75.123}

    await local_store.save_checkpoint("ckpt-001", tensors, metadata)
    loaded_tensors, loaded_metadata = await local_store.load_checkpoint("ckpt-001")

    np.testing.assert_array_equal(loaded_tensors["mo_coeff"], tensors["mo_coeff"])
    np.testing.assert_array_equal(loaded_tensors["mo_occ"], tensors["mo_occ"])
    assert loaded_metadata["iteration"] == 5
    assert loaded_metadata["e_tot"] == -75.123


@pytest.mark.asyncio
async def test_list_checkpoints(local_store: LocalStore):
    """List returns only checkpoints with valid manifests."""
    tensors = {"x": np.zeros(10)}

    await local_store.save_checkpoint("ckpt-001", tensors, {"i": 1})
    await local_store.save_checkpoint("ckpt-002", tensors, {"i": 2})

    results = await local_store.list_checkpoints("ckpt")
    assert len(results) == 2

    # Filter by prefix
    results = await local_store.list_checkpoints("ckpt-001")
    assert len(results) == 1


@pytest.mark.asyncio
async def test_delete_checkpoint(local_store: LocalStore):
    tensors = {"x": np.zeros(10)}
    await local_store.save_checkpoint("ckpt-del", tensors, {})

    await local_store.delete_checkpoint("ckpt-del")

    results = await local_store.list_checkpoints("ckpt-del")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_load_nonexistent_raises(local_store: LocalStore):
    with pytest.raises(CheckpointReadError):
        await local_store.load_checkpoint("nonexistent")


@pytest.mark.asyncio
async def test_large_tensor_roundtrip(local_store: LocalStore):
    """Verify large tensors survive save/load without corruption."""
    # Simulate a medium CCSD t2 tensor
    t2 = np.random.rand(10, 10, 30, 30).astype(np.float64)
    tensors = {"t2": t2}

    await local_store.save_checkpoint("ckpt-large", tensors, {"method": "ccsd"})
    loaded, _meta = await local_store.load_checkpoint("ckpt-large")

    np.testing.assert_array_equal(loaded["t2"], t2)


@_needs_zstd
@pytest.mark.asyncio
async def test_local_compressed_roundtrip(tmp_path):
    """LocalStore with compress=True: tensors survive save/load."""
    store = LocalStore(base_dir=tmp_path, job_id="compress-job", compress=True)
    tensors = {
        "mo_coeff": np.random.rand(20, 20).astype(np.float64),
        "mo_occ": np.array([2.0, 2.0, 0.0], dtype=np.float64),
    }
    await store.save_checkpoint("ckpt-c001", tensors, {"method": "scf"})
    loaded, meta = await store.load_checkpoint("ckpt-c001")

    np.testing.assert_array_equal(loaded["mo_coeff"], tensors["mo_coeff"])
    np.testing.assert_array_equal(loaded["mo_occ"], tensors["mo_occ"])
    assert meta["method"] == "scf"


@_needs_zstd
@pytest.mark.asyncio
async def test_local_compress_creates_zst_files(tmp_path):
    """compress=True writes .bin.zst files, not .npy files."""
    store = LocalStore(base_dir=tmp_path, job_id="compress-job", compress=True)
    tensor = np.random.rand(50).astype(np.float64)
    await store.save_checkpoint("ckpt-c002", {"data": tensor}, {"method": "test"})

    ckpt_dir = store._ckpt_dir / "ckpt-c002"
    assert (ckpt_dir / "data.bin.zst").exists()
    assert not (ckpt_dir / "data.npy").exists()


@_needs_zstd
@pytest.mark.asyncio
async def test_local_compress_reduces_size(tmp_path):
    """Compressed checkpoint is smaller than uncompressed for compressible data."""
    plain_store = LocalStore(base_dir=tmp_path / "plain", job_id="j", compress=False)
    zstd_store = LocalStore(base_dir=tmp_path / "zstd", job_id="j", compress=True)

    # Zeros compress very well
    tensor = np.zeros((100, 100), dtype=np.float64)
    await plain_store.save_checkpoint("ckpt", {"data": tensor}, {"method": "test"})
    await zstd_store.save_checkpoint("ckpt", {"data": tensor}, {"method": "test"})

    plain_size = (plain_store._ckpt_dir / "ckpt" / "data.npy").stat().st_size
    zstd_size = (zstd_store._ckpt_dir / "ckpt" / "data.bin.zst").stat().st_size
    assert zstd_size < plain_size


@_needs_zstd
@pytest.mark.asyncio
async def test_local_compress_manifest_records_compression(tmp_path):
    """compress=True records compression='zstd' in the manifest."""
    import json

    store = LocalStore(base_dir=tmp_path, job_id="j", compress=True)
    tensor = np.zeros((5,), dtype=np.float64)
    await store.save_checkpoint("ckpt-m", {"x": tensor}, {"method": "test"})

    manifest_path = store._ckpt_dir / "ckpt-m" / "_manifest.json"
    data = json.loads(manifest_path.read_text())
    assert data.get("compression") == "zstd"


# ---------------------------------------------------------------------------
# TensorSpec and CheckpointManifest protocol tests (v0.9.0)
# ---------------------------------------------------------------------------

from spot_checkpoint.protocol import CheckpointManifest, TensorSpec


def test_tensorspec_invalid_num_shards_raises() -> None:
    """TensorSpec raises ValueError when num_shards < 1."""
    with pytest.raises(ValueError, match="num_shards must be >= 1"):
        TensorSpec(
            shape=(10,),
            dtype="float64",
            nbytes=80,
            num_shards=0,
            shard_size=80,
            checksums=[],
        )


def test_tensorspec_checksum_count_mismatch_raises() -> None:
    """TensorSpec raises ValueError when checksums length != num_shards."""
    with pytest.raises(ValueError, match="checksums length"):
        TensorSpec(
            shape=(10,),
            dtype="float64",
            nbytes=80,
            num_shards=2,
            shard_size=40,
            checksums=["abc123"],  # only 1, but num_shards=2
        )


def test_manifest_schema_version_roundtrip() -> None:
    """to_dict/from_dict preserves schema_version=1."""
    spec = TensorSpec(
        shape=(4,), dtype="float64", nbytes=32,
        num_shards=1, shard_size=32, checksums=["deadbeef"],
    )
    manifest = CheckpointManifest(
        checkpoint_id="ckpt-v1",
        method="test",
        timestamp=1000.0,
        total_bytes=32,
        tensor_specs={"data": spec},
        metadata={},
        schema_version=1,
    )
    d = manifest.to_dict()
    assert d["schema_version"] == 1
    restored = CheckpointManifest.from_dict(d)
    assert restored.schema_version == 1


def test_manifest_schema_version_default_on_old_manifest() -> None:
    """from_dict returns schema_version=1 when key is absent (old manifest)."""
    spec = TensorSpec(
        shape=(4,), dtype="float64", nbytes=32,
        num_shards=1, shard_size=32, checksums=["deadbeef"],
    )
    manifest = CheckpointManifest(
        checkpoint_id="ckpt-old",
        method="test",
        timestamp=1000.0,
        total_bytes=32,
        tensor_specs={"data": spec},
        metadata={},
    )
    d = manifest.to_dict()
    del d["schema_version"]  # simulate old format
    restored = CheckpointManifest.from_dict(d)
    assert restored.schema_version == 1
