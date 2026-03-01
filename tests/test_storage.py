"""Tests for LocalStore checkpoint storage."""

import numpy as np
import pytest

from spot_checkpoint.protocol import CheckpointCorruptedError, CheckpointLoadError
from spot_checkpoint.storage import LocalStore


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
    with pytest.raises(CheckpointLoadError):
        await local_store.load_checkpoint("nonexistent")


@pytest.mark.asyncio
async def test_large_tensor_roundtrip(local_store: LocalStore):
    """Verify large tensors survive save/load without corruption."""
    # Simulate a medium CCSD t2 tensor
    t2 = np.random.rand(10, 10, 30, 30).astype(np.float64)
    tensors = {"t2": t2}

    await local_store.save_checkpoint("ckpt-large", tensors, {"method": "ccsd"})
    loaded, meta = await local_store.load_checkpoint("ckpt-large")

    np.testing.assert_array_equal(loaded["t2"], t2)
