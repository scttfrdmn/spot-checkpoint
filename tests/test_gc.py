"""Tests for gc.garbage_collect()."""

from __future__ import annotations

import json

import numpy as np

from spot_checkpoint.gc import garbage_collect
from spot_checkpoint.storage import LocalStore


async def _save(store: LocalStore, ckpt_id: str, ts_offset: float = 0.0) -> None:
    """Save a minimal checkpoint and patch its timestamp for deterministic ordering."""
    tensor = np.zeros((2,), dtype=np.float64)
    await store.save_checkpoint(ckpt_id, {"data": tensor}, {"method": "fake"})
    manifest_path = store._ckpt_dir / ckpt_id / "_manifest.json"
    data = json.loads(manifest_path.read_text())
    data["timestamp"] = 1000.0 + ts_offset
    manifest_path.write_text(json.dumps(data))


async def test_gc_empty_store(local_store: LocalStore) -> None:
    result = await garbage_collect(local_store, prefix="")
    assert result == {"total": 0, "kept": 0, "deleted": 0}


async def test_gc_no_keep_keeps_all(local_store: LocalStore) -> None:
    await _save(local_store, "ckpt-001", 0.0)
    await _save(local_store, "ckpt-002", 10.0)
    await _save(local_store, "ckpt-003", 20.0)

    result = await garbage_collect(local_store, prefix="")

    assert result["deleted"] == 0
    assert result["kept"] == 3
    assert result["total"] == 3
    remaining = await local_store.list_checkpoints("")
    assert len(remaining) == 3


async def test_gc_keep_n_deletes_oldest(local_store: LocalStore) -> None:
    for i, offset in enumerate([0.0, 10.0, 20.0, 30.0], 1):
        await _save(local_store, f"ckpt-{i:03d}", offset)

    result = await garbage_collect(local_store, prefix="", keep=2)

    assert result["total"] == 4
    assert result["kept"] == 2
    assert result["deleted"] == 2

    remaining = await local_store.list_checkpoints("")
    remaining_ids = {c["checkpoint_id"] for c in remaining}
    assert remaining_ids == {"ckpt-003", "ckpt-004"}


async def test_gc_keep_gte_total_deletes_nothing(local_store: LocalStore) -> None:
    await _save(local_store, "ckpt-001", 0.0)
    await _save(local_store, "ckpt-002", 10.0)

    result = await garbage_collect(local_store, prefix="", keep=5)

    assert result["deleted"] == 0
    assert result["total"] == 2


async def test_gc_keep_zero_deletes_all(local_store: LocalStore) -> None:
    for i in range(1, 4):
        await _save(local_store, f"ckpt-{i:03d}", float(i * 10))

    result = await garbage_collect(local_store, prefix="", keep=0)

    assert result["total"] == 3
    assert result["deleted"] == 3
    assert result["kept"] == 0

    remaining = await local_store.list_checkpoints("")
    assert remaining == []


async def test_gc_keep_one(local_store: LocalStore) -> None:
    for i, offset in enumerate([0.0, 10.0, 20.0], 1):
        await _save(local_store, f"ckpt-{i:03d}", offset)

    result = await garbage_collect(local_store, prefix="", keep=1)

    assert result["deleted"] == 2

    remaining = await local_store.list_checkpoints("")
    assert len(remaining) == 1
    assert remaining[0]["checkpoint_id"] == "ckpt-003"


async def test_gc_prefix_forwarded(local_store: LocalStore) -> None:
    await _save(local_store, "scf-001", 0.0)
    await _save(local_store, "scf-002", 10.0)
    await _save(local_store, "ccsd-001", 5.0)

    result = await garbage_collect(local_store, prefix="scf-", keep=1)

    assert result["total"] == 2
    assert result["deleted"] == 1
    assert result["kept"] == 1

    all_remaining = await local_store.list_checkpoints("")
    remaining_ids = {c["checkpoint_id"] for c in all_remaining}
    assert "ccsd-001" in remaining_ids
    assert "scf-001" not in remaining_ids
    assert "scf-002" in remaining_ids


async def test_gc_return_dict_keys(local_store: LocalStore) -> None:
    await _save(local_store, "ckpt-001", 0.0)

    result = await garbage_collect(local_store, prefix="", keep=1)
    assert set(result.keys()) == {"total", "kept", "deleted"}

    result2 = await garbage_collect(local_store, prefix="")
    assert set(result2.keys()) == {"total", "kept", "deleted"}
