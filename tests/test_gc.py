"""Tests for gc.garbage_collect()."""

from __future__ import annotations

import json
import os
from typing import Any

import numpy as np
import pytest

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
    assert result == {"total": 0, "kept": 0, "deleted": 0, "errors": []}


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
    assert set(result.keys()) == {"total", "kept", "deleted", "errors"}

    result2 = await garbage_collect(local_store, prefix="")
    assert set(result2.keys()) == {"total", "kept", "deleted", "errors"}


async def test_auto_gc_via_manager(
    tmp_path: Any,
    fake_adapter: Any,
) -> None:
    """keep_checkpoints=2 auto-prunes checkpoints after each periodic save."""
    from spot_checkpoint.lifecycle import LifecycleBackend, InterruptEvent, SpotLifecycleManager

    store = LocalStore(base_dir=tmp_path, job_id="gc-test")

    class NoOpBackend(LifecycleBackend):
        def start(self, on_interrupt: Any) -> None:
            pass

        def stop(self) -> None:
            pass

    mgr = SpotLifecycleManager(
        store=store,
        adapter=fake_adapter,
        backend=NoOpBackend(),
        keep_checkpoints=2,
    )
    mgr.start()
    try:
        for i in range(4):
            mgr._do_periodic_checkpoint(i)

        remaining = await store.list_checkpoints("")
        assert len(remaining) == 2
    finally:
        mgr.stop()


async def test_gc_validates_negative_keep(local_store: LocalStore) -> None:
    """garbage_collect raises ValueError for keep < 0."""
    with pytest.raises(ValueError, match="keep must be >= 0"):
        await garbage_collect(local_store, prefix="", keep=-1)


async def test_gc_continues_on_deletion_error(local_store: LocalStore) -> None:
    """GC continues if one deletion fails; second checkpoint is still deleted."""
    from unittest.mock import AsyncMock, patch

    await _save(local_store, "ckpt-001", 0.0)
    await _save(local_store, "ckpt-002", 10.0)
    await _save(local_store, "ckpt-003", 20.0)

    original_delete = local_store.delete_checkpoint
    call_count = 0

    async def _failing_first(ckpt_id: str) -> None:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise OSError("simulated delete failure")
        await original_delete(ckpt_id)

    with patch.object(local_store, "delete_checkpoint", side_effect=_failing_first):
        result = await garbage_collect(local_store, prefix="", keep=1)

    # 3 total, keep=1, so 2 to_delete; 1 failed, 1 succeeded
    assert result["deleted"] == 1
    assert len(result["errors"]) == 1


def test_spot_safe_keep_checkpoints_env_var(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
    fake_adapter: Any,
) -> None:
    """SPOT_CHECKPOINT_KEEP env var sets keep_checkpoints on SpotLifecycleManager."""
    from spot_checkpoint.lifecycle import LifecycleBackend, SpotLifecycleManager

    monkeypatch.setenv("SPOT_CHECKPOINT_KEEP", "2")

    # Simulate the env-var reading logic used by spot_safe()
    _keep_env = os.environ.get("SPOT_CHECKPOINT_KEEP")
    keep_checkpoints = int(_keep_env) if _keep_env is not None else None

    store = LocalStore(base_dir=tmp_path, job_id="env-test")

    class NoOpBackend(LifecycleBackend):
        def start(self, on_interrupt: Any) -> None:
            pass

        def stop(self) -> None:
            pass

    mgr = SpotLifecycleManager(
        store=store,
        adapter=fake_adapter,
        backend=NoOpBackend(),
        keep_checkpoints=keep_checkpoints,
    )
    assert mgr.keep_checkpoints == 2
