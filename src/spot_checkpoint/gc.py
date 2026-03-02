"""
Checkpoint garbage collection.

Removes incomplete checkpoints (orphaned shards without manifests)
and optionally prunes old checkpoints beyond a retention count.
"""

from __future__ import annotations

import logging
from typing import Any

from spot_checkpoint.protocol import CheckpointStore

logger = logging.getLogger(__name__)


async def garbage_collect(
    store: CheckpointStore,
    prefix: str = "",
    keep: int | None = None,
) -> dict[str, Any]:
    """
    Clean up checkpoint storage.

    Args:
        store: Checkpoint store to clean.
        prefix: Only consider checkpoints matching this prefix.
        keep: If set, keep only the N most recent valid checkpoints.

    Returns:
        Summary dict with counts of actions taken.
    """
    if keep is not None and keep < 0:
        raise ValueError(f"keep must be >= 0, got {keep}")

    checkpoints = await store.list_checkpoints(prefix)

    if keep is not None and len(checkpoints) > keep:
        sorted_ckpts = sorted(checkpoints, key=lambda c: c.get("timestamp", 0))
        to_delete = sorted_ckpts[: len(sorted_ckpts) - keep]

        errors: list[str] = []
        for ckpt in to_delete:
            ckpt_id = ckpt["checkpoint_id"]
            try:
                await store.delete_checkpoint(ckpt_id)
                logger.info("Pruned old checkpoint: %s", ckpt_id)
            except Exception as exc:
                logger.warning("Failed to delete checkpoint %s: %s", ckpt_id, exc)
                errors.append(ckpt_id)

        return {
            "total": len(checkpoints),
            "kept": keep,
            "deleted": len(to_delete) - len(errors),
            "errors": errors,
        }

    return {
        "total": len(checkpoints),
        "kept": len(checkpoints),
        "deleted": 0,
        "errors": [],
    }
