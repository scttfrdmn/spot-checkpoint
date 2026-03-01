"""Tests for CheckpointPayload and protocol types."""

import numpy as np

from spot_checkpoint.protocol import CheckpointPayload


def test_payload_total_bytes():
    payload = CheckpointPayload(
        tensors={
            "a": np.zeros((10, 10), dtype=np.float64),
            "b": np.zeros((5,), dtype=np.float64),
        },
        metadata={"iteration": 1},
        method="test",
    )
    assert payload.total_bytes == 10 * 10 * 8 + 5 * 8


def test_payload_tensor_summary():
    payload = CheckpointPayload(
        tensors={"mo_coeff": np.zeros((100, 100), dtype=np.float64)},
        metadata={},
        method="scf",
    )
    summary = payload.tensor_summary
    assert "mo_coeff" in summary
    assert "(100, 100)" in summary["mo_coeff"]


def test_payload_empty_tensors():
    payload = CheckpointPayload(
        tensors={},
        metadata={"iteration": 0},
        method="empty",
    )
    assert payload.total_bytes == 0
