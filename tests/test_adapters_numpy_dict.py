"""Tests for NumpyDictAdapter."""

from __future__ import annotations

import numpy as np
import pytest

from spot_checkpoint.adapters.numpy_dict import NumpyDictAdapter
from spot_checkpoint.protocol import AdapterError


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _State:
    """Simple mutable state container for tests."""

    def __init__(self) -> None:
        self.x = np.array([1.0, 2.0, 3.0])
        self.grad = np.array([0.1, 0.2, 0.3])
        self.iteration = 5


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip() -> None:
    """checkpoint_state() returns a payload with the expected tensor keys."""
    state = _State()
    adapter = NumpyDictAdapter(
        get_state=lambda: {"x": state.x, "grad": state.grad},
        set_state=lambda s: None,
    )
    payload = adapter.checkpoint_state()
    assert "x" in payload.tensors
    assert "grad" in payload.tensors
    assert payload.method == "generic"
    np.testing.assert_array_equal(payload.tensors["x"], state.x)
    np.testing.assert_array_equal(payload.tensors["grad"], state.grad)


def test_restore_roundtrip() -> None:
    """checkpoint → restore preserves tensor values in a new state container."""
    original = _State()
    original.x = np.array([7.0, 8.0, 9.0])

    restored: dict[str, np.ndarray] = {}

    adapter = NumpyDictAdapter(
        get_state=lambda: {"x": original.x},
        set_state=lambda s: restored.update(s),
    )
    payload = adapter.checkpoint_state()

    # Mutate original to prove restore is independent
    original.x = np.zeros(3)

    adapter.restore_state(payload)
    np.testing.assert_array_equal(restored["x"], np.array([7.0, 8.0, 9.0]))


def test_empty_state_raises() -> None:
    """get_state returning {} raises AdapterError."""
    adapter = NumpyDictAdapter(
        get_state=lambda: {},
        set_state=lambda s: None,
    )
    with pytest.raises(AdapterError, match="empty dict"):
        adapter.checkpoint_state()


def test_without_metadata_callbacks() -> None:
    """Omitting get/set_metadata does not raise; metadata is empty dict."""
    state = _State()
    adapter = NumpyDictAdapter(
        get_state=lambda: {"x": state.x},
        set_state=lambda s: None,
    )
    payload = adapter.checkpoint_state()
    assert payload.metadata == {}
    # restore with no-op set_metadata should not raise
    adapter.restore_state(payload)


def test_with_metadata_callbacks() -> None:
    """get/set_metadata are called correctly during checkpoint and restore."""
    state = _State()
    meta_out: dict = {}

    adapter = NumpyDictAdapter(
        get_state=lambda: {"x": state.x},
        set_state=lambda s: None,
        get_metadata=lambda: {"iter": state.iteration},
        set_metadata=lambda m: meta_out.update(m),
    )
    payload = adapter.checkpoint_state()
    assert payload.metadata["iter"] == 5

    adapter.restore_state(payload)
    assert meta_out["iter"] == 5


def test_size_estimate() -> None:
    """checkpoint_size_estimate returns sum of array nbytes."""
    x = np.ones(100, dtype=np.float64)   # 800 bytes
    y = np.ones(50, dtype=np.float32)    # 200 bytes
    adapter = NumpyDictAdapter(
        get_state=lambda: {"x": x, "y": y},
        set_state=lambda s: None,
    )
    assert adapter.checkpoint_size_estimate == 800 + 200


def test_custom_method_name() -> None:
    """The method param is stored in the payload."""
    adapter = NumpyDictAdapter(
        get_state=lambda: {"z": np.zeros(3)},
        set_state=lambda s: None,
        method="my-custom-solver",
    )
    payload = adapter.checkpoint_state()
    assert payload.method == "my-custom-solver"


def test_tensors_are_copies() -> None:
    """Tensors in the payload are independent copies, not references."""
    arr = np.array([1.0, 2.0, 3.0])
    adapter = NumpyDictAdapter(
        get_state=lambda: {"a": arr},
        set_state=lambda s: None,
    )
    payload = adapter.checkpoint_state()
    # Mutate original — payload should be unaffected
    arr[:] = 0.0
    np.testing.assert_array_equal(payload.tensors["a"], np.array([1.0, 2.0, 3.0]))
