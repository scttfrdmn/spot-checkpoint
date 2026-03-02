"""Tests for PyTorchTrainingAdapter.

All tests skip if torch is not installed.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn as nn  # noqa: E402
import torch.optim as optim  # noqa: E402

from spot_checkpoint.adapters.torch import PyTorchTrainingAdapter  # noqa: E402
from spot_checkpoint.protocol import AdapterError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_model() -> nn.Module:
    """Return a tiny Linear model (easy to compare weights)."""
    model = nn.Linear(4, 2)
    # Fix weights for reproducibility
    nn.init.ones_(model.weight)
    nn.init.zeros_(model.bias)
    return model


def _adam_optimizer(model: nn.Module) -> optim.Optimizer:
    return optim.Adam(model.parameters(), lr=1e-3)


def _do_one_step(model: nn.Module, optimizer: optim.Optimizer) -> float:
    """Run one synthetic training step; return scalar loss."""
    optimizer.zero_grad()
    x = torch.randn(8, 4)
    out = model(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()
    return float(loss.item())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_checkpoint_contains_model_tensors() -> None:
    """Payload tensors include 'model/weight' and 'model/bias'."""
    model = _small_model()
    optimizer = _adam_optimizer(model)
    adapter = PyTorchTrainingAdapter(model, optimizer)
    payload = adapter.checkpoint_state()
    assert "model/weight" in payload.tensors
    assert "model/bias" in payload.tensors
    assert payload.method == "pytorch-training"


def test_checkpoint_contains_optimizer_tensors() -> None:
    """After one Adam step, opt/state/0/exp_avg appears in tensors."""
    model = _small_model()
    optimizer = _adam_optimizer(model)
    _do_one_step(model, optimizer)
    adapter = PyTorchTrainingAdapter(model, optimizer, epoch=1, step=1)
    payload = adapter.checkpoint_state()
    # Adam accumulates exp_avg for param 0
    assert any(k.startswith("opt/state/0/") for k in payload.tensors)


def test_restore_roundtrip() -> None:
    """checkpoint → new model/optimizer → restore → weights match original."""
    model = _small_model()
    optimizer = _adam_optimizer(model)
    _do_one_step(model, optimizer)

    adapter = PyTorchTrainingAdapter(model, optimizer, epoch=2, step=10)
    adapter.loss = 0.42
    payload = adapter.checkpoint_state()

    # New model with different weights
    new_model = _small_model()
    nn.init.constant_(new_model.weight, 0.5)
    new_optimizer = _adam_optimizer(new_model)
    new_adapter = PyTorchTrainingAdapter(new_model, new_optimizer)
    new_adapter.restore_state(payload)

    np.testing.assert_allclose(
        new_model.weight.detach().numpy(),
        model.weight.detach().numpy(),
        rtol=1e-5,
    )
    assert new_adapter.epoch == 2
    assert new_adapter.step == 10
    assert new_adapter.loss == pytest.approx(0.42)


def test_restore_optimizer_state() -> None:
    """Restored optimizer state_dict has the same structure as original."""
    model = _small_model()
    optimizer = _adam_optimizer(model)
    _do_one_step(model, optimizer)

    adapter = PyTorchTrainingAdapter(model, optimizer)
    payload = adapter.checkpoint_state()

    new_model = _small_model()
    new_optimizer = _adam_optimizer(new_model)
    new_adapter = PyTorchTrainingAdapter(new_model, new_optimizer)
    new_adapter.restore_state(payload)

    orig_state = optimizer.state_dict()["state"]
    restored_state = new_optimizer.state_dict()["state"]
    assert set(orig_state.keys()) == set(restored_state.keys())


def test_size_estimate() -> None:
    """checkpoint_size_estimate is a positive integer."""
    model = _small_model()
    optimizer = _adam_optimizer(model)
    adapter = PyTorchTrainingAdapter(model, optimizer)
    estimate = adapter.checkpoint_size_estimate
    assert isinstance(estimate, int)
    assert estimate > 0


def test_empty_model_raises() -> None:
    """Model with no parameters raises AdapterError at checkpoint time."""
    class EmptyModel(nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
            return x

    model = EmptyModel()
    optimizer = optim.Adam([], lr=1e-3)
    adapter = PyTorchTrainingAdapter(model, optimizer)
    with pytest.raises(AdapterError, match="no parameters"):
        adapter.checkpoint_state()


def test_epoch_step_metadata() -> None:
    """Metadata fields epoch, step, loss are stored and restored correctly."""
    model = _small_model()
    optimizer = _adam_optimizer(model)
    adapter = PyTorchTrainingAdapter(model, optimizer, epoch=5, step=200, loss=0.1)
    payload = adapter.checkpoint_state()
    assert payload.metadata["epoch"] == 5
    assert payload.metadata["step"] == 200
    assert payload.metadata["loss"] == pytest.approx(0.1)


def test_torch_version_in_metadata() -> None:
    """torch_version key is stored in checkpoint metadata."""
    model = _small_model()
    optimizer = _adam_optimizer(model)
    adapter = PyTorchTrainingAdapter(model, optimizer)
    payload = adapter.checkpoint_state()
    assert "torch_version" in payload.metadata
    assert isinstance(payload.metadata["torch_version"], str)


def test_multiple_roundtrips() -> None:
    """Multiple checkpoint/restore cycles remain consistent."""
    model = _small_model()
    optimizer = _adam_optimizer(model)

    adapter = PyTorchTrainingAdapter(model, optimizer)
    for i in range(3):
        _do_one_step(model, optimizer)
        adapter.step = i + 1
        adapter.epoch = i
        payload = adapter.checkpoint_state()

        new_model = _small_model()
        new_optimizer = _adam_optimizer(new_model)
        new_adapter = PyTorchTrainingAdapter(new_model, new_optimizer)
        new_adapter.restore_state(payload)
        assert new_adapter.step == i + 1
