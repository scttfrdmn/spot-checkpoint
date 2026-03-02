"""Tests for ScipyOptimizeAdapter and ScipySparseLinalgAdapter."""

from __future__ import annotations

import numpy as np
import pytest

from spot_checkpoint.adapters.scipy_opt import (
    ScipyOptimizeAdapter,
    ScipySparseLinalgAdapter,
)
from spot_checkpoint.protocol import AdapterError


# ---------------------------------------------------------------------------
# ScipyOptimizeAdapter
# ---------------------------------------------------------------------------

class TestScipyOptimizeAdapter:
    def test_callback_updates_x(self) -> None:
        """callback(xk) stores xk in adapter.x."""
        x0 = np.zeros(5)
        adapter = ScipyOptimizeAdapter(x0)
        xk = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        adapter.callback(xk)
        np.testing.assert_array_equal(adapter.x, xk)

    def test_callback_copy_independence(self) -> None:
        """callback stores a copy; mutating xk does not affect adapter.x."""
        adapter = ScipyOptimizeAdapter(np.zeros(3))
        xk = np.array([1.0, 2.0, 3.0])
        adapter.callback(xk)
        xk[:] = 0.0
        np.testing.assert_array_equal(adapter.x, np.array([1.0, 2.0, 3.0]))

    def test_iteration_counter(self) -> None:
        """5 callback calls increment adapter.iteration to 5."""
        adapter = ScipyOptimizeAdapter(np.zeros(3))
        for i in range(5):
            adapter.callback(np.full(3, float(i)))
        assert adapter.iteration == 5

    def test_checkpoint_roundtrip(self) -> None:
        """tensors['x'] is present and matches current iterate."""
        adapter = ScipyOptimizeAdapter(np.array([3.14, 2.71]))
        payload = adapter.checkpoint_state()
        assert "x" in payload.tensors
        assert payload.method == "scipy-optimize"
        np.testing.assert_array_almost_equal(payload.tensors["x"], [3.14, 2.71])

    def test_checkpoint_metadata(self) -> None:
        """Metadata contains iteration, fun, and optimize_method."""
        adapter = ScipyOptimizeAdapter(np.ones(2), method="CG")
        adapter.iteration = 42
        adapter.fun = -3.5
        payload = adapter.checkpoint_state()
        assert payload.metadata["iteration"] == 42
        assert payload.metadata["fun"] == pytest.approx(-3.5)
        assert payload.metadata["optimize_method"] == "CG"

    def test_restore_roundtrip(self) -> None:
        """checkpoint → restore → values match original iterate."""
        x0 = np.array([1.0, 2.0, 3.0])
        adapter = ScipyOptimizeAdapter(x0.copy())
        adapter.callback(x0 * 2)
        adapter.fun = -1.0
        payload = adapter.checkpoint_state()

        new_adapter = ScipyOptimizeAdapter(np.zeros(3))
        new_adapter.restore_state(payload)
        np.testing.assert_array_equal(new_adapter.x, x0 * 2)
        assert new_adapter.iteration == 1
        assert new_adapter.fun == pytest.approx(-1.0)

    def test_restore_missing_tensor_raises(self) -> None:
        """restore_state raises AdapterError if 'x' tensor is absent."""
        from spot_checkpoint.protocol import CheckpointPayload
        payload = CheckpointPayload(
            tensors={},
            metadata={},
            method="scipy-optimize",
        )
        adapter = ScipyOptimizeAdapter(np.zeros(3))
        with pytest.raises(AdapterError, match="missing required tensor"):
            adapter.restore_state(payload)

    def test_size_estimate(self) -> None:
        """checkpoint_size_estimate returns x.nbytes."""
        x0 = np.ones(100, dtype=np.float64)
        adapter = ScipyOptimizeAdapter(x0)
        assert adapter.checkpoint_size_estimate == 100 * 8

    def test_initial_x_is_copy(self) -> None:
        """Constructor copies x0; mutating x0 does not affect adapter.x."""
        x0 = np.array([1.0, 2.0])
        adapter = ScipyOptimizeAdapter(x0)
        x0[:] = 99.0
        np.testing.assert_array_equal(adapter.x, np.array([1.0, 2.0]))


# ---------------------------------------------------------------------------
# ScipySparseLinalgAdapter
# ---------------------------------------------------------------------------

class TestScipySparseLinalgAdapter:
    def test_callback_updates_x(self) -> None:
        """callback(xk) stores xk in adapter.x."""
        adapter = ScipySparseLinalgAdapter(np.zeros(4))
        xk = np.array([0.1, 0.2, 0.3, 0.4])
        adapter.callback(xk)
        np.testing.assert_array_equal(adapter.x, xk)

    def test_callback_copy_independence(self) -> None:
        """callback stores a copy; mutating xk does not affect adapter.x."""
        adapter = ScipySparseLinalgAdapter(np.zeros(3))
        xk = np.array([1.0, 2.0, 3.0])
        adapter.callback(xk)
        xk[:] = 0.0
        np.testing.assert_array_equal(adapter.x, np.array([1.0, 2.0, 3.0]))

    def test_iteration_counter(self) -> None:
        """5 callback calls increment adapter.iteration to 5."""
        adapter = ScipySparseLinalgAdapter(np.zeros(3))
        for i in range(5):
            adapter.callback(np.full(3, float(i)))
        assert adapter.iteration == 5

    def test_checkpoint_roundtrip(self) -> None:
        """tensors['x'] is present, method is 'sparse-linalg'."""
        x0 = np.array([0.5, 1.5, 2.5])
        adapter = ScipySparseLinalgAdapter(x0.copy())
        payload = adapter.checkpoint_state()
        assert "x" in payload.tensors
        assert payload.method == "sparse-linalg"
        np.testing.assert_array_equal(payload.tensors["x"], x0)

    def test_checkpoint_metadata(self) -> None:
        """Metadata contains iteration and residual."""
        adapter = ScipySparseLinalgAdapter(np.ones(3))
        adapter.iteration = 10
        adapter.residual = 1e-6
        payload = adapter.checkpoint_state()
        assert payload.metadata["iteration"] == 10
        assert payload.metadata["residual"] == pytest.approx(1e-6)

    def test_restore_roundtrip(self) -> None:
        """checkpoint → restore → values match saved iterate."""
        x0 = np.array([1.0, 2.0, 3.0])
        adapter = ScipySparseLinalgAdapter(np.zeros(3))
        adapter.callback(x0)
        adapter.residual = 5e-4
        payload = adapter.checkpoint_state()

        new_adapter = ScipySparseLinalgAdapter(np.zeros(3))
        new_adapter.restore_state(payload)
        np.testing.assert_array_equal(new_adapter.x, x0)
        assert new_adapter.iteration == 1
        assert new_adapter.residual == pytest.approx(5e-4)

    def test_restore_missing_tensor_raises(self) -> None:
        """restore_state raises AdapterError if 'x' tensor is absent."""
        from spot_checkpoint.protocol import CheckpointPayload
        payload = CheckpointPayload(
            tensors={},
            metadata={},
            method="sparse-linalg",
        )
        adapter = ScipySparseLinalgAdapter(np.zeros(3))
        with pytest.raises(AdapterError, match="missing required tensor"):
            adapter.restore_state(payload)

    def test_size_estimate(self) -> None:
        """checkpoint_size_estimate returns x.nbytes."""
        x0 = np.ones(200, dtype=np.float64)
        adapter = ScipySparseLinalgAdapter(x0)
        assert adapter.checkpoint_size_estimate == 200 * 8

    def test_initial_x_is_copy(self) -> None:
        """Constructor copies x0; mutating x0 does not affect adapter.x."""
        x0 = np.array([1.0, 2.0, 3.0])
        adapter = ScipySparseLinalgAdapter(x0)
        x0[:] = 0.0
        np.testing.assert_array_equal(adapter.x, np.array([1.0, 2.0, 3.0]))


# ---------------------------------------------------------------------------
# Integration-style: simulate a few callback rounds
# ---------------------------------------------------------------------------

def test_scipy_optimize_with_actual_minimize() -> None:
    """ScipyOptimizeAdapter works with scipy.optimize.minimize in-process."""
    scipy_optimize = pytest.importorskip("scipy.optimize")

    x0 = np.array([5.0, 5.0])
    adapter = ScipyOptimizeAdapter(x0.copy())

    def rosenbrock(x: np.ndarray) -> float:
        return float((1 - x[0]) ** 2 + 100 * (x[1] - x[0] ** 2) ** 2)

    result = scipy_optimize.minimize(
        rosenbrock, adapter.x, method="L-BFGS-B", callback=adapter.callback
    )
    assert result.success
    assert adapter.iteration > 0
    # The stored x should be close to the last iterate (near [1, 1])
    assert adapter.x is not None


def test_scipy_sparse_linalg_with_actual_cg() -> None:
    """ScipySparseLinalgAdapter works with scipy.sparse.linalg.cg in-process."""
    scipy_sparse_linalg = pytest.importorskip("scipy.sparse.linalg")
    scipy_sparse = pytest.importorskip("scipy.sparse")

    n = 10
    A = scipy_sparse.diags([2.0] * n) + scipy_sparse.diags([-1.0] * (n - 1), 1) + \
        scipy_sparse.diags([-1.0] * (n - 1), -1)
    b = np.ones(n)
    x0 = np.zeros(n)
    adapter = ScipySparseLinalgAdapter(x0.copy())

    x, info = scipy_sparse_linalg.cg(A, b, x0=adapter.x, callback=adapter.callback)
    assert info == 0  # converged
    assert adapter.iteration > 0
