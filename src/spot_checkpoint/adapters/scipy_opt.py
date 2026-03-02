"""
SciPy optimization and sparse linear algebra checkpoint adapters.

Both adapters follow a callback-pattern: pass ``adapter.callback`` as the
``callback=`` argument to the relevant SciPy function, then checkpoint/restore
``adapter`` via the normal ``SpotLifecycleManager`` API.

On restore, pass ``adapter.x`` as the initial guess to resume from the last
saved iterate — the standard warm-restart approach for these solvers.
Internal optimizer state (e.g., Hessian approximations) is intentionally
not preserved; it is reconstructed by the solver on the next few iterations.
"""

from __future__ import annotations

import numpy as np

from spot_checkpoint.protocol import AdapterError, CheckpointPayload


class ScipyOptimizeAdapter:
    """Adapter for ``scipy.optimize.minimize`` — checkpoint/restore via callback.

    Pass ``adapter.callback`` as ``callback=`` to ``scipy.optimize.minimize``.
    After each iteration the current iterate ``xk`` is saved.  On interrupt,
    checkpoint; on restart, pass ``adapter.x`` as ``x0`` to ``minimize()``.

    Internal optimizer state (Hessian approximation for L-BFGS-B, etc.) is
    not preserved — this is standard practice; the solver rebuilds it in a
    few iterations.

    Args:
        x0: Initial parameter vector.  Copied internally.
        method: Optimization method name (e.g. ``"L-BFGS-B"``, ``"CG"``).
            Stored in metadata only; does not affect adapter behaviour.

    Example:
        >>> from scipy.optimize import minimize
        >>> adapter = ScipyOptimizeAdapter(x0=np.zeros(10))
        >>> result = minimize(f, adapter.x, method="L-BFGS-B",
        ...                   callback=adapter.callback)
    """

    def __init__(self, x0: np.ndarray, method: str = "L-BFGS-B") -> None:
        self.x: np.ndarray = x0.copy()
        self.iteration: int = 0
        self.fun: float | None = None
        self._method = method

    def callback(self, xk: np.ndarray) -> None:
        """Callback passed to ``scipy.optimize.minimize``.

        Updates the stored iterate and increments the iteration counter.

        Args:
            xk: Current parameter vector from the optimizer.
        """
        self.x = xk.copy()
        self.iteration += 1

    def checkpoint_state(self) -> CheckpointPayload:
        """Capture current iterate.

        Returns:
            Payload with ``"x"`` tensor and iteration metadata.
        """
        return CheckpointPayload(
            tensors={"x": self.x.copy()},
            metadata={
                "iteration": self.iteration,
                "fun": self.fun,
                "optimize_method": self._method,
            },
            method="scipy-optimize",
        )

    def restore_state(self, payload: CheckpointPayload) -> None:
        """Restore iterate from a checkpoint payload.

        Args:
            payload: Previously saved checkpoint.

        Raises:
            AdapterError: If ``"x"`` tensor is missing from payload.
        """
        if "x" not in payload.tensors:
            raise AdapterError("Checkpoint payload missing required tensor 'x'")
        self.x = payload.tensors["x"].copy()
        self.iteration = int(payload.metadata.get("iteration", 0))
        fun_val = payload.metadata.get("fun")
        self.fun = float(fun_val) if fun_val is not None else None

    @property
    def checkpoint_size_estimate(self) -> int:
        """Bytes occupied by the current iterate vector."""
        return int(self.x.nbytes)


class ScipySparseLinalgAdapter:
    """Adapter for ``scipy.sparse.linalg`` iterative solvers (cg, gmres, etc.).

    Pass ``adapter.callback`` as ``callback=`` to the solver.  On restart,
    pass ``adapter.x`` as ``x0`` to continue from the last saved iterate.

    Supported solvers: ``cg``, ``gmres``, ``minres``, ``bicgstab``, etc. —
    any that accept a ``callback`` receiving the current solution vector.

    Args:
        x0: Initial solution vector.  Copied internally.

    Example:
        >>> from scipy.sparse.linalg import cg
        >>> adapter = ScipySparseLinalgAdapter(x0=np.zeros(n))
        >>> x, info = cg(A, b, x0=adapter.x, callback=adapter.callback)
    """

    def __init__(self, x0: np.ndarray) -> None:
        self.x: np.ndarray = x0.copy()
        self.iteration: int = 0
        self.residual: float | None = None

    def callback(self, xk: np.ndarray) -> None:
        """Callback passed to the iterative solver.

        Args:
            xk: Current solution vector from the solver.
        """
        self.x = xk.copy()
        self.iteration += 1

    def checkpoint_state(self) -> CheckpointPayload:
        """Capture current solution vector.

        Returns:
            Payload with ``"x"`` tensor and iteration metadata.
        """
        return CheckpointPayload(
            tensors={"x": self.x.copy()},
            metadata={
                "iteration": self.iteration,
                "residual": self.residual,
            },
            method="sparse-linalg",
        )

    def restore_state(self, payload: CheckpointPayload) -> None:
        """Restore solution vector from a checkpoint payload.

        Args:
            payload: Previously saved checkpoint.

        Raises:
            AdapterError: If ``"x"`` tensor is missing from payload.
        """
        if "x" not in payload.tensors:
            raise AdapterError("Checkpoint payload missing required tensor 'x'")
        self.x = payload.tensors["x"].copy()
        self.iteration = int(payload.metadata.get("iteration", 0))
        res_val = payload.metadata.get("residual")
        self.residual = float(res_val) if res_val is not None else None

    @property
    def checkpoint_size_estimate(self) -> int:
        """Bytes occupied by the current solution vector."""
        return int(self.x.nbytes)


__all__: list[str] = ["ScipyOptimizeAdapter", "ScipySparseLinalgAdapter"]
