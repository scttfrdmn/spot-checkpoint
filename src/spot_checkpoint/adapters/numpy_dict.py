"""
Generic numpy-dict checkpoint adapter.

Functional-style adapter for any iterative computation whose state fits
in a ``dict[str, np.ndarray]`` plus a JSON-serializable metadata dict.
No solver-specific dependencies — works with any custom iterative code.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from spot_checkpoint.protocol import AdapterError, CheckpointPayload


class NumpyDictAdapter:
    """Generic adapter for any numpy-dict-based iterative computation.

    Accepts callables that get/set state, making it easy to wrap any custom
    solver without subclassing.

    Args:
        get_state: Returns the current computation state as a dict of arrays.
        set_state: Restores computation state from a dict of arrays.
        get_metadata: Optional; returns JSON-serializable scalar metadata.
        set_metadata: Optional; restores metadata (iteration count, etc.).
        method: Identifier string stored in the checkpoint (default: ``"generic"``).

    Example:
        >>> adapter = NumpyDictAdapter(
        ...     get_state=lambda: {"x": solver.x, "grad": solver.grad},
        ...     set_state=lambda s: setattr(solver, "x", s["x"]),
        ...     get_metadata=lambda: {"iter": solver.iteration},
        ...     set_metadata=lambda m: setattr(solver, "iteration", m["iter"]),
        ... )
    """

    def __init__(
        self,
        get_state: Callable[[], dict[str, np.ndarray]],
        set_state: Callable[[dict[str, np.ndarray]], None],
        get_metadata: Callable[[], dict[str, Any]] | None = None,
        set_metadata: Callable[[dict[str, Any]], None] | None = None,
        method: str = "generic",
    ) -> None:
        self._get_state = get_state
        self._set_state = set_state
        self._get_metadata = get_metadata
        self._set_metadata = set_metadata
        self._method = method

    def checkpoint_state(self) -> CheckpointPayload:
        """Extract current state into a checkpoint payload.

        Returns:
            Payload containing all tensors and metadata.

        Raises:
            AdapterError: If ``get_state()`` returns an empty dict.
        """
        state = self._get_state()
        if not state:
            raise AdapterError(
                "get_state() returned empty dict — nothing to checkpoint"
            )
        meta: dict[str, Any] = self._get_metadata() if self._get_metadata else {}
        return CheckpointPayload(
            tensors={k: np.array(v) for k, v in state.items()},
            metadata=meta,
            method=self._method,
        )

    def restore_state(self, payload: CheckpointPayload) -> None:
        """Restore computation state from a checkpoint payload.

        Args:
            payload: Previously saved checkpoint.
        """
        self._set_state(payload.tensors)
        if self._set_metadata is not None and payload.metadata:
            self._set_metadata(payload.metadata)

    @property
    def checkpoint_size_estimate(self) -> int:
        """Total bytes of all arrays returned by ``get_state()``."""
        return sum(np.asarray(v).nbytes for v in self._get_state().values())
