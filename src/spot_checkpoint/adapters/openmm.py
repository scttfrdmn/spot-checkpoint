# ruff: noqa: RUF002, RUF003  (× is intentional Unicode math in docstrings/comments)
"""
OpenMM molecular dynamics checkpoint adapter.

Saves atomic positions, velocities, periodic box vectors (if present), and
key simulation metadata (step, time, potential energy).

OpenMM is an optional dependency — this module imports it lazily and fails
with a clear message if not installed.

Units:
    All values are stored in OpenMM native units (nm, nm/ps, kJ/mol) after
    stripping the unit object via ``.value_in_unit()``.  They are restored by
    multiplying by the corresponding unit object.

Restore notes:
    After ``restore_state()``, call ``simulation.step(N)`` to continue the
    trajectory.  The ``simulation.currentStep`` counter is also restored so
    that output reporters use the correct step number.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

from spot_checkpoint.protocol import AdapterError, CheckpointPayload

if TYPE_CHECKING:
    import openmm
    import openmm.app


class OpenMMAdapter:
    """Checkpoint adapter for an OpenMM ``Simulation`` object.

    Saves positions, velocities, box vectors, and scalar metadata.
    On restore, all state is pushed back into the simulation context so
    the caller can continue with ``simulation.step(N)``.

    Args:
        simulation: The ``openmm.app.Simulation`` instance to checkpoint.

    Raises:
        AdapterError: At checkpoint or restore time if the simulation context
            is unavailable or state retrieval fails.

    Example:
        >>> adapter = OpenMMAdapter(simulation)
        >>> # ... lifecycle manager calls adapter.checkpoint_state() as needed
        >>> # On restart:
        >>> adapter.restore_state(payload)
        >>> simulation.step(remaining_steps)
    """

    def __init__(self, simulation: openmm.app.Simulation) -> None:
        self.simulation = simulation

    def checkpoint_state(self) -> CheckpointPayload:
        """Capture positions, velocities, box vectors, and metadata.

        Returns:
            Payload with position/velocity/box tensors and scalar metadata.

        Raises:
            AdapterError: If state cannot be retrieved from the context.
        """
        try:
            from openmm import unit
        except ImportError as e:
            raise AdapterError(
                "openmm is required for OpenMMAdapter. "
                "Install it with: conda install -c conda-forge openmm"
            ) from e

        try:
            ctx = self.simulation.context
            state = ctx.getState(
                getPositions=True,
                getVelocities=True,
                getEnergy=True,
                enforcePeriodicBox=True,
            )
        except Exception as exc:
            raise AdapterError(
                f"Failed to retrieve state from OpenMM context: {exc}"
            ) from exc

        positions = np.asarray(
            state.getPositions(asNumpy=True).value_in_unit(unit.nanometer),
            dtype=np.float64,
        )
        velocities = np.asarray(
            state.getVelocities(asNumpy=True).value_in_unit(
                unit.nanometer / unit.picosecond
            ),
            dtype=np.float64,
        )

        tensors: dict[str, np.ndarray] = {
            "positions": positions,
            "velocities": velocities,
        }

        # Box vectors — only for periodic systems
        try:
            bv = state.getPeriodicBoxVectors(asNumpy=True).value_in_unit(
                unit.nanometer
            )
            tensors["box_vectors"] = np.asarray(bv, dtype=np.float64)
        except Exception:
            pass  # non-periodic system; box_vectors omitted

        metadata: dict[str, Any] = {
            "step": self.simulation.currentStep,
            "time_ps": float(
                state.getTime().value_in_unit(unit.picosecond)
            ),
            "n_atoms": self.simulation.topology.getNumAtoms(),
            "potential_energy_kj_mol": float(
                state.getPotentialEnergy().value_in_unit(
                    unit.kilojoule_per_mole
                )
            ),
        }

        return CheckpointPayload(
            tensors=tensors,
            metadata=metadata,
            method="openmm-md",
        )

    def restore_state(self, payload: CheckpointPayload) -> None:
        """Push saved positions, velocities, and box vectors into the context.

        Args:
            payload: Previously saved checkpoint payload.

        Raises:
            AdapterError: If required tensors are missing or restore fails.
        """
        try:
            from openmm import unit
        except ImportError as e:
            raise AdapterError(
                "openmm is required for OpenMMAdapter. "
                "Install it with: conda install -c conda-forge openmm"
            ) from e

        if "positions" not in payload.tensors:
            raise AdapterError(
                "Checkpoint payload missing required tensor 'positions'"
            )
        if "velocities" not in payload.tensors:
            raise AdapterError(
                "Checkpoint payload missing required tensor 'velocities'"
            )

        try:
            ctx = self.simulation.context
            ctx.setPositions(
                payload.tensors["positions"] * unit.nanometer
            )
            ctx.setVelocities(
                payload.tensors["velocities"]
                * unit.nanometer
                / unit.picosecond
            )
            if "box_vectors" in payload.tensors:
                bv = payload.tensors["box_vectors"]
                ctx.setPeriodicBoxVectors(
                    *[bv[i] * unit.nanometer for i in range(3)]
                )
        except Exception as exc:
            raise AdapterError(
                f"Failed to restore state into OpenMM context: {exc}"
            ) from exc

        self.simulation.currentStep = int(payload.metadata.get("step", 0))

    @property
    def checkpoint_size_estimate(self) -> int:
        """Estimated bytes: positions + velocities arrays (N×3 float64 each)."""
        n_atoms = self.simulation.topology.getNumAtoms()
        # positions + velocities = 2 × N × 3 × 8 bytes; +72 for 3×3 box
        return int(n_atoms * 3 * 8 * 2 + 72)


__all__: list[str] = ["OpenMMAdapter"]
