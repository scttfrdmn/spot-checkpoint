"""Tests for OpenMMAdapter.

All tests skip if openmm is not installed.

A minimal 3-atom water molecule in vacuum is used as the test system.
"""

from __future__ import annotations

import numpy as np
import pytest

openmm = pytest.importorskip("openmm")
openmm_app = pytest.importorskip("openmm.app")
openmm_unit = pytest.importorskip("openmm.unit")

from openmm import unit  # type: ignore[import-not-found]  # noqa: E402
from openmm.app import (  # type: ignore[import-not-found]  # noqa: E402
    Simulation,
    Topology,
)

from spot_checkpoint.adapters.openmm import OpenMMAdapter  # noqa: E402
from spot_checkpoint.protocol import AdapterError  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_water_simulation() -> Simulation:
    """Build a minimal 3-atom (H2O) vacuum simulation with a simple force field."""
    import openmm  # type: ignore[import-not-found]
    from openmm.app import Element  # type: ignore[import-not-found]

    # Build topology
    topology = Topology()
    chain = topology.addChain()
    residue = topology.addResidue("HOH", chain)
    O = topology.addAtom("O", Element.getBySymbol("O"), residue)
    H1 = topology.addAtom("H1", Element.getBySymbol("H"), residue)
    H2 = topology.addAtom("H2", Element.getBySymbol("H"), residue)
    topology.addBond(O, H1)
    topology.addBond(O, H2)

    # Minimal force field: harmonic bonds + nonbonded
    system = openmm.System()
    system.addParticle(16.0)   # O
    system.addParticle(1.0)    # H1
    system.addParticle(1.0)    # H2

    # Harmonic bond O-H1
    bond_force = openmm.HarmonicBondForce()
    bond_force.addBond(0, 1, 0.096 * unit.nanometer, 5e5 * unit.kilojoule_per_mole / unit.nanometer**2)
    bond_force.addBond(0, 2, 0.096 * unit.nanometer, 5e5 * unit.kilojoule_per_mole / unit.nanometer**2)
    system.addForce(bond_force)

    # Integrator
    integrator = openmm.VerletIntegrator(0.001 * unit.picosecond)

    # Simulation
    platform = openmm.Platform.getPlatformByName("Reference")
    simulation = Simulation(topology, system, integrator, platform)

    # Initial positions (nm)
    positions = [
        openmm.Vec3(0.0, 0.0, 0.0) * unit.nanometer,
        openmm.Vec3(0.096, 0.0, 0.0) * unit.nanometer,
        openmm.Vec3(-0.024, 0.090, 0.0) * unit.nanometer,
    ]
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)

    return simulation


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simulation() -> Simulation:
    return _make_water_simulation()


@pytest.fixture
def adapter(simulation: Simulation) -> OpenMMAdapter:
    return OpenMMAdapter(simulation)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_checkpoint_positions_shape(adapter: OpenMMAdapter) -> None:
    """Positions array has shape (3, 3) — 3 atoms × 3 coordinates."""
    payload = adapter.checkpoint_state()
    assert "positions" in payload.tensors
    assert payload.tensors["positions"].shape == (3, 3)
    assert payload.tensors["positions"].dtype == np.float64


def test_checkpoint_velocities_shape(adapter: OpenMMAdapter) -> None:
    """Velocities array has shape (3, 3)."""
    payload = adapter.checkpoint_state()
    assert "velocities" in payload.tensors
    assert payload.tensors["velocities"].shape == (3, 3)
    assert payload.tensors["velocities"].dtype == np.float64


def test_checkpoint_method(adapter: OpenMMAdapter) -> None:
    """Payload method is 'openmm-md'."""
    payload = adapter.checkpoint_state()
    assert payload.method == "openmm-md"


def test_checkpoint_metadata_fields(adapter: OpenMMAdapter) -> None:
    """Metadata contains step, time_ps, n_atoms, potential_energy_kj_mol."""
    payload = adapter.checkpoint_state()
    meta = payload.metadata
    assert "step" in meta
    assert "time_ps" in meta
    assert "n_atoms" in meta
    assert "potential_energy_kj_mol" in meta
    assert meta["n_atoms"] == 3


def test_step_metadata(simulation: Simulation) -> None:
    """After step(5), metadata['step'] equals 5."""
    simulation.step(5)
    adapter = OpenMMAdapter(simulation)
    payload = adapter.checkpoint_state()
    assert payload.metadata["step"] == 5


def test_restore_roundtrip(simulation: Simulation) -> None:
    """Perturb positions → checkpoint → restore → positions match original."""
    adapter = OpenMMAdapter(simulation)
    payload = adapter.checkpoint_state()
    original_positions = payload.tensors["positions"].copy()

    # Perturb positions
    ctx = simulation.context
    import openmm  # type: ignore[import-not-found]
    perturbed = [
        openmm.Vec3(*(original_positions[i] + 0.5)) * unit.nanometer
        for i in range(3)
    ]
    ctx.setPositions(perturbed)

    # Restore
    adapter.restore_state(payload)

    # Check
    state = ctx.getState(getPositions=True)
    restored = np.asarray(
        state.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    )
    np.testing.assert_allclose(restored, original_positions, atol=1e-10)


def test_restore_step_counter(simulation: Simulation) -> None:
    """restore_state sets simulation.currentStep from metadata."""
    simulation.step(7)
    adapter = OpenMMAdapter(simulation)
    payload = adapter.checkpoint_state()
    assert payload.metadata["step"] == 7

    # Reset step counter
    simulation.currentStep = 0
    adapter.restore_state(payload)
    assert simulation.currentStep == 7


def test_size_estimate(adapter: OpenMMAdapter) -> None:
    """checkpoint_size_estimate is a positive integer."""
    estimate = adapter.checkpoint_size_estimate
    assert isinstance(estimate, int)
    assert estimate > 0


def test_restore_missing_positions_raises(adapter: OpenMMAdapter) -> None:
    """restore_state raises AdapterError if 'positions' tensor is missing."""
    from spot_checkpoint.protocol import CheckpointPayload
    payload = CheckpointPayload(
        tensors={"velocities": np.zeros((3, 3))},
        metadata={},
        method="openmm-md",
    )
    with pytest.raises(AdapterError, match="positions"):
        adapter.restore_state(payload)


def test_restore_missing_velocities_raises(adapter: OpenMMAdapter) -> None:
    """restore_state raises AdapterError if 'velocities' tensor is missing."""
    from spot_checkpoint.protocol import CheckpointPayload
    payload = CheckpointPayload(
        tensors={"positions": np.zeros((3, 3))},
        metadata={},
        method="openmm-md",
    )
    with pytest.raises(AdapterError, match="velocities"):
        adapter.restore_state(payload)
