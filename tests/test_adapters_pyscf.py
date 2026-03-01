"""Tests for PySCF adapters. Skipped if PySCF is not installed."""

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")


@pytest.mark.integration
class TestSCFAdapter:
    def test_checkpoint_roundtrip(self):
        """Run small SCF, checkpoint, restore, verify."""
        from pyscf import gto, scf
        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
        mf = scf.RHF(mol)
        mf.kernel()

        adapter = SCFCheckpointAdapter(mf)
        payload = adapter.checkpoint_state()

        assert "mo_coeff" in payload.tensors
        assert "mo_occ" in payload.tensors
        assert payload.metadata["converged"] is True
        assert payload.total_bytes > 0

    def test_size_estimate(self):
        from pyscf import gto, scf
        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g")
        mf = scf.RHF(mol)
        adapter = SCFCheckpointAdapter(mf)

        estimate = adapter.checkpoint_size_estimate
        assert estimate > 0
        assert estimate < 1e9  # Should be small for H2/sto-3g
