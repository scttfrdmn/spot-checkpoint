"""Tests for PySCF adapters. Skipped if PySCF is not installed."""

import types

import numpy as np
import pytest

pyscf = pytest.importorskip("pyscf")


@pytest.mark.integration
class TestSCFAdapter:
    def test_checkpoint_roundtrip(self):
        """Run small SCF, checkpoint, verify chkfile blob is stored."""
        from pyscf import gto, scf

        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        adapter = SCFCheckpointAdapter(mf)
        payload = adapter.checkpoint_state()

        assert "chkfile" in payload.tensors
        assert payload.tensors["chkfile"].dtype == np.uint8
        assert payload.metadata["converged"] is True
        assert payload.total_bytes > 0

    def test_size_estimate(self):
        from pyscf import gto, scf

        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        adapter = SCFCheckpointAdapter(mf)

        estimate = adapter.checkpoint_size_estimate
        assert estimate > 0
        assert estimate < 1e9  # Should be small for H2/sto-3g

    def test_restore_roundtrip(self):
        """Checkpoint after convergence, restore into fresh solver, verify energy matches."""
        from pyscf import gto, scf

        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        adapter = SCFCheckpointAdapter(mf)
        payload = adapter.checkpoint_state()

        mf2 = scf.RHF(mol)
        adapter2 = SCFCheckpointAdapter(mf2)
        adapter2.restore_state(payload)

        # restore_state sets init_guess='chkfile'; kernel() reads the saved MOs
        e_tot2 = mf2.kernel()
        assert e_tot2 == pytest.approx(mf.e_tot, rel=1e-6)
        np.testing.assert_array_almost_equal(mf2.mo_coeff, mf.mo_coeff)

    def test_adapter_error_before_kernel(self):
        """checkpoint_state raises AdapterError if SCF has not been run."""
        from pyscf import gto, scf

        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter
        from spot_checkpoint.protocol import AdapterError

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        # mo_coeff is None before kernel()
        assert mf.mo_coeff is None

        adapter = SCFCheckpointAdapter(mf)
        with pytest.raises(AdapterError):
            adapter.checkpoint_state()

    def test_metadata_fields(self):
        """Verify payload metadata contains expected fields with correct values."""
        from pyscf import gto, scf

        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()

        adapter = SCFCheckpointAdapter(mf)
        payload = adapter.checkpoint_state()

        assert payload.method == "scf"
        assert payload.metadata["converged"] is True
        assert payload.metadata["e_tot"] == pytest.approx(mf.e_tot, rel=1e-10)
        assert "chkfile" in payload.tensors


@pytest.mark.integration
class TestCCSDAdapter:
    @pytest.fixture(autouse=True)
    def setup_ccsd(self):
        from pyscf import cc, gto, scf

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        mycc = cc.CCSD(mf)
        mycc.kernel()

        self.mol = mol
        self.mf = mf
        self.mycc = mycc

    def test_checkpoint_roundtrip(self):
        """Checkpoint CCSD; verify tensors and method."""
        from spot_checkpoint.adapters.pyscf import CCSDCheckpointAdapter

        adapter = CCSDCheckpointAdapter(self.mycc)
        payload = adapter.checkpoint_state()

        assert "t1" in payload.tensors
        assert "t2" in payload.tensors
        assert payload.method == "ccsd"
        assert payload.total_bytes > 0

    def test_restore_roundtrip(self):
        """Restore CCSD amplitudes into fresh solver; verify they match."""
        from pyscf import cc

        from spot_checkpoint.adapters.pyscf import CCSDCheckpointAdapter

        adapter = CCSDCheckpointAdapter(self.mycc)
        payload = adapter.checkpoint_state()

        mycc2 = cc.CCSD(self.mf)
        adapter2 = CCSDCheckpointAdapter(mycc2)
        adapter2.restore_state(payload)

        np.testing.assert_array_almost_equal(mycc2.t1, self.mycc.t1)
        np.testing.assert_array_almost_equal(mycc2.t2, self.mycc.t2)

    def test_adapter_error_before_kernel(self):
        """checkpoint_state raises AdapterError if CCSD has not been run."""
        from pyscf import cc

        from spot_checkpoint.adapters.pyscf import CCSDCheckpointAdapter
        from spot_checkpoint.protocol import AdapterError

        mycc2 = cc.CCSD(self.mf)
        assert mycc2.t1 is None

        adapter = CCSDCheckpointAdapter(mycc2)
        with pytest.raises(AdapterError):
            adapter.checkpoint_state()

    def test_metadata_fields(self):
        """Verify CCSD payload metadata contains expected fields."""
        from spot_checkpoint.adapters.pyscf import CCSDCheckpointAdapter

        adapter = CCSDCheckpointAdapter(self.mycc)
        payload = adapter.checkpoint_state()

        assert payload.metadata["method"] == "ccsd"
        assert payload.metadata["converged"] is True
        assert payload.metadata["e_corr"] == pytest.approx(self.mycc.e_corr, rel=1e-10)

    def test_size_estimate(self):
        """Size estimate should be positive and reasonable for H2/sto-3g."""
        from pyscf import cc

        from spot_checkpoint.adapters.pyscf import CCSDCheckpointAdapter

        mycc2 = cc.CCSD(self.mf)
        # mo_occ is available from underlying SCF without running kernel
        adapter = CCSDCheckpointAdapter(mycc2)
        estimate = adapter.checkpoint_size_estimate

        assert estimate > 0
        assert estimate < 1e9


@pytest.mark.integration
class TestCASSCFAdapter:
    @pytest.fixture(autouse=True)
    def setup_casscf(self):
        from pyscf import gto, mcscf, scf

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        mc = mcscf.CASSCF(mf, 2, 2)
        mc.kernel()

        self.mol = mol
        self.mf = mf
        self.mc = mc

    def test_checkpoint_roundtrip(self):
        """Checkpoint CASSCF; verify mo_coeff, ci, and method."""
        from spot_checkpoint.adapters.pyscf import CASSCFCheckpointAdapter

        adapter = CASSCFCheckpointAdapter(self.mc)
        payload = adapter.checkpoint_state()

        assert "mo_coeff" in payload.tensors
        assert "ci" in payload.tensors  # H2 CAS(2,2) CI is numpy array
        assert payload.method == "casscf"

    def test_restore_roundtrip(self):
        """Restore CASSCF state into fresh solver; verify mo_coeff and ci."""
        from pyscf import mcscf

        from spot_checkpoint.adapters.pyscf import CASSCFCheckpointAdapter

        adapter = CASSCFCheckpointAdapter(self.mc)
        payload = adapter.checkpoint_state()

        mc2 = mcscf.CASSCF(self.mf, 2, 2)
        adapter2 = CASSCFCheckpointAdapter(mc2)
        adapter2.restore_state(payload)

        np.testing.assert_array_almost_equal(mc2.mo_coeff, self.mc.mo_coeff)
        assert mc2.ci is not None

    def test_adapter_error_before_kernel(self):
        """checkpoint_state raises AdapterError if mo_coeff is None."""
        from pyscf import mcscf

        from spot_checkpoint.adapters.pyscf import CASSCFCheckpointAdapter
        from spot_checkpoint.protocol import AdapterError

        mc2 = mcscf.CASSCF(self.mf, 2, 2)
        mc2.mo_coeff = None

        adapter = CASSCFCheckpointAdapter(mc2)
        with pytest.raises(AdapterError):
            adapter.checkpoint_state()

    def test_metadata_fields(self):
        """Verify CASSCF payload metadata contains expected fields."""
        from spot_checkpoint.adapters.pyscf import CASSCFCheckpointAdapter

        adapter = CASSCFCheckpointAdapter(self.mc)
        payload = adapter.checkpoint_state()

        assert payload.metadata["method"] == "casscf"
        assert payload.metadata["ncas"] == 2
        # PySCF stores nelecas as (nalpha, nbeta) tuple
        nelecas = payload.metadata["nelecas"]
        assert (nelecas == 2) or (isinstance(nelecas, tuple) and sum(nelecas) == 2)
        assert payload.metadata["e_tot"] == pytest.approx(self.mc.e_tot, rel=1e-10)

    def test_size_estimate(self):
        """Size estimate should be positive and reasonable for H2 CAS(2,2)."""
        from pyscf import mcscf

        from spot_checkpoint.adapters.pyscf import CASSCFCheckpointAdapter

        mc2 = mcscf.CASSCF(self.mf, 2, 2)
        adapter = CASSCFCheckpointAdapter(mc2)
        estimate = adapter.checkpoint_size_estimate

        assert estimate > 0
        assert estimate < 1e9


@pytest.mark.integration
class TestPySCFWithLocalStore:
    async def test_scf_save_restore_via_store(self, tmp_path):
        """Full pipeline: SCF → save_checkpoint → load_checkpoint → restore."""
        from pyscf import gto, scf

        from spot_checkpoint.adapters.pyscf import SCFCheckpointAdapter
        from spot_checkpoint.protocol import CheckpointPayload
        from spot_checkpoint.storage import LocalStore

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        e_tot_original = mf.e_tot

        adapter = SCFCheckpointAdapter(mf)
        payload = adapter.checkpoint_state()

        store = LocalStore(base_dir=tmp_path, job_id="scf-test")
        await store.save_checkpoint("ckpt-scf", payload.tensors, payload.metadata)

        tensors, metadata = await store.load_checkpoint("ckpt-scf")

        mf2 = scf.RHF(mol)
        adapter2 = SCFCheckpointAdapter(mf2)
        adapter2.restore_state(CheckpointPayload(tensors=tensors, metadata=metadata, method="scf"))

        # restore_state sets init_guess='chkfile'; kernel() reads the saved MOs
        e_tot2 = mf2.kernel()
        assert e_tot2 == pytest.approx(e_tot_original, rel=1e-6)
        np.testing.assert_array_almost_equal(mf2.mo_coeff, mf.mo_coeff)
        assert metadata["e_tot"] == pytest.approx(e_tot_original, rel=1e-10)

    async def test_ccsd_save_restore_via_store(self, tmp_path):
        """Full pipeline: CCSD → save_checkpoint → load_checkpoint → restore."""
        from pyscf import cc, gto, scf

        from spot_checkpoint.adapters.pyscf import CCSDCheckpointAdapter
        from spot_checkpoint.protocol import CheckpointPayload
        from spot_checkpoint.storage import LocalStore

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        mycc = cc.CCSD(mf)
        mycc.kernel()

        adapter = CCSDCheckpointAdapter(mycc)
        payload = adapter.checkpoint_state()

        store = LocalStore(base_dir=tmp_path, job_id="ccsd-test")
        await store.save_checkpoint("ckpt-ccsd", payload.tensors, payload.metadata)

        tensors, metadata = await store.load_checkpoint("ckpt-ccsd")

        mycc2 = cc.CCSD(mf)
        adapter2 = CCSDCheckpointAdapter(mycc2)
        adapter2.restore_state(CheckpointPayload(tensors=tensors, metadata=metadata, method="ccsd"))

        np.testing.assert_array_almost_equal(mycc2.t1, mycc.t1)
        np.testing.assert_array_almost_equal(mycc2.t2, mycc.t2)
        assert metadata["e_corr"] == pytest.approx(mycc.e_corr, rel=1e-10)


@pytest.mark.integration
class TestCASSCFExternalSolver:
    """Tests for CASSCFCheckpointAdapter with a mocked external FCI solver."""

    @pytest.fixture(autouse=True)
    def setup_external_solver(self, tmp_path):
        from pyscf import gto, mcscf, scf

        from spot_checkpoint.adapters.pyscf import CASSCFCheckpointAdapter

        mol = gto.M(atom="H 0 0 0; H 0 0 0.74", basis="sto-3g", verbose=0)
        mf = scf.RHF(mol)
        mf.kernel()
        mc = mcscf.CASSCF(mf, 2, 2)
        mc.kernel()

        # Simulate an external solver: replace ci with a non-array object
        # and wire up a scratch dir on fcisolver
        mps_dir = tmp_path / "mps_scratch"
        mps_dir.mkdir()
        (mps_dir / "wave.mps").write_bytes(b"\x00\x01\x02\x03")
        (mps_dir / "config.json").write_text('{"sweep": 10}')

        mc.ci = types.SimpleNamespace()  # non-numpy CI placeholder
        mc.fcisolver.scratch = str(mps_dir)

        self.mol = mol
        self.mf = mf
        self.mc = mc
        self.mps_dir = mps_dir
        self.adapter = CASSCFCheckpointAdapter(mc)

    def test_checkpoint_tars_mps_directory(self):
        """Payload contains ci_mps_tar (uint8), ci_mps_dir, ci_external_solver; no 'ci' key."""
        payload = self.adapter.checkpoint_state()

        assert "ci_mps_tar" in payload.tensors
        assert "ci" not in payload.tensors
        assert payload.tensors["ci_mps_tar"].dtype == np.uint8
        assert "ci_mps_dir" in payload.metadata
        assert "ci_external_solver" in payload.metadata

    def test_restore_untars_to_path(self, tmp_path):
        """restore_state with ci_mps_tar payload extracts files to the recorded path."""
        from pyscf import mcscf

        from spot_checkpoint.adapters.pyscf import CASSCFCheckpointAdapter
        from spot_checkpoint.protocol import CheckpointPayload

        # Build a minimal tar payload pointing at a fresh destination directory
        dest_dir = tmp_path / "restored_mps"
        # Create the tar data from the original mps_dir
        import io as _io
        import tarfile as _tf
        buf = _io.BytesIO()
        with _tf.open(fileobj=buf, mode="w:gz") as tar:
            tar.add(self.mps_dir, arcname=".")
        arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)

        payload = CheckpointPayload(
            tensors={"mo_coeff": self.mc.mo_coeff, "ci_mps_tar": arr},
            metadata={
                "ci_mps_dir": str(dest_dir),
                "ci_external_solver": "SimpleNamespace",
                "method": "casscf",
                "ncas": 2,
                "nelecas": 2,
                "e_tot": None,
                "e_cas": None,
            },
            method="casscf",
        )

        mc2 = mcscf.CASSCF(self.mf, 2, 2)
        adapter2 = CASSCFCheckpointAdapter(mc2)
        adapter2.restore_state(payload)

        assert dest_dir.exists()
        restored_files = {f.name for f in dest_dir.iterdir()}
        assert "wave.mps" in restored_files
        assert "config.json" in restored_files

    def test_graceful_skip_when_no_dir_found(self, caplog):
        """When ci is non-array and no MPS dir can be found, a warning is logged."""
        import logging

        from pyscf import mcscf

        from spot_checkpoint.adapters.pyscf import CASSCFCheckpointAdapter

        mc2 = mcscf.CASSCF(self.mf, 2, 2)
        mc2.kernel()
        # Replace ci with a bare object with no recognised attrs
        mc2.ci = object()
        # Ensure fcisolver has no scratch-like attrs
        for attr in ("scratch", "runtimedir", "tmpdir", "workdir", "directory", "path"):
            if hasattr(mc2.fcisolver, attr):
                delattr(mc2.fcisolver, attr)

        adapter2 = CASSCFCheckpointAdapter(mc2)
        with caplog.at_level(logging.WARNING, logger="spot_checkpoint.adapters.pyscf"):
            payload = adapter2.checkpoint_state()

        assert "ci_mps_tar" not in payload.tensors
        assert "ci" not in payload.tensors
        assert any("no MPS directory found" in r.message for r in caplog.records)

    def test_metadata_contains_solver_type(self):
        """ci_external_solver in metadata equals type(mc.ci).__name__."""
        payload = self.adapter.checkpoint_state()
        assert payload.metadata["ci_external_solver"] == type(self.mc.ci).__name__

    def test_size_estimate_uses_dir_size(self):
        """Size estimate is at least as large as the raw files in the MPS directory."""
        raw_size = sum(
            f.stat().st_size for f in self.mps_dir.rglob("*") if f.is_file()
        )
        estimate = self.adapter.checkpoint_size_estimate
        assert estimate >= raw_size
