"""
PySCF checkpoint adapters — SCF, CCSD, CASSCF.

Each adapter wraps a PySCF solver object and implements the Checkpointable
protocol, extracting/restoring the minimal state needed to resume an
interrupted calculation.

PySCF is an optional dependency — these adapters import lazily and fail
with a clear message if PySCF is not installed.

See docs/ARCHITECTURE.md Layer 2 for design details and size estimates.
"""

from __future__ import annotations

import io
import logging
import tarfile
import time
from pathlib import Path
from typing import Any

import numpy as np

from spot_checkpoint.protocol import AdapterError, CheckpointPayload

logger = logging.getLogger(__name__)

# Attribute names to probe on mc.fcisolver / mc.ci to locate the MPS scratch dir
_MPS_DIR_ATTRS = ("scratch", "runtimedir", "tmpdir", "workdir", "directory", "path")


def _tar_directory(src_dir: Path) -> np.ndarray:
    """Tar-gzip a directory and return its bytes as a uint8 numpy array."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(src_dir, arcname=".")
    return np.frombuffer(buf.getvalue(), dtype=np.uint8)


def _untar_directory(data: np.ndarray, dest_dir: Path) -> None:
    """Extract a tar-gzip uint8 array into dest_dir."""
    buf = io.BytesIO(data.tobytes())
    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        tar.extractall(dest_dir)


def _find_mps_dir(mc: Any) -> Path | None:
    """Probe common attributes on mc.fcisolver and mc.ci to find the MPS directory.

    Returns the first Path that resolves to an existing directory, or None.
    """
    candidates: list[Any] = []
    if hasattr(mc, "fcisolver"):
        candidates.append(mc.fcisolver)
    if hasattr(mc, "ci") and mc.ci is not None:
        candidates.append(mc.ci)

    for obj in candidates:
        for attr in _MPS_DIR_ATTRS:
            val = getattr(obj, attr, None)
            if val is None:
                continue
            p = Path(str(val))
            if p.is_dir():
                return p
    return None


class SCFCheckpointAdapter:
    """
    Wraps a PySCF SCF solver (RHF, UHF, ROHF, RKS, UKS).

    State captured:
      - mo_coeff: MO coefficients (nao × nmo)
      - mo_occ: occupation numbers (nmo)
      - mo_energy: orbital energies (nmo)

    Checkpoint size: nao × nmo × 8 × 3 — typically < 100MB.
    """

    def __init__(self, mf: Any) -> None:
        self.mf = mf

    def checkpoint_state(self) -> CheckpointPayload:
        if self.mf.mo_coeff is None:
            raise AdapterError("SCF solver has no state to checkpoint (not started?)")

        tensors: dict[str, np.ndarray] = {
            "mo_coeff": np.asarray(self.mf.mo_coeff),
            "mo_occ": np.asarray(self.mf.mo_occ),
            "mo_energy": np.asarray(self.mf.mo_energy),
        }

        metadata: dict[str, Any] = {
            "e_tot": float(self.mf.e_tot) if self.mf.e_tot else None,
            "converged": bool(self.mf.converged) if hasattr(self.mf, "converged") else None,
            "method": "scf",
        }

        # Try to get iteration count (internal attribute)
        if hasattr(self.mf, "_iter"):
            metadata["iteration"] = self.mf._iter

        return CheckpointPayload(
            tensors=tensors,
            metadata=metadata,
            method="scf",
            timestamp=time.time(),
        )

    def restore_state(self, payload: CheckpointPayload) -> None:
        self.mf.mo_coeff = payload.tensors["mo_coeff"]
        self.mf.mo_occ = payload.tensors["mo_occ"]
        self.mf.mo_energy = payload.tensors["mo_energy"]
        logger.info(
            "SCF state restored: e_tot=%s, iteration=%s",
            payload.metadata.get("e_tot"),
            payload.metadata.get("iteration"),
        )

    @property
    def checkpoint_size_estimate(self) -> int:
        nao = self.mf.mol.nao
        # mo_coeff (nao × nao) + mo_occ (nao) + mo_energy (nao), all float64
        return nao * nao * 8 + nao * 8 * 2


class CCSDCheckpointAdapter:
    """
    Wraps a PySCF CCSD solver (CCSD, RCCSD, UCCSD).

    State captured:
      - t1: singles amplitudes (nocc × nvir) — small
      - t2: doubles amplitudes (nocc × nocc × nvir × nvir) — BIG

    This is where S3 sharding matters most. t2 can be 18GB+ for
    large molecules.
    """

    def __init__(self, mycc: Any) -> None:
        self.mycc = mycc

    def checkpoint_state(self) -> CheckpointPayload:
        if self.mycc.t1 is None or self.mycc.t2 is None:
            raise AdapterError("CCSD solver has no amplitudes to checkpoint (not started?)")

        tensors: dict[str, np.ndarray] = {
            "t1": np.asarray(self.mycc.t1),
            "t2": np.asarray(self.mycc.t2),
        }

        metadata: dict[str, Any] = {
            "e_corr": float(self.mycc.e_corr) if self.mycc.e_corr else None,
            "converged": bool(self.mycc.converged) if hasattr(self.mycc, "converged") else None,
            "method": "ccsd",
        }

        return CheckpointPayload(
            tensors=tensors,
            metadata=metadata,
            method="ccsd",
            timestamp=time.time(),
        )

    def restore_state(self, payload: CheckpointPayload) -> None:
        self.mycc.t1 = payload.tensors["t1"]
        self.mycc.t2 = payload.tensors["t2"]
        logger.info(
            "CCSD state restored: e_corr=%s, t2 shape=%s (%.1f MB)",
            payload.metadata.get("e_corr"),
            self.mycc.t2.shape,
            self.mycc.t2.nbytes / 1e6,
        )

    @property
    def checkpoint_size_estimate(self) -> int:
        nocc = int(np.sum(self.mycc.mo_occ > 0))
        nvir = len(self.mycc.mo_occ) - nocc
        # t2 dominates: nocc^2 × nvir^2 × 8 bytes, plus ~10% for t1 and overhead
        return int(nocc**2 * nvir**2 * 8 * 1.1)


class CASSCFCheckpointAdapter:
    """
    Wraps a PySCF CASSCF solver.

    State captured:
      - mo_coeff: MO coefficients
      - ci: CI vector (numpy array, standard FCI)
      - ci_mps_tar: tar-gzipped MPS directory (external solver, e.g. Block2/DMRG)

    For external solvers (Block2/DMRG, Dice/SHCI), the MPS state
    lives on disk — this adapter detects the scratch directory and
    captures it as a uint8 numpy array (tar-gzip).
    """

    def __init__(self, mc: Any) -> None:
        self.mc = mc

    def checkpoint_state(self) -> CheckpointPayload:
        if self.mc.mo_coeff is None:
            raise AdapterError("CASSCF solver has no state to checkpoint")

        tensors: dict[str, np.ndarray] = {
            "mo_coeff": np.asarray(self.mc.mo_coeff),
        }

        if self.mc.ci is not None:
            if isinstance(self.mc.ci, np.ndarray):
                tensors["ci"] = self.mc.ci
            else:
                mps_dir = _find_mps_dir(self.mc)
                if mps_dir is not None:
                    tensors["ci_mps_tar"] = _tar_directory(mps_dir)
                    metadata["ci_mps_dir"] = str(mps_dir)
                    metadata["ci_external_solver"] = type(self.mc.ci).__name__
                    logger.info(
                        "CASSCF external solver MPS tarred: dir=%s, bytes=%d",
                        mps_dir,
                        tensors["ci_mps_tar"].nbytes,
                    )
                else:
                    logger.warning(
                        "CI vector is not a numpy array (type=%s) and no MPS "
                        "directory found — external solver CI cannot be checkpointed.",
                        type(self.mc.ci).__name__,
                    )

        metadata: dict[str, Any] = {
            "e_tot": float(self.mc.e_tot) if self.mc.e_tot else None,
            "e_cas": float(self.mc.e_cas) if hasattr(self.mc, "e_cas") and self.mc.e_cas else None,
            "ncas": self.mc.ncas,
            "nelecas": self.mc.nelecas,
            "method": "casscf",
        }

        return CheckpointPayload(
            tensors=tensors,
            metadata=metadata,
            method="casscf",
            timestamp=time.time(),
        )

    def restore_state(self, payload: CheckpointPayload) -> None:
        self.mc.mo_coeff = payload.tensors["mo_coeff"]
        if "ci" in payload.tensors:
            self.mc.ci = payload.tensors["ci"]
        elif "ci_mps_tar" in payload.tensors:
            mps_dir = Path(payload.metadata["ci_mps_dir"])
            mps_dir.mkdir(parents=True, exist_ok=True)
            _untar_directory(payload.tensors["ci_mps_tar"], mps_dir)
            logger.info("CASSCF external solver MPS restored to %s", mps_dir)
        logger.info(
            "CASSCF state restored: e_tot=%s, ncas=%s, nelecas=%s",
            payload.metadata.get("e_tot"),
            payload.metadata.get("ncas"),
            payload.metadata.get("nelecas"),
        )

    @property
    def checkpoint_size_estimate(self) -> int:
        nao = self.mc.mol.nao
        mo_bytes = nao * nao * 8

        # CI vector size depends on active space
        ncas = self.mc.ncas
        nelecas = self.mc.nelecas
        if isinstance(nelecas, (list, tuple)):
            nalpha, nbeta = nelecas
        else:
            nalpha = nbeta = nelecas // 2

        if self.mc.ci is None or isinstance(self.mc.ci, np.ndarray):
            # Rough estimate: C(ncas, nalpha) × C(ncas, nbeta) × 8 bytes
            from math import comb
            ci_elements = comb(ncas, nalpha) * comb(ncas, nbeta)
            ci_bytes = ci_elements * 8
        else:
            mps_dir = _find_mps_dir(self.mc)
            ci_bytes = (
                sum(f.stat().st_size for f in mps_dir.rglob("*") if f.is_file())
                if mps_dir is not None
                else 0
            )

        return mo_bytes + ci_bytes
