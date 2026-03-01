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

import logging
import time
from typing import Any

import numpy as np

from spot_checkpoint.protocol import AdapterError, CheckpointPayload

logger = logging.getLogger(__name__)


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
      - ci: CI vector (can be enormous for large active spaces)

    For external solvers (Block2/DMRG, Dice/SHCI), the MPS state
    lives on disk — this adapter captures it as a binary blob.

    TODO: External solver support is deferred to v2.
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
                logger.warning(
                    "CI vector is not a numpy array (type=%s) — external solver? "
                    "Skipping CI checkpoint. External solver support is TODO.",
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

        # Rough estimate: C(ncas, nalpha) × C(ncas, nbeta) × 8 bytes
        from math import comb
        ci_elements = comb(ncas, nalpha) * comb(ncas, nbeta)
        ci_bytes = ci_elements * 8

        return mo_bytes + ci_bytes
