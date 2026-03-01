"""
Restore and continue an interrupted SCF calculation.

This example shows the two-call pattern for spot-safe PySCF jobs:
  1. spot_restore() — load any previously written checkpoint into the solver
     before the kernel starts. No-op if no checkpoint exists (fresh start).
  2. spot_safe()    — register the checkpoint callback so the solver writes
     checkpoints periodically and on spot interruption.

Usage on a spot instance:
    python restore_and_continue_scf.py

The same script works whether this is the first run or a restart after
preemption — spot_restore() handles both cases transparently.
"""

from pyscf import gto, scf

from spot_checkpoint import spot_restore, spot_safe

# ---------------------------------------------------------------------------
# Molecule and solver setup
# ---------------------------------------------------------------------------

mol = gto.M(
    atom="O 0 0 0; H 0 .757 .587; H 0 -.757 .587",
    basis="aug-cc-pvtz",
    verbose=4,
)
mf = scf.RHF(mol)

BUCKET = "my-checkpoints"

# ---------------------------------------------------------------------------
# Step 1: Restore from the latest checkpoint (no-op if none exists)
# ---------------------------------------------------------------------------

restored = spot_restore(mf, bucket=BUCKET)
if restored:
    print("Restored from previous checkpoint — continuing SCF from saved MO coefficients.")
else:
    print("No checkpoint found — starting SCF from scratch.")

# ---------------------------------------------------------------------------
# Step 2: Attach the spot-safe callback and run the kernel
#
# The callback writes periodic checkpoints every 5 minutes and an emergency
# checkpoint if a spot interruption signal is detected.
# ---------------------------------------------------------------------------

mf.callback = spot_safe(mf, bucket=BUCKET, periodic_interval=300)
mf.kernel()

print(f"Converged: {mf.converged}  E(RHF) = {mf.e_tot:.8f} Eh")
