"""
Minimal example: spot-safe SCF calculation.

Run on a spot instance with automatic checkpoint/restart:
    python basic_scf.py

The spot_safe() call auto-detects the environment (spore.host, Slurm, or bare EC2)
and adds periodic + emergency checkpointing to the SCF solver.
"""

from pyscf import gto, scf
from spot_checkpoint import spot_safe

# Build molecule
mol = gto.M(
    atom="""
        O  0.000  0.000  0.000
        H  0.000  0.757  0.587
        H  0.000 -0.757  0.587
    """,
    basis="aug-cc-pvtz",
)

# Run SCF with spot-safe checkpointing
mf = scf.RHF(mol)
mf.callback = spot_safe(mf, bucket="my-checkpoints")
mf.kernel()

print(f"SCF energy: {mf.e_tot:.10f}")
