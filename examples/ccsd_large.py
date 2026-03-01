"""
CCSD calculation with S3-sharded checkpointing.

For large CCSD calculations, the t2 amplitude tensor can be 10-200GB.
The S3ShardedStore writes this across parallel prefixed keys to exploit
S3's horizontal scaling — 500MB/s+ aggregate throughput.
"""

from pyscf import cc, gto, scf
from spot_checkpoint import spot_safe

mol = gto.M(
    atom="""
        O  0.000  0.000  0.000
        H  0.000  0.757  0.587
        H  0.000 -0.757  0.587
    """,
    basis="aug-cc-pvtz",
)

# SCF first
mf = scf.RHF(mol)
mf.callback = spot_safe(mf, bucket="my-checkpoints")
mf.kernel()

# CCSD with spot-safe checkpointing
mycc = cc.CCSD(mf)
mycc.callback = spot_safe(
    mycc,
    bucket="my-checkpoints",
    # Tune for large tensors
    shard_size=128 * 1024 * 1024,  # 128MB shards
    max_concurrency=64,             # More parallel streams
)
mycc.kernel()

print(f"CCSD correlation energy: {mycc.e_corr:.10f}")
print(f"Total energy: {mycc.e_tot:.10f}")
