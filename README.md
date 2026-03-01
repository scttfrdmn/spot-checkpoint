# spot-checkpoint

Fault-tolerant checkpoint/restart for iterative scientific computations on preemptible cloud instances.

**Take your existing PySCF script, add two lines, run on spot at 70% discount with automatic fault tolerance.**

```python
from pyscf import gto, scf
from spot_checkpoint import spot_safe, spot_restore

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='aug-cc-pvtz')
mf = scf.RHF(mol)

spot_restore(mf, bucket="my-checkpoints")          # no-op on first run
mf.callback = spot_safe(mf, bucket="my-checkpoints")
mf.kernel()
```

## What It Does

- **Periodic checkpointing** every 5 minutes (configurable) using PySCF's native restart paths
- **Emergency checkpoint** on spot interruption — writes state within the 2-minute warning window
- **One-line restore** (`spot_restore`) on re-launch — picks up from the latest checkpoint
- **S3 sharding** — parallel writes across prefixed keys for fast saves of large tensors (18+ GB CCSD t2)
- **Optional compression** — zstd reduces storage and transfer cost by 30–50% for float64 arrays

## Installation

```bash
pip install spot-checkpoint

# With PySCF adapters
pip install spot-checkpoint[pyscf]

# With CLI tools
pip install spot-checkpoint[cli]

# With zstd compression
pip install spot-checkpoint[compress]

# Everything
pip install spot-checkpoint[pyscf,cli,compress]
```

## Quick Start

### SCF

```python
from pyscf import gto, scf
from spot_checkpoint import spot_safe, spot_restore

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='aug-cc-pvtz')
mf = scf.RHF(mol)

restored = spot_restore(mf, bucket="my-bucket")
if restored:
    print("Resuming from checkpoint")

mf.callback = spot_safe(mf, bucket="my-bucket")
mf.kernel()
```

### CCSD

```python
from pyscf import gto, scf, cc
from spot_checkpoint import spot_safe, spot_restore

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='aug-cc-pvtz')
mf = scf.RHF(mol).run()

mycc = cc.CCSD(mf)
spot_restore(mycc, bucket="my-bucket")
mycc.callback = spot_safe(mycc, bucket="my-bucket")
mycc.kernel(mycc.t1, mycc.t2)   # t1/t2 are None on fresh start — PySCF handles it
```

### Async / Jupyter

```python
from spot_checkpoint import spot_safe_async, spot_restore_async

restored = await spot_restore_async(mf, bucket="my-bucket")
mf.callback = await spot_safe_async(mf, bucket="my-bucket")
mf.kernel()
```

## Environment Variables

All parameters can be set via environment variables — no hardcoded bucket names in scripts:

| Variable | Default | Description |
|---|---|---|
| `SPOT_CHECKPOINT_BUCKET` | — | S3 bucket (required if `bucket=` not passed) |
| `SPOT_CHECKPOINT_INTERVAL` | `300` | Periodic checkpoint interval (seconds) |
| `SPOT_CHECKPOINT_SHARD_SIZE` | `67108864` | Shard size in bytes (64 MB) |
| `SPOT_CHECKPOINT_MAX_CONCURRENCY` | `32` | Parallel S3 streams |
| `SLURM_JOB_ID` | — | Auto-detected; used as job_id namespace |
| `SPAWN_INSTANCE_ID` | — | Auto-detected; used as job_id on spore.host |

With env vars set you can reduce the script to:

```bash
export SPOT_CHECKPOINT_BUCKET=my-bucket
export SPOT_CHECKPOINT_INTERVAL=120
```

```python
spot_restore(mf)
mf.callback = spot_safe(mf)
mf.kernel()
```

## Three Deployment Models

| Environment | Backend | How It Works |
|---|---|---|
| **spore.host** | `SporeLifecycleBackend` | Watches spored's signal file, writes completion signal |
| **Slurm** | `SlurmLifecycleBackend` | Handles SIGTERM preemption, requests requeue |
| **Bare EC2** | `DirectEC2Backend` | Polls instance metadata directly with IMDSv2 |

The right backend is auto-detected from environment variables. You don't have to choose.

## Supported Solvers

| Method | Checkpoint Size | Restart Path |
|---|---|---|
| SCF/DFT (RHF, UHF, RKS, UKS) | < 10 MB | PySCF native HDF5 chkfile |
| CCSD/RCCSD/UCCSD | 1–200 GB (t2 amplitudes) | `kernel(t1, t2)` — PySCF documented restart |
| CASSCF | MO + CI vector or MPS tar | `kernel(mo_coeff)` — PySCF documented restart |

## CLI

```bash
# List all checkpoints for a job
spot-checkpoint list /path/to/store my-job-id
spot-checkpoint list s3://my-bucket my-job-id

# Show details for the latest (or a specific) checkpoint
spot-checkpoint info /path/to/store my-job-id
spot-checkpoint info /path/to/store my-job-id ckpt-20260301-001

# Validate checkpoint integrity (re-verify checksums)
spot-checkpoint validate /path/to/store my-job-id
spot-checkpoint validate s3://my-bucket my-job-id ckpt-20260301-001

# Restore tensors to .npy files + metadata.json
spot-checkpoint restore /path/to/store my-job-id --output ./restored/

# Garbage collect old checkpoints
spot-checkpoint gc /path/to/store my-job-id --keep 3

# Benchmark write/read throughput
spot-checkpoint bench /tmp/bench --size-mb 256
spot-checkpoint bench s3://my-bucket --size-mb 1024

# All commands support --json for machine-readable output
spot-checkpoint list --json s3://my-bucket my-job-id
```

## Advanced: S3 Configuration

```python
from spot_checkpoint import spot_safe

# VPC endpoint (default, no data transfer fees)
mf.callback = spot_safe(mf, bucket="my-bucket", endpoint_strategy="vpc")

# Transfer acceleration
mf.callback = spot_safe(mf, bucket="my-bucket", endpoint_strategy="accelerate")

# Custom endpoint (MinIO, LocalStack, etc.)
mf.callback = spot_safe(mf, bucket="my-bucket", endpoint_url="http://localhost:9000")

# With zstd compression
mf.callback = spot_safe(mf, bucket="my-bucket", compress=True)
```

## Architecture

Three layers:

1. **Storage** (`LocalStore`, `S3ShardedStore`) — tensor serialisation with manifest-last atomicity, xxhash checksums, optional zstd compression
2. **Adapters** (`SCFCheckpointAdapter`, `CCSDCheckpointAdapter`, `CASSCFCheckpointAdapter`) — extract/restore minimal solver state using PySCF's documented restart paths
3. **Lifecycle** (`SpotLifecycleManager`) — connects interrupt signals to checkpoint operations; wraps three deployment backends

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the full design specification.

## Examples

- [Basic SCF on Slurm](examples/slurm_submit.sh)
- [Restore and continue SCF](examples/restore_and_continue_scf.py)

## License

Apache 2.0
