# spot-checkpoint

Fault-tolerant checkpoint/restart for iterative scientific computations on preemptible cloud instances.

**Take your existing PySCF script, add one line, run on spot at 70% discount with automatic fault tolerance.**

```python
from pyscf import gto, scf
from spot_checkpoint import spot_safe

mol = gto.M(atom='O 0 0 0; H 0 0.757 0.587; H 0 -0.757 0.587', basis='aug-cc-pvtz')
mf = scf.RHF(mol)
mf.callback = spot_safe(mf, bucket="my-checkpoints")
mf.kernel()
```

## What It Does

- **Periodic checkpointing** of iterative solver state (MO coefficients, CCSD amplitudes, CI vectors)
- **Emergency checkpoint** on spot interruption — writes state to S3 within the 2-minute warning window
- **Automatic restart** from the last checkpoint when the job resumes
- **S3 sharding** for large tensors — parallel writes across prefixed keys exploit S3's horizontal scaling

## Three Deployment Models

| Environment | Backend | How It Works |
|---|---|---|
| **spore.host** | `SporeLifecycleBackend` | Watches spored's signal file, writes completion signal |
| **Slurm** | `SlurmLifecycleBackend` | Handles SIGTERM preemption, requests requeue |
| **Bare EC2** | `DirectEC2Backend` | Polls instance metadata directly |

The right backend is auto-detected. You don't have to think about it.

## Supported Solvers

| Method | Checkpoint Size | Restart Cost |
|---|---|---|
| SCF/DFT | < 100 MB | Few iterations |
| CCSD | 1-200 GB (t2 amplitudes) | Resume from amplitudes |
| CASSCF | Varies (CI vector) | Resume from MO + CI |

## Installation

```bash
pip install spot-checkpoint

# With PySCF adapters
pip install spot-checkpoint[pyscf]
```

## Documentation

- [Architecture & Design](docs/ARCHITECTURE.md) — full technical specification
- [Examples](examples/) — usage patterns for different scenarios

## Status

Early development. Layer 3 (lifecycle manager) is implemented. Layer 1 (S3 sharding) and Layer 2 (PySCF adapters) are in progress.

## License

Apache 2.0
