# spot-checkpoint: Architecture & Design Specification

## Purpose

A Python library that makes iterative scientific computations fault-tolerant on preemptible cloud instances (AWS spot, Slurm-preemptible nodes). First target is PySCF quantum chemistry; the architecture generalizes to any iterative solver.

The pitch: take your existing PySCF script, add one line, run on spot at 70% discount with automatic fault tolerance.

```python
from pyscf import gto, scf
from spot_checkpoint import spot_safe

mol = gto.M(atom='...', basis='aug-cc-pvtz')
mf = scf.RHF(mol)
mf.callback = spot_safe(mf, bucket="my-checkpoints")
mf.kernel()
```

## Positioning

Complements CloudRamp (economic case for spot in research computing) and spore.host (instance lifecycle management). This library handles the application-level checkpoint/restart that makes spot instances practically usable for long-running scientific calculations.

## Three-Layer Architecture

```
┌─────────────────────────────────────────────────────┐
│  Layer 3: Spot Lifecycle Manager                    │
│  Connects interrupt detection to checkpoint engine  │
│                                                     │
│  Backends: SporeLifecycleBackend (spore.host)       │
│            SlurmLifecycleBackend (Slurm/SIGTERM)    │
│            DirectEC2Backend (raw spot instances)     │
├─────────────────────────────────────────────────────┤
│  Layer 2: Computation State Protocol                │
│  Checkpointable interface + domain adapters         │
│                                                     │
│  Adapters: SCFCheckpointAdapter                     │
│            CCSDCheckpointAdapter                    │
│            CASSCFCheckpointAdapter                  │
│            (extensible to Psi4, ASE, ML, etc.)      │
├─────────────────────────────────────────────────────┤
│  Layer 1: Checkpoint Storage Engine                 │
│  S3 sharded tensor persistence                      │
│                                                     │
│  Impls:   S3ShardedStore (production)               │
│           LocalStore (testing/development)           │
└─────────────────────────────────────────────────────┘
```

Each layer has a clean protocol boundary. Layer 1 knows nothing about chemistry. Layer 2 knows nothing about cloud infrastructure. Layer 3 ties them together.

---

## Layer 1: Checkpoint Storage Engine

### Purpose

Backend-agnostic tensor persistence. Takes named numpy arrays, persists them, returns them on restore.

### S3 Sharding Strategy

S3 scales to 3,500 PUT/s and 5,500 GET/s **per prefix**. A monolithic 20GB upload is bottlenecked on a single connection. Sharding across prefixed keys exploits S3's horizontal scaling.

Key layout:
```
s3://bucket/{job_id}/ckpt/{checkpoint_id}/{tensor_name}/shard-{N:04d}
s3://bucket/{job_id}/ckpt/{checkpoint_id}/_manifest.json
```

Each `{tensor_name}` gives an independent S3 prefix partition. With 32 concurrent PUTs across 32 prefixes, aggregate throughput reaches 500MB/s+ on instances with 25Gbps networking.

### Manifest-Last Atomic Commits

The `_manifest.json` is written last. It contains:
- Tensor names, shapes, dtypes
- Shard layout (count, size, dimension slicing)
- xxhash checksums per shard
- Metadata (iteration, convergence, method parameters)

A checkpoint is only valid if its manifest exists. Incomplete checkpoints (killed mid-write) leave orphaned shards that garbage collection sweeps. No distributed transactions needed.

### Interface

```python
class CheckpointStore(Protocol):
    async def save_checkpoint(
        self,
        checkpoint_id: str,
        tensors: dict[str, np.ndarray],
        metadata: dict[str, Any],
    ) -> CheckpointManifest: ...

    async def load_checkpoint(
        self,
        checkpoint_id: str,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]: ...

    async def list_checkpoints(
        self,
        prefix: str,
    ) -> list[dict[str, Any]]: ...

    async def delete_checkpoint(
        self,
        checkpoint_id: str,
    ) -> None: ...
```

### S3ShardedStore Implementation

Constructor parameters:
- `bucket: str` — S3 bucket name
- `job_id: str` — job identifier (becomes key prefix)
- `shard_size: int = 64 * 1024 * 1024` — shard size in bytes (64MB default)
- `max_concurrency: int = 32` — parallel upload streams
- `endpoint_strategy: Literal["standard", "accelerate", "vpc"] = "vpc"` — S3 endpoint type
- `checksum_algorithm: str = "xxhash"` — fast checksums over SHA256

Save workflow:
1. For each tensor, compute sharding plan (contiguous chunks along largest dimension)
2. Serialize each shard to bytes (numpy `.tobytes()`)
3. Launch parallel PUTs across all shards of all tensors using asyncio + aioboto3
4. Write `_manifest.json` with shapes, dtypes, shard layouts, checksums
5. Manifest write is the commit point

Load workflow:
1. Read `_manifest.json`
2. Parallel GET all shards
3. Verify checksums
4. Reassemble tensors from shards
5. Return tensors dict + metadata dict

### LocalStore Implementation

File-system based store for testing and local development. Same interface, uses directories instead of S3 prefixes. No sharding needed (local disk is not prefix-partitioned).

### Throughput Math

Large CCSD T2 tensor: 50 occupied × 300 virtual → 50² × 300² × 8 bytes ≈ 18GB

With 32 concurrent PUT streams, 64MB shards, ~500MB/s aggregate throughput via VPC endpoint:
- 18GB / 500MB/s ≈ 36 seconds
- Well within 2-minute spot window minus 30s headroom

For extreme cases (huge basis sets), use instances with 25Gbps+ networking and S3 Transfer Acceleration.

### Garbage Collection

A background or CLI-invocable sweep that:
1. Lists all keys under `s3://bucket/{job_id}/ckpt/`
2. Identifies checkpoint directories without `_manifest.json`
3. Deletes orphaned shards
4. Optionally prunes old checkpoints (keep N most recent)

---

## Layer 2: Computation State Protocol

### Purpose

Abstract what "my current state" means for any iterative solver. The seam between domain-specific science code and generic infrastructure.

### Core Protocol

```python
class Checkpointable(Protocol):
    def checkpoint_state(self) -> CheckpointPayload: ...
    def restore_state(self, payload: CheckpointPayload) -> None: ...

    @property
    def checkpoint_size_estimate(self) -> int: ...

@dataclass
class CheckpointPayload:
    tensors: dict[str, np.ndarray]
    metadata: dict[str, Any]    # iteration, convergence, params
    method: str                 # "rhf", "ccsd", "casscf", etc.
    timestamp: float

    @property
    def total_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors.values())
```

### PySCF Adapters

#### SCFCheckpointAdapter

Wraps `pyscf.scf.hf.SCF` and subclasses (RHF, UHF, ROHF, RKS, UKS).

State captured:
- `mo_coeff` — MO coefficients (nao × nmo float64)
- `mo_occ` — occupation numbers (nmo float64)
- `mo_energy` — orbital energies (nmo float64)

Metadata: iteration count, total energy, convergence flag.

Checkpoint size: `nao × nmo × 8 × 3`. Typically < 100MB even for large molecules. Trivial to checkpoint.

Restore: set `mf.mo_coeff`, `mf.mo_occ`, `mf.mo_energy` then call `mf.kernel()` — PySCF uses these as the initial guess and converges in a few iterations.

#### CCSDCheckpointAdapter

Wraps `pyscf.cc.ccsd.CCSD` (and RCCSD, UCCSD).

State captured:
- `t1` — singles amplitudes (nocc × nvir float64), small
- `t2` — doubles amplitudes (nocc × nocc × nvir × nvir float64), BIG

Metadata: iteration count, correlation energy, convergence flag.

Checkpoint size estimate: `nocc² × nvir² × 8 × 1.1` (t2 dominates, 10% overhead for t1 + metadata).

This is where S3 sharding matters most. Example sizes:
- Small molecule (10 occ, 50 vir): 10² × 50² × 8 = 20MB
- Medium molecule (30 occ, 200 vir): 30² × 200² × 8 = 2.9GB
- Large molecule (50 occ, 300 vir): 50² × 300² × 8 = 18GB
- Very large (100 occ, 500 vir): 100² × 500² × 8 = 200GB (may exceed memory anyway)

Restore: set `mycc.t1`, `mycc.t2` then call `mycc.kernel()`.

#### CASSCFCheckpointAdapter

Wraps `pyscf.mcscf.casci.CASSCF`.

State captured:
- `mo_coeff` — MO coefficients
- `ci` — CI vector (can be enormous for large active spaces)

For external solvers (Block2/DMRG, Dice/SHCI):
- MPS state lives on disk, not in Python memory
- Adapter must capture the solver's scratch directory contents
- Block2 supports MPS snapshots — adapter reads those files and includes them as binary blobs

Metadata: macro iteration, total energy, CAS energy, ncas, nelecas.

This is the hardest adapter because of the external solver coordination.

### Future Adapters (not in v1)

- **Psi4**: HDF5-based internal checkpointing, adapter would wrap wavefunction object
- **ASE**: Geometry optimization trajectories
- **ML training loops**: Model weights + optimizer state (PyTorch/JAX)

---

## Layer 3: Spot Lifecycle Manager

### Purpose

Connects interrupt detection (environment-specific) to checkpoint persistence (generic). The user-facing orchestrator.

### Three Backends

Backends detect interrupts and fire a callback. They do NOT handle checkpointing.

#### SporeLifecycleBackend

For instances managed by spore.host's `spored` agent.

spored already handles:
- EC2 metadata polling for spot interruption
- `wall(1)` user notifications
- DNS cleanup and registry deregistration
- Instance termination/stop/hibernate

The backend watches `/tmp/spawn-spot-interruption.json` (written by spored on spot detection). After emergency checkpoint, writes `/tmp/spawn-job-complete` so spored takes its configured `on_complete` action.

Does NOT duplicate any spored functionality. Thin integration only.

**Note:** spored currently polls on a 1-minute tick, not the 5-second interval documented. Issue filed: mycelium#151. The checkpoint library's 1-second file watch partially compensates.

#### SlurmLifecycleBackend

For jobs running under Slurm (including Slurm bursting to spot via ParallelCluster or spore.host).

Handles:
- `SIGTERM` — Slurm preemption (GraceTime is the deadline)
- `SIGUSR1` — approaching wall-time limit (via `--signal=B:USR1@N` in sbatch)

After emergency checkpoint, calls `scontrol requeue` so Slurm resubmits the job.

Uses `SLURM_CHECKPOINT_DIR` env var for local checkpoint staging if set.

#### DirectEC2Backend

Fallback for raw EC2 spot instances without spored or Slurm. Does its own IMDSv2 metadata polling (2-second interval). This is what most "getting started" tutorials would use.

### Auto-Detection

`detect_backend()` checks environment:
1. `SLURM_JOB_ID` set → SlurmLifecycleBackend
2. `systemctl is-active spored` → SporeLifecycleBackend
3. Otherwise → DirectEC2Backend

### SpotLifecycleManager

The orchestrator that ties backend + store + adapter together.

Two integration modes:

**PySCF callback mode:**
```python
mgr = SpotLifecycleManager(store, adapter)
solver.callback = mgr.make_callback()
solver.kernel()
```

The callback fires each solver iteration. It checks:
- Is there a pending interrupt? → emergency checkpoint, then exit
- Has periodic_interval elapsed? → background checkpoint

**Manual loop mode:**
```python
with mgr:
    for step in iteration:
        do_work(step)
        mgr.check(step)
```

### Emergency Checkpoint Flow

1. Backend detects interrupt, fires `_on_interrupt(event)`
2. Event stored with deadline timestamp
3. On next solver iteration (callback) or `check()` call:
   a. Extract state via `adapter.checkpoint_state()`
   b. Estimate write time from `payload.total_bytes` / expected throughput
   c. Log feasibility warning if tight
   d. Blocking write via `store.save_checkpoint()`
   e. Backend-specific post-actions:
      - Slurm: `scontrol requeue` if configured
      - spore.host: write completion file
      - Direct: exit
   f. `raise SystemExit(0)`

### Restore Flow

Before starting computation:
```python
restored = await mgr.restore_latest()
if restored:
    print(f"Resumed from checkpoint")
solver.kernel()  # Picks up from restored state
```

---

## Package Structure

```
spot-checkpoint/
├── pyproject.toml
├── README.md
├── CLAUDE.md
├── docs/
│   └── ARCHITECTURE.md          ← this document
├── src/
│   └── spot_checkpoint/
│       ├── __init__.py          ← exports spot_safe, SpotLifecycleManager
│       ├── storage.py           ← Layer 1: CheckpointStore, S3ShardedStore, LocalStore
│       ├── protocol.py          ← Layer 2: Checkpointable, CheckpointPayload
│       ├── lifecycle.py         ← Layer 3: SpotLifecycleManager + backends
│       ├── gc.py                ← Checkpoint garbage collection
│       ├── adapters/
│       │   ├── __init__.py
│       │   └── pyscf.py         ← SCF, CCSD, CASSCF adapters
│       └── cli.py               ← CLI for gc, list checkpoints, restore info
├── tests/
│   ├── conftest.py              ← fixtures: mock S3, fake solvers, temp dirs
│   ├── test_storage.py          ← S3ShardedStore + LocalStore
│   ├── test_protocol.py         ← CheckpointPayload serialization
│   ├── test_lifecycle.py        ← All 3 backends + manager
│   ├── test_adapters_pyscf.py   ← PySCF adapter round-trips
│   ├── test_gc.py               ← Garbage collection
│   └── test_integration.py      ← End-to-end: fake solver + local store + lifecycle
├── examples/
│   ├── basic_scf.py             ← Minimal SCF + spot_safe example
│   ├── ccsd_large.py            ← CCSD with S3 sharding
│   └── slurm_submit.sh          ← sbatch script with checkpoint/restart
└── benchmarks/
    └── s3_throughput.py          ← Measure sharded write throughput
```

## Dependencies

### Required
- `numpy` — tensor handling
- `aioboto3` / `aiobotocore` — async S3 operations
- `xxhash` — fast checksums

### Optional
- `pyscf` — only needed if using PySCF adapters
- `boto3` — sync fallback if async not desired

### Dev
- `pytest`, `pytest-asyncio` — testing
- `moto` — S3 mocking
- `ruff` — linting
- `mypy` — type checking

## Testing Strategy

### Unit Tests (no AWS, no PySCF)

- **Storage**: Use `moto` to mock S3. Test shard/reassemble round-trips for various tensor shapes. Test manifest-last atomicity (simulate kill mid-write, verify incomplete checkpoint ignored). Test LocalStore equivalence.
- **Protocol**: Test CheckpointPayload serialization. Test size estimates vs actual sizes.
- **Lifecycle**: Test each backend in isolation:
  - SporeLifecycleBackend: write signal file to tmp, verify callback fires with correct deadline
  - SlurmLifecycleBackend: send SIGTERM to test process, verify callback fires
  - DirectEC2Backend: mock HTTP responses from metadata endpoint
  - Manager: fake Checkpointable + LocalStore, verify periodic and emergency checkpoint flows

### Integration Tests (optional, need PySCF installed)

- Round-trip: run small SCF, checkpoint, restore, verify convergence from checkpoint
- Round-trip: run small CCSD, checkpoint t1/t2, restore, verify energies match

### Benchmarks (manual, need real S3)

- Measure sharded write throughput at various concurrency levels
- Compare shard sizes (32MB, 64MB, 128MB) for different tensor sizes
- Validate throughput math from architecture doc

## CLI

```bash
# List checkpoints for a job
spot-checkpoint list --bucket my-bucket --job-id ccsd-h2o

# Show details of latest checkpoint
spot-checkpoint info --bucket my-bucket --job-id ccsd-h2o

# Garbage collect orphaned shards
spot-checkpoint gc --bucket my-bucket --job-id ccsd-h2o --keep 3

# Benchmark S3 throughput
spot-checkpoint bench --bucket my-bucket --size 10GB --concurrency 32
```

## Configuration

Environment variables (all optional, sensible defaults):
- `SPOT_CHECKPOINT_BUCKET` — default S3 bucket
- `SPOT_CHECKPOINT_INTERVAL` — periodic checkpoint interval in seconds (default: 300)
- `SPOT_CHECKPOINT_SHARD_SIZE` — shard size in bytes (default: 67108864 = 64MB)
- `SPOT_CHECKPOINT_MAX_CONCURRENCY` — parallel S3 streams (default: 32)
- `SPOT_CHECKPOINT_HEADROOM` — seconds reserved from interrupt deadline (default: 30)

## Open Questions for Implementation

1. **Async vs sync API surface**: The storage layer is naturally async (S3 I/O). Should the user-facing API (`spot_safe`, `make_callback`) hide the async entirely, or expose it? Current design uses `asyncio.run()` internally — fine for PySCF (single-threaded), may conflict with existing event loops in other contexts.

2. **CASSCF external solver files**: Block2 MPS snapshots can be large binary files on disk. Should the adapter tar them and upload as a single "tensor" blob, or use a separate file-upload path in the store?

3. **Checkpoint retention policy**: Keep N most recent? Keep all until manual GC? Time-based expiry? Configurable per job?

4. **Multi-node MPI jobs**: Some PySCF calculations run across multiple nodes. Checkpoint coordination across nodes is a harder problem (barrier sync, consistent snapshot). Defer to v2?

5. **Compression**: zstd on shards before upload? Trades CPU for bandwidth. Probably worth it for SCF (small, compressible MO coefficients), less clear for CCSD (dense floating point, low compressibility).
