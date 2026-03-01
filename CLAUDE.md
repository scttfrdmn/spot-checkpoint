# CLAUDE.md

## What This Is

Checkpoint/restart library for iterative scientific computations on preemptible instances. First target: PySCF quantum chemistry on AWS spot. See `docs/ARCHITECTURE.md` for full design.

## Project Layout

```
src/spot_checkpoint/
├── __init__.py        # exports: spot_safe, SpotLifecycleManager
├── storage.py         # Layer 1: S3ShardedStore, LocalStore
├── protocol.py        # Layer 2: Checkpointable, CheckpointPayload
├── lifecycle.py       # Layer 3: backends + SpotLifecycleManager  [DONE]
├── gc.py              # Checkpoint garbage collection
├── adapters/
│   └── pyscf.py       # SCF, CCSD, CASSCF adapters
└── cli.py             # CLI (list, info, gc, bench)
```

## Build Order

Implement in dependency order:
1. `protocol.py` — dataclasses and protocols (no deps)
2. `storage.py` — S3ShardedStore + LocalStore
3. `adapters/pyscf.py` — PySCF wrappers
4. `lifecycle.py` — already written, may need updates to import from protocol/storage
5. `gc.py` — garbage collection
6. `cli.py` — typer CLI
7. `__init__.py` — public API surface
8. Tests throughout

## Python Standards

- Python 3.10+ (PySCF requires 3.8+, we use 3.10 for `match` and `X | Y` union types)
- `src/` layout with `pyproject.toml` (no setup.py)
- Type hints everywhere, `mypy --strict` clean
- `ruff` for linting and formatting
- `pytest` + `pytest-asyncio` for tests
- Docstrings: Google style

## Code Style

- Protocols over ABC where possible
- `dataclass` for value types, `Protocol` for interfaces
- Async for I/O (S3 operations), sync wrappers for user-facing API
- No global state
- Errors: custom exception hierarchy rooted at `SpotCheckpointError`

## Testing

- `moto` for S3 mocking — never hit real AWS in unit tests
- `conftest.py` provides: mock S3 bucket, temp directories, fake solver objects
- PySCF adapter tests: skip if PySCF not installed (`pytest.importorskip("pyscf")`)
- Target 80%+ coverage
- Integration test: fake iterative solver + LocalStore + lifecycle manager round-trip

## Dependencies

Required: `numpy`, `aioboto3`, `xxhash`
Optional: `pyscf` (adapters only)
Dev: `pytest`, `pytest-asyncio`, `moto[s3]`, `ruff`, `mypy`

## Key Design Decisions

- **Manifest-last atomicity**: checkpoint valid only if `_manifest.json` exists
- **S3 prefix sharding**: each tensor name → independent S3 partition for parallel writes
- **spore.host integration**: watch signal file, don't duplicate spored's work (see `docs/ARCHITECTURE.md` Layer 3)
- **Slurm integration**: SIGTERM handler for preemption, SIGUSR1 for wall-time warning
- **Auto-detection**: `detect_backend()` picks spore.host > Slurm > direct EC2 based on environment

## Do Not

- Duplicate spored functionality (metadata polling, DNS, notifications)
- Hard-depend on PySCF — adapters are optional imports
- Use `boto3` sync client in the storage engine — async throughout
- Use SHA256 for checksums — xxhash for speed
- Write monolithic S3 objects — always shard

## References

- `docs/ARCHITECTURE.md` — full design spec with interfaces, throughput math, and open questions
- spore.host agent code: `github.com/scttfrdmn/mycelium/spawn/pkg/agent/agent.go`
- spore.host issue #151: spored spot polling interval mismatch
