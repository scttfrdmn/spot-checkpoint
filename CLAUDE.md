# CLAUDE.md

## What This Is

Checkpoint/restart library for iterative scientific computations on preemptible instances. First target: PySCF quantum chemistry on AWS spot. See `docs/ARCHITECTURE.md` for full design.

## Project Layout

```
src/spot_checkpoint/
├── __init__.py        # exports: spot_safe, SpotLifecycleManager
├── storage.py         # Layer 1: S3ShardedStore, LocalStore
├── protocol.py        # Layer 2: Checkpointable, CheckpointPayload
├── lifecycle.py       # Layer 3: backends + SpotLifecycleManager
├── gc.py              # Checkpoint garbage collection
├── adapters/
│   └── pyscf.py       # SCF, CCSD, CASSCF adapters
└── cli.py             # CLI (list, info, gc, bench)
```

## Project Tracking

All work is tracked on GitHub:

- **Project board**: https://github.com/users/scttfrdmn/projects/30
- **Issues**: https://github.com/scttfrdmn/spot-checkpoint/issues
- **Milestones**: https://github.com/scttfrdmn/spot-checkpoint/milestones

Use GitHub issues for all tasks, bugs, and enhancements. Do not maintain standalone tracking documents.

### Milestones

| Milestone | Scope |
|-----------|-------|
| v0.1.0 | Fix & Foundation — import fixes, LocalStore tests green, CHANGELOG |
| v0.2.0 | S3 Backend — full S3ShardedStore with sharding + moto tests |
| v0.3.0 | CLI & GC — typer CLI, gc tests, developer experience |
| v0.4.0 | Integration — PySCF integration tests, 80%+ coverage |

### Labels

- Type: `bug`, `enhancement`, `testing`, `documentation`, `refactor`
- Component: `component: storage`, `component: protocol`, `component: lifecycle`, `component: adapters`, `component: cli`, `component: gc`
- Priority: `P1: critical`, `P2: high`, `P3: medium`

## Versioning and Changelog

- Versions follow [Semantic Versioning 2.0.0](https://semver.org/)
- `CHANGELOG.md` follows [Keep a Changelog 1.1.0](https://keepachangelog.com/en/1.1.0/)
- Update `CHANGELOG.md` with every PR under `[Unreleased]`; move to a version heading on release

## Python Standards

- Python 3.10+ (PySCF requires 3.8+, we use 3.10 for `match` and `X | Y` union types)
- `src/` layout with `pyproject.toml` (no setup.py)
- Type hints everywhere, `mypy --strict` clean
- `ruff` for linting and formatting
- `pytest` + `pytest-asyncio` for tests
- Docstrings: Google style

## Tooling

- Use `uv` for all package management and running tools:
  - `uv sync --extra dev` — install all dev dependencies
  - `uv run pytest` — run tests
  - `uv run ruff check .` — lint
  - `uv run mypy src/` — type check
- Never use `pip` or `python` directly — always `uv run` or `uv sync`
- AWS: always use `AWS_PROFILE=aws` — e.g. `AWS_PROFILE=aws aws ...`, `AWS_PROFILE=aws cdk ...`

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
Optional: `pyscf` (adapters only), `typer` + `rich` (CLI)
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
- Use `pip` or bare `python` — use `uv`

## References

- `docs/ARCHITECTURE.md` — full design spec with interfaces, throughput math, and open questions
- spore.host agent code: `github.com/scttfrdmn/mycelium/spawn/pkg/agent/agent.go`
- spore.host issue #151: spored spot polling interval mismatch
