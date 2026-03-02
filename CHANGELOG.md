# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.10.0] - 2026-03-01

### Added
- **lifecycle.py**: `SpotLifecycleManager.complete(keep)` — explicit completion hook with
  optional GC; sets `_completed = True` to prevent double-cleanup (closes #55)
- **lifecycle.py**: `SpotLifecycleManager.complete_async(keep)` — awaitable variant of
  `complete()` for callers inside a running event loop (closes #55)
- **lifecycle.py**: `cleanup_on_complete: int | None` constructor param — auto-triggers
  `complete()` on clean `__exit__` when set; skipped on exception to preserve checkpoints
  for restart (closes #56)
- **lifecycle.py**: `spot_complete(bucket, job_id, keep, ...)` and `spot_complete_async(...)`
  — standalone top-level helpers that resolve store from env vars and run GC; mirrors the
  `spot_restore` / `spot_status` pattern (closes #61)
- **__init__.py**: `spot_complete` and `spot_complete_async` exported from package
  `__all__` (closes #58)
- **cli.py**: `spot-checkpoint complete <location> <job_id> [--keep N]` subcommand —
  signals job success and runs GC from the command line (closes #59)
- **infra/run_smoke.py**: New automated FIS smoke-test runner (stdlib + boto3) that runs
  the full two-instance scenario non-interactively and prints PASS/FAIL with timing (closes #60)
- **tests/test_lifecycle.py**: 5 new tests covering `complete()`, `cleanup_on_complete`,
  `spot_complete()`, and `spot_complete_async()` (closes #62)
- **tests/test_lifecycle_e2e.py**: New file with 4 end-to-end scenario tests using
  `FakeCheckpointAdapter` + `LocalStore` (no S3, no PySCF) — happy path cleanup, keep-latest,
  no-cleanup default, interrupt+restore+complete, exception-preserves (closes #62)
- **tests/test_cli.py**: 2 new tests for `complete` command (`--keep 0` and `--keep 1`)

### Changed
- **infra/smoke_stack.py**: `SpotLifecycleManager` gains `keep_checkpoints=3` to limit
  accumulation during run; subprocess completion-marker hack replaced with `spot_complete(keep=1)`
  (closes #59)
- `pyproject.toml` + `__init__.py`: version bumped to `0.10.0`

## [0.9.0] - 2026-03-02

### Fixed
- **adapters/pyscf.py**: Falsy-check on energy values replaced with `is not None` in
  `SCFCheckpointAdapter`, `CCSDCheckpointAdapter`, and `CASSCFCheckpointAdapter` — energy
  values of exactly `0.0` are now stored correctly instead of as `None` (closes #40)
- **adapters/pyscf.py**: Temp chkfile created by `SCFCheckpointAdapter.checkpoint_state()`
  and `restore_state()` is now registered with `atexit` for cleanup on process exit (closes #50)
- **gc.py**: `garbage_collect()` no longer aborts on first deletion failure — each deletion
  is wrapped in `try/except/continue`; failed IDs are returned in `result["errors"]`.
  Added `ValueError` guard for `keep < 0` (closes #41)
- **lifecycle.py**: `SlurmLifecycleBackend.request_requeue()` now uses `subprocess.run()`
  instead of `os.system()`; logs an error if `scontrol requeue` returns non-zero (closes #42)
- **lifecycle.py**: `SpotLifecycleManager._run_async()` raises `RuntimeError` instead of
  `AssertionError` when called before `start()` or after `stop()` (closes #43)
- **cli.py**: `_make_store()` raises `ValueError` on empty S3 bucket name (`s3://`) (closes #48)
- **cli.py**: `int(os.environ.get(...))` calls replaced with `_env_int_cli()` helper that
  logs a warning and falls back to the default on invalid input (closes #46)
- **cli.py**: `restore` command logs a `WARNING` when overwriting an existing `.npy` file (closes #47)
- **cli.py**: `bench` command now deletes the benchmark checkpoint artifact in a `finally`
  block so cleanup runs even if read/write raises (closes #45)

### Added
- **protocol.py**: `CheckpointManifest` gains `schema_version: int = 1` field; included in
  `to_dict()` output and read back in `from_dict()` with default `1` for old manifests (closes #51)
- **protocol.py**: `TensorSpec.__post_init__()` validates `num_shards >= 1` and
  `len(checksums) == num_shards` at construction time (closes #49)
- **lifecycle.py**: job_id fallback now includes hostname —
  `f"pyscf-{socket.gethostname()}-{os.getpid()}"` — reducing collision risk on shared
  systems (closes #44)
- **lifecycle.py**: Clarifying inline comment on IMDSv2 token expiry logic in
  `DirectEC2Backend._maybe_refresh_token()`
- **lifecycle.py**: `spot_safe_async()` docstring updated to clarify the function is
  `async` for caller compatibility, not because it performs async I/O (closes #52)
- **gc.py**: `garbage_collect()` return dict now always includes `"errors": []` key for
  consistency across both the keep and no-keep paths
- 15 new tests: `test_gc.py` (2), `test_lifecycle.py` (2), `test_direct_ec2_backend.py` (1),
  `test_cli.py` (3), `test_adapters_pyscf.py` (3), `test_storage.py` (4)

### Changed
- `pyproject.toml`: version bumped to `0.9.0`
- `__init__.py`: `__version__` updated to `"0.9.0"`

## [0.8.0] - 2026-03-01

### Added
- `SpotLifecycleManager`: new `keep_checkpoints: int | None = None` parameter — after each
  periodic or emergency checkpoint, automatically calls `garbage_collect()` to prune old
  checkpoints beyond the retention limit (closes #36)
- `spot_safe()`: new `keep_checkpoints` parameter; reads `SPOT_CHECKPOINT_KEEP` env var when
  not passed explicitly; forwards to `SpotLifecycleManager` (closes #36)
- `spot_status()` and `spot_status_async()` — query the latest checkpoint's metadata without
  loading tensors; returns a flat dict with `checkpoint_id`, `method`, `timestamp`,
  `total_bytes`, and all user metadata keys merged at top level; `None` if no checkpoints
  exist (closes #37)
- `spot_status` and `spot_status_async` exported from `spot_checkpoint.__init__` (closes #37)
- `_status_from_store()` internal async helper used by `spot_status_async` and directly
  testable with any `CheckpointStore` (closes #37)
- `spot-checkpoint status LOCATION JOB_ID` CLI subcommand — shows latest checkpoint metadata
  in a rich table; `--json` output; exits non-zero when no checkpoints found (closes #37)
- `_detect_adapter_class()`: k-point SCF solvers (`KRHF`, `KUHF`, `KROHF`, `KRKS`, `KUKS`,
  `KSCF`) now detected and mapped to `SCFCheckpointAdapter` — k-point solvers use the same
  HDF5 chkfile format as gamma-point SCF (closes #38)
- `tests/test_pyscf_lifecycle_integration.py`: 3 real PySCF integration tests —
  emergency checkpoint + restore round-trip, empty-store restore returns False,
  keep_checkpoints prunes correctly during SCF (closes #34)
- `pytest-cov>=4.0` added to dev optional dependencies (closes #39)
- `.github/workflows/ci.yml`: CI workflow with ruff lint, mypy typecheck, pytest with
  `--cov-fail-under=80` coverage gate, and Codecov upload (closes #39)
- `README.md`: CI and codecov badges (closes #39)

### Changed
- `pyproject.toml`: version bumped to `0.8.0`
- `__init__.py`: `__version__` updated to `"0.8.0"`

## [0.7.0] - 2026-03-01

### Added
- `spot_safe()` / `spot_restore()`: `bucket` parameter now optional — falls back to
  `SPOT_CHECKPOINT_BUCKET` env var; raises `ValueError` with clear message if missing (closes #29)
- `SPOT_CHECKPOINT_INTERVAL`, `SPOT_CHECKPOINT_SHARD_SIZE`, `SPOT_CHECKPOINT_MAX_CONCURRENCY`
  env vars read by `spot_safe()` / `spot_restore()` — zero-config scripts on configured hosts (closes #29)
- `spot_safe_async()` and `spot_restore_async()` — async-native variants that `await` store
  operations directly; safe in Jupyter notebooks, FastAPI, and other running event loops (closes #32)
- `spot_safe_async` and `spot_restore_async` exported from `spot_checkpoint.__init__` (closes #32)
- `spot-checkpoint validate LOCATION JOB_ID [CHECKPOINT_ID]` CLI subcommand — loads checkpoint
  and re-verifies all per-tensor checksums; exits non-zero on corruption; `--json` output (closes #30)
- `LocalStore`: optional `compress: bool = False` parameter — mirrors `S3ShardedStore` compression
  behaviour; saves raw bytes as `{name}.bin.zst` (zstd level 3) and records `compression: "zstd"`
  in manifest; load path auto-detects and decompresses (closes #31)
- `tests/test_storage.py`: 4 LocalStore compression tests (closes #31)
- `tests/test_cli.py`: 5 `validate` command tests (closes #30)
- `README.md`: full documentation — install, quick start, env vars, CLI reference, architecture
  overview (closes #27)
- `.github/workflows/publish.yml`: PyPI publish workflow triggered on GitHub release; uses OIDC
  trusted publisher (`pypa/gh-action-pypi-publish`); runs lint + typecheck + tests before build (closes #28)
- `pyproject.toml`: added `typer>=0.9` and `rich>=13.0` to `dev` optional dependencies so CLI
  tests run without installing the `cli` extra
- `_env_int()` helper in `lifecycle.py` for reading integer environment variables

### Changed
- `SCFCheckpointAdapter`: replaced manual numpy extraction of `mo_coeff`/`mo_occ`/`mo_energy`
  with PySCF's native `pyscf.scf.chkfile.dump_scf()` — stores a single HDF5 blob (`"chkfile"`
  uint8 tensor); `restore_state()` writes it to a temp file and sets `mf.init_guess = "chkfile"`
  so `mf.kernel()` reads the saved MOs directly (PySCF's documented restart path) (closes #33)
- `CCSDCheckpointAdapter`: docstring clarified — saving t1/t2 and calling
  `mycc.kernel(t1, t2)` is PySCF's own documented CCSD restart; no logic change
- `CASSCFCheckpointAdapter`: docstring clarified — `mc.kernel(mo_coeff)` is PySCF's
  documented CASSCF restart; no logic change
- `tests/test_adapters_pyscf.py`: updated SCF tests — `test_checkpoint_roundtrip` checks
  `"chkfile"` key; `test_restore_roundtrip` and `test_scf_save_restore_via_store` call
  `mf.kernel()` after `restore_state()` then assert energy convergence
- `pyproject.toml`: version bumped to `0.7.0`
- `__init__.py`: `__version__` updated to `"0.7.0"`

## [0.6.0] - 2026-02-28

### Added
- `spot_restore()` convenience function in `lifecycle.py` — one-liner to restore a PySCF solver
  from the latest S3 checkpoint before calling `spot_safe()`; returns `True` if a checkpoint was
  found and restored, `False` for a fresh start (closes #22)
- `spot_restore` exported from `spot_checkpoint.__init__` (closes #22)
- CLI `restore` subcommand — `spot-checkpoint restore LOCATION JOB_ID` dumps tensors as `.npy`
  files and `metadata.json` to an output directory; supports `--checkpoint-id`, `--output`,
  `--json` flags (closes #23)
- `examples/restore_and_continue_scf.py` — canonical two-call pattern showing `spot_restore()`
  followed by `spot_safe()` (closes #24)
- `tests/test_restore_roundtrip.py` — 4 round-trip integration tests using `LocalStore`:
  no-checkpoint returns False, latest selected from two checkpoints, specific state recovered,
  full cycle with periodic checkpoint + restore (closes #25)
- `S3ShardedStore` optional `compress: bool = False` parameter — compresses shards with zstd
  level 3 before upload; manifest records `"compression": "zstd"`; load path auto-decompresses
  based on manifest field (closes #26)
- `CheckpointManifest.compression: str | None` field — backwards compatible (missing key → None)
  (closes #26)
- `pyproject.toml`: `compress = ["zstandard>=0.21"]` optional dependency extra (closes #26)
- `tests/test_storage_s3.py`: 3 compression tests — `test_compressed_roundtrip`,
  `test_compress_reduces_size`, `test_roundtrip_parameterized` (closes #26)

### Fixed
- `pyproject.toml`: version bumped from `0.1.0` to `0.5.0` to match the actual release state
  (closes #21)
- `__init__.py`: `__version__` string updated to `"0.5.0"` (closes #21)
- `examples/slurm_submit.sh`: removed misleading comment claiming restore is automatic; added
  explicit note that `spot_restore()` must be called in the Python script (closes #24)

## [0.5.0] - 2026-02-28

### Added
- `adapters/pyscf.py`: CASSCF external solver support — `_find_mps_dir`, `_tar_directory`, `_untar_directory` helpers; `CASSCFCheckpointAdapter` now detects Block2/DMRG scratch directories, tars them to `ci_mps_tar` (uint8 numpy array), and untars on restore (closes #14)
- `tests/test_external_solver_helpers.py`: 8 new tests for `_tar_directory`, `_untar_directory`, and `_find_mps_dir` helpers — no PySCF dependency
- `tests/test_adapters_pyscf.py`: `TestCASSCFExternalSolver` class — 5 mock-based integration tests for external solver checkpoint/restore path
- `tests/test_adapters_pyscf.py`: Expanded PySCF adapter test suite — 15 new tests covering SCF/CCSD/CASSCF restore roundtrips, AdapterError before kernel, metadata field assertions, size estimates, and end-to-end save/load via LocalStore (closes #13)
- `cli.py`: Full typer CLI implementation — `list`, `info`, `gc`, `bench` subcommands with `--json` output; `_make_store` helper supports both local paths and `s3://` URIs (closes #11)
- `tests/test_gc.py`: 8 tests for `garbage_collect()` covering empty store, no-keep, keep-N, keep ≥ total, keep-zero, keep-one, prefix forwarding, and return-dict shape (closes #5)
- `tests/test_cli.py`: ~25 CLI tests using `typer.testing.CliRunner` across 5 test classes (TestHelp, TestListCommand, TestInfoCommand, TestGcCommand, TestBenchCommand) (closes #12)
- `storage.py`: `S3ShardedStore` fully implemented — parallel sharded PUTs/GETs, manifest-last atomicity, xxhash checksums, batch deletes (closes #7, #8, #9, #10)
- `storage.py`: `S3ShardedStore` gains optional `endpoint_url` parameter for custom/VPC endpoints and test overrides
- `tests/test_storage_s3.py`: 8 tests covering roundtrip, sharding, corruption, missing manifest, prefix filtering, incomplete checkpoint exclusion, delete, and large tensor roundtrip
- `tests/conftest.py`: `moto_server` (ThreadedMotoServer) and `s3_store` fixtures for aioboto3-compatible S3 testing
- `pyproject.toml`: added `flask` and `flask-cors` as dev dependencies for `moto[server]`

### Added
- `.github/workflows/smoke-test.yml`: manual `workflow_dispatch` CI job —
  deploys CDK stack, launches spot instance, waits for checkpoint, injects FIS
  spot interruption, relaunches restore instance, asserts completion marker
  written with `restored_from_iteration > 0`; CDK stack always destroyed in
  cleanup steps (resolves #18)
- `infra/smoke_stack.py`: CDK stack — versioned S3 checkpoint bucket (7-day
  lifecycle), IAM instance role (`ec2:CreateTags` added for self-tagging),
  `c5.large` spot launch template (AL2023, IMDSv2 required, fake-solver
  user-data with S3 completion marker on finish) (resolves #17)
- `infra/app.py`: CDK app entry point
- `infra/fis_experiment.json`: FIS experiment template — targets instances
  tagged `spot-checkpoint-smoke-test=true`, sends 2-min interruption notice
  then terminates (`aws:ec2:send-spot-instance-interruptions`)
- `infra/requirements.txt`: `aws-cdk-lib` + `constructs` deps
- `infra/README.md`: step-by-step deploy and smoke-test runbook
- `DirectEC2Backend`: full IMDSv2 token flow — 6-hour TTL, cached token with
  automatic refresh before expiry, graceful IMDSv1 fallback when PUT fails
- `tests/test_direct_ec2_backend.py`: 11 tests covering token acquisition,
  TTL header, refresh-near-expiry, IMDSv1 fallback, and interrupt detection
  (resolves #16)

### Fixed
- `lifecycle.py`: replaced all `Optional[X]` with `X | None` (UP045), removed
  unused `Optional` import, suppressed intentional `B027` on
  `signal_completion`, shortened over-length docstring line (resolves #19)
- `lifecycle.py`: `_get_imds_token` return cast to `str(...)` — fixes
  `no-any-return` under mypy `--strict`
- `storage.py`: added `# type: ignore[import-untyped]` for aioboto3/botocore
  imports; fixed `shard_bytes` annotation `tuple[bytes, ...]` → `list[bytes]`
  to match `asyncio.gather` return; added `strict=True` to `zip()` (B905)
  (resolves #20)
- `adapters/pyscf.py`: cast `checkpoint_size_estimate` returns to `int(...)`
  — fixes two `no-any-return` errors; suppressed `RUF002`/`RUF003` for
  intentional `×` Unicode math in docstrings (resolves #20)
- `__init__.py`: fixed import sort (I001); suppressed `RUF022` on `__all__`
  (grouped by category, not alphabetically)
- `cli.py`: updated `Annotated` import to `typing` (UP035); `Optional[X]` →
  `X | None` (UP045)
- `protocol.py`: fixed import sort (I001)
- `pyproject.toml`: added mypy overrides — `ignore_missing_imports` for
  optional `typer`/`rich` deps; `disallow_untyped_decorators = false` for
  `cli` module
- `SpotLifecycleManager` now runs its async event loop on a dedicated background
  daemon thread and uses `asyncio.run_coroutine_threadsafe` instead of
  `loop.run_until_complete` — fixes `RuntimeError: Cannot run the event loop while
  another loop is running` when `check()` is called from an async context
  (resolves #15)
- `storage.py`: corrected import names (`CheckpointCorruptionError`, `CheckpointReadError`, `TensorSpec`) to match `protocol.py`
- `storage.py`: fixed `LocalStore.save_checkpoint` to construct `CheckpointManifest` with correct fields (`tensor_specs`, `method`); use `manifest.to_dict()` for serialization
- `storage.py`: fixed `load_checkpoint` to read `tensor_specs` key from manifest JSON
- `adapters/pyscf.py`: added missing `AdapterError` to `protocol.py` exception hierarchy
- `tests/test_storage.py`: updated exception imports to match `protocol.py` names
- `tests/test_integration.py`: fixed `from tests.conftest` import (not a package)
- `pyproject.toml`: removed non-existent `numpy-stubs` dev dependency (numpy ships its own stubs)

## [0.1.0] - 2026-02-28

### Added
- `protocol.py`: `SpotCheckpointError` hierarchy, `CheckpointPayload`, `CheckpointManifest`, `TensorSpec`, `Checkpointable` and `CheckpointStore` protocols
- `storage.py`: `LocalStore` filesystem backend; `S3ShardedStore` stub (methods raise `NotImplementedError`)
- `lifecycle.py`: `SporeLifecycleBackend`, `SlurmLifecycleBackend`, `DirectEC2Backend`; `SpotLifecycleManager`; `detect_backend()`; `spot_safe()` convenience API
- `gc.py`: `garbage_collect()` async function
- `adapters/pyscf.py`: `SCFCheckpointAdapter`, `CCSDCheckpointAdapter`, `CASSCFCheckpointAdapter`
- `cli.py`: placeholder (not yet implemented)
- `tests/`: `conftest.py` with `FakeSolver`/`FakeCheckpointAdapter` fixtures; unit tests for protocol, storage, lifecycle
- `docs/ARCHITECTURE.md`: full three-layer design spec with throughput math and open questions
- `pyproject.toml`: `hatchling` build, `uv`-compatible dependency spec

[Unreleased]: https://github.com/scttfrdmn/spot-checkpoint/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/scttfrdmn/spot-checkpoint/compare/v0.7.0...v0.8.0
[0.7.0]: https://github.com/scttfrdmn/spot-checkpoint/compare/v0.6.0...v0.7.0
[0.6.0]: https://github.com/scttfrdmn/spot-checkpoint/compare/v0.5.0...v0.6.0
[0.5.0]: https://github.com/scttfrdmn/spot-checkpoint/compare/v0.1.0...v0.5.0
[0.1.0]: https://github.com/scttfrdmn/spot-checkpoint/releases/tag/v0.1.0
