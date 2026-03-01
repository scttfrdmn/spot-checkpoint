# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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

[Unreleased]: https://github.com/scttfrdmn/spot-checkpoint/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/scttfrdmn/spot-checkpoint/releases/tag/v0.1.0
