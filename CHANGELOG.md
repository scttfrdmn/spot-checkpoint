# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `tests/test_adapters_pyscf.py`: Expanded PySCF adapter test suite — 15 new tests covering SCF/CCSD/CASSCF restore roundtrips, AdapterError before kernel, metadata field assertions, size estimates, and end-to-end save/load via LocalStore (closes #13)
- `cli.py`: Full typer CLI implementation — `list`, `info`, `gc`, `bench` subcommands with `--json` output; `_make_store` helper supports both local paths and `s3://` URIs (closes #11)
- `tests/test_gc.py`: 8 tests for `garbage_collect()` covering empty store, no-keep, keep-N, keep ≥ total, keep-zero, keep-one, prefix forwarding, and return-dict shape (closes #5)
- `tests/test_cli.py`: ~25 CLI tests using `typer.testing.CliRunner` across 5 test classes (TestHelp, TestListCommand, TestInfoCommand, TestGcCommand, TestBenchCommand) (closes #12)
- `storage.py`: `S3ShardedStore` fully implemented — parallel sharded PUTs/GETs, manifest-last atomicity, xxhash checksums, batch deletes (closes #7, #8, #9, #10)
- `storage.py`: `S3ShardedStore` gains optional `endpoint_url` parameter for custom/VPC endpoints and test overrides
- `tests/test_storage_s3.py`: 8 tests covering roundtrip, sharding, corruption, missing manifest, prefix filtering, incomplete checkpoint exclusion, delete, and large tensor roundtrip
- `tests/conftest.py`: `moto_server` (ThreadedMotoServer) and `s3_store` fixtures for aioboto3-compatible S3 testing
- `pyproject.toml`: added `flask` and `flask-cors` as dev dependencies for `moto[server]`

### Fixed
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
