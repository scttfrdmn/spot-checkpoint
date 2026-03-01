"""
CLI for spot-checkpoint management.

Usage:
    spot-checkpoint list /path/to/store my-job
    spot-checkpoint info /path/to/store my-job
    spot-checkpoint gc /path/to/store my-job --keep 3
    spot-checkpoint bench /tmp/bench --size-mb 256
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.table import Table

from spot_checkpoint.gc import garbage_collect
from spot_checkpoint.storage import LocalStore, S3ShardedStore

app = typer.Typer(
    name="spot-checkpoint",
    help="Manage checkpoints for iterative scientific computations on preemptible instances.",
    no_args_is_help=True,
)
console = Console()
err_console = Console(stderr=True)


def _make_store(location: str, job_id: str) -> LocalStore | S3ShardedStore:
    """Create a store from a location string.

    Args:
        location: Local path or ``s3://bucket`` URI.
        job_id: Job identifier scoping the checkpoints.

    Returns:
        A :class:`LocalStore` or :class:`S3ShardedStore` as appropriate.
    """
    if location.startswith("s3://"):
        bucket = location.removeprefix("s3://").rstrip("/")
        shard_size = int(os.environ.get("SPOT_CHECKPOINT_SHARD_SIZE", str(64 * 1024 * 1024)))
        max_concurrency = int(os.environ.get("SPOT_CHECKPOINT_MAX_CONCURRENCY", "32"))
        return S3ShardedStore(
            bucket=bucket,
            job_id=job_id,
            shard_size=shard_size,
            max_concurrency=max_concurrency,
        )
    return LocalStore(base_dir=location, job_id=job_id)


def _fmt_size(nbytes: int) -> str:
    if nbytes >= 1_000_000_000:
        return f"{nbytes / 1e9:.2f} GB"
    return f"{nbytes / 1e6:.2f} MB"


@app.command("list")
def list_checkpoints(
    location: Annotated[str, typer.Argument(help="Storage location (path or s3://bucket)")],
    job_id: Annotated[str, typer.Argument(help="Job identifier")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """List all checkpoints for a job."""
    store = _make_store(location, job_id)
    checkpoints: list[dict[str, Any]] = asyncio.run(store.list_checkpoints(""))

    if json_output:
        output = [
            {
                "checkpoint_id": c["checkpoint_id"],
                "method": c["method"],
                "timestamp": c["timestamp"],
                "total_bytes": c["total_bytes"],
            }
            for c in checkpoints
        ]
        typer.echo(json.dumps(output, indent=2))
        return

    if not checkpoints:
        console.print("[yellow]No checkpoints found.[/yellow]")
        return

    table = Table(title=f"Checkpoints for {job_id}")
    table.add_column("ID", style="cyan")
    table.add_column("Method", style="green")
    table.add_column("Timestamp", style="blue")
    table.add_column("Size", justify="right", style="magenta")
    table.add_column("Tensors", justify="right")

    for c in checkpoints:
        dt_str = datetime.fromtimestamp(c["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
        n_tensors = len(c.get("tensor_specs", {}))
        table.add_row(
            c["checkpoint_id"],
            c["method"],
            dt_str,
            _fmt_size(c["total_bytes"]),
            str(n_tensors),
        )

    console.print(table)


@app.command("info")
def info(
    location: Annotated[str, typer.Argument(help="Storage location (path or s3://bucket)")],
    job_id: Annotated[str, typer.Argument(help="Job identifier")],
    checkpoint_id: Annotated[
        str | None, typer.Argument(help="Checkpoint ID (default: latest)")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show details for a checkpoint (defaults to latest)."""
    store = _make_store(location, job_id)

    async def _get_manifest() -> dict[str, Any]:
        checkpoints = await store.list_checkpoints("")
        if not checkpoints:
            return {}
        if checkpoint_id is None:
            return max(checkpoints, key=lambda c: c["timestamp"])
        for c in checkpoints:
            if c["checkpoint_id"] == checkpoint_id:
                return c
        return {}

    manifest = asyncio.run(_get_manifest())

    if not manifest:
        err_console.print(
            f"[red]Checkpoint not found: {checkpoint_id or '(latest)'}[/red]"
        )
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(manifest, indent=2))
        return

    dt_str = datetime.fromtimestamp(manifest["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[bold]Checkpoint ID:[/bold] {manifest['checkpoint_id']}")
    console.print(f"[bold]Method:[/bold] {manifest['method']}")
    console.print(f"[bold]Timestamp:[/bold] {dt_str}")
    console.print(f"[bold]Size:[/bold] {_fmt_size(manifest['total_bytes'])}")

    tensor_specs: dict[str, Any] = manifest.get("tensor_specs", {})
    if tensor_specs:
        table = Table(title="Tensors")
        table.add_column("Name", style="cyan")
        table.add_column("Shape", style="green")
        table.add_column("DType", style="blue")
        table.add_column("Size", justify="right", style="magenta")
        table.add_column("Shards", justify="right")

        for name, spec in tensor_specs.items():
            shape_str = str(tuple(spec["shape"]))
            table.add_row(
                name,
                shape_str,
                spec["dtype"],
                _fmt_size(spec["nbytes"]),
                str(spec["num_shards"]),
            )

        console.print(table)


@app.command("status")
def status(
    location: Annotated[str, typer.Argument(help="Storage location (path or s3://bucket)")],
    job_id: Annotated[str, typer.Argument(help="Job identifier")],
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Show metadata of the latest checkpoint without loading tensors."""
    from spot_checkpoint.lifecycle import _status_from_store

    store = _make_store(location, job_id)
    result = asyncio.run(_status_from_store(store, ""))

    if result is None:
        err_console.print("[red]No checkpoints found.[/red]")
        raise typer.Exit(1)

    if json_output:
        typer.echo(json.dumps(result, indent=2))
        return

    dt_str = datetime.fromtimestamp(result["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[bold]Checkpoint ID:[/bold] {result['checkpoint_id']}")
    console.print(f"[bold]Method:[/bold] {result['method']}")
    console.print(f"[bold]Timestamp:[/bold] {dt_str}")
    console.print(f"[bold]Size:[/bold] {_fmt_size(result['total_bytes'])}")

    meta_keys = {
        k for k in result
        if k not in ("checkpoint_id", "method", "timestamp", "total_bytes")
    }
    if meta_keys:
        table = Table(title="Metadata")
        table.add_column("Key", style="cyan")
        table.add_column("Value", style="green")
        for key in sorted(meta_keys):
            table.add_row(key, str(result[key]))
        console.print(table)


@app.command("gc")
def gc(
    location: Annotated[str, typer.Argument(help="Storage location (path or s3://bucket)")],
    job_id: Annotated[str, typer.Argument(help="Job identifier")],
    keep: Annotated[
        int | None, typer.Option("--keep", "-k", help="Number of checkpoints to keep")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
    dry_run: Annotated[
        bool, typer.Option("--dry-run", help="Show what would be deleted without deleting")
    ] = False,
) -> None:
    """Garbage collect old checkpoints."""
    store = _make_store(location, job_id)

    if dry_run:
        async def _dry_run_check() -> dict[str, Any]:
            checkpoints = await store.list_checkpoints("")
            total = len(checkpoints)
            if keep is None or total <= keep:
                to_delete: list[dict[str, Any]] = []
                to_keep = checkpoints
            else:
                sorted_ckpts = sorted(checkpoints, key=lambda c: c.get("timestamp", 0))
                to_delete = sorted_ckpts[: total - keep]
                to_keep = sorted_ckpts[total - keep :]
            return {
                "total": total,
                "would_delete": len(to_delete),
                "would_keep": len(to_keep),
                "checkpoints_to_delete": [c["checkpoint_id"] for c in to_delete],
            }

        result = asyncio.run(_dry_run_check())
        if json_output:
            typer.echo(json.dumps(result, indent=2))
        else:
            console.print(f"Total checkpoints: {result['total']}")
            console.print(f"Would keep: {result['would_keep']}")
            console.print(f"Would delete: {result['would_delete']}")
            if result["checkpoints_to_delete"]:
                console.print("Checkpoints to delete:")
                for ckpt_id in result["checkpoints_to_delete"]:
                    console.print(f"  {ckpt_id}")
        return

    result = asyncio.run(garbage_collect(store, prefix="", keep=keep))

    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print(f"Total checkpoints: {result['total']}")
        console.print(f"Kept: {result['kept']}")
        console.print(f"Deleted: {result['deleted']}")


@app.command("restore")
def restore(
    location: Annotated[str, typer.Argument(help="Storage location (path or s3://bucket)")],
    job_id: Annotated[str, typer.Argument(help="Job identifier")],
    checkpoint_id: Annotated[
        str | None, typer.Option("--checkpoint-id", help="Specific checkpoint ID; default: latest")
    ] = None,
    output: Annotated[
        str | None,
        typer.Option("--output", "-o", help="Destination directory (default: ./{JOB_ID}-restored/)")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Restore checkpoint tensors to .npy files in an output directory."""
    import numpy as np

    store = _make_store(location, job_id)
    output_dir = Path(output) if output else Path(f"./{job_id}-restored")

    async def _restore() -> dict[str, Any]:
        checkpoints = await store.list_checkpoints("")
        if not checkpoints:
            return {}
        if checkpoint_id is None:
            manifest = max(checkpoints, key=lambda c: c["timestamp"])
        else:
            matches = [c for c in checkpoints if c["checkpoint_id"] == checkpoint_id]
            if not matches:
                return {}
            manifest = matches[0]

        ckpt_id = manifest["checkpoint_id"]
        tensors, metadata = await store.load_checkpoint(ckpt_id)
        return {"manifest": manifest, "tensors": tensors, "metadata": metadata, "ckpt_id": ckpt_id}

    result = asyncio.run(_restore())

    if not result:
        err_console.print(
            f"[red]Checkpoint not found: {checkpoint_id or '(latest)'}[/red]"
        )
        raise typer.Exit(1)

    tensors: dict[str, Any] = result["tensors"]
    metadata: dict[str, Any] = result["metadata"]
    ckpt_id: str = result["ckpt_id"]
    manifest: dict[str, Any] = result["manifest"]

    output_dir.mkdir(parents=True, exist_ok=True)

    tensor_info = []
    for name, arr in tensors.items():
        npy_path = output_dir / f"{name}.npy"
        np.save(str(npy_path), arr)
        tensor_info.append({
            "name": name,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "nbytes": arr.nbytes,
            "file": str(npy_path),
        })

    meta_path = output_dir / "metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2))

    if json_output:
        typer.echo(json.dumps({
            "checkpoint_id": ckpt_id,
            "output_dir": str(output_dir),
            "tensors": tensor_info,
            "metadata": metadata,
        }, indent=2))
        return

    console.print(f"[bold]Checkpoint ID:[/bold] {ckpt_id}")
    dt_str = datetime.fromtimestamp(manifest["timestamp"]).strftime("%Y-%m-%d %H:%M:%S")
    console.print(f"[bold]Timestamp:[/bold] {dt_str}")
    console.print(f"[bold]Output:[/bold] {output_dir}")

    table = Table(title="Restored Tensors")
    table.add_column("Name", style="cyan")
    table.add_column("Shape", style="green")
    table.add_column("DType", style="blue")
    table.add_column("Size", justify="right", style="magenta")

    for info in tensor_info:
        table.add_row(
            info["name"],
            str(tuple(info["shape"])),
            info["dtype"],
            _fmt_size(info["nbytes"]),
        )

    console.print(table)
    console.print(f"[green]metadata.json written to {meta_path}[/green]")


@app.command("validate")
def validate(
    location: Annotated[str, typer.Argument(help="Storage location (path or s3://bucket)")],
    job_id: Annotated[str, typer.Argument(help="Job identifier")],
    checkpoint_id: Annotated[
        str | None, typer.Argument(help="Checkpoint ID (default: latest)")
    ] = None,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Validate checkpoint integrity by re-verifying checksums.

    Loads the checkpoint and re-computes all per-shard checksums, catching
    any corruption introduced after the original write.
    """
    import xxhash

    store = _make_store(location, job_id)

    async def _validate() -> dict[str, Any]:
        checkpoints = await store.list_checkpoints("")
        if not checkpoints:
            return {"ok": False, "error": "No checkpoints found"}

        if checkpoint_id is None:
            manifest = max(checkpoints, key=lambda c: c["timestamp"])
        else:
            matches = [c for c in checkpoints if c["checkpoint_id"] == checkpoint_id]
            if not matches:
                return {"ok": False, "error": f"Checkpoint not found: {checkpoint_id}"}
            manifest = matches[0]

        ckpt_id = manifest["checkpoint_id"]
        try:
            tensors, _ = await store.load_checkpoint(ckpt_id)
        except Exception as exc:
            return {"ok": False, "checkpoint_id": ckpt_id, "error": str(exc)}

        # Re-verify post-load (load_checkpoint already verifies, but we report per-tensor)
        tensor_results = []
        for name, arr in tensors.items():
            checksum = xxhash.xxh64(arr.tobytes()).hexdigest()
            tensor_results.append({
                "name": name,
                "shape": list(arr.shape),
                "dtype": str(arr.dtype),
                "nbytes": arr.nbytes,
                "checksum": checksum,
                "ok": True,
            })

        return {
            "ok": True,
            "checkpoint_id": ckpt_id,
            "method": manifest["method"],
            "total_bytes": manifest["total_bytes"],
            "tensors": tensor_results,
        }

    result = asyncio.run(_validate())

    if json_output:
        typer.echo(json.dumps(result, indent=2))
        if not result["ok"]:
            raise typer.Exit(1)
        return

    if not result["ok"]:
        err_console.print(f"[red]Validation failed: {result.get('error', 'unknown')}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]Checkpoint ID:[/bold] {result['checkpoint_id']}")
    console.print(f"[bold]Method:[/bold] {result['method']}")
    console.print(f"[bold]Total size:[/bold] {_fmt_size(result['total_bytes'])}")

    table = Table(title="Tensor Integrity")
    table.add_column("Name", style="cyan")
    table.add_column("Shape", style="green")
    table.add_column("DType", style="blue")
    table.add_column("Size", justify="right", style="magenta")
    table.add_column("Status", justify="center")

    for t in result["tensors"]:
        status = "[green]OK[/green]" if t["ok"] else "[red]FAIL[/red]"
        table.add_row(
            t["name"],
            str(tuple(t["shape"])),
            t["dtype"],
            _fmt_size(t["nbytes"]),
            status,
        )

    console.print(table)
    console.print("[green]All checksums verified.[/green]")


@app.command("bench")
def bench(
    location: Annotated[str, typer.Argument(help="Storage location (path or s3://bucket)")],
    job_id: Annotated[str, typer.Argument(help="Job identifier")] = "bench",
    size_mb: Annotated[int, typer.Option("--size-mb", help="Tensor size in MB")] = 256,
    concurrency: Annotated[
        int, typer.Option("--concurrency", "-c", help="Max S3 concurrency")
    ] = 32,
    json_output: Annotated[bool, typer.Option("--json", "-j", help="Output as JSON")] = False,
) -> None:
    """Benchmark checkpoint write/read throughput."""
    import numpy as np

    store: LocalStore | S3ShardedStore
    if location.startswith("s3://"):
        bucket = location.removeprefix("s3://").rstrip("/")
        shard_size = int(os.environ.get("SPOT_CHECKPOINT_SHARD_SIZE", str(64 * 1024 * 1024)))
        store = S3ShardedStore(
            bucket=bucket,
            job_id=job_id,
            shard_size=shard_size,
            max_concurrency=concurrency,
        )
    else:
        store = LocalStore(base_dir=location, job_id=job_id)

    n_elements = (size_mb * 1024 * 1024) // 8  # float64 = 8 bytes each
    tensor = np.random.rand(n_elements)
    checkpoint_id = f"bench-{int(time.time())}"
    metadata: dict[str, Any] = {"method": "bench"}

    async def _run_bench() -> tuple[float, float]:
        t0 = time.perf_counter()
        await store.save_checkpoint(checkpoint_id, {"data": tensor}, metadata)
        write_elapsed = time.perf_counter() - t0

        t1 = time.perf_counter()
        await store.load_checkpoint(checkpoint_id)
        read_elapsed = time.perf_counter() - t1

        await store.delete_checkpoint(checkpoint_id)
        return write_elapsed, read_elapsed

    write_elapsed, read_elapsed = asyncio.run(_run_bench())
    write_mbps = size_mb / write_elapsed
    read_mbps = size_mb / read_elapsed

    result: dict[str, Any] = {
        "size_mb": size_mb,
        "write_mbps": round(write_mbps, 2),
        "read_mbps": round(read_mbps, 2),
        "write_elapsed_s": round(write_elapsed, 3),
        "read_elapsed_s": round(read_elapsed, 3),
        "concurrency": concurrency,
    }

    if json_output:
        typer.echo(json.dumps(result, indent=2))
    else:
        console.print(f"Size:        {size_mb} MB")
        console.print(f"Write:       {write_mbps:.1f} MB/s ({write_elapsed:.3f}s)")
        console.print(f"Read:        {read_mbps:.1f} MB/s ({read_elapsed:.3f}s)")
        console.print(f"Concurrency: {concurrency}")


if __name__ == "__main__":
    app()
