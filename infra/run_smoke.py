"""
Automated FIS smoke-test runner for spot-checkpoint.

Runs the full two-instance scenario non-interactively:
  1.  Launch spot instance from launch template
  2.  Poll EC2 until instance is tagged (max 3 min, 15s intervals)
  3.  Confirm ≥1 checkpoint in S3 (at least one periodic checkpoint written)
  4.  Start FIS experiment
  5.  Poll until instance state = terminated (max 4 min)
  6.  Verify checkpoint manifest exists in S3 (emergency checkpoint written)
  7.  Launch second instance from same template (restore run)
  8.  Poll until second instance terminates cleanly (exit 0, max 5 min)
  9.  Verify exactly 1 checkpoint remains (spot_complete(keep=1) ran correctly)
  10. Print PASS / FAIL with timing summary

Usage:
    # Original fake-solver (backward-compatible):
    python run_smoke.py \\
        --bucket spot-checkpoint-smoke-xxxx \\
        --template-id lt-xxxx \\
        --fis-template EXTxxx

    # Single adapter:
    python run_smoke.py \\
        --bucket ... --fis-template ... \\
        --adapter numpy-dict \\
        --template-id-numpy-dict lt-xxxx

    # All adapters in sequence:
    python run_smoke.py \\
        --bucket ... --fis-template ... \\
        --adapter all \\
        --template-id-numpy-dict lt-xxx \\
        --template-id-scipy-opt lt-xxx \\
        --template-id-scipy-sparse lt-xxx \\
        --template-id-torch lt-xxx \\
        --template-id-openmm lt-xxx
"""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any

import boto3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _log(msg: str) -> None:
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _fail(reason: str) -> None:
    _log(f"FAIL — {reason}")
    sys.exit(1)


def _poll(
    description: str,
    check_fn: Any,
    max_seconds: float,
    interval: float = 15.0,
) -> Any:
    """Poll until check_fn() returns a truthy result or timeout expires.

    Args:
        description: Human-readable description for logging.
        check_fn: Callable returning the result on success, or falsy on not-yet-ready.
        max_seconds: Maximum time to wait in seconds.
        interval: Seconds between polls.

    Returns:
        The truthy result from check_fn.

    Raises:
        SystemExit: If timeout expires.
    """
    deadline = time.time() + max_seconds
    _log(f"Waiting for: {description} (max {max_seconds:.0f}s)")
    while time.time() < deadline:
        result = check_fn()
        if result:
            return result
        time.sleep(interval)
    _fail(f"Timeout waiting for: {description}")


def _list_checkpoints(s3: Any, bucket: str, job_id: str) -> list[str]:
    """Return list of checkpoint manifest keys for a job."""
    prefix = f"{job_id}/"
    paginator = s3.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key: str = obj["Key"]
            if key.endswith("/_manifest.json"):
                keys.append(key)
    return keys


def _launch_instance(ec2: Any, template_id: str) -> str:
    """Launch one spot instance and return its instance ID."""
    resp = ec2.run_instances(
        LaunchTemplate={"LaunchTemplateId": template_id, "Version": "$Latest"},
        MinCount=1,
        MaxCount=1,
    )
    instance_id: str = resp["Instances"][0]["InstanceId"]
    _log(f"Launched instance: {instance_id}")
    return instance_id


def _instance_is_tagged(ec2: Any, instance_id: str, tag_key: str) -> bool:
    """Return True when the instance has the expected smoke-test tag."""
    try:
        resp = ec2.describe_instances(InstanceIds=[instance_id])
    except Exception:
        return False  # instance not visible yet — retry
    reservations = resp.get("Reservations", [])
    if not reservations:
        return False
    tags = reservations[0]["Instances"][0].get("Tags", [])
    return any(t["Key"] == tag_key for t in tags)


def _instance_state(ec2: Any, instance_id: str) -> str:
    """Return current instance state name."""
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    return resp["Reservations"][0]["Instances"][0]["State"]["Name"]


def _run_fis_experiment(fis: Any, template_id: str) -> str:
    """Start a FIS experiment and return the experiment ID."""
    resp = fis.start_experiment(experimentTemplateId=template_id)
    exp_id: str = resp["experiment"]["id"]
    _log(f"FIS experiment started: {exp_id}")
    return exp_id


# ---------------------------------------------------------------------------
# Core smoke-test scenario
# ---------------------------------------------------------------------------

def _run_adapter_smoke(
    ec2_client: Any,
    s3_client: Any,
    fis_client: Any,
    bucket: str,
    template_id: str,
    fis_template: str,
    job_id: str,
    smoke_tag: str,
    tag_poll_seconds: float = 180,
) -> float:
    """Run the 10-step smoke scenario for one adapter.

    Args:
        ec2_client: boto3 EC2 client.
        s3_client: boto3 S3 client.
        fis_client: boto3 FIS client.
        bucket: S3 checkpoint bucket name.
        template_id: EC2 launch template ID.
        fis_template: FIS experiment template ID.
        job_id: Job identifier used in benchmark (S3 prefix).
        smoke_tag: Tag key applied by benchmark to the running instance.
        tag_poll_seconds: Max seconds to wait for instance tag (default 180).
            Use 600 for adapters with large install times (e.g. torch).

    Returns:
        Elapsed time in seconds.

    Raises:
        SystemExit: On any verification failure.
    """
    t_start = time.time()

    # -----------------------------------------------------------------------
    # Step 1: Launch first instance
    # -----------------------------------------------------------------------
    _log(f"=== [{job_id}] Phase 1: First instance (interrupted by FIS) ===")
    instance1_id = _launch_instance(ec2_client, template_id)

    # -----------------------------------------------------------------------
    # Step 2: Wait for instance to tag itself
    # -----------------------------------------------------------------------
    _poll(
        f"instance {instance1_id} to tag itself",
        lambda: _instance_is_tagged(ec2_client, instance1_id, smoke_tag),
        max_seconds=tag_poll_seconds,
        interval=15,
    )
    _log(f"Instance {instance1_id} is running and tagged")

    # -----------------------------------------------------------------------
    # Step 3: Confirm ≥1 periodic checkpoint written
    # -----------------------------------------------------------------------
    _poll(
        "≥1 checkpoint in S3",
        lambda: _list_checkpoints(s3_client, bucket, job_id),
        max_seconds=120,
        interval=15,
    )
    _log("Periodic checkpoint confirmed in S3")

    # -----------------------------------------------------------------------
    # Step 4: Start FIS experiment
    # -----------------------------------------------------------------------
    _run_fis_experiment(fis_client, fis_template)

    # -----------------------------------------------------------------------
    # Step 5: Wait for instance termination
    # -----------------------------------------------------------------------
    _poll(
        f"instance {instance1_id} to terminate",
        lambda: _instance_state(ec2_client, instance1_id) == "terminated",
        max_seconds=240,
        interval=15,
    )
    _log(f"Instance {instance1_id} terminated")

    # -----------------------------------------------------------------------
    # Step 6: Verify emergency checkpoint exists
    # -----------------------------------------------------------------------
    ckpts_after_interrupt = _list_checkpoints(s3_client, bucket, job_id)
    if not ckpts_after_interrupt:
        _fail("No checkpoint found after FIS interruption — emergency checkpoint not written")
    emergency_ckpts = [k for k in ckpts_after_interrupt if "emergency" in k]
    if not emergency_ckpts:
        _log(
            f"WARNING: no 'emergency' checkpoint found; {len(ckpts_after_interrupt)} "
            "checkpoint(s) exist (may be periodic only)"
        )
    else:
        _log(f"Emergency checkpoint confirmed: {emergency_ckpts[-1]}")

    # -----------------------------------------------------------------------
    # Step 7: Launch second instance (restore run)
    # -----------------------------------------------------------------------
    _log(f"=== [{job_id}] Phase 2: Second instance (restore + complete) ===")
    instance2_id = _launch_instance(ec2_client, template_id)

    # -----------------------------------------------------------------------
    # Step 8: Wait for second instance to terminate cleanly
    # -----------------------------------------------------------------------
    _poll(
        f"instance {instance2_id} to terminate",
        lambda: _instance_state(ec2_client, instance2_id) == "terminated",
        max_seconds=300,
        interval=15,
    )
    _log(f"Instance {instance2_id} terminated")

    # -----------------------------------------------------------------------
    # Step 9: Verify exactly 1 checkpoint remains (spot_complete ran)
    # -----------------------------------------------------------------------
    final_ckpts = _list_checkpoints(s3_client, bucket, job_id)
    if len(final_ckpts) != 1:
        _fail(
            f"Expected exactly 1 checkpoint after completion, found {len(final_ckpts)}: "
            + str(final_ckpts)
        )
    _log(f"Exactly 1 checkpoint retained as archive: {final_ckpts[0]}")

    elapsed = time.time() - t_start
    _log(f"=== [{job_id}] PASS — elapsed: {elapsed:.0f}s ===")
    return elapsed


# ---------------------------------------------------------------------------
# Adapter configuration table
# ---------------------------------------------------------------------------

#: Maps adapter name → (job_id, tag_poll_seconds)
_ADAPTER_DEFAULTS: dict[str, tuple[str, float]] = {
    "fake-solver": ("smoke-test", 180),
    "numpy-dict": ("smoke-numpy-dict", 180),
    "scipy-opt": ("smoke-scipy-opt", 180),
    "scipy-sparse": ("smoke-scipy-sparse", 180),
    "torch": ("smoke-torch", 600),
    "openmm": ("smoke-openmm", 180),
}

_ADAPTER_CHOICES = list(_ADAPTER_DEFAULTS.keys()) + ["all"]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Automated FIS smoke-test for spot-checkpoint")
    parser.add_argument("--bucket", required=True, help="S3 checkpoint bucket")
    parser.add_argument("--fis-template", required=True, help="FIS experiment template ID")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--smoke-tag", default="spot-checkpoint-smoke-test",
                        help="Tag key applied by benchmark to running instance")

    # Adapter selection
    parser.add_argument(
        "--adapter",
        choices=_ADAPTER_CHOICES,
        default="fake-solver",
        help="Which adapter smoke test to run (default: fake-solver for backward compat)",
    )

    # Original template-id (fake-solver)
    parser.add_argument("--template-id", default=None,
                        help="EC2 launch template ID for fake-solver benchmark")

    # Per-adapter template IDs (v0.12.0)
    parser.add_argument("--template-id-numpy-dict", default=None,
                        help="Launch template ID for NumpyDictAdapter smoke test")
    parser.add_argument("--template-id-scipy-opt", default=None,
                        help="Launch template ID for ScipyOptimizeAdapter smoke test")
    parser.add_argument("--template-id-scipy-sparse", default=None,
                        help="Launch template ID for ScipySparseLinalgAdapter smoke test")
    parser.add_argument("--template-id-torch", default=None,
                        help="Launch template ID for PyTorchTrainingAdapter smoke test")
    parser.add_argument("--template-id-openmm", default=None,
                        help="Launch template ID for OpenMMAdapter smoke test")

    args = parser.parse_args()

    # Map adapter name → template-id arg
    template_id_map: dict[str, str | None] = {
        "fake-solver": args.template_id,
        "numpy-dict": args.template_id_numpy_dict,
        "scipy-opt": args.template_id_scipy_opt,
        "scipy-sparse": args.template_id_scipy_sparse,
        "torch": args.template_id_torch,
        "openmm": args.template_id_openmm,
    }

    # Determine which adapters to run
    if args.adapter == "all":
        adapters_to_run = list(_ADAPTER_DEFAULTS.keys())
    else:
        adapters_to_run = [args.adapter]

    # Validate template IDs up front
    missing = []
    for adapter in adapters_to_run:
        if not template_id_map[adapter]:
            flag = "--template-id" if adapter == "fake-solver" else f"--template-id-{adapter}"
            missing.append(flag)
    if missing:
        parser.error(f"Missing required template IDs for selected adapters: {', '.join(missing)}")

    session = boto3.Session(region_name=args.region)
    ec2_client = session.client("ec2")
    s3_client = session.client("s3")
    fis_client = session.client("fis")

    results: list[tuple[str, str, float]] = []  # (adapter, pass/fail, elapsed)

    for adapter in adapters_to_run:
        job_id, tag_poll_seconds = _ADAPTER_DEFAULTS[adapter]
        tid = template_id_map[adapter]
        assert tid is not None  # validated above

        _log(f"\n{'='*60}")
        _log(f"Running smoke test: adapter={adapter} job_id={job_id}")
        _log(f"{'='*60}")

        try:
            elapsed = _run_adapter_smoke(
                ec2_client=ec2_client,
                s3_client=s3_client,
                fis_client=fis_client,
                bucket=args.bucket,
                template_id=tid,
                fis_template=args.fis_template,
                job_id=job_id,
                smoke_tag=args.smoke_tag,
                tag_poll_seconds=tag_poll_seconds,
            )
            results.append((adapter, "PASS", elapsed))
        except SystemExit:
            results.append((adapter, "FAIL", 0.0))
            if args.adapter != "all":
                sys.exit(1)
            # In 'all' mode: continue so we can print a summary

    # Print summary for multi-adapter runs
    if len(results) > 1:
        _log("\n" + "=" * 60)
        _log("SUMMARY")
        _log("=" * 60)
        any_fail = False
        for adapter, status, elapsed in results:
            marker = "✓" if status == "PASS" else "✗"
            elapsed_str = f"{elapsed:.0f}s" if elapsed else "—"
            _log(f"  {marker}  {adapter:<20} {status}  {elapsed_str}")
            if status == "FAIL":
                any_fail = True
        _log("=" * 60)
        if any_fail:
            sys.exit(1)


if __name__ == "__main__":
    main()
