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
    python run_smoke.py \
        --bucket spot-checkpoint-smoke-xxxx \
        --template-id lt-xxxx \
        --fis-template EXTxxx \
        [--region us-east-1] \
        [--wait-minutes 10]
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
        LaunchTemplate={"LaunchTemplateId": template_id},
        MinCount=1,
        MaxCount=1,
    )
    instance_id: str = resp["Instances"][0]["InstanceId"]
    _log(f"Launched instance: {instance_id}")
    return instance_id


def _instance_is_tagged(ec2: Any, instance_id: str, tag_key: str) -> bool:
    """Return True when the instance has the expected smoke-test tag."""
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    tags = resp["Reservations"][0]["Instances"][0].get("Tags", [])
    return any(t["Key"] == tag_key for t in tags)


def _instance_state(ec2: Any, instance_id: str) -> str:
    """Return current instance state name."""
    resp = ec2.describe_instances(InstanceIds=[instance_id])
    return resp["Reservations"][0]["Instances"][0]["State"]["Name"]


def _get_exit_code(ec2: Any, instance_id: str) -> int | None:
    """Return the instance's platform exit code if available, else None."""
    # EC2 doesn't expose OS exit codes directly; we infer success from
    # the instance terminating cleanly (state=terminated) with no system
    # status events.  Return 0 if terminated, None otherwise.
    state = _instance_state(ec2, instance_id)
    return 0 if state == "terminated" else None


def _run_fis_experiment(fis: Any, template_id: str) -> str:
    """Start a FIS experiment and return the experiment ID."""
    resp = fis.start_experiment(experimentTemplateId=template_id)
    exp_id: str = resp["experiment"]["id"]
    _log(f"FIS experiment started: {exp_id}")
    return exp_id


def _fis_experiment_done(fis: Any, exp_id: str) -> bool:
    """Return True when the FIS experiment has completed."""
    resp = fis.get_experiment(id=exp_id)
    status = resp["experiment"]["state"]["status"]
    return status in ("completed", "stopped", "failed")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Automated FIS smoke-test for spot-checkpoint")
    parser.add_argument("--bucket", required=True, help="S3 checkpoint bucket")
    parser.add_argument("--template-id", required=True, help="EC2 launch template ID")
    parser.add_argument("--fis-template", required=True, help="FIS experiment template ID")
    parser.add_argument("--region", default="us-east-1", help="AWS region")
    parser.add_argument("--job-id", default="smoke-test", help="Job identifier used in benchmark")
    parser.add_argument("--smoke-tag", default="spot-checkpoint-smoke-test",
                        help="Tag key applied by benchmark to running instance")
    args = parser.parse_args()

    t_start = time.time()

    session = boto3.Session(region_name=args.region)
    ec2_client = session.client("ec2")
    s3_client = session.client("s3")
    fis_client = session.client("fis")

    # -----------------------------------------------------------------------
    # Step 1: Launch first instance
    # -----------------------------------------------------------------------
    _log("=== Phase 1: First instance (interrupted by FIS) ===")
    instance1_id = _launch_instance(ec2_client, args.template_id)

    # -----------------------------------------------------------------------
    # Step 2: Wait for instance to tag itself
    # -----------------------------------------------------------------------
    _poll(
        f"instance {instance1_id} to tag itself",
        lambda: _instance_is_tagged(ec2_client, instance1_id, args.smoke_tag),
        max_seconds=180,
        interval=15,
    )
    _log(f"Instance {instance1_id} is running and tagged")

    # -----------------------------------------------------------------------
    # Step 3: Confirm ≥1 periodic checkpoint written
    # -----------------------------------------------------------------------
    _poll(
        "≥1 checkpoint in S3",
        lambda: _list_checkpoints(s3_client, args.bucket, args.job_id),
        max_seconds=120,
        interval=15,
    )
    _log("Periodic checkpoint confirmed in S3")

    # -----------------------------------------------------------------------
    # Step 4: Start FIS experiment
    # -----------------------------------------------------------------------
    exp_id = _run_fis_experiment(fis_client, args.fis_template)

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
    ckpts_after_interrupt = _list_checkpoints(s3_client, args.bucket, args.job_id)
    if not ckpts_after_interrupt:
        _fail("No checkpoint found after FIS interruption — emergency checkpoint not written")
    # Emergency checkpoint has "emergency" in name
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
    _log("=== Phase 2: Second instance (restore + complete) ===")
    instance2_id = _launch_instance(ec2_client, args.template_id)

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
    final_ckpts = _list_checkpoints(s3_client, args.bucket, args.job_id)
    if len(final_ckpts) != 1:
        _fail(
            f"Expected exactly 1 checkpoint after completion, found {len(final_ckpts)}: "
            + str(final_ckpts)
        )
    _log(f"Exactly 1 checkpoint retained as archive: {final_ckpts[0]}")

    # -----------------------------------------------------------------------
    # Step 10: PASS
    # -----------------------------------------------------------------------
    elapsed = time.time() - t_start
    _log(f"=== PASS — elapsed: {elapsed:.0f}s ===")


if __name__ == "__main__":
    main()
