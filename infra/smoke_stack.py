"""
CDK stack: minimal AWS infrastructure for spot-checkpoint FIS smoke testing.

Resources created:
  - S3 bucket    — checkpoint storage (versioned, 7-day lifecycle)
  - IAM role     — EC2 instance profile with S3 + IMDS permissions
  - Launch template — c5.large spot, AL2023, IMDSv2 required, user-data script
"""

from __future__ import annotations

import json
from textwrap import dedent

import aws_cdk as cdk
import aws_cdk.aws_ec2 as ec2
import aws_cdk.aws_iam as iam
import aws_cdk.aws_s3 as s3
from constructs import Construct

#: Tag applied to running instances so the FIS experiment can target them.
SMOKE_TEST_TAG = "spot-checkpoint-smoke-test"

#: SSM parameter path for the latest AL2023 x86_64 AMI.
AL2023_SSM_PARAM = "/aws/service/ami-amazon-linux-latest/al2023-ami-kernel-default-x86_64"


class SpotCheckpointSmokeStack(cdk.Stack):
    """Minimal infrastructure for a real-world spot interruption smoke test."""

    def __init__(self, scope: Construct, construct_id: str, **kwargs: object) -> None:
        super().__init__(scope, construct_id, **kwargs)  # type: ignore[arg-type]

        # ------------------------------------------------------------------
        # S3 checkpoint bucket
        # ------------------------------------------------------------------
        bucket = s3.Bucket(
            self,
            "CheckpointBucket",
            versioned=True,
            removal_policy=cdk.RemovalPolicy.DESTROY,
            auto_delete_objects=True,
            lifecycle_rules=[
                s3.LifecycleRule(
                    id="expire-old-checkpoints",
                    enabled=True,
                    expiration=cdk.Duration.days(7),
                    noncurrent_version_expiration=cdk.Duration.days(1),
                )
            ],
            block_public_access=s3.BlockPublicAccess.BLOCK_ALL,
            encryption=s3.BucketEncryption.S3_MANAGED,
        )

        # ------------------------------------------------------------------
        # IAM instance role
        # ------------------------------------------------------------------
        role = iam.Role(
            self,
            "InstanceRole",
            assumed_by=iam.ServicePrincipal("ec2.amazonaws.com"),
            description="spot-checkpoint smoke-test instance role",
        )

        # Checkpoint bucket access
        bucket.grant_read_write(role)
        bucket.grant_delete(role)

        # Self-identification and self-tagging (DirectEC2Backend + smoke-test tag)
        role.add_to_policy(
            iam.PolicyStatement(
                sid="EC2SelfManage",
                actions=["ec2:DescribeInstances", "ec2:CreateTags"],
                resources=["*"],
            )
        )

        # SSM read-only (useful for remote access via Session Manager)
        role.add_managed_policy(
            iam.ManagedPolicy.from_aws_managed_policy_name(
                "AmazonSSMManagedInstanceCore"
            )
        )

        instance_profile = iam.CfnInstanceProfile(
            self,
            "InstanceProfile",
            roles=[role.role_name],
        )

        # ------------------------------------------------------------------
        # AMI — latest AL2023 resolved via SSM parameter at synth time
        # ------------------------------------------------------------------
        ami = ec2.MachineImage.from_ssm_parameter(
            AL2023_SSM_PARAM,
            os=ec2.OperatingSystemType.LINUX,
        )

        # ------------------------------------------------------------------
        # User-data script
        # ------------------------------------------------------------------
        user_data = ec2.UserData.for_linux()
        user_data.add_commands(
            dedent(f"""\
                #!/bin/bash
                set -euxo pipefail
                exec > >(tee /var/log/spot-checkpoint-smoke.log) 2>&1

                # ---- Tag this instance so FIS can target it ----
                INSTANCE_ID=$(TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \\
                    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600") && \\
                    curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \\
                    http://169.254.169.254/latest/meta-data/instance-id)
                REGION=$(TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \\
                    -H "X-aws-ec2-metadata-token-ttl-seconds: 21600") && \\
                    curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \\
                    http://169.254.169.254/latest/meta-data/placement/region)
                aws ec2 create-tags --region "$REGION" --resources "$INSTANCE_ID" \\
                    --tags Key={SMOKE_TEST_TAG},Value=true

                # ---- Install dependencies ----
                dnf install -y python3.11 python3.11-pip git
                pip3.11 install --quiet "spot-checkpoint[cli]"

                # ---- Write benchmark script ----
                cat > /root/run_benchmark.py << 'PYSCRIPT'
                {_benchmark_script(bucket.bucket_name)}
                PYSCRIPT

                # ---- Run (foreground so the instance stays alive under FIS) ----
                python3.11 /root/run_benchmark.py
            """),
        )

        # ------------------------------------------------------------------
        # Launch template — c5.large spot, IMDSv2 required
        # ------------------------------------------------------------------
        launch_template = ec2.LaunchTemplate(
            self,
            "LaunchTemplate",
            instance_type=ec2.InstanceType("c5.large"),
            machine_image=ami,
            role=role,
            user_data=user_data,
            require_imdsv2=True,
            spot_options=ec2.LaunchTemplateSpotOptions(
                request_type=ec2.SpotRequestType.ONE_TIME,
            ),
            launch_template_name="spot-checkpoint-smoke",
        )

        # ------------------------------------------------------------------
        # CloudFormation outputs
        # ------------------------------------------------------------------
        cdk.CfnOutput(self, "BucketName", value=bucket.bucket_name,
                      description="Checkpoint S3 bucket")
        cdk.CfnOutput(self, "LaunchTemplateId", value=launch_template.launch_template_id or "",
                      description="Launch template ID for manual launch / FIS target")
        cdk.CfnOutput(self, "InstanceProfileArn", value=instance_profile.attr_arn,
                      description="IAM instance profile ARN")


def _benchmark_script(bucket_name: str) -> str:
    """Return the Python benchmark script embedded in user-data."""
    return dedent(f"""\
        \"\"\"
        Smoke-test benchmark: fake iterative solver with spot-checkpoint.

        Runs 200 iterations of a trivial computation, checkpointing every 30 s.
        On spot interruption DirectEC2Backend fires an emergency checkpoint and
        the instance exits cleanly.  After restart the manager restores from the
        latest checkpoint and resumes from where it left off.
        \"\"\"
        import asyncio
        import logging
        import time

        import numpy as np

        from spot_checkpoint import spot_complete
        from spot_checkpoint.lifecycle import SpotLifecycleManager, detect_backend
        from spot_checkpoint.protocol import Checkpointable, CheckpointPayload
        from spot_checkpoint.storage import S3ShardedStore

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        log = logging.getLogger("smoke")

        BUCKET = "{bucket_name}"
        JOB_ID = "smoke-test"
        TOTAL_ITERS = 60
        ITER_SLEEP = 0.5   # seconds per iteration — short for CI speed


        class FakeSolver:
            \"\"\"Trivial iterative solver that accumulates a sum.\"\"\"

            def __init__(self) -> None:
                self.iteration = 0
                self.value = np.zeros(1024, dtype=np.float64)   # 8 kB state

            def step(self) -> None:
                self.value += np.random.default_rng(self.iteration).random(1024)
                self.iteration += 1


        class FakeSolverAdapter(Checkpointable):
            def __init__(self, solver: FakeSolver) -> None:
                self._solver = solver

            def checkpoint_state(self) -> CheckpointPayload:
                return CheckpointPayload(
                    tensors={{"value": self._solver.value.copy()}},
                    metadata={{"iteration": self._solver.iteration}},
                    method="fake-solver",
                    timestamp=time.time(),
                )

            def restore_state(self, payload: CheckpointPayload) -> None:
                self._solver.value = payload.tensors["value"].copy()
                self._solver.iteration = int(payload.metadata.get("iteration", 0))
                log.info("Restored from iteration %d", self._solver.iteration)


        def main() -> None:
            solver = FakeSolver()
            adapter = FakeSolverAdapter(solver)
            store = S3ShardedStore(bucket=BUCKET, job_id=JOB_ID)
            backend = detect_backend()
            mgr = SpotLifecycleManager(
                store=store,
                adapter=adapter,
                backend=backend,
                periodic_interval=15.0,   # short for CI — ensures checkpoint before FIS fires
                checkpoint_id_prefix=JOB_ID,
                keep_checkpoints=3,
            )

            # Attempt restore from previous run
            asyncio.run(mgr.restore_latest())

            with mgr:
                start_iter = solver.iteration
                log.info("Starting from iteration %d / %d", start_iter, TOTAL_ITERS)

                for i in range(start_iter, TOTAL_ITERS):
                    solver.step()
                    mgr.check(i)
                    time.sleep(ITER_SLEEP)

                log.info("Benchmark complete — final value norm: %.6f",
                         float(np.linalg.norm(solver.value)))

                # Prune old checkpoints, keep latest as archive for inspection
                log.info("All %d iterations complete — pruning old checkpoints", TOTAL_ITERS)
                spot_complete(bucket=BUCKET, job_id=JOB_ID, keep=1)
                log.info("Smoke test PASSED — 1 checkpoint retained as archive")


        if __name__ == "__main__":
            main()
    """)
