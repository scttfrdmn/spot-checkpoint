"""
CDK stack: minimal AWS infrastructure for spot-checkpoint FIS smoke testing.

Resources created:
  - S3 bucket    — checkpoint storage (versioned, 7-day lifecycle)
  - IAM role     — EC2 instance profile with S3 + IMDS permissions
  - Launch template — c5.large spot, AL2023, IMDSv2 required, user-data script
  - 5 additional launch templates for adapter smoke tests (v0.12.0)
"""

from __future__ import annotations

import base64
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

#: Wheel version installed in all user-data scripts.
WHEEL_VERSION = "0.11.0"


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

        # Staging bucket — read wheel during install
        staging_bucket = s3.Bucket.from_bucket_name(
            self, "StagingBucket", "spot-checkpoint-staging-942542972736"
        )
        staging_bucket.grant_read(role)

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
        # User-data script (original fake-solver benchmark)
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
                # Install from staging wheel (pre-PyPI dev build)
                WHEEL_FILE="spot_checkpoint-{WHEEL_VERSION}-py3-none-any.whl"
                aws s3 cp "s3://spot-checkpoint-staging-942542972736/wheels/$WHEEL_FILE" "/tmp/$WHEEL_FILE"
                pip3.11 install --quiet "/tmp/$WHEEL_FILE[cli]"

                # ---- Write benchmark script (base64; bucket resolved at deploy time via env) ----
                export SPOT_CHECKPOINT_BUCKET={bucket.bucket_name}
                python3.11 -c "import base64; open('/root/run_benchmark.py','w').write(base64.b64decode('{base64.b64encode(_benchmark_script().encode()).decode()}').decode())"

                # ---- Run (foreground so the instance stays alive under FIS) ----
                python3.11 /root/run_benchmark.py
                # Self-terminate after successful completion (shutdown-behavior=terminate)
                shutdown -h now
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
            instance_initiated_shutdown_behavior=ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
            launch_template_name="spot-checkpoint-smoke",
        )

        # ------------------------------------------------------------------
        # Adapter smoke test launch templates (v0.12.0)
        # ------------------------------------------------------------------
        lt_numpy_dict = self._make_lt(
            lt_id="LaunchTemplateNumpyDict",
            lt_name="spot-checkpoint-smoke-numpy-dict",
            script_encoded=base64.b64encode(_benchmark_script_numpy_dict().encode()).decode(),
            extra_pip="",
            bucket_name=bucket.bucket_name,
            ami=ami,
            role=role,
        )

        lt_scipy_opt = self._make_lt(
            lt_id="LaunchTemplateScipyOpt",
            lt_name="spot-checkpoint-smoke-scipy-opt",
            script_encoded=base64.b64encode(_benchmark_script_scipy_opt().encode()).decode(),
            extra_pip="scipy",
            bucket_name=bucket.bucket_name,
            ami=ami,
            role=role,
        )

        lt_scipy_sparse = self._make_lt(
            lt_id="LaunchTemplateScipySparse",
            lt_name="spot-checkpoint-smoke-scipy-sparse",
            script_encoded=base64.b64encode(_benchmark_script_scipy_sparse().encode()).decode(),
            extra_pip="scipy",
            bucket_name=bucket.bucket_name,
            ami=ami,
            role=role,
        )

        lt_torch = self._make_lt(
            lt_id="LaunchTemplateTorch",
            lt_name="spot-checkpoint-smoke-torch",
            script_encoded=base64.b64encode(_benchmark_script_torch().encode()).decode(),
            extra_pip="torch --index-url https://download.pytorch.org/whl/cpu",
            bucket_name=bucket.bucket_name,
            ami=ami,
            role=role,
        )

        lt_openmm = self._make_lt(
            lt_id="LaunchTemplateOpenMM",
            lt_name="spot-checkpoint-smoke-openmm",
            script_encoded=base64.b64encode(_benchmark_script_openmm().encode()).decode(),
            extra_pip="openmm",
            bucket_name=bucket.bucket_name,
            ami=ami,
            role=role,
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
        cdk.CfnOutput(self, "LaunchTemplateIdNumpyDict",
                      value=lt_numpy_dict.launch_template_id or "",
                      description="Launch template ID for NumpyDictAdapter smoke test")
        cdk.CfnOutput(self, "LaunchTemplateIdScipyOpt",
                      value=lt_scipy_opt.launch_template_id or "",
                      description="Launch template ID for ScipyOptimizeAdapter smoke test")
        cdk.CfnOutput(self, "LaunchTemplateIdScipySparse",
                      value=lt_scipy_sparse.launch_template_id or "",
                      description="Launch template ID for ScipySparseLinalgAdapter smoke test")
        cdk.CfnOutput(self, "LaunchTemplateIdTorch",
                      value=lt_torch.launch_template_id or "",
                      description="Launch template ID for PyTorchTrainingAdapter smoke test")
        cdk.CfnOutput(self, "LaunchTemplateIdOpenMM",
                      value=lt_openmm.launch_template_id or "",
                      description="Launch template ID for OpenMMAdapter smoke test")

    def _make_lt(
        self,
        lt_id: str,
        lt_name: str,
        script_encoded: str,
        extra_pip: str,
        bucket_name: str,
        ami: ec2.IMachineImage,
        role: iam.Role,
        instance_type: str = "c5.large",
    ) -> ec2.LaunchTemplate:
        """Create a parameterized launch template for an adapter smoke test.

        Args:
            lt_id: CDK construct ID.
            lt_name: EC2 launch template name.
            script_encoded: Base64-encoded benchmark Python script.
            extra_pip: Additional pip packages to install (empty string = none).
            bucket_name: S3 bucket name token (resolved at deploy time).
            ami: Machine image to use.
            role: IAM role for the instance profile.
            instance_type: EC2 instance type (default c5.large).

        Returns:
            The created LaunchTemplate construct.
        """
        pip_line = f"pip3.11 install --quiet {extra_pip}" if extra_pip else ""
        ud = ec2.UserData.for_linux()
        ud.add_commands(
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

                # ---- Install spot-checkpoint + adapter deps ----
                dnf install -y python3.11 python3.11-pip git
                WHEEL_FILE="spot_checkpoint-{WHEEL_VERSION}-py3-none-any.whl"
                aws s3 cp "s3://spot-checkpoint-staging-942542972736/wheels/$WHEEL_FILE" "/tmp/$WHEEL_FILE"
                pip3.11 install --quiet "/tmp/$WHEEL_FILE[cli]"
                {pip_line}

                # ---- Write + run benchmark script ----
                export SPOT_CHECKPOINT_BUCKET={bucket_name}
                python3.11 -c "import base64; open('/root/run_benchmark.py','w').write(base64.b64decode('{script_encoded}').decode())"
                python3.11 /root/run_benchmark.py
                shutdown -h now
            """),
        )
        return ec2.LaunchTemplate(
            self,
            lt_id,
            instance_type=ec2.InstanceType(instance_type),
            machine_image=ami,
            role=role,
            user_data=ud,
            require_imdsv2=True,
            spot_options=ec2.LaunchTemplateSpotOptions(
                request_type=ec2.SpotRequestType.ONE_TIME,
            ),
            instance_initiated_shutdown_behavior=ec2.InstanceInitiatedShutdownBehavior.TERMINATE,
            launch_template_name=lt_name,
        )


def _benchmark_script() -> str:
    """Return the Python benchmark script embedded in user-data.

    The bucket name is read at runtime from the SPOT_CHECKPOINT_BUCKET env var,
    which is set in user-data after CloudFormation resolves the bucket name token.
    """
    return dedent("""\
        \"\"\"
        Smoke-test benchmark: fake iterative solver with spot-checkpoint.

        Runs 60 iterations of a trivial computation, checkpointing every 15 s.
        On spot interruption DirectEC2Backend fires an emergency checkpoint and
        the instance exits cleanly.  After restart the manager restores from the
        latest checkpoint and resumes from where it left off.
        \"\"\"
        import asyncio
        import logging
        import os
        import time

        import numpy as np

        from spot_checkpoint import spot_complete
        from spot_checkpoint.lifecycle import SpotLifecycleManager, detect_backend
        from spot_checkpoint.protocol import Checkpointable, CheckpointPayload
        from spot_checkpoint.storage import S3ShardedStore

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        log = logging.getLogger("smoke")

        BUCKET = os.environ["SPOT_CHECKPOINT_BUCKET"]
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
                    tensors={"value": self._solver.value.copy()},
                    metadata={"iteration": self._solver.iteration},
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


def _benchmark_script_numpy_dict() -> str:
    """Benchmark script for NumpyDictAdapter: gradient descent on quadratic."""
    return dedent("""\
        \"\"\"
        Smoke-test benchmark: NumpyDictAdapter — gradient descent on quadratic f(x).

        Runs 200 iterations of gradient descent on f(x) = 0.5*x'Ax - b'x (n=5000, SPD).
        Each iteration sleeps 0.5 s → ~100 s total; periodic checkpoints every 15 s
        ensure >= 6 checkpoints before FIS fires the 2-min interruption notice.
        \"\"\"
        import asyncio
        import logging
        import os
        import time

        import numpy as np

        from spot_checkpoint import spot_complete
        from spot_checkpoint.adapters.numpy_dict import NumpyDictAdapter
        from spot_checkpoint.lifecycle import SpotLifecycleManager, detect_backend
        from spot_checkpoint.storage import S3ShardedStore

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        log = logging.getLogger("smoke-numpy-dict")

        BUCKET = os.environ["SPOT_CHECKPOINT_BUCKET"]
        JOB_ID = "smoke-numpy-dict"
        TOTAL_ITERS = 200
        ITER_SLEEP = 0.5


        class GradientDescentSolver:
            def __init__(self, n=5000):
                rng = np.random.default_rng(0)
                Q = rng.standard_normal((n, n))
                self.A = Q.T @ Q / n + np.eye(n)
                self.b = rng.standard_normal(n)
                self.x = np.zeros(n)
                self.grad = np.zeros(n)
                self.loss = float("inf")
                self.iteration = 0
                self._lr = 0.01

            def step(self):
                self.grad = self.A @ self.x - self.b
                self.x -= self._lr * self.grad
                self.loss = 0.5 * float(self.x @ self.A @ self.x) - float(self.b @ self.x)
                self.iteration += 1


        def main():
            solver = GradientDescentSolver()
            adapter = NumpyDictAdapter(
                get_state=lambda: {"x": solver.x.copy(), "grad": solver.grad.copy()},
                set_state=lambda s: (
                    setattr(solver, "x", s["x"].copy()),
                    setattr(solver, "grad", s["grad"].copy()),
                ),
                get_metadata=lambda: {"iteration": solver.iteration, "loss": solver.loss},
                set_metadata=lambda m: (
                    setattr(solver, "iteration", int(m["iteration"])),
                    setattr(solver, "loss", float(m.get("loss", "inf"))),
                ),
                method="gradient-descent",
            )
            store = S3ShardedStore(bucket=BUCKET, job_id=JOB_ID)
            backend = detect_backend()
            mgr = SpotLifecycleManager(
                store=store,
                adapter=adapter,
                backend=backend,
                periodic_interval=15.0,
                checkpoint_id_prefix=JOB_ID,
                keep_checkpoints=3,
            )

            asyncio.run(mgr.restore_latest())

            with mgr:
                start_iter = solver.iteration
                log.info("Starting from iteration %d / %d (loss=%.6f)",
                         start_iter, TOTAL_ITERS, solver.loss)

                for i in range(start_iter, TOTAL_ITERS):
                    solver.step()
                    mgr.check(i)
                    if i % 20 == 0:
                        log.info("iter=%d loss=%.6f |grad|=%.6f",
                                 solver.iteration, solver.loss,
                                 float(np.linalg.norm(solver.grad)))
                    time.sleep(ITER_SLEEP)

                log.info("Complete — final loss: %.6f", solver.loss)
                spot_complete(bucket=BUCKET, job_id=JOB_ID, keep=1)
                log.info("NumpyDictAdapter smoke test PASSED — 1 checkpoint retained")


        if __name__ == "__main__":
            main()
    """)


def _benchmark_script_scipy_opt() -> str:
    """Benchmark script for ScipyOptimizeAdapter: L-BFGS-B on Rosenbrock."""
    return dedent("""\
        \"\"\"
        Smoke-test benchmark: ScipyOptimizeAdapter — L-BFGS-B on extended Rosenbrock.

        Slices minimize() into maxiter=50 sub-calls so mgr.check() fires between
        slices.  FIS fires during a minimize() call; SIGTERM triggers emergency ckpt.
        \"\"\"
        import asyncio
        import logging
        import os
        import time

        import numpy as np
        import scipy.optimize

        from spot_checkpoint import spot_complete
        from spot_checkpoint.adapters.scipy_opt import ScipyOptimizeAdapter
        from spot_checkpoint.lifecycle import SpotLifecycleManager, detect_backend
        from spot_checkpoint.storage import S3ShardedStore

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        log = logging.getLogger("smoke-scipy-opt")

        BUCKET = os.environ["SPOT_CHECKPOINT_BUCKET"]
        JOB_ID = "smoke-scipy-opt"
        MAX_ITER = 2000
        N = 1000


        def rosenbrock(x):
            \"\"\"Extended Rosenbrock function on R^n.\"\"\"
            return float(np.sum(100.0 * (x[1:] - x[:-1] ** 2) ** 2 + (1.0 - x[:-1]) ** 2))


        def rosenbrock_grad(x):
            \"\"\"Gradient of extended Rosenbrock.\"\"\"
            g = np.zeros_like(x)
            g[:-1] += -400.0 * x[:-1] * (x[1:] - x[:-1] ** 2) - 2.0 * (1.0 - x[:-1])
            g[1:] += 200.0 * (x[1:] - x[:-1] ** 2)
            return g


        def main():
            adapter = ScipyOptimizeAdapter(x0=np.zeros(N), method="L-BFGS-B")
            store = S3ShardedStore(bucket=BUCKET, job_id=JOB_ID)
            backend = detect_backend()
            mgr = SpotLifecycleManager(
                store=store,
                adapter=adapter,
                backend=backend,
                periodic_interval=15.0,
                checkpoint_id_prefix=JOB_ID,
                keep_checkpoints=3,
            )

            asyncio.run(mgr.restore_latest())

            with mgr:
                log.info("Starting from iteration %d (fun=%s)",
                         adapter.iteration, adapter.fun)
                converged = False

                def combined_callback(xk):
                    adapter.callback(xk)
                    mgr.check(adapter.iteration)
                    if adapter.iteration % 50 == 0:
                        log.info("iter=%d fun=%.6e", adapter.iteration,
                                 rosenbrock(adapter.x))

                while not converged and adapter.iteration < MAX_ITER:
                    result = scipy.optimize.minimize(
                        rosenbrock,
                        adapter.x,
                        jac=rosenbrock_grad,
                        method="L-BFGS-B",
                        callback=combined_callback,
                        options={"maxiter": 50},
                    )
                    adapter.fun = float(result.fun)
                    if result.success:
                        converged = True
                        log.info("Converged at iteration %d, fun=%.6e",
                                 adapter.iteration, adapter.fun)
                    elif result.status == 1:
                        # maxiter reached — continue outer loop
                        pass
                    else:
                        log.warning("Optimizer stopped: %s", result.message)
                        break
                    # Throttle so periodic checkpoints have time to fire
                    time.sleep(0.05)

                log.info("Complete — iteration=%d converged=%s", adapter.iteration, converged)
                spot_complete(bucket=BUCKET, job_id=JOB_ID, keep=1)
                log.info("ScipyOptimizeAdapter smoke test PASSED — 1 checkpoint retained")


        if __name__ == "__main__":
            main()
    """)


def _benchmark_script_scipy_sparse() -> str:
    """Benchmark script for ScipySparseLinalgAdapter: CG on tridiagonal SPD."""
    return dedent("""\
        \"\"\"
        Smoke-test benchmark: ScipySparseLinalgAdapter — CG on tridiagonal SPD system.

        n=20000 tridiagonal system Ax=b; callback-driven periodic checkpoints.
        Each CG call limited to maxiter=50 so mgr.check() fires between batches.
        \"\"\"
        import asyncio
        import logging
        import os
        import time

        import numpy as np
        import scipy.sparse
        import scipy.sparse.linalg

        from spot_checkpoint import spot_complete
        from spot_checkpoint.adapters.scipy_opt import ScipySparseLinalgAdapter
        from spot_checkpoint.lifecycle import SpotLifecycleManager, detect_backend
        from spot_checkpoint.storage import S3ShardedStore

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        log = logging.getLogger("smoke-scipy-sparse")

        BUCKET = os.environ["SPOT_CHECKPOINT_BUCKET"]
        JOB_ID = "smoke-scipy-sparse"
        N = 20000
        MAX_ITER = 2000


        def build_system(n):
            \"\"\"Build tridiagonal SPD matrix: 2 on diagonal, -1 on off-diagonals.\"\"\"
            diags = [np.full(n, 2.0), np.full(n - 1, -1.0), np.full(n - 1, -1.0)]
            A = scipy.sparse.diags(diags, [0, -1, 1], format="csr")
            rng = np.random.default_rng(42)
            b = rng.standard_normal(n)
            return A, b


        def main():
            A, b = build_system(N)
            adapter = ScipySparseLinalgAdapter(x0=np.zeros(N))
            store = S3ShardedStore(bucket=BUCKET, job_id=JOB_ID)
            backend = detect_backend()
            mgr = SpotLifecycleManager(
                store=store,
                adapter=adapter,
                backend=backend,
                periodic_interval=15.0,
                checkpoint_id_prefix=JOB_ID,
                keep_checkpoints=3,
            )

            asyncio.run(mgr.restore_latest())

            with mgr:
                log.info("Starting from iteration %d", adapter.iteration)
                converged = False

                def combined_callback(xk):
                    adapter.callback(xk)
                    mgr.check(adapter.iteration)
                    if adapter.iteration % 100 == 0:
                        residual = float(np.linalg.norm(A @ xk - b))
                        log.info("iter=%d residual=%.6e", adapter.iteration, residual)

                while not converged and adapter.iteration < MAX_ITER:
                    x, info = scipy.sparse.linalg.cg(
                        A, b, x0=adapter.x,
                        callback=combined_callback,
                        maxiter=50,
                    )
                    if info == 0:
                        converged = True
                        log.info("CG converged at iteration %d", adapter.iteration)
                    elif adapter.iteration >= MAX_ITER:
                        log.warning("Max iterations reached without convergence")
                        break
                    # else: continue from adapter.x (updated by callback)
                    time.sleep(0.02)

                log.info("Complete — iteration=%d converged=%s", adapter.iteration, converged)
                spot_complete(bucket=BUCKET, job_id=JOB_ID, keep=1)
                log.info("ScipySparseLinalgAdapter smoke test PASSED — 1 checkpoint retained")


        if __name__ == "__main__":
            main()
    """)


def _benchmark_script_torch() -> str:
    """Benchmark script for PyTorchTrainingAdapter: MLP training (CPU)."""
    return dedent("""\
        \"\"\"
        Smoke-test benchmark: PyTorchTrainingAdapter — 3-layer MLP on synthetic data (CPU).

        Trains for 2000 steps x 0.1s = ~200s; periodic checkpoints every 15s.
        CPU-only torch (~300 MB install, ~4 min); tag_poll_seconds must be >= 600.
        \"\"\"
        import asyncio
        import logging
        import os
        import time

        import numpy as np
        import torch
        import torch.nn as nn

        from spot_checkpoint import spot_complete
        from spot_checkpoint.adapters.torch import PyTorchTrainingAdapter
        from spot_checkpoint.lifecycle import SpotLifecycleManager, detect_backend
        from spot_checkpoint.storage import S3ShardedStore

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        log = logging.getLogger("smoke-torch")

        BUCKET = os.environ["SPOT_CHECKPOINT_BUCKET"]
        JOB_ID = "smoke-torch"
        TOTAL_STEPS = 2000
        STEP_SLEEP = 0.1


        def main():
            model = nn.Sequential(
                nn.Linear(64, 128), nn.ReLU(),
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, 1),
            )
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
            adapter = PyTorchTrainingAdapter(model, optimizer, step=0)
            store = S3ShardedStore(bucket=BUCKET, job_id=JOB_ID)
            backend = detect_backend()
            mgr = SpotLifecycleManager(
                store=store,
                adapter=adapter,
                backend=backend,
                periodic_interval=15.0,
                checkpoint_id_prefix=JOB_ID,
                keep_checkpoints=3,
            )

            asyncio.run(mgr.restore_latest())
            start_step = adapter.step
            log.info("Starting from step %d / %d", start_step, TOTAL_STEPS)

            rng = np.random.default_rng(42)

            with mgr:
                for step in range(start_step, TOTAL_STEPS):
                    x = torch.from_numpy(rng.standard_normal((64, 64)).astype(np.float32))
                    y = torch.from_numpy(rng.standard_normal((64, 1)).astype(np.float32))
                    optimizer.zero_grad()
                    loss = nn.functional.mse_loss(model(x), y)
                    loss.backward()
                    optimizer.step()
                    adapter.step = step
                    adapter.loss = float(loss.item())
                    mgr.check(step)
                    if step % 200 == 0:
                        log.info("step=%d loss=%.6f", step, adapter.loss)
                    time.sleep(STEP_SLEEP)

                log.info("Complete — final loss: %.6f", adapter.loss)
                spot_complete(bucket=BUCKET, job_id=JOB_ID, keep=1)
                log.info("PyTorchTrainingAdapter smoke test PASSED — 1 checkpoint retained")


        if __name__ == "__main__":
            main()
    """)


def _benchmark_script_openmm() -> str:
    """Benchmark script for OpenMMAdapter: LJ argon fluid simulation."""
    return dedent("""\
        \"\"\"
        Smoke-test benchmark: OpenMMAdapter — 512-particle LJ argon fluid (Langevin NVT).

        Topology and force field built programmatically — no external PDB files needed.
        2000 blocks x simulation.step(100) x ~0.1s/block = ~200s total.
        \"\"\"
        import asyncio
        import logging
        import math
        import os
        import time

        import numpy as np
        import openmm
        import openmm.app
        import openmm.unit as unit

        from spot_checkpoint import spot_complete
        from spot_checkpoint.adapters.openmm import OpenMMAdapter
        from spot_checkpoint.lifecycle import SpotLifecycleManager, detect_backend
        from spot_checkpoint.storage import S3ShardedStore

        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
        log = logging.getLogger("smoke-openmm")

        BUCKET = os.environ["SPOT_CHECKPOINT_BUCKET"]
        JOB_ID = "smoke-openmm"
        N_ATOMS = 512
        TOTAL_BLOCKS = 2000
        STEPS_PER_BLOCK = 100


        def build_argon_simulation(n_atoms=N_ATOMS):
            \"\"\"Build an LJ argon simulation programmatically (no external files).\"\"\"
            # System
            system = openmm.System()
            box_edge = 3.5 * unit.nanometer
            system.setDefaultPeriodicBoxVectors(
                openmm.Vec3(box_edge / unit.nanometer, 0, 0),
                openmm.Vec3(0, box_edge / unit.nanometer, 0),
                openmm.Vec3(0, 0, box_edge / unit.nanometer),
            )
            for _ in range(n_atoms):
                system.addParticle(39.948 * unit.amu)

            # LJ non-bonded force (argon parameters)
            nbforce = openmm.NonbondedForce()
            sigma = 0.3405 * unit.nanometer
            epsilon = 0.010323 * unit.kilojoule_per_mole
            for _ in range(n_atoms):
                nbforce.addParticle(0.0, sigma, epsilon)
            nbforce.setNonbondedMethod(openmm.NonbondedForce.CutoffPeriodic)
            nbforce.setCutoffDistance(1.2 * unit.nanometer)
            system.addForce(nbforce)

            # Integrator
            integrator = openmm.LangevinMiddleIntegrator(
                300 * unit.kelvin,
                1.0 / unit.picosecond,
                0.004 * unit.picosecond,
            )

            # Minimal topology (N atoms, no bonds)
            topology = openmm.app.Topology()
            chain = topology.addChain()
            for i in range(n_atoms):
                res = topology.addResidue(f"AR{i}", chain)
                topology.addAtom("AR", openmm.app.element.argon, res)

            platform = openmm.Platform.getPlatformByName("CPU")
            simulation = openmm.app.Simulation(topology, system, integrator, platform)

            # Place atoms on approximate FCC grid
            cell_size = box_edge / unit.nanometer
            atoms_per_side = math.ceil(n_atoms ** (1.0 / 3.0))
            spacing = cell_size / atoms_per_side
            positions = []
            count = 0
            for ix in range(atoms_per_side):
                for iy in range(atoms_per_side):
                    for iz in range(atoms_per_side):
                        if count >= n_atoms:
                            break
                        positions.append([ix * spacing, iy * spacing, iz * spacing])
                        count += 1
                    if count >= n_atoms:
                        break
                if count >= n_atoms:
                    break
            simulation.context.setPositions(
                [openmm.Vec3(p[0], p[1], p[2]) for p in positions]
            )
            simulation.context.setVelocitiesToTemperature(300 * unit.kelvin)
            return simulation


        def main():
            simulation = build_argon_simulation()
            adapter = OpenMMAdapter(simulation)
            store = S3ShardedStore(bucket=BUCKET, job_id=JOB_ID)
            backend = detect_backend()
            mgr = SpotLifecycleManager(
                store=store,
                adapter=adapter,
                backend=backend,
                periodic_interval=15.0,
                checkpoint_id_prefix=JOB_ID,
                keep_checkpoints=3,
            )

            asyncio.run(mgr.restore_latest())

            # Determine start block from adapter metadata
            state = simulation.context.getState(getPositions=True)
            start_block = int(
                simulation.context.getStepCount() // STEPS_PER_BLOCK
            )
            log.info("Starting from block %d / %d", start_block, TOTAL_BLOCKS)

            with mgr:
                for block in range(start_block, TOTAL_BLOCKS):
                    simulation.step(STEPS_PER_BLOCK)
                    mgr.check(block)
                    if block % 200 == 0:
                        state = simulation.context.getState(getEnergy=True)
                        pe = state.getPotentialEnergy().value_in_unit(
                            unit.kilojoule_per_mole
                        )
                        log.info("block=%d PE=%.2f kJ/mol", block, pe)

                log.info("Simulation complete — %d blocks", TOTAL_BLOCKS)
                spot_complete(bucket=BUCKET, job_id=JOB_ID, keep=1)
                log.info("OpenMMAdapter smoke test PASSED — 1 checkpoint retained")


        if __name__ == "__main__":
            main()
    """)
