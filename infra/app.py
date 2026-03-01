#!/usr/bin/env python3
"""CDK app entry point for the spot-checkpoint smoke-test stack."""

import aws_cdk as cdk
from smoke_stack import SpotCheckpointSmokeStack

app = cdk.App()

SpotCheckpointSmokeStack(
    app,
    "SpotCheckpointSmoke",
    env=cdk.Environment(
        account=app.node.try_get_context("account"),
        region=app.node.try_get_context("region") or "us-east-1",
    ),
    description="spot-checkpoint FIS smoke-test stack (issue #17)",
)

app.synth()
