# spot-checkpoint smoke-test infrastructure

Minimal CDK stack + FIS experiment for validating spot interruption
handling on a real AWS spot instance.

## What this creates

| Resource | Purpose |
|----------|---------|
| S3 bucket | Checkpoint storage (versioned, 7-day expiry) |
| IAM role + instance profile | S3 read/write/delete + SSM Session Manager |
| EC2 launch template | `c5.large` spot, AL2023, IMDSv2 required |
| FIS experiment template | Sends 2-min interruption notice, then terminates |

The launch template user-data installs `spot-checkpoint` and runs a
fake iterative solver (`/root/run_benchmark.py`) that checkpoints every
30 seconds.  On interruption the `DirectEC2Backend` detects the IMDS
notice, writes an emergency checkpoint, and exits.  Relaunching from the
same template restores from the last checkpoint.

## Prerequisites

```bash
# Node.js 18+ (for CDK CLI)
node --version

# CDK CLI
npm install -g aws-cdk

# Python deps for this stack
pip install -r requirements.txt

# AWS credentials with sufficient permissions
aws sts get-caller-identity
```

### IAM permissions required to deploy

The deploying principal needs at minimum:

- `cloudformation:*`
- `s3:*` (bucket creation)
- `iam:CreateRole`, `iam:AttachRolePolicy`, `iam:PassRole`
- `ec2:CreateLaunchTemplate`, `ec2:RunInstances`
- `ssm:GetParameter` (to resolve the AL2023 AMI)

## Deploy

```bash
cd infra/

# Bootstrap CDK in your account/region (first time only)
cdk bootstrap aws://ACCOUNT_ID/us-east-1

# Preview what will be created
cdk diff

# Deploy
cdk deploy --context account=ACCOUNT_ID --context region=us-east-1
```

The deploy outputs three values you'll need:

```
SpotCheckpointSmoke.BucketName       = spot-checkpoint-smoke-xxxx
SpotCheckpointSmoke.LaunchTemplateId = lt-0123456789abcdef0
SpotCheckpointSmoke.InstanceProfileArn = arn:aws:iam::...
```

## Run the smoke test

### 1. Create the FIS IAM role (one-time)

FIS needs a role that can send interruption notices:

```bash
aws iam create-role \
  --role-name FISRole \
  --assume-role-policy-document '{
    "Version": "2012-10-17",
    "Statement": [{
      "Effect": "Allow",
      "Principal": {"Service": "fis.amazonaws.com"},
      "Action": "sts:AssumeRole"
    }]
  }'

aws iam attach-role-policy \
  --role-name FISRole \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSFaultInjectionSimulatorEC2Access
```

### 2. Register the FIS experiment template

Fill in the two `<REPLACE:...>` placeholders in `fis_experiment.json`, then:

```bash
aws fis create-experiment-template \
  --cli-input-json file://fis_experiment.json \
  --query 'experimentTemplate.id' --output text
# → EXT0123456789
```

### 3. Launch a spot instance

```bash
LAUNCH_TEMPLATE_ID=lt-0123456789abcdef0   # from CDK output

aws ec2 run-instances \
  --launch-template LaunchTemplateId=$LAUNCH_TEMPLATE_ID,Version='$Latest' \
  --count 1 \
  --query 'Instances[0].InstanceId' --output text
```

Wait ~60 seconds for user-data to complete (the instance tags itself
`spot-checkpoint-smoke-test=true` once the benchmark starts).

### 4. Start the FIS experiment

```bash
TEMPLATE_ID=EXT0123456789   # from step 2

aws fis start-experiment \
  --experiment-template-id $TEMPLATE_ID \
  --query 'experiment.id' --output text
# → EXP0123456789
```

FIS sends a 2-minute interruption notice to the instance, then
terminates it.  Watch the benchmark log:

```bash
INSTANCE_ID=i-0123456789abcdef0

aws ssm start-session --target $INSTANCE_ID
# Inside the session:
tail -f /var/log/spot-checkpoint-smoke.log
```

### 5. Verify the checkpoint was written

```bash
BUCKET=spot-checkpoint-smoke-xxxx   # from CDK output

aws s3 ls s3://$BUCKET/smoke-test/ --recursive | sort
```

You should see shard files and a `_manifest.json` written shortly
before the instance was terminated.

### 6. Relaunch and verify restore

Rerun step 3 with the same launch template.  The new instance will
restore from the checkpoint and log:

```
Restored from iteration N
Starting from iteration N / 200
```

### 7. Check the FIS experiment result

```bash
aws fis get-experiment --id EXP0123456789 \
  --query 'experiment.state'
```

Expected: `{"status": "completed", "reason": "Experiment completed."}`

## Tear down

```bash
# Delete all checkpoints first (bucket has auto_delete_objects=True but
# versioned objects need explicit clearing for fast deletion)
aws s3 rm s3://$BUCKET --recursive

# Destroy the CDK stack
cdk destroy --context account=ACCOUNT_ID --context region=us-east-1
```

## Troubleshooting

| Symptom | Likely cause |
|---------|-------------|
| FIS experiment fails with `NoTargetsResolved` | Instance not yet tagged; wait 60 s after launch |
| No checkpoint files in S3 | Check `/var/log/spot-checkpoint-smoke.log` via SSM |
| `cdk deploy` fails on AMI resolution | Run in a region where AL2023 SSM param exists (`us-east-1`, `us-west-2`, `eu-west-1`, etc.) |
| Instance terminates before benchmark starts | c5.large spot capacity; try `--instance-type c5a.large` |
