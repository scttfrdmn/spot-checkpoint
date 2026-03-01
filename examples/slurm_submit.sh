#!/bin/bash
#SBATCH --job-name=pyscf-ccsd
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=spot
#SBATCH --signal=B:USR1@120
#SBATCH --requeue
#SBATCH --output=pyscf-%j.out

# --signal=B:USR1@120 : Send SIGUSR1 120 seconds before wall-time limit
# --requeue           : Allow Slurm to requeue on preemption
#
# spot-checkpoint's SlurmLifecycleBackend handles:
#   SIGTERM → emergency checkpoint + scontrol requeue
#   SIGUSR1 → checkpoint before wall-time expiry
#
# IMPORTANT: Restore is NOT automatic. Your Python script must call
# spot_restore() before spot_safe() to reload from the last checkpoint.
# See examples/restore_and_continue_scf.py for the two-call pattern:
#
#   from spot_checkpoint import spot_restore, spot_safe
#   spot_restore(solver, bucket=os.environ["SPOT_CHECKPOINT_BUCKET"])
#   solver.callback = spot_safe(solver, bucket=os.environ["SPOT_CHECKPOINT_BUCKET"])
#   solver.kernel()

module load python/3.11
source ~/venvs/pyscf/bin/activate

export SPOT_CHECKPOINT_BUCKET="my-lab-checkpoints"

python ccsd_large.py
