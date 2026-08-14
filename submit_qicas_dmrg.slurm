#!/bin/bash
#SBATCH --job-name=qicas_dmrg
#SBATCH --account=hpc-prf-qehpc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=qicas_dmrg_%j.out
#SBATCH --error=qicas_dmrg_%j.err
#
# Usage:
#   sbatch submit_qicas_dmrg.slurm                          # interactive
#   sbatch submit_qicas_dmrg.slurm --export=SYSTEM=MnBr4_sextet
#   sbatch submit_qicas_dmrg.slurm --export=CONFIG=my_system.yaml
#   sbatch submit_qicas_dmrg.slurm --export=TEST_ONLY=1
#
# Monitor:
#   grep "RESULT\|gorb\|macro iter\|ROHF\|QICAS\|converged" qicas_dmrg_JOBID.out

source ~/.block2_fix/block2_env.sh

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================"
echo "Job:      ${SLURM_JOB_ID}"
echo "CPUs:     ${SLURM_CPUS_PER_TASK}"
echo "Node:     $(hostname)"
echo "Started:  $(date)"
echo "======================================"

# Build argument list
ARGS=""

if [ "${TEST_ONLY}" = "1" ]; then
    echo "Running in TEST MODE"
    ARGS="--test"
elif [ -n "${SYSTEM}" ]; then
    echo "System: ${SYSTEM}"
    ARGS="--system ${SYSTEM} --M 350 --M-qicas 100 --max-macro 150"
elif [ -n "${CONFIG}" ]; then
    echo "Config: ${CONFIG}"
    ARGS="--config ${CONFIG} --M 350 --M-qicas 100 --max-macro 150"
else
    echo "No system specified — running interactively"
    ARGS="--M 350 --M-qicas 100 --max-macro 150"
fi

python3 "${SCRIPT_DIR}/run_qicas_dmrg.py" ${ARGS} 2>&1

echo "Finished: $(date)"
