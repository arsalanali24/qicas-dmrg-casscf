#!/bin/bash
#SBATCH --job-name=dmrg_casscf_hs
#SBATCH --account=hpc-prf-qehpc
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=120G
#SBATCH --time=24:00:00
#SBATCH --output=logs/dmrg_casscf_%x_%j.out
#SBATCH --error=logs/dmrg_casscf_%x_%j.err
#
# Usage:
#   sbatch --job-name=MnCl4 submit_dmrg_casscf_hs.slurm --export=SYSTEM=MnCl4
#   sbatch --job-name=MnBr4 submit_dmrg_casscf_hs.slurm --export=SYSTEM=MnBr4
#
# Parameters (override via --export):
#   SYSTEM      : MnCl4 | MnBr4 | ... (must match SYSTEMS dict)  [required]
#   M           : DMRG bond dimension for CASSCF  (default 350)
#   M_QICAS     : DMRG bond dimension for QICAS   (default 100)
#   MAX_MACRO   : max CASSCF macro iterations      (default 150)
#   WORK_DIR    : scratch directory for DMRG       (default /scratch/.../dmrg_SYSTEM)
#
# Monitor:
#   grep "RESULT\|converged\|macro iter\|ROHF\|F_QI\|QICAS active" logs/dmrg_casscf_*.out
#
# ── Critical settings (do NOT change without reason) ──────────────────────
# conv_tol_grad = 3e-4  : DMRG noise requires relaxed gradient tolerance
# conv_tol      = 1e-8  : 0.01 mHa energy convergence
# init_guess    = atom  : avoids wrong local min (~500 mHa error with minao)
# ROHF          : required for HS TM; UHF reference causes non-convergence
# WINDOW_SIZE   : set per-system in SYSTEMS dict (26 for MnCl4/MnBr4)
# step4→step5   : must use step4 to get inactive/active split before step5

echo "=============================="
echo "  DMRG-CASSCF HS Pipeline"
echo "  System  : ${SYSTEM:-NOT SET}"
echo "  Job     : $SLURM_JOB_ID"
echo "  Node    : $SLURMD_NODENAME"
echo "  Start   : $(date)"
echo "=============================="

# Validate required parameter
if [ -z "$SYSTEM" ]; then
    echo "ERROR: SYSTEM not set. Use --export=SYSTEM=MnCl4"
    exit 1
fi

# Environment
source /pc2/users/h/hpcmual/.block2_fix/block2_env.sh

# Threading — match cpus-per-task
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Defaults
M=${M:-350}
M_QICAS=${M_QICAS:-100}
MAX_MACRO=${MAX_MACRO:-100}
WORK_DIR=${WORK_DIR:-/scratch/hpc-prf-qehpc/hpcmual/dmrg_${SYSTEM}_$$}

REPO_DIR=/pc2/users/h/hpcmual/qicas-dmrg-casscf

mkdir -p logs "$WORK_DIR"

echo "  M       : $M"
echo "  M_QICAS : $M_QICAS"
echo "  Macros  : $MAX_MACRO"
echo "  WorkDir : $WORK_DIR"
echo ""

cd "$REPO_DIR"

python3 dmrg_casscf_hs.py \
    ${SKIP_HF:+--skip-hf-run} \
    ${SKIP_CASSCF:+--skip-casscf} \
    --system   "$SYSTEM"   \
    --M        "$M"        \
    --M-qicas  "$M_QICAS"  \
    --max-macro "$MAX_MACRO" \
    --work-dir "$WORK_DIR"

EXIT_CODE=$?

echo ""
echo "=============================="
echo "  Exit code : $EXIT_CODE"
echo "  End       : $(date)"
echo "=============================="

exit $EXIT_CODE
