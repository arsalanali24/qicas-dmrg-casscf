# QICAS + DMRG-CASSCF Pipeline

Pipeline for high-spin transition metal complexes on Noctua2 (PC²).

## Quick start
```bash
source ~/.block2_fix/block2_env.sh
python3 run_qicas_dmrg.py --test                              # validate first
python3 run_qicas_dmrg.py --system MnBr4_sextet              # known system
python3 run_qicas_dmrg.py --config systems/MySystem.yaml     # new system
sbatch submit_qicas_dmrg.slurm --export=SYSTEM=MnBr4_sextet  # via SLURM
```

## New chat — paste CONTEXT.md to restore full context instantly.
