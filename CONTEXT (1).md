# QICAS + DMRG-CASSCF Pipeline — Context File
## Paste this into a new chat to restore full context instantly

---

## Project
QICAS vs AutoCAS benchmark for transition metal complexes.
PI: Dr. Werner Dobrautz | Cluster: Noctua2 (PC² Paderborn)
User: hpcmual | Account: hpc-prf-qehpc | Env: `source ~/.block2_fix/block2_env.sh`

## What we are doing
Running QICAS orbital optimization followed by DMRG-CASSCF with two initializations:
- **Run 1:** ROHF canonical orbitals (baseline)
- **Run 2:** QICAS-optimized orbitals (comparison)

Goal: show QICAS-init converges faster / finds lower energy minimum than HF-init.

## Key script
**`run_qicas_dmrg.py`** — single entry point for any new system.

```bash
# Test environment first (always do this for a new system)
python3 run_qicas_dmrg.py --test

# Run known system
python3 run_qicas_dmrg.py --system MnBr4_sextet

# Run new system (interactive)
python3 run_qicas_dmrg.py

# Via SLURM
sbatch submit_qicas_dmrg.slurm --export=SYSTEM=MnBr4_sextet
sbatch submit_qicas_dmrg.slurm --export=TEST_ONLY=1
```

## Critical settings (do NOT change without reason)
| Setting | Value | Why |
|---------|-------|-----|
| `init_guess` | `atom` | minao finds wrong local min for HS TM (~500 mHa error) |
| `conv_tol_grad` | `3e-4` | PySCF default; 1e-4 too tight for DMRG noise at M=350 |
| `conv_tol_energy` | `1e-8` | 0.01 mHa |
| `work_dir` | **absolute path** | block2 nests relative paths → node0/dmrg.e not found |
| DMRG schedule | monotonically increasing M | decreasing M gives wrong wavefunction |
| ECP | `def2-svp` for 4d/5d metals | without ECP energies are ~774 Ha off |

## Known qicas_benchmark.py API (exact signatures)
```python
window                                   = qb.determine_window(mol, mf)
# window keys: actual_window_size, n_elec_active, socc_window_rel,
#              window_start, window_end, n_frozen

mc, e, t                                 = qb.run_dmrg_on_window(mol, mf, window, max_M)
gamma, Gamma, s_vals, occ_nums          = qb.step3_extract_rdms(mc, n_orbs, n_elec)      # 4 values
U, gamma_opt, Gamma_opt, history        = qb.step5_optimize_orbitals(γ, Γ, inactive, active)  # 4 values
n_qi, n_e, s, occ, idx, active_rel     = qb.step6_determine_active_space(γ_opt, Γ_opt, n, socc)  # 6 values
mo_qicas, ncore                         = qb.build_casci_mo_coeff(mo, n_mo, win, np.array(active_rel), occ, n_e)  # tuple
```

## Validated results
| System | ΔCASCI (mHa) | HF-init | QICAS-init | Key finding |
|--------|-------------|---------|------------|-------------|
| MnBr4 sextet | −560 | ✗ 715 iters | ✓ 219 iters | QICAS converges, HF fails |
| MnCl4 sextet | −239 | ✗ local min A | ✗ local min B (3.7 mHa lower) | QICAS finds better minimum |

## Cluster paths
```
Pipeline:  ~/activeml/qio/QIO-master/examples_benchmark/qicas_benchmark.py
Scratch:   /scratch/hpc-prf-qehpc/hpcmual/
Results:   ~/activeml/qio/QIO-master/examples_benchmark/pilot/
```

## Common errors and fixes
1. `node0/dmrg.e not found` → work_dir must be absolute path
2. `ROHF energy ~500 mHa wrong` → use `init_guess='atom'`
3. `|gorb| = 4.49 at Macro 1` → QICAS MOs from different HF reference — rerun QICAS fresh
4. `determine_window() unexpected keyword` → takes only `(mol, mf)`, no size arg
5. `too many values to unpack` → step3 returns 4, step5 returns 4, step6 returns 6
6. `'tuple' has no attribute shape` → build_casci_mo_coeff returns (mo_coeff, ncore)
7. `DMRG schedule decreasing` → clip intermediates to min(step_M, target_M)
8. `conv_tol_grad too tight` → use 3e-4 not 1e-4 for DMRG-CASSCF

## To add a new system
1. Run `python3 run_qicas_dmrg.py --test` to validate environment
2. Run `python3 run_qicas_dmrg.py` and enter geometry interactively
3. Or add to `KNOWN_SYSTEMS` dict in `run_qicas_dmrg.py` and use `--system`
4. Results saved to `work_dir/result_SYSTEMNAME.json`
