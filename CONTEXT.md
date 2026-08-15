# QICAS + DMRG-CASSCF Pipeline — Context File
## Paste this into a new chat to restore full context instantly

---

## Project
QICAS vs AutoCAS benchmark for transition metal complexes.
PI: Dr. Werner Dobrautz | Cluster: Noctua2 (PC² Paderborn)
User: hpcmual | Account: hpc-prf-qehpc | Env: `source ~/.block2_fix/block2_env.sh`

## What this pipeline does
Runs QICAS orbital optimization followed by DMRG-CASSCF with two initializations:
- **Run 1:** ROHF canonical orbitals (baseline)
- **Run 2:** QICAS-optimized orbitals (comparison)

Goal: show QICAS-init converges faster / finds lower energy minimum than HF-init.

## Key script
**`dmrg_casscf_hs.py`** — single entry point for any system in SYSTEMS dict.

```bash
# Standard submission
sbatch --job-name=MnCl4 submit_dmrg_casscf_hs.slurm --export=SYSTEM=MnCl4

# Custom parameters
sbatch --job-name=MnBr4 submit_dmrg_casscf_hs.slurm \
    --export=SYSTEM=MnBr4,M=350,M_QICAS=100,MAX_MACRO=150

# Monitor
grep "RESULT\|converged\|macro iter" logs/dmrg_casscf_*.out
```

## Adding a new system
Add to `SYSTEMS` dict in `dmrg_casscf_hs.py`:
```python
"NewSystem": {
    "name":       "CSD_name_for_json",
    "spin_2s":    5,          # 2S = Nalpha - Nbeta
    "charge":     -2,
    "n_active":   14,         # QICAS active orbitals (from prior run or estimate)
    "n_elec":     21,         # QICAS active electrons
    "autocas_no": 10,         # AutoCAS reference n_orbs (for step4 partition)
    "window_size": 26,        # DMRG window: 26 (HS), 24 (MS), 22 (LS)
    "basis":      "def2-svp",
    "ecp":        None,       # "def2-svp" for 4d/5d metals; None for 3d
    "atoms": """Metal  x  y  z
Ligand  x  y  z
...""",
}
```

## Critical settings (do NOT change without reason)
| Setting | Value | Why |
|---------|-------|-----|
| `init_guess` | `atom` | minao finds wrong local min for HS TM (~500 mHa error) |
| Reference | **ROHF** | UHF reference causes non-convergence for HS TM (confirmed MnCl4) |
| `conv_tol_grad` | `3e-4` | DMRG noise at M=350 prevents tighter convergence |
| `conv_tol` | `1e-8` | 0.01 mHa energy convergence |
| `work_dir` | **absolute path** | block2 nests relative paths → node0/dmrg.e not found |
| DMRG schedule | monotonically increasing M | decreasing M gives wrong wavefunction |
| ECP | `def2-svp` for 4d/5d metals | without ECP energies are ~774 Ha off |
| `WINDOW_SIZE` | per-system (26 for HS) | must set `qb.WINDOW_SIZE` before `determine_window()` |
| step4 before step5 | **mandatory** | step5 needs inactive/active split from step4; passing socc_rel only gives wrong F_QI |

## QICAS API — correct call sequence
```python
# 1. Set window size BEFORE calling determine_window
qb.WINDOW_SIZE = sys_dict["window_size"]
window = qb.determine_window(mol, mf)

# 2. Run DMRG on window
mc_dmrg, e_dmrg, t_dmrg = qb.run_dmrg_on_window(mol, mf, window, max_M=100)

# 3. Extract RDMs
gamma, Gamma, s_vals, occ_nums = qb.step3_extract_rdms(mc_dmrg, n_orbs, n_elec)

# 4. Get initial inactive/active split (REQUIRED before step5)
fqi_initial, sorted_idx, active_set, inactive_list = qb.step4_initial_fqi(
    gamma, Gamma, s_vals, occ_nums, n_cas_target)

# 5. F_QI orbital optimization (use inactive_list from step4, NOT socc_rel)
U, gamma_opt, Gamma_opt, history = qb.step5_optimize_orbitals(
    gamma, Gamma, inactive_list, sorted(active_set))

# 6. Determine final active space
n_qi, n_e, s_opt, occ_opt, idx, active_rel = qb.step6_determine_active_space(
    gamma_opt, Gamma_opt, n_orbs, socc_rel)
```

## Validated results
| System | ΔCASCI (mHa) | HF-init (ROHF) | QICAS-init | Key finding |
|--------|-------------|---------|------------|-------------|
| MnBr4 sextet | −560 | ✗ 715 iters (UHF) | ✓ 219 iters | QICAS converges, HF fails |
| MnCl4 sextet | −214 | ✓ **95 iters** (ROHF) | running | ROHF ref fixes convergence |

## Why ROHF not UHF
UHF reference for HS TM complexes causes DMRG-CASSCF non-convergence:
- UHF breaks spin symmetry → orbital gradient oscillates near saddle point
- ROHF gives canonical orbitals with correct spin symmetry
- MnCl4 sextet: UHF HF-init failed at 400 iters; ROHF HF-init **converged in 95 iters**
- The pipeline (`qicas_casscf_benchmark.py`) uses UHF internally — do NOT use it for CASSCF
- Use `dmrg_casscf_hs.py` which is ROHF throughout

## Common errors and fixes
1. `node0/dmrg.e not found` → work_dir must be absolute path
2. `ROHF energy ~500 mHa wrong` → use `init_guess='atom'`
3. `F_QI pre==post (no change)` → forgot step4; passing empty inactive_indices to step5
4. `wrong active space (13e,10o) instead of (21e,14o)` → WINDOW_SIZE not set before determine_window()
5. `CASSCF not converging (400+ iters)` → using UHF reference; switch to ROHF
6. `too many values to unpack` → step3 returns 4, step5 returns 4, step6 returns 6
7. `'tuple' has no attribute shape` → build_casci_mo_coeff returns (mo_coeff, ncore)
8. `DMRG schedule decreasing` → clip intermediates to min(step_M, target_M)
9. `num_thrds=1 in log` → misleading header; check dmrg.conf for actual thread count

## Cluster paths
```
Repo:      ~/qicas-dmrg-casscf/
Pipeline:  ~/activeml/qio/QIO-master/examples_benchmark/qicas_benchmark.py
Scratch:   /scratch/hpc-prf-qehpc/hpcmual/
Results:   ~/qicas-dmrg-casscf/results/
Logs:      ~/qicas-dmrg-casscf/logs/
```

## Window sizes by spin category
| 2S | Category | window_size | M_QICAS | M_CASSCF |
|---|---|---|---|---|
| ≥4 | HIGH | 26 | 100 | 350 |
| 2–3 | MEDIUM | 24 | 100 | 350 |
| 0–1 | LOW | 22 | 100 | 350 |

## Quick start for new system
```bash
# 1. Add system to SYSTEMS dict in dmrg_casscf_hs.py
# 2. Submit
sbatch --job-name=MySystem submit_dmrg_casscf_hs.slurm --export=SYSTEM=MySystem
# 3. Monitor
tail -f logs/dmrg_casscf_MySystem_*.out
# 4. Results saved to
cat results/dmrg_casscf_MySystem_M350.json
```
