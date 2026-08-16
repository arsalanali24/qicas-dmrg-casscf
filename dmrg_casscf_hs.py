#!/usr/bin/env python3
"""
dmrg_casscf_hs.py
─────────────────
Runs DMRG-CASSCF for MnCl4 and MnBr4 sextets with TWO initializations:
  Run 1: ROHF canonical orbitals  (baseline)
  Run 2: QICAS-optimized orbitals (generated fresh, same HF reference)

Both runs use IDENTICAL HF settings so MOs are directly comparable.
The QICAS step uses qicas_benchmark.py from the existing pipeline.

Usage:
  source ~/.block2_fix/block2_env.sh
  python3 dmrg_casscf_hs.py --system MnBr4 --M 350 --max-macro 150
"""

import argparse, json, logging, os, sys, time
import numpy as np

log = logging.getLogger("dmrg_casscf")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")

# ── System definitions ─────────────────────────────────────────────────────
SYSTEMS = {
    "MnCl4": {
        "name":     "CSD_MnCl4_2m_tet_spin5",
        "spin_2s":  5, "charge": -2,
        "n_active": 14, "n_elec": 21,
        "autocas_no": 10, "window_size": 26,
        "basis":    "def2-svp", "ecp": None,
        "atoms": """Mn  0.000000  0.000000  0.000000
Cl  1.356773  1.356773  1.356773
Cl -1.356773 -1.356773  1.356773
Cl -1.356773  1.356773 -1.356773
Cl  1.356773 -1.356773 -1.356773""",
    },
    "MnBr4": {
        "name":     "CSD_MnBr4_2m_tet_spin5",
        "spin_2s":  5, "charge": -2,
        "n_active": 13, "n_elec": 19,
        "autocas_no": 10, "window_size": 26,
        "basis":    "def2-svp", "ecp": None,
        "atoms": """Mn  0.000000  0.000000  0.000000
Br  1.491296  1.491296  1.491296
Br -1.491296 -1.491296  1.491296
Br -1.491296  1.491296 -1.491296
Br  1.491296 -1.491296 -1.491296""",
    },

    "MnCl4_quartet": {
        "name":       "CSD_MnCl4_2m_tet_spin3",
        "spin_2s":    3, "charge": -2,
        "n_active":   14, "n_elec": 21,
        "autocas_no": 12, "window_size": 26,
        "basis":      "def2-svp", "ecp": None,
        "atoms": """Mn  0.000000  0.000000  0.000000
Cl  1.356773  1.356773  1.356773
Cl -1.356773 -1.356773  1.356773
Cl -1.356773  1.356773 -1.356773
Cl  1.356773 -1.356773 -1.356773""",
    },
    "MnBr4_quartet": {
        "name":       "CSD_MnBr4_2m_tet_spin3",
        "spin_2s":    3, "charge": -2,
        "n_active":   13, "n_elec": 19,
        "autocas_no": 12, "window_size": 26,
        "basis":      "def2-svp", "ecp": None,
        "atoms": """Mn  0.000000  0.000000  0.000000
Br  1.443376  1.443376  1.443376
Br -1.443376 -1.443376  1.443376
Br -1.443376  1.443376 -1.443376
Br  1.443376 -1.443376 -1.443376""",
    },

    "VBr6_triplet": {
        "name":       "CSD_VBr6_3m_oct_spin2",
        "spin_2s":    2, "charge": -3,
        "n_active":   10, "n_elec": 18,
        "autocas_no": 10, "window_size": 22,
        "basis":      "def2-svp", "ecp": None,
        "atoms": """V   0.000000  0.000000  0.000000
Br  2.318000  0.000000  0.000000
Br -2.318000  0.000000  0.000000
Br  0.000000  2.318000  0.000000
Br  0.000000 -2.318000  0.000000
Br  0.000000  0.000000  2.318000
Br  0.000000  0.000000 -2.318000""",
    },
    "NiBr6_triplet": {
        "name":       "CSD_NiBr6_4m_oct_spin2",
        "spin_2s":    2, "charge": -4,
        "n_active":   12, "n_elec": 22,
        "autocas_no": 12, "window_size": 22,
        "basis":      "def2-svp", "ecp": None,
        "atoms": """Ni  0.000000  0.000000  0.000000
Br  2.530000  0.000000  0.000000
Br -2.530000  0.000000  0.000000
Br  0.000000  2.530000  0.000000
Br  0.000000 -2.530000  0.000000
Br  0.000000  0.000000  2.530000
Br  0.000000  0.000000 -2.530000""",
    },
    "VBr6_doublet": {
        "name":       "CSD_VBr6_2m_oct_spin1",
        "spin_2s":    1, "charge": -2,
        "n_active":   12, "n_elec": 23,
        "autocas_no": 12, "window_size": 20,
        "basis":      "def2-svp", "ecp": None,
        "atoms": """V   0.000000  0.000000  0.000000
Br  2.318000  0.000000  0.000000
Br -2.318000  0.000000  0.000000
Br  0.000000  2.318000  0.000000
Br  0.000000 -2.318000  0.000000
Br  0.000000  0.000000  2.318000
Br  0.000000  0.000000 -2.318000""",
    },

    "MnBr4_singlet": {
        "name":       "CSD_MnBr4_1m_oct_spin0",
        "spin_2s":    0, "charge": -1,
        "n_active":   14, "n_elec": 24,
        "autocas_no": 13, "window_size": 20,
        "basis":      "def2-svp", "ecp": None,
        "atoms": """Mn  0.000000  0.000000  0.000000
Br  2.630000  0.000000  0.000000
Br -2.630000  0.000000  0.000000
Br  0.000000  2.630000  0.000000
Br  0.000000 -2.630000  0.000000
Br  0.000000  0.000000  2.630000
Br  0.000000  0.000000 -2.630000""",
    },
}


# ── DMRG warmup schedule ───────────────────────────────────────────────────
def build_dmrg_schedule(target_M):
    m1 = min(64, target_M);  m2 = min(128, target_M)
    m3 = min(200, target_M); m4 = min(250, target_M)
    if target_M <= 200:
        return dict(
            sweeps=[4, 4, 8, 8, 16, 16],
            maxMs =[m1, m2, m3, target_M, target_M, target_M],
            tols  =[1e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            noises=[1e-3, 1e-4, 1e-4, 1e-5, 0, 0])
    elif target_M <= 350:
        return dict(
            sweeps=[4, 4, 4, 8, 8, 8, 16, 16],
            maxMs =[m1, m2, m2, m3, m3, target_M, target_M, target_M],
            tols  =[1e-5, 1e-5, 1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-9],
            noises=[1e-3, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 0, 0])
    else:
        return dict(
            sweeps=[4, 4, 4, 8, 8, 8, 8, 16, 16],
            maxMs =[m1, m2, m2, m3, m3, m4, target_M, target_M, target_M],
            tols  =[1e-5, 1e-5, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7, 1e-8, 1e-9],
            noises=[1e-3, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 0, 0])


# ── Build molecule + ROHF (pipeline-identical settings) ───────────────────
def build_mol_and_hf(sys_dict):
    from pyscf import gto, scf
    mol = gto.Mole()
    mol.atom      = sys_dict["atoms"]
    mol.basis     = sys_dict["basis"]
    mol.charge    = sys_dict["charge"]
    mol.spin      = sys_dict["spin_2s"]   # 2S
    mol.verbose   = 4
    mol.max_memory = 32000
    if sys_dict.get("ecp"):
        mol.ecp = sys_dict["ecp"]
    mol.build()

    # ── ROHF: identical to qicas_benchmark.py (no level_shift, no damping) ──
    mf = scf.ROHF(mol)
    mf.max_cycle = 200        # same as pipeline
    mf.init_guess = 'atom'  # avoids local minimum from minao
    mf.conv_tol  = 1e-12      # PySCF ROHF default
    # No level_shift — pipeline does not use it
    mf.kernel()
    log.info(f"ROHF: E = {mf.e_tot:.10f} Ha   converged = {mf.converged}")
    if not mf.converged:
        log.warning("ROHF not converged — trying with level_shift=0.1 as fallback")
        mf2 = scf.ROHF(mol)
        mf2.max_cycle  = 400
        mf2.conv_tol   = 1e-10
        mf2.level_shift = 0.1
        mf2.kernel()
        if mf2.converged or mf2.e_tot < mf.e_tot:
            mf = mf2
            log.info(f"ROHF (level_shift=0.1): E = {mf.e_tot:.10f}  converged={mf.converged}")
    return mol, mf


# ── Run QICAS to get optimized MOs (same HF reference) ────────────────────
def run_qicas(mol, mf, sys_dict, M_qicas, work_dir):
    """
    Run QICAS orbital optimization using the existing pipeline.
    Returns QICAS-rotated MO coefficients in the same AO basis as mf.
    """
    log.info("\n" + "="*60)
    log.info("QICAS STEP: Generating optimized orbitals")
    log.info("="*60)

    # Add the pipeline scripts directory to path
    pipeline_scripts = os.path.expanduser(
        "~/activeml/qio/QIO-master/examples_benchmark")
    if pipeline_scripts not in sys.path:
        sys.path.insert(0, pipeline_scripts)

    try:
        import qicas_benchmark as qb
    except ImportError:
        log.error("Cannot import qicas_benchmark — check pipeline path")
        return None

    ncas  = sys_dict["n_active"]
    nelec = sys_dict["n_elec"]
    spin  = sys_dict["spin_2s"]

    # Step 2: Determine frontier window from ROHF occupations
    qb.WINDOW_SIZE = sys_dict.get("window_size", 26)  # per-system window
    qb.SYSTEM['spin'] = sys_dict['spin_2s']  # fix assert n_socc == SYSTEM['spin']
    window = qb.determine_window(mol, mf)
    n_orbs          = window["actual_window_size"]
    n_elec_active   = window["n_elec_active"]
    socc_rel        = window["socc_window_rel"]
    log.info(f"  Window: {n_orbs} orbitals  "
             f"(range [{window['window_start']}, {window['window_end']}])")

    # Step 2: Run DMRG on window at M=M_qicas
    log.info(f"  Running DMRG at M={M_qicas}...")
    # Fix: override relative scratch path in qicas_benchmark before DMRG runs
    _qicas_scratch = os.path.join(work_dir, f"qicas_dmrg_{os.getpid()}")
    os.makedirs(_qicas_scratch, exist_ok=True)
    import pyscf.dmrgscf as _dmrgscf
    _orig_DMRGCI = _dmrgscf.DMRGCI
    class _PatchedDMRGCI(_orig_DMRGCI):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self.runtimeDir = _qicas_scratch
            self.scratchDirectory = _qicas_scratch
    _dmrgscf.DMRGCI = _PatchedDMRGCI
    mc_dmrg, e_dmrg, t_dmrg = qb.run_dmrg_on_window(
        mol, mf, window, max_M=M_qicas)
    _dmrgscf.DMRGCI = _orig_DMRGCI  # restore
    log.info(f"  DMRG energy: {e_dmrg:.10f} Ha  (t={t_dmrg:.1f}s)")

    # Step 3: Extract 1- and 2-RDMs from DMRG wavefunction
    gamma, Gamma, s_vals, occ_nums = qb.step3_extract_rdms(mc_dmrg, n_orbs, n_elec_active)
    log.info(f"  RDMs extracted: shape={gamma.shape}")

    # Step 4: Compute initial entropy/occupation profile

    # Step 5: Optimize orbitals via F_QI minimization
    log.info("  Running F_QI orbital rotation (step 5)...")
    # Step 4: determine initial inactive/active split (correct approach)
    n_cas_target = sys_dict.get('autocas_no', 10)
    fqi_initial, sorted_idx, active_set, inactive_list = qb.step4_initial_fqi(
        gamma, Gamma, s_vals, occ_nums, n_cas_target)
    inactive_indices = inactive_list
    active_indices   = sorted(active_set)
    U_total, gamma_opt, Gamma_opt, opt_history = qb.step5_optimize_orbitals(
        gamma, Gamma, inactive_indices, active_indices)
    log.info("  F_QI optimization done")

    # Step 6: Determine final active space from optimized RDMs
    n_qi, n_e_qi, s_opt, occ_opt, sorted_idx, active_window_rel = qb.step6_determine_active_space(
        gamma_opt, Gamma_opt, n_orbs, socc_rel)
    log.info(f"  QICAS active orbitals (window-relative): {active_window_rel}")
    # Apply F_QI rotation directly to window columns — preserves MO ordering for CASSCF
    win_start = window["window_start"]
    win_end   = window["window_end"]
    mo_qicas  = mf.mo_coeff.copy()
    mo_qicas[:, win_start:win_end] = mf.mo_coeff[:, win_start:win_end] @ U_total
    log.info(f"  F_QI rotation applied to window [{win_start}:{win_end}]")
    log.info(f"  MO shape: {mo_qicas.shape}")
    log.info(f"  QICAS MO shape: {mo_qicas.shape}")

    # Save for reuse
    mo_path = os.path.join(work_dir, f"qicas_mo_{sys_dict['name']}.npy")
    np.save(mo_path, mo_qicas)
    log.info(f"  Saved QICAS MOs: {mo_path}")

    return mo_qicas


# ── DMRG-CASSCF runner ─────────────────────────────────────────────────────
def run_casscf(mol, mf, sys_dict, mo_start, M, max_cycle_macro,
               work_dir, label):
    from pyscf import mcscf, dmrgscf

    ncas  = sys_dict["n_active"]
    nelec = sys_dict["n_elec"]
    spin  = sys_dict["spin_2s"]
    n_alpha = (nelec + spin) // 2
    n_beta  = (nelec - spin) // 2

    log.info(f"\n{'─'*50}")
    log.info(f"{label}: CAS({nelec}e,{ncas}o)  M={M}")
    log.info(f"{'─'*50}")

    mc = mcscf.CASSCF(mf, ncas, (n_alpha, n_beta))
    mc.conv_tol_grad   = 3e-4    # orbital gradient — publication standard
    mc.conv_tol        = 1e-8    # energy = 0.01 mHa
    mc.max_cycle_macro = max_cycle_macro
    mc.max_cycle_micro = 10
    mc.verbose         = 4

    # DMRG as CI solver
    block2_exe = os.popen("which block2main").read().strip()
    if not block2_exe:
        raise RuntimeError("block2main not found — source block2_env.sh")

    dmrgscf.settings.BLOCKEXE  = block2_exe
    dmrgscf.settings.MPIPREFIX = ""

    scratch = os.path.join(work_dir, label.replace(" ", "_").replace(":", ""))
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(os.path.join(scratch, "node0"), exist_ok=True)

    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=M, tol=1e-9)
    mc.fcisolver.runtimeDir       = scratch
    mc.fcisolver.scratchDirectory = scratch
    mc.fcisolver.threads          = int(os.environ.get("OMP_NUM_THREADS", 8))
    mc.fcisolver.memory           = int(mol.max_memory / 1000)

    sched = build_dmrg_schedule(M)
    mc.fcisolver.scheduleSweeps   = sched["sweeps"]
    mc.fcisolver.scheduleMaxMs    = sched["maxMs"]
    mc.fcisolver.scheduleTols     = sched["tols"]
    mc.fcisolver.scheduleNoises   = sched["noises"]
    mc.fcisolver.maxIter          = 30
    mc.fcisolver.twodot_to_onedot = 20

    iter_count = [0]
    last_e = [float('nan')]
    def callback(envs):
        iter_count[0] += 1
        e    = envs.get("e_tot",    float("nan"))
        de   = envs.get("de",       float("nan"))
        gorb = envs.get("norm_gorb",float("nan"))
        log.info(f"  [{label}] Macro {iter_count[0]:4d}: "
                 f"E={e:.10f}  dE={de:+.2e}  |gorb|={gorb:.2e}")
        last_e[0] = envs.get("e_tot", float("nan"))
        if iter_count[0] >= max_cycle_macro:
            raise KeyboardInterrupt(f"max_cycle_macro={max_cycle_macro} reached")

    mc.callback = callback

    t0 = time.time()
    try:
        e_tot = mc.kernel(mo_start)[0]
    except KeyboardInterrupt:
        log.info(f"  [{label}] Stopped at max_cycle_macro={max_cycle_macro}")
        e_tot = last_e[0]
    t_wall = time.time() - t0

    log.info(f"\n  {label} RESULT:")
    log.info(f"    E           = {e_tot:.10f} Ha")
    log.info(f"    converged   = {mc.converged}")
    log.info(f"    macro_iters = {iter_count[0]}")
    log.info(f"    wall_time   = {t_wall/3600:.2f} h")

    return {
        "e": float(e_tot),
        "converged": bool(mc.converged),
        "n_iters": iter_count[0],
        "t_h": round(t_wall/3600, 3),
    }


# ── Main ───────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--system",    choices=list(SYSTEMS.keys()), required=True)
    p.add_argument("--M",         type=int, default=350,
                   help="DMRG bond dimension for CASSCF (default 350)")
    p.add_argument("--M-qicas",   type=int, default=100,
                   help="DMRG bond dimension for QICAS step (default 100)")
    p.add_argument("--max-macro", type=int, default=150,
                   help="Max CASSCF macro-iterations per run (default 150)")
    p.add_argument("--work-dir",  type=str, default="./tmp_dmrg_casscf")
    p.add_argument("--skip-hf-run", action="store_true",
                   help="Skip Run 1 (HF-init), run QICAS-init only")
    p.add_argument("--skip-casscf", action="store_true",
                   help="Run QICAS only, skip CASSCF runs (for active space verification)")
    p.add_argument("--skip-qicas", action="store_true",
                   help="Skip QICAS step, run HF-init only")
    args = p.parse_args()

    sys_dict = SYSTEMS[args.system]
    work_dir = os.path.abspath(args.work_dir)
    os.makedirs(work_dir, exist_ok=True)

    t_total = time.time()

    # ── Step 1: Build molecule and ROHF ───────────────────────────────────
    mol, mf = build_mol_and_hf(sys_dict)

    result = {
        "system":          sys_dict["name"],
        "cas":             f"({sys_dict['n_elec']}e,{sys_dict['n_active']}o)",
        "spin_2s":         sys_dict["spin_2s"],
        "charge":          sys_dict["charge"],
        "M_casscf":        args.M,
        "M_qicas":         args.M_qicas,
        "conv_tol_grad":   3e-4,
        "conv_tol_energy": 1e-8,
        "e_rohf":          float(mf.e_tot),
    }

    if not args.skip_hf_run and not args.skip_casscf:
        # ── Step 2: Run 1 — ROHF initialization ──────────────────────────────
        log.info("\n" + "="*60)
        log.info("RUN 1: DMRG-CASSCF from ROHF canonical orbitals")
        log.info("="*60)
        r1 = run_casscf(mol, mf, sys_dict, mf.mo_coeff,
                        args.M, args.max_macro, work_dir, "HF-init")
        result.update({
            "e_casscf_hf_init":      r1["e"],
            "converged_hf_init":     r1["converged"],
            "n_macro_iter_hf_init":  r1["n_iters"],
            "t_h_hf_init":           r1["t_h"],
        })
    else:
        r1 = None
        log.info("Skipping Run 1 (HF-init) — --skip-hf-run")

    # ── Step 3: QICAS step — generate compatible MOs ─────────────────────
    mo_qicas = None
    if not args.skip_qicas:
        # Check if MOs already saved from a prior run
        mo_cache = os.path.join(work_dir, f"qicas_mo_{sys_dict['name']}.npy")
        if os.path.exists(mo_cache):
            mo_qicas = np.load(mo_cache)
            log.info(f"Loaded cached QICAS MOs: {mo_cache}  shape={mo_qicas.shape}")
        else:
            mo_qicas = run_qicas(mol, mf, sys_dict, args.M_qicas, work_dir)

    # ── Step 4: Run 2 — QICAS initialization ─────────────────────────────
    if mo_qicas is not None:
        log.info("\n" + "="*60)
        if args.skip_casscf:
            log.info("--skip-casscf: QICAS done, skipping CASSCF runs")
            return
        log.info("RUN 2: DMRG-CASSCF from QICAS-optimized orbitals")
        log.info("="*60)
        r2 = run_casscf(mol, mf, sys_dict, mo_qicas,
                        args.M, args.max_macro, work_dir, "QICAS-init")
        speedup = r1["n_iters"] - r2["n_iters"]
        result.update({
            "e_casscf_qicas_init":      r2["e"],
            "converged_qicas_init":     r2["converged"],
            "n_macro_iter_qicas_init":  r2["n_iters"],
            "t_h_qicas_init":           r2["t_h"],
            "iter_speedup_qicas_vs_hf": speedup,
            "e_diff_hf_vs_qicas_mha":   (r1["e"] - r2["e"]) * 1000,
        })
    else:
        log.info("Skipping Run 2 (no QICAS MOs available)")

    # ── Summary ───────────────────────────────────────────────────────────
    result["t_total_h"] = round((time.time() - t_total) / 3600, 3)

    log.info("\n" + "="*60)
    log.info(f"FINAL SUMMARY: {sys_dict['name']}")
    log.info("="*60)
    log.info(f"  E(ROHF)           = {mf.e_tot:.10f} Ha")
    log.info(f"  E(CASSCF HF-init) = {r1['e']:.10f} Ha  "
             f"conv={r1['converged']}  iters={r1['n_iters']}")
    if mo_qicas is not None:
        log.info(f"  E(CASSCF QI-init) = {r2['e']:.10f} Ha  "
                 f"conv={r2['converged']}  iters={r2['n_iters']}")
        if speedup is not None:
            log.info(f"  Iteration speedup = {speedup:+d} "
                     f"({'QICAS faster' if speedup > 0 else 'HF faster' if speedup < 0 else 'same'})")
        if r1 is not None:
            log.info(f"  Energy difference = {(r1['e']-r2['e'])*1000:+.3f} mHa "
                     f"({'same minimum' if abs(r1['e']-r2['e'])*1000 < 1 else 'different minima'})")

    out = f"dmrg_casscf_{args.system}_M{args.M}.json"
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"\nSaved: {out}")

    # One-liner for SLURM log grep
    print(f"\nRESULT HF-init:   E={r1['e']:.10f}  "
          f"conv={r1['converged']}  iters={r1['n_iters']}")
    if mo_qicas is not None:
        print(f"RESULT QICAS-init: E={r2['e']:.10f}  "
              f"conv={r2['converged']}  iters={r2['n_iters']}  "
              f"speedup={speedup:+d}")


if __name__ == "__main__":
    main()
