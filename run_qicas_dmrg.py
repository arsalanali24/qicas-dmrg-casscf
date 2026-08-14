#!/usr/bin/env python3
"""
run_qicas_dmrg.py
─────────────────
Complete pipeline for a new transition metal system:

  Step 1: ROHF  (atom initial guess — correct for HS open-shell TM)
  Step 2: QICAS (determine window → DMRG at M_qicas → F_QI rotation)
  Step 3: DMRG-CASSCF Run 1 — HF-init   (baseline)
  Step 4: DMRG-CASSCF Run 2 — QICAS-init (comparison)

All steps use the SAME HF reference so MOs are directly comparable.

KNOWN FIXES (from MnBr4/MnCl4 debugging):
  - init_guess='atom'  mandatory for HS TM (minao finds wrong local min)
  - work_dir must be absolute path (block2 nests relative paths)
  - DMRG schedule must be monotonically increasing in M
  - step3 returns 4 values, step5 returns 4, step6 returns 6
  - build_casci_mo_coeff returns (mo_coeff, ncore)
  - conv_tol_grad = 3e-4 (PySCF default; 1e-4 too tight for DMRG noise)

Usage:
  # Interactive (asks for system info):
  python3 run_qicas_dmrg.py

  # From config file:
  python3 run_qicas_dmrg.py --config my_system.yaml

  # Skip CASSCF, run QICAS only:
  python3 run_qicas_dmrg.py --qicas-only

  # Use existing QICAS MOs (skip QICAS step):
  python3 run_qicas_dmrg.py --qicas-mo path/to/qicas_mo.npy

  # Test mode (validates environment with MnBr4 at M=50):
  python3 run_qicas_dmrg.py --test
"""

import argparse
import json
import logging
import os
import sys
import time
import textwrap
import numpy as np

log = logging.getLogger("qicas_dmrg")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[logging.StreamHandler()]
)

# ── Known systems (add new ones here or pass via --config) ─────────────────
KNOWN_SYSTEMS = {
    "MnBr4_sextet": {
        "name":     "CSD_MnBr4_2m_tet_spin5",
        "spin_2s":  5,
        "charge":   -2,
        "n_active": 13,
        "n_elec":   19,
        "basis":    "def2-svp",
        "ecp":      None,
        "atoms": """Mn  0.000000  0.000000  0.000000
Br  1.491296  1.491296  1.491296
Br -1.491296 -1.491296  1.491296
Br -1.491296  1.491296 -1.491296
Br  1.491296 -1.491296 -1.491296""",
        "notes": "CSD geometry, Mn-Br=2.583 Å, validated ΔCASCI=-560 mHa",
    },
    "MnCl4_sextet": {
        "name":     "CSD_MnCl4_2m_tet_spin5",
        "spin_2s":  5,
        "charge":   -2,
        "n_active": 14,
        "n_elec":   21,
        "basis":    "def2-svp",
        "ecp":      None,
        "atoms": """Mn  0.000000  0.000000  0.000000
Cl  1.356773  1.356773  1.356773
Cl -1.356773 -1.356773  1.356773
Cl -1.356773  1.356773 -1.356773
Cl  1.356773 -1.356773 -1.356773""",
        "notes": "CSD geometry, Mn-Cl=2.350 Å, validated ΔCASCI=-239 mHa",
    },
}

# ── Test system (fast validation before new calculations) ─────────────────
TEST_SYSTEM = {
    "name":     "TEST_MnBr4_sextet",
    "spin_2s":  5,
    "charge":   -2,
    "n_active": 13,
    "n_elec":   19,
    "basis":    "def2-svp",
    "ecp":      None,
    "atoms":    KNOWN_SYSTEMS["MnBr4_sextet"]["atoms"],
    "M_qicas":  50,     # fast for testing
    "M_casscf": 100,
    "max_macro": 5,
    "expected_e_rohf_range": (-11438.43, -11438.41),  # sanity check
}


# ── Environment detection ──────────────────────────────────────────────────
def detect_environment():
    """Auto-detect block2, pipeline path, scratch directory."""
    env = {}

    # block2main executable
    block2 = os.popen("which block2main 2>/dev/null").read().strip()
    if not block2:
        # Try common locations
        candidates = [
            os.path.expanduser("~/envs/block2env/bin/block2main"),
            os.path.expanduser("~/.block2_fix/block2env/bin/block2main"),
        ]
        for c in candidates:
            if os.path.exists(c):
                block2 = c
                break
    env["block2_exe"] = block2
    env["block2_ok"]  = bool(block2)

    # QICAS pipeline directory
    pipeline_candidates = [
        os.path.expanduser(
            "~/activeml/qio/QIO-master/examples_benchmark"),
        os.path.expanduser(
            "~/qio/QIO-master/examples_benchmark"),
    ]
    env["pipeline_dir"] = None
    for p in pipeline_candidates:
        if os.path.exists(os.path.join(p, "qicas_benchmark.py")):
            env["pipeline_dir"] = p
            break

    # Scratch directory
    scratch_candidates = [
        f"/scratch/hpc-prf-qehpc/{os.environ.get('USER', 'user')}",
        "/scratch",
        "/tmp",
    ]
    env["scratch"] = next(
        (s for s in scratch_candidates if os.path.isdir(s)),
        os.path.expanduser("~/tmp_qicas"))

    return env


def check_environment(env):
    """Print environment status and raise if critical components missing."""
    log.info("─" * 50)
    log.info("Environment check:")
    log.info(f"  block2main:    {'✓ ' + env['block2_exe'] if env['block2_ok'] else '✗ NOT FOUND'}")
    log.info(f"  pipeline:      {'✓ ' + env['pipeline_dir'] if env['pipeline_dir'] else '✗ NOT FOUND'}")
    log.info(f"  scratch:       {env['scratch']}")
    log.info(f"  OMP_threads:   {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    log.info("─" * 50)

    if not env["block2_ok"]:
        raise RuntimeError(
            "block2main not found. Run: source ~/.block2_fix/block2_env.sh")
    if not env["pipeline_dir"]:
        raise RuntimeError(
            "qicas_benchmark.py not found. Check pipeline path.")


# ── Interactive system builder ────────────────────────────────────────────
def ask_system_interactively():
    """Ask user for system information interactively."""
    print("\n" + "="*60)
    print("QICAS + DMRG-CASSCF Setup")
    print("="*60)

    # Check known systems
    if KNOWN_SYSTEMS:
        print("\nKnown systems:")
        for i, (key, s) in enumerate(KNOWN_SYSTEMS.items()):
            print(f"  [{i}] {key}  ({s['notes']})")
        print(f"  [{len(KNOWN_SYSTEMS)}] New system")
        choice = input(f"\nSelect [0-{len(KNOWN_SYSTEMS)}]: ").strip()
        if choice.isdigit() and int(choice) < len(KNOWN_SYSTEMS):
            key = list(KNOWN_SYSTEMS.keys())[int(choice)]
            sys_dict = dict(KNOWN_SYSTEMS[key])
            print(f"Using: {key}")
            return sys_dict

    # New system
    print("\nEnter new system details:")
    name    = input("System name (e.g. IrBr6_quintet): ").strip()
    charge  = int(input("Charge (e.g. -2): ").strip())
    spin_2s = int(input("Spin 2S (e.g. 4 for quintet): ").strip())
    basis   = input("Basis set [def2-svp]: ").strip() or "def2-svp"
    n_active= int(input("QICAS n_active (from previous QICAS run, or estimate): ").strip())
    n_elec  = int(input("QICAS n_elec (active electrons): ").strip())

    print("\nEnter geometry (XYZ format, blank line to finish):")
    print("Example:  Ir  0.0  0.0  0.0")
    atoms_lines = []
    while True:
        line = input()
        if not line.strip():
            break
        atoms_lines.append(line)
    atoms = "\n".join(atoms_lines)

    # ECP for 4d/5d metals
    metal = atoms_lines[0].split()[0] if atoms_lines else ""
    metals_4d5d = ["Mo","Tc","Ru","Rh","Pd","Ag","Cd",
                   "W","Re","Os","Ir","Pt","Au","Hg"]
    ecp = "def2-svp" if metal in metals_4d5d else None
    if ecp:
        print(f"  Auto-detected 4d/5d metal ({metal}) → ECP = def2-svp")

    return {
        "name":     name,
        "spin_2s":  spin_2s,
        "charge":   charge,
        "n_active": n_active,
        "n_elec":   n_elec,
        "basis":    basis,
        "ecp":      ecp,
        "atoms":    atoms,
    }


# ── DMRG schedule ─────────────────────────────────────────────────────────
def build_dmrg_schedule(target_M):
    """Monotonically increasing M schedule. Never decreases."""
    m1 = min(64,  target_M)
    m2 = min(128, target_M)
    m3 = min(200, target_M)
    m4 = min(250, target_M)

    if target_M <= 100:
        return dict(
            sweeps=[4,  4,  8,  8, 16],
            maxMs =[m1, m2, target_M, target_M, target_M],
            tols  =[1e-5, 1e-5, 1e-6, 1e-7, 1e-8],
            noises=[1e-3, 1e-4, 1e-4, 0, 0])
    elif target_M <= 200:
        return dict(
            sweeps=[4,  4,  8,  8, 16, 16],
            maxMs =[m1, m2, m3, target_M, target_M, target_M],
            tols  =[1e-5, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9],
            noises=[1e-3, 1e-4, 1e-4, 1e-5, 0, 0])
    elif target_M <= 350:
        return dict(
            sweeps=[4,  4,  4,  8,  8,  8, 16, 16],
            maxMs =[m1, m2, m2, m3, m3, target_M, target_M, target_M],
            tols  =[1e-5, 1e-5, 1e-6, 1e-6, 1e-7, 1e-7, 1e-8, 1e-9],
            noises=[1e-3, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 0, 0])
    else:
        return dict(
            sweeps=[4,  4,  4,  8,  8,  8,  8, 16, 16],
            maxMs =[m1, m2, m2, m3, m3, m4, target_M, target_M, target_M],
            tols  =[1e-5, 1e-5, 1e-6, 1e-6, 1e-7, 1e-7, 1e-7, 1e-8, 1e-9],
            noises=[1e-3, 1e-4, 1e-4, 1e-4, 1e-5, 1e-5, 1e-5, 0, 0])


# ── Step 1: ROHF ──────────────────────────────────────────────────────────
def run_rohf(sys_dict):
    """
    Run ROHF with atom initial guess.
    CRITICAL: atom init_guess avoids wrong local minimum for HS TM.
    minao (PySCF default) gives ~500 mHa higher energy for [MnBr4]^2-.
    """
    from pyscf import gto, scf

    mol = gto.Mole()
    mol.atom      = sys_dict["atoms"]
    mol.basis     = sys_dict["basis"]
    mol.charge    = sys_dict["charge"]
    mol.spin      = sys_dict["spin_2s"]
    mol.verbose   = 4
    mol.max_memory = 32000
    if sys_dict.get("ecp"):
        mol.ecp = sys_dict["ecp"]
    mol.build()

    log.info(f"Molecule: {mol.nelectron} electrons, {mol.nao_nr()} AOs")

    # atom init_guess — mandatory for HS TM, avoids minao local minimum
    mf = scf.ROHF(mol)
    mf.max_cycle  = 200
    mf.init_guess = 'atom'
    mf.kernel()
    log.info(f"ROHF: E = {mf.e_tot:.10f} Ha  converged = {mf.converged}")

    if not mf.converged:
        log.warning("ROHF not converged — retrying with stability analysis")
        mo_new = mf.stability()[0]
        mf.kernel(mo_new)
        log.info(f"ROHF (stability): E = {mf.e_tot:.10f}  converged = {mf.converged}")

    if not mf.converged:
        log.warning("ROHF still not converged — using best result")

    return mol, mf


# ── Step 2: QICAS ─────────────────────────────────────────────────────────
def run_qicas(mol, mf, sys_dict, M_qicas, work_dir, pipeline_dir):
    """
    Run QICAS orbital rotation using qicas_benchmark.py pipeline.
    All API calls use exact signatures confirmed for this pipeline version:
      determine_window(mol, mf)                              → window dict
      run_dmrg_on_window(mol, mf, window, max_M)            → mc, e, t
      step3_extract_rdms(mc, n_orbs, n_elec)                → γ, Γ, s, occ  [4 values]
      step5_optimize_orbitals(γ, Γ, inactive, active)       → U, γ_opt, Γ_opt, hist [4 values]
      step6_determine_active_space(γ_opt, Γ_opt, n, socc)   → n_qi, n_e, s, occ, idx, active [6 values]
      build_casci_mo_coeff(mo, n_mo, win, active, occ, n_e) → (mo_coeff, ncore) [tuple]
    """
    if pipeline_dir not in sys.path:
        sys.path.insert(0, pipeline_dir)

    try:
        import qicas_benchmark as qb
    except ImportError as e:
        raise RuntimeError(f"Cannot import qicas_benchmark from {pipeline_dir}: {e}")

    ncas = sys_dict["n_active"]

    log.info("\n" + "="*60)
    log.info("STEP 2: QICAS orbital optimization")
    log.info(f"  M_qicas = {M_qicas}  target_ncas = {ncas}")
    log.info("="*60)

    # 2a: Determine frontier window from ROHF occupations
    window        = qb.determine_window(mol, mf)
    n_orbs        = window["actual_window_size"]
    n_elec_active = window["n_elec_active"]
    socc_rel      = window["socc_window_rel"]
    log.info(f"  Window: {n_orbs} orbitals  "
             f"[{window['window_start']}–{window['window_end']}]  "
             f"({window['n_frozen']} frozen core)")

    # 2b: DMRG on window
    log.info(f"  Running DMRG at M={M_qicas}...")
    mc_dmrg, e_dmrg, t_dmrg = qb.run_dmrg_on_window(
        mol, mf, window, max_M=M_qicas)
    log.info(f"  DMRG: E = {e_dmrg:.10f} Ha  t = {t_dmrg:.1f}s")

    # 2c: Extract 1- and 2-RDMs (returns 4 values)
    gamma, Gamma, s_vals, occ_nums = qb.step3_extract_rdms(
        mc_dmrg, n_orbs, n_elec_active)
    log.info(f"  RDMs extracted  shape = {gamma.shape}")
    log.info(f"  Initial s_vals (top 5): {np.sort(s_vals)[::-1][:5].round(4)}")

    # 2d: F_QI orbital rotation (returns 4 values: U, γ_opt, Γ_opt, history)
    log.info("  Running F_QI orbital rotation...")
    inactive_idx = [i for i in range(n_orbs) if i not in socc_rel]
    active_idx   = list(socc_rel)
    U_total, gamma_opt, Gamma_opt, opt_history = qb.step5_optimize_orbitals(
        gamma, Gamma, inactive_idx, active_idx)
    fqi_i = opt_history[0]  if opt_history else 0
    fqi_f = opt_history[-1] if opt_history else 0
    log.info(f"  F_QI: {fqi_i:.4f} → {fqi_f:.4f}  "
             f"(reduction = {(1-fqi_f/fqi_i)*100:.1f}%)" if fqi_i else "")

    # 2e: Determine active space (returns 6 values)
    n_qi, n_e_qi, s_opt, occ_opt, sorted_idx, active_window_rel = \
        qb.step6_determine_active_space(
            gamma_opt, Gamma_opt, n_orbs, socc_rel)
    log.info(f"  QICAS active: {n_qi} orbitals, {n_e_qi} electrons")
    log.info(f"  Active (window-relative): {list(active_window_rel)}")

    # 2f: Build full MO matrix (returns tuple: (mo_coeff, ncore))
    n_mo = mol.nao_nr()
    mo_qicas, ncore_qicas = qb.build_casci_mo_coeff(
        mf.mo_coeff, n_mo, window,
        np.array(active_window_rel),
        occ_opt, n_elec_active)
    log.info(f"  QICAS MO shape: {mo_qicas.shape}  ncore = {ncore_qicas}")

    # Save MOs
    mo_path = os.path.join(work_dir, f"qicas_mo_{sys_dict['name']}.npy")
    np.save(mo_path, mo_qicas)
    log.info(f"  Saved: {mo_path}")

    return mo_qicas, {
        "n_active_qicas": int(n_qi),
        "n_elec_qicas":   int(n_e_qi),
        "fqi_initial":    float(fqi_i),
        "fqi_final":      float(fqi_f),
        "e_dmrg":         float(e_dmrg),
        "mo_path":        mo_path,
    }


# ── Step 3/4: DMRG-CASSCF ─────────────────────────────────────────────────
def run_casscf(mol, mf, sys_dict, mo_start, M, max_macro,
               work_dir, label, env):
    """
    Run DMRG-CASSCF from a given set of starting MOs.
    conv_tol_grad = 3e-4 (PySCF default, appropriate for DMRG noise at M=350)
    conv_tol_energy = 1e-8 Ha (0.01 mHa)
    """
    from pyscf import mcscf, dmrgscf

    ncas   = sys_dict["n_active"]
    nelec  = sys_dict["n_elec"]
    spin   = sys_dict["spin_2s"]
    n_alpha = (nelec + spin) // 2
    n_beta  = (nelec - spin) // 2

    log.info(f"\n{'─'*50}")
    log.info(f"{label}: CAS({nelec}e,{ncas}o)  M={M}  max_macro={max_macro}")
    log.info(f"{'─'*50}")

    mc = mcscf.CASSCF(mf, ncas, (n_alpha, n_beta))
    mc.conv_tol_grad   = 3e-4   # PySCF default — appropriate for DMRG noise
    mc.conv_tol        = 1e-8   # 0.01 mHa energy threshold
    mc.max_cycle_macro = max_macro
    mc.max_cycle_micro = 10
    mc.verbose         = 4

    # Scratch directory per run (isolated — prevents block2 file collisions)
    scratch = os.path.abspath(
        os.path.join(work_dir, label.replace(" ", "_").replace(":", "")))
    os.makedirs(scratch, exist_ok=True)
    os.makedirs(os.path.join(scratch, "node0"), exist_ok=True)

    dmrgscf.settings.BLOCKEXE  = env["block2_exe"]
    dmrgscf.settings.MPIPREFIX = ""

    mc.fcisolver = dmrgscf.DMRGCI(mol, maxM=M, tol=1e-9)
    mc.fcisolver.runtimeDir       = scratch
    mc.fcisolver.scratchDirectory = scratch   # must be absolute
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
    def callback(envs):
        iter_count[0] += 1
        e    = envs.get("e_tot",    float("nan"))
        de   = envs.get("de",       float("nan"))
        gorb = envs.get("norm_gorb",float("nan"))
        log.info(f"  [{label}] Macro {iter_count[0]:4d}: "
                 f"E={e:.10f}  dE={de:+.2e}  |gorb|={gorb:.2e}")

    mc.callback = callback

    t0    = time.time()
    e_tot = mc.kernel(mo_start)[0]
    t_s   = time.time() - t0

    log.info(f"\n  {label}: E={e_tot:.10f}  conv={mc.converged}  "
             f"iters={iter_count[0]}  t={t_s/3600:.2f}h")

    return {
        "e":         float(e_tot),
        "converged": bool(mc.converged),
        "n_iters":   iter_count[0],
        "t_h":       round(t_s/3600, 3),
    }


# ── Test mode ─────────────────────────────────────────────────────────────
def run_test(env):
    """Fast validation with MnBr4 at M=50, 5 macro-iterations."""
    log.info("\n" + "="*60)
    log.info("TEST MODE: MnBr4 sextet at M=50 (5 macro iterations)")
    log.info("="*60)

    ts = TEST_SYSTEM
    work_dir = os.path.join(env["scratch"], "qicas_dmrg_test")
    os.makedirs(work_dir, exist_ok=True)

    mol, mf = run_rohf(ts)

    # Sanity check HF energy
    lo, hi = ts["expected_e_rohf_range"]
    if not (lo < mf.e_tot < hi):
        log.error(f"ROHF energy {mf.e_tot:.6f} outside expected range [{lo}, {hi}]")
        log.error("Check: atom initial guess, geometry, charge, spin")
        return False
    log.info(f"  ROHF sanity check PASSED: {mf.e_tot:.6f} in [{lo}, {hi}]")

    _, qicas_info = run_qicas(
        mol, mf, ts, ts["M_qicas"], work_dir, env["pipeline_dir"])

    log.info(f"  QICAS PASSED: n_active={qicas_info['n_active_qicas']}  "
             f"F_QI: {qicas_info['fqi_initial']:.3f}→{qicas_info['fqi_final']:.3f}")

    mo_qicas = np.load(qicas_info["mo_path"])
    r = run_casscf(mol, mf, ts, mo_qicas,
                   ts["M_casscf"], ts["max_macro"], work_dir, "TEST", env)

    log.info(f"  CASSCF test: E={r['e']:.6f}  iters={r['n_iters']}")
    log.info("TEST PASSED ✓  Environment is correctly configured.")
    return True


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config",      type=str,  default=None,
                   help="YAML config file with system definition")
    p.add_argument("--system",      type=str,  default=None,
                   choices=list(KNOWN_SYSTEMS.keys()),
                   help="Use a known system directly")
    p.add_argument("--qicas-only",  action="store_true",
                   help="Run ROHF+QICAS only, skip CASSCF")
    p.add_argument("--skip-hf-run", action="store_true",
                   help="Skip CASSCF Run 1 (HF-init), run QICAS-init only")
    p.add_argument("--qicas-mo",    type=str,  default=None,
                   help="Path to existing QICAS MOs (skip QICAS step)")
    p.add_argument("--M",           type=int,  default=350,
                   help="DMRG bond dimension for CASSCF (default 350)")
    p.add_argument("--M-qicas",     type=int,  default=100,
                   help="DMRG bond dimension for QICAS step (default 100)")
    p.add_argument("--max-macro",   type=int,  default=150,
                   help="Max CASSCF macro-iterations (default 150)")
    p.add_argument("--work-dir",    type=str,  default=None,
                   help="Working/scratch directory (auto-detected if not set)")
    p.add_argument("--test",        action="store_true",
                   help="Run validation test with MnBr4 before proceeding")
    args = p.parse_args()

    # ── Detect environment ─────────────────────────────────────────────
    env = detect_environment()
    check_environment(env)

    # ── Work directory ─────────────────────────────────────────────────
    work_dir = os.path.abspath(
        args.work_dir or
        os.path.join(env["scratch"], "qicas_dmrg",
                     f"run_{time.strftime('%Y%m%d_%H%M%S')}"))
    os.makedirs(work_dir, exist_ok=True)
    log.info(f"Work directory: {work_dir}")

    # ── Test mode ──────────────────────────────────────────────────────
    if args.test:
        ok = run_test(env)
        sys.exit(0 if ok else 1)

    # ── Load system definition ─────────────────────────────────────────
    if args.config:
        try:
            import yaml
            with open(args.config) as f:
                sys_dict = yaml.safe_load(f)
            log.info(f"Loaded system from {args.config}")
        except ImportError:
            # Fallback to JSON if yaml not available
            with open(args.config) as f:
                sys_dict = json.load(f)
    elif args.system:
        sys_dict = dict(KNOWN_SYSTEMS[args.system])
        log.info(f"Using known system: {args.system}")
    else:
        sys_dict = ask_system_interactively()

    log.info(f"\nSystem: {sys_dict['name']}")
    log.info(f"  CAS({sys_dict['n_elec']}e, {sys_dict['n_active']}o)  "
             f"2S={sys_dict['spin_2s']}  charge={sys_dict['charge']}")

    result = {
        "system":    sys_dict["name"],
        "cas":       f"({sys_dict['n_elec']}e,{sys_dict['n_active']}o)",
        "spin_2s":   sys_dict["spin_2s"],
        "charge":    sys_dict["charge"],
        "M_casscf":  args.M,
        "M_qicas":   args.M_qicas,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    # ── Step 1: ROHF ──────────────────────────────────────────────────
    mol, mf = run_rohf(sys_dict)
    result["e_rohf"] = float(mf.e_tot)

    # ── Step 2: QICAS ─────────────────────────────────────────────────
    mo_qicas   = None
    qicas_info = None

    if args.qicas_mo:
        if os.path.exists(args.qicas_mo):
            mo_qicas = np.load(args.qicas_mo)
            log.info(f"Loaded existing QICAS MOs: {args.qicas_mo}  "
                     f"shape={mo_qicas.shape}")
        else:
            log.warning(f"QICAS MO file not found: {args.qicas_mo}")

    if mo_qicas is None:
        mo_qicas, qicas_info = run_qicas(
            mol, mf, sys_dict, args.M_qicas, work_dir, env["pipeline_dir"])
        result.update(qicas_info)

    if args.qicas_only:
        log.info("\nQICAS-only mode — stopping before CASSCF.")
        _save(result, work_dir, sys_dict)
        return

    # ── Step 3: DMRG-CASSCF Run 1 (HF-init) ──────────────────────────
    if not args.skip_hf_run:
        log.info("\n" + "="*60)
        log.info("RUN 1: DMRG-CASSCF from ROHF orbitals")
        log.info("="*60)
        r1 = run_casscf(mol, mf, sys_dict, mf.mo_coeff,
                        args.M, args.max_macro, work_dir, "HF-init", env)
        result.update({
            "e_casscf_hf_init":     r1["e"],
            "converged_hf_init":    r1["converged"],
            "n_macro_hf_init":      r1["n_iters"],
            "t_h_hf_init":          r1["t_h"],
        })
    else:
        r1 = None
        log.info("Skipping Run 1 (--skip-hf-run)")

    # ── Step 4: DMRG-CASSCF Run 2 (QICAS-init) ────────────────────────
    log.info("\n" + "="*60)
    log.info("RUN 2: DMRG-CASSCF from QICAS orbitals")
    log.info("="*60)
    r2 = run_casscf(mol, mf, sys_dict, mo_qicas,
                    args.M, args.max_macro, work_dir, "QICAS-init", env)
    result.update({
        "e_casscf_qicas_init":  r2["e"],
        "converged_qicas_init": r2["converged"],
        "n_macro_qicas_init":   r2["n_iters"],
        "t_h_qicas_init":       r2["t_h"],
    })

    if r1 is not None:
        speedup = r1["n_iters"] - r2["n_iters"]
        e_diff  = (r1["e"] - r2["e"]) * 1000
        result["iter_speedup"]      = speedup
        result["e_diff_mha"]        = round(e_diff, 4)
        result["qicas_lower_energy"]= e_diff > 0

    # ── Summary ───────────────────────────────────────────────────────
    _save(result, work_dir, sys_dict)
    _print_summary(result, r1, r2)


def _save(result, work_dir, sys_dict):
    out = os.path.join(work_dir, f"result_{sys_dict['name']}.json")
    with open(out, "w") as f:
        json.dump(result, f, indent=2)
    log.info(f"\nSaved: {out}")


def _print_summary(result, r1, r2):
    log.info("\n" + "="*60)
    log.info("FINAL SUMMARY")
    log.info("="*60)
    log.info(f"  E(ROHF)             = {result.get('e_rohf','N/A'):.10f} Ha")
    if r1:
        log.info(f"  E(CASSCF HF-init)   = {r1['e']:.10f} Ha  "
                 f"conv={r1['converged']}  iters={r1['n_iters']}")
    log.info(f"  E(CASSCF QICAS-init)= {r2['e']:.10f} Ha  "
             f"conv={r2['converged']}  iters={r2['n_iters']}")
    if r1:
        speedup = result.get("iter_speedup", "?")
        e_diff  = result.get("e_diff_mha",   "?")
        log.info(f"  Iteration speedup   = {speedup:+d}  "
                 f"({'QICAS faster' if isinstance(speedup,int) and speedup>0 else 'HF faster'})")
        log.info(f"  Energy difference   = {e_diff:+.3f} mHa  "
                 f"({'QICAS lower' if isinstance(e_diff,float) and e_diff>0 else 'HF lower'})")
    log.info("="*60)

    # One-liner for SLURM grep
    if r1:
        print(f"\nRESULT HF-init:    E={r1['e']:.10f}  "
              f"conv={r1['converged']}  iters={r1['n_iters']}")
    print(f"RESULT QICAS-init: E={r2['e']:.10f}  "
          f"conv={r2['converged']}  iters={r2['n_iters']}")


if __name__ == "__main__":
    main()
