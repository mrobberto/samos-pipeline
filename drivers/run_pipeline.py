#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SAMOS master pipeline driver.

PURPOSE
-------
Run the SAMOS reduction pipeline in canonical stage order using the active
target configuration module. The driver launches existing step scripts as
subprocesses and does not re-implement the science logic of the pipeline.

DESIGN PRINCIPLES
-----------------
- Keep the driver lightweight and easy to maintain.
- Use config-driven canonical products and directories.
- Treat the script registry below as the authoritative execution order.
- Keep stage semantics explicit:
    Step09 = ABAB OH-clean stage
    Step10 = telluric correction
    Step11 = photometric flux calibration
- Allow partial execution with --from-step / --to-step / --only.
- Allow parity-restricted runs for set-based stages.
- Optionally run registered QC companions after each stage.

MAINTENANCE NOTES
-----------------
- Update SCRIPT_REGISTRY when the operational stage structure changes.
- Update QC_REGISTRY when the preferred QC scripts change.
- Update OUTPUT_CHECKS whenever a stage's canonical output contract changes.
- The driver should prefer canonical config variables such as:
    EXTRACT1D_WAV
    EXTRACT1D_OHCLEAN
    EXTRACT1D_TELLCOR
    STEP11_RADEC
    STEP11_PHOTCAT
    EXTRACT1D_FLUXCAL
- Avoid adding target-specific filesystem logic here; that belongs in the
  active target configuration profile.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
from pathlib import Path
import os
import shlex
import subprocess
import sys
import time
from typing import Iterable, Sequence


@dataclass(frozen=True)
class Stage:
    """One pipeline stage entry used by the master driver registry."""
    key: str
    script: str
    description: str
    sets: tuple[str, ...] = ()
    args_template: str = ""

    @property
    def is_set_based(self) -> bool:
        return len(self.sets) > 0
    

# -----------------------------------------------------------------------------
# Stage registry
# -----------------------------------------------------------------------------
# The order here is the authoritative pipeline execution order.
#
# IMPORTANT:
# - Step09 is a single ABAB OH-clean stage.
# - The historical 09a/09b/09bm/09c subdivision is no longer operational.
# - Step11c should point to the currently adopted production script.
# -----------------------------------------------------------------------------
SCRIPT_REGISTRY: tuple[Stage, ...] = (
    Stage("00",   "pipeline/step00_orient/step00_rotate.py",                        "Rotate frames to the standard orientation"),
    Stage("01",   "pipeline/step01_bias/step01_masterbias.py",                      "Build master bias"),
    Stage("02",   "pipeline/step02_biascorr/step02_biascorr.py",                    "Apply bias correction"),
    Stage("03",   "pipeline/step03_crclean/step03_crclean.py",                      "Cosmic-ray cleaning"),
    Stage("03.5", "pipeline/step03p5_rowstripe/step03p5_remove_rowstripe.py",       "Remove row-wise striping and match quadrant pedestals"),
    Stage("04",   "pipeline/step04_traces/step04_make_traces.py",                   "Build slit traces and geometry",                  sets=("EVEN", "ODD"), args_template="--set {set}"),
    Stage("05",   "pipeline/step05_pixflat/step05_build_pixflat.py",                "Build pixel flat from quartz differences",        sets=("EVEN", "ODD"), args_template="--set {set}"),
    Stage("06a",  "pipeline/step06_science_rectify/step06a_make_final_science.py",  "Build FinalScience mosaic"),
    Stage("06b",  "pipeline/step06_science_rectify/step06b_apply_pixflat_clip.py",  "Apply pixel flat to FinalScience",                sets=("EVEN", "ODD"), args_template="--traceset {set}"),
    Stage("06c",  "pipeline/step06_science_rectify/step06c_rectify_tracecoords.py", "Rectify slitlets into TRACECOORDS",               sets=("EVEN", "ODD"), args_template="--traceset {set}"),
    Stage("07a",  "pipeline/step07_wavecal/step07a_make_arc_diff.py",               "Build arc-difference frame"),
    Stage("07b",  "pipeline/step07_wavecal/step07b_apply_pixflat_arc.py",           "Apply pixel flat to arc frame"),
    Stage("07c",  "pipeline/step07_wavecal/step07c_extract_arc_1d.py",              "Extract rectified 1D arc slit spectra",           sets=("EVEN", "ODD"), args_template="--traceset {set}"),
    Stage("07d",  "pipeline/step07_wavecal/step07d_find_line_shifts.py",            "Measure initial relative arc shifts",             sets=("EVEN", "ODD"), args_template="--traceset {set}"),
    Stage("07e",  "pipeline/step07_wavecal/step07e_refine_stack_arc.py",            "Refine arc-stack alignment",                      sets=("EVEN", "ODD"), args_template="--set {set}"),
    Stage("07f",  "pipeline/step07_wavecal/step07f_build_master_arc.py",            "Build aligned master arc"),
    Stage("07g",  "pipeline/step07_wavecal/step07g_solve_wavelength.py",            "Fit global wavelength solution"),
    Stage("07h",  "pipeline/step07_wavecal/step07h_propagate_wavesol.py",           "Propagate wavelength solution to all slit arcs"),
    Stage("08a",  "pipeline/step08_extract1d/step08a_extract_1d.py",                "Ridge-guided optimal extraction",                 sets=("EVEN", "ODD"), args_template="--set {set}"),
    Stage("08b",  "pipeline/step08_extract1d/step08b_merge_even_odd.py",            "Merge EVEN and ODD extracted spectra"),
    Stage("08c",  "pipeline/step08_extract1d/step08c_attach_wavelength.py",         "Attach wavelength vectors to extracted spectra"),
    Stage("09",   "pipeline/step09_oh_refine/step09_abab_driver.py",                "Full OH cleanup and preferred-spectrum selection (A/B/A/B)"),
    Stage("10a",  "pipeline/step10_telluric/step10a_build_telluric_template.py",    "Build empirical O2 telluric template"),
    Stage("10b",  "pipeline/step10_telluric/step10b_apply_telluric.py",             "Apply O2 telluric correction"),
    Stage("11a",  "pipeline/step11_fluxcal/step11a_extract_header_radec_resilient.py", "Extract RA/DEC and slit metadata"),
    Stage("11b",  "pipeline/step11_fluxcal/step11b_query_skymapper.py",             "Query SkyMapper photometry"),
    Stage("11c",  "pipeline/step11_fluxcal/step11c_fluxcal_b.py",                   "Apply photometric flux calibration"),
)

# -----------------------------------------------------------------------------
# QC registry
# -----------------------------------------------------------------------------
# These are the preferred QC companions for the current operational pipeline.
# Keep this list aligned with the canonical QC scripts actually used in the
# notebooks and science validation workflow.
# -----------------------------------------------------------------------------   
QC_REGISTRY: dict[str, tuple[str, ...]] = {
    "04":  ("qc/step04/qc_step04_trace_quicklooks.py",),
    "05":  ("qc/step05/qc_step05_pixflat.py",),
    "06a": ("qc/step06/qc_step06a_mosaic_final.py",),
    "06b": ("qc/step06/qc_step06b_inspector_final.py",),
    "06c": ("qc/step06/qc_step06c_quicklooks_final.py",),
    "07g": ("qc/step07/qc07g_inspect_wavelength_solution.py",),
    "07h": ("qc/step07/qc07h_arc_wavelength_products.py",),
    "08a": ("qc/step08/qc_step08_extract.py",),
    "08c": ("qc/step08/qc_step08c_wavelength_alignment.py",),
    "09":  (
        "qc/step09/qc_step09_preferred_all_slits.py",
        "qc/step09/qc_step09_final_mosaic.py",
    ),
    "10b": ("qc/step10/qc_step10_final_mosaic.py",),
    "11c": ("qc/step11/qc_step11_grid_patched_v2.py", "qc/step11/qc_step11_summary_b.py"),
}

# -----------------------------------------------------------------------------
# Output contract checks
# -----------------------------------------------------------------------------
# These are lightweight post-stage assertions using canonical variables from the
# active target config. They are intended to catch broken file flow early.
# -----------------------------------------------------------------------------
OUTPUT_CHECKS: dict[str, tuple[str, ...]] = {
    "06c": ("SCI_EVEN_TRACECOORDS", "SCI_ODD_TRACECOORDS"),
    "07a": ("MASTER_ARC_DIFF",),
    "07g": ("WAVESOL_ALL_FITS",),
    "07h": ("ARC_1D_WAVELENGTH_ALL",),
    "08a": ("EXTRACT1D_EVEN", "EXTRACT1D_ODD"),
    "08b": ("EXTRACT1D_ALL",),
    "08c": ("EXTRACT1D_WAV",),
    "09":  ("EXTRACT1D_OHCLEAN",),
    "10a": ("TELLURIC_TEMPLATE",),
    "10b": ("EXTRACT1D_TELLCOR",),
    "11c": ("EXTRACT1D_FLUXCAL", "FLUXCAL_SUMMARY_CSV"),
}


def stage_index(stage_key: str) -> int:
    keys = [s.key for s in SCRIPT_REGISTRY]
    if stage_key not in keys:
        raise KeyError(f"Unknown stage '{stage_key}'. Valid stages: {', '.join(keys)}")
    return keys.index(stage_key)


def expand_requested_stages(from_step: str | None, to_step: str | None, only_steps: Sequence[str] | None) -> list[Stage]:
    if only_steps:
        wanted = set(only_steps)
        out = [s for s in SCRIPT_REGISTRY if s.key in wanted]
        missing = wanted - {s.key for s in out}
        if missing:
            raise KeyError(f"Unknown stages in --only: {', '.join(sorted(missing))}")
        return out

    start = 0 if from_step is None else stage_index(from_step)
    stop = len(SCRIPT_REGISTRY) - 1 if to_step is None else stage_index(to_step)
    if stop < start:
        raise ValueError("--to-step must not come before --from-step")
    return list(SCRIPT_REGISTRY[start:stop + 1])


def normalize_sets(user_set: str) -> tuple[str, ...]:
    tag = user_set.strip().upper()
    if tag == "ALL":
        return ("EVEN", "ODD")
    if tag not in {"EVEN", "ODD"}:
        raise ValueError("--set must be EVEN, ODD, or ALL")
    return (tag,)


def _pick_first_existing(*vals):
    for v in vals:
        if not v:
            continue
        p = Path(v)
        if p.exists():
            return str(p)
    return ""

# -----------------------------------------------------------------------------
# Per-stage argument formatting
# -----------------------------------------------------------------------------
# Most stages use static args_template values from SCRIPT_REGISTRY.
# Stages with config-driven inputs/outputs or backward-compatible overrides are
# handled explicitly here.
# -----------------------------------------------------------------------------
def format_stage_args(stage: Stage, set_name: str | None, args: argparse.Namespace, cfg_module) -> list[str]:

    if stage.key == "09":
        # Step09 is now a single ABAB OH-clean stage.
        # It consumes the Step08 wavelength-attached extraction and writes into
        # the canonical Step09 ABAB directory defined by the active config.
        infile = str(getattr(cfg_module, "EXTRACT1D_WAV", ""))
        outdir = str(getattr(cfg_module, "ST09_ABAB", ""))
    
        vals: list[str] = []
        if infile:
            vals.extend(["--in-fits", infile])
        if outdir:
            vals.extend(["--outdir", outdir])
        return vals
    
    if stage.key == "11a":
        vals: list[str] = []
        infile = (
            args.step11a_infile
            or str(getattr(cfg_module, "EXTRACT1D_TELLCOR", ""))
            or str(getattr(cfg_module, "STEP11_INPUT_SPECTRA", ""))
        )
        outcsv = (
            args.step11a_outcsv
            or str(getattr(cfg_module, "STEP11_RADEC", ""))
            or str(Path(getattr(cfg_module, "ST11_FLUXCAL")) / "slit_trace_radec_all.csv")
        )
        if not infile:
            raise ValueError("Step11a requires a config EXTRACT1D_TELLCOR/STEP11_INPUT_SPECTRA or --step11a-infile")
        vals.extend(["--infile", infile, "--out", outcsv])

        even_geom = (
            getattr(cfg_module, "EVEN_TRACES_GEOM", None)
            or getattr(cfg_module, "EVEN_TRACE_GEOM", None)
            or getattr(cfg_module, "EVEN_TRACES_GEOMETRY", None)
        )
        odd_geom = (
            getattr(cfg_module, "ODD_TRACES_GEOM", None)
            or getattr(cfg_module, "ODD_TRACE_GEOM", None)
            or getattr(cfg_module, "ODD_TRACES_GEOMETRY", None)
        )
        if even_geom:
            vals.extend(["--even-geom", str(even_geom)])
        if odd_geom:
            vals.extend(["--odd-geom", str(odd_geom)])
        return vals

    if stage.key == "11c":
        # Keep override support, but default to config-driven discovery if available.
        extract = (
            args.step11c_extract
            or str(getattr(cfg_module, "STEP11_INPUT_SPECTRA", ""))
            or str(getattr(cfg_module, "EXTRACT1D_TELLCOR", ""))
        )
        phot = (
            args.step11c_photcsv
            or str(getattr(cfg_module, "STEP11_PHOTCAT", ""))
            or str(getattr(cfg_module, "SKYMAPPER_CSV", ""))
            or str(getattr(cfg_module, "PHOTCSV", ""))
            or _pick_first_existing(
                Path(getattr(cfg_module, "ST11_FLUXCAL", "")) / "slit_trace_radec_skymapper_all.csv",
                Path(getattr(cfg_module, "ST11_FLUXCAL", "")) / "skymapper_photometry.csv",
                Path(getattr(cfg_module, "ST11_FLUXCAL", "")) / "skymapper.csv",
                Path(getattr(cfg_module, "ST11_FLUXCAL", "")) / "step11b_skymapper.csv",
            )
        )
        
        vals: list[str] = []
        if extract:
            vals.append(extract)
        if phot:
            vals.append(phot)
        return vals

    if not stage.args_template:
        return []
    s = stage.args_template.format(set=set_name) if set_name else stage.args_template
    return shlex.split(s)


def resolve_script(repo_root: Path, rel_path: str) -> Path:
    path = repo_root / rel_path
    if not path.exists():
        raise FileNotFoundError(f"Missing script: {path}")
    return path

# Build the subprocess command using the current Python executable so that the
# driver and all launched stage scripts run in the same environment.
def build_command(script_path: Path, extra_args: Sequence[str]) -> list[str]:
    return [str(sys.executable), str(script_path), *[str(x) for x in extra_args]]

# Execute one subprocess command from the repository root.
# We prepend repo_root to PYTHONPATH so that 'import config' and related local
# imports resolve consistently inside all stage and QC scripts.
def run_one_command(cmd: Sequence[str], cwd: Path, dry_run: bool, verbose: bool, extra_env: dict[str, str] | None = None) -> int:
    if verbose or dry_run:
        print("[CMD]", " ".join(shlex.quote(x) for x in cmd))
    if dry_run:
        return 0

    env = os.environ.copy()
    old_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([str(cwd)] + ([old_pp] if old_pp else []))
    if extra_env:
        env.update(extra_env)

    proc = subprocess.run(cmd, cwd=str(cwd), env=env)
    return int(proc.returncode)


def iter_stage_runs(stage: Stage, selected_sets: tuple[str, ...]) -> Iterable[tuple[Stage, str | None]]:
    if stage.is_set_based:
        for s in stage.sets:
            if s in selected_sets:
                yield stage, s
    else:
        yield stage, None

def print_plan(stages: Sequence[Stage], selected_sets: tuple[str, ...], run_qc: bool) -> None:
    print("\nPlanned pipeline execution:\n")
    for s in stages:
        if s.is_set_based:
            chosen = [x for x in s.sets if x in selected_sets]
            suffix = f" [{', '.join(chosen)}]"
        else:
            suffix = ""
        qc = f" | QC: {', '.join(QC_REGISTRY.get(s.key, ()))}" if run_qc and s.key in QC_REGISTRY else ""
        print(f"  {s.key:>4}  {s.script}{suffix}  —  {s.description}{qc}")
    print()


def import_target_config(module_name: str):
    mod = importlib.import_module(module_name)
    if hasattr(mod, "ensure_directories"):
        mod.ensure_directories()
    return mod

# Validate that a stage produced its canonical outputs according to the active
# target config. These are lightweight contract checks, not scientific QA.
def validate_stage_outputs(stage_key: str, cfg_module, selected_sets: tuple[str, ...]) -> list[Path]:
    required_names = list(OUTPUT_CHECKS.get(stage_key, ()))
    paths: list[Path] = []

    # Backward-compatible fallback if config has not yet been extended.
    if stage_key == "07h" and not hasattr(cfg_module, "ARC_1D_WAVELENGTH_ALL"):
        required_names = ["WAVESOL_ALL_FITS"]

    for name in required_names:
        if not hasattr(cfg_module, name):
            raise AttributeError(f"Config module is missing required product variable: {name}")
        p = Path(getattr(cfg_module, name))

        if stage_key == "06c":
            if "EVEN" not in selected_sets and name == "SCI_EVEN_TRACECOORDS":
                continue
            if "ODD" not in selected_sets and name == "SCI_ODD_TRACECOORDS":
                continue
        if stage_key == "08a":
            if "EVEN" not in selected_sets and name == "EXTRACT1D_EVEN":
                continue
            if "ODD" not in selected_sets and name == "EXTRACT1D_ODD":
                continue

        if not p.exists():
            raise FileNotFoundError(f"Expected output missing after {stage_key}: {name} -> {p}")
        paths.append(p)

    return paths


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the SAMOS pipeline sequentially.")
    ap.add_argument("--repo-root", type=str, default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--config", type=str, default="config.reductions.run8_dolidze25")
    ap.add_argument("--from-step", type=str, default="04")
    ap.add_argument("--to-step", type=str, default="11c")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--set", type=str, default="ALL")
    ap.add_argument("--run-qc", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--skip-checks", action="store_true")
    ap.add_argument("--step11a-infile", type=str, default="")
    ap.add_argument("--step11a-outcsv", type=str, default="")
    ap.add_argument("--step11c-extract", type=str, default="")
    ap.add_argument("--step11c-photcsv", type=str, default="")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    print(f"[INFO] repo_root = {repo_root}")
    print(f"[INFO] config    = {args.config}")

    if not repo_root.exists():
        raise FileNotFoundError(repo_root)

    cfg_module = import_target_config(args.config)
    selected_sets = normalize_sets(args.set)
    stages = expand_requested_stages(args.from_step, args.to_step, args.only)

    print_plan(stages, selected_sets, args.run_qc)

    t0 = time.time()
    failures: list[str] = []

    for stage in stages:
        
        for stage_obj, set_name in iter_stage_runs(stage, selected_sets):
            t_stage = time.time()
            label = f"{stage_obj.key}:{set_name}" if set_name else stage_obj.key
            print(f"\n=== Running {label} — {stage_obj.description} ===")

            try:
                script_path = resolve_script(repo_root, stage_obj.script)
            except FileNotFoundError as exc:
                failures.append(f"{label} (missing script)")
                print(f"[ERROR] {exc}")
                if not args.keep_going:
                    break
                continue

            cmd = build_command(script_path, format_stage_args(stage_obj, set_name, args, cfg_module))
            rc = run_one_command(
                cmd,
                cwd=repo_root,
                dry_run=args.dry_run,
                verbose=args.verbose,
                extra_env={"SAMOS_REPO_ROOT": str(repo_root)},
            )
            if rc != 0:
                failures.append(f"{label} (rc={rc})")
                print(f"[FAIL] {label} returned {rc}")
                if not args.keep_going:
                    dt = time.time() - t0
                    print(f"\nStopped after first failure. Elapsed: {dt:.1f} s")
                    return rc
                continue

            print(f"[OK] {label} ({time.time() - t_stage:.1f}s)")

            last_set_for_stage = (
                (not stage_obj.is_set_based)
                or (set_name == tuple(s for s in stage_obj.sets if s in selected_sets)[-1])
            )
            if last_set_for_stage and (not args.skip_checks) and (not args.dry_run):
                try:
                    checked = validate_stage_outputs(stage_obj.key, cfg_module, selected_sets)
                    if checked:
                        print(f"[CHECK] {stage_obj.key} outputs OK ({len(checked)} file(s))")
                except Exception as exc:
                    failures.append(f"{stage_obj.key} output-check ({exc})")
                    print(f"[FAIL] Output check for {stage_obj.key}: {exc}")
                    if not args.keep_going:
                        dt = time.time() - t0
                        print(f"\nStopped after failed output check. Elapsed: {dt:.1f} s")
                        return 1

            if args.run_qc:
                for qc_rel in QC_REGISTRY.get(stage_obj.key, ()):
                    try:
                        qc_path = resolve_script(repo_root, qc_rel)
                    except FileNotFoundError as exc:
                        print(f"[SKIP QC] {exc}")
                        continue
            
                    # ----------------------------
                    # Build QC arguments
                    # ----------------------------
                    qc_args: list[str] = []
                    qc_str = str(qc_path)
                    
                    # --- Step09 closeout QC ---
                    if qc_str.endswith("qc/step09/qc_step09_preferred_all_slits.py"):
                        root = getattr(cfg_module, "ST09_ABAB")
                        qc_args = [
                            "--root", str(root),
                            "--out-pdf", str(Path(root) / "qc_step09_preferred_all_slits.pdf"),
                        ]
                    
                    elif qc_str.endswith("qc/step09/qc_step09_final_mosaic.py"):
                        root = getattr(cfg_module, "ST09_ABAB")
                        qc_args = [
                            "--infile", str(cfg_module.EXTRACT1D_OHCLEAN),
                            "--outdir", str(Path(root) / "qc_step09"),
                            "--column", "STELLAR",
                            "--show-pref",
                        ]
                    
                    # --- Step10 closeout QC ---
                    elif qc_str.endswith("qc/step10/qc_step10_final_mosaic.py"):
                        qc_args = [
                            "--infile", str(cfg_module.EXTRACT1D_TELLCOR),
                            "--outdir", str(Path(cfg_module.ST10_TELLURIC) / "qc_step10"),
                            "--column", "FLUX_TELLCOR_O2",
                        ]
                    
                    # --- Step11 summary QC ---
                    elif qc_str.endswith("qc/step11/qc_step11_summary_b.py"):
                        qc_args = [
                            "--extract", str(cfg_module.EXTRACT1D_FLUXCAL),
                            "--photcat", str(cfg_module.STEP11_PHOTCAT),
                            "--tracecoords", f"{cfg_module.SCI_EVEN_TRACECOORDS}|{cfg_module.SCI_ODD_TRACECOORDS}",
                            "--image", str(
                                getattr(
                                    cfg_module,
                                    "SISI_IMAGE_FITS",
                                    repo_root / "calibration" / "sisi" / "Coadd_i_median_078-082_ff_flipx_wcs_manual.fits"
                                )
                            ),
                            "--outpdf", str(Path(cfg_module.ST11_FLUXCAL) / "qc_step11" / "qc_step11_summary_pages.pdf"),
                        ]
                    
                    # --- Step11 grid QC ---
                    # Grid QC auto-discovers its inputs from config/defaults.
                    elif qc_str.endswith("qc/step11/qc_step11_grid_patched_v2.py"):
                        qc_args = []
                    
                    # --- Generic fallback for set-based QC ---
                    elif set_name is not None:
                        if qc_str.endswith("qc/step04/qc_step04_trace_quicklooks.py"):
                            qc_args = ["--traceset", set_name]
                        else:
                            qc_args = ["--set", set_name]
        
            
                    # ----------------------------
                    # Build and run QC command
                    # ----------------------------
                    qc_cmd = build_command(qc_path, qc_args)
            
                    qlabel = f"QC:{stage_obj.key}:{set_name}" if set_name else f"QC:{stage_obj.key}"
            
                    qrc = run_one_command(
                        qc_cmd,
                        cwd=repo_root,
                        dry_run=args.dry_run,
                        verbose=args.verbose,
                        extra_env={"SAMOS_REPO_ROOT": str(repo_root)},
                    )
            
                    if qrc != 0:
                        failures.append(f"{qlabel} (rc={qrc})")
                        print(f"[FAIL] {qlabel} returned {qrc}")
                        if not args.keep_going:
                            dt = time.time() - t0
                            print(f"\nStopped after QC failure. Elapsed: {dt:.1f} s")
                            return qrc
                    else:
                        print(f"[OK] {qlabel}")

        if failures and not args.keep_going:
            break

    dt = time.time() - t0
    print("\n=== Pipeline run complete ===")
    print(f"Elapsed time: {dt:.1f} s")
    if failures:
        print("Failures:")
        for f in failures:
            print("  -", f)
        return 1

    print("All requested stages completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
