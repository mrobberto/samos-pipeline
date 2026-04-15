#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SAMOS master pipeline driver.

PURPOSE
-------
Run the SAMOS reduction pipeline in a controlled sequence, with optional stage
selection, parity selection (EVEN / ODD), dry-run mode, and optional QC
execution.

CURRENT OPERATIONAL USE
-----------------------
This driver is now intended primarily for end-to-end reductions from Step04 to
Step11 using config-driven discovery and canonical products. Earlier steps can
still be launched explicitly, but the default operational flow is the science
pipeline from traces onward.

DESIGN GOALS
------------
- Keep the driver lightweight and easy to edit.
- Do not re-implement the science logic of each step.
- Launch the existing step scripts as subprocesses using the current Python.
- Make stage order explicit and easy to maintain.
- Allow partial runs, e.g.:
    Step06a -> Step08c
    Step07a -> Step07h
    Step08a EVEN only
- Optionally run QC scripts after each stage when available.
- Keep the scientific order explicit:
    Step09 = OH refine
    Step10 = telluric
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


# -----------------------------------------------------------------------------
# Stage model
# -----------------------------------------------------------------------------
@dataclass(frozen=True)
class Stage:
    """One pipeline stage entry."""

    key: str
    script: str
    description: str
    sets: tuple[str, ...] = ()
    args_template: str = ""

    @property
    def is_set_based(self) -> bool:
        return len(self.sets) > 0


# -----------------------------------------------------------------------------
# Registry
#
# The order here is the authoritative driver order.
#
# IMPORTANT:
# - Step09 is OH refine
# - Step10 is telluric
#
# For now, some script paths still point to their historical pipeline folders
# until the repository folders are renamed. The *semantic* order is already
# cleanly swapped here.
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
    Stage("07c", "pipeline/step07_wavecal/step07c_extract_arc_1d.py",               "Extract rectified 1D arc slit spectra",            sets=("EVEN", "ODD"), args_template="--traceset {set}"),
    Stage("07d", "pipeline/step07_wavecal/step07d_find_line_shifts.py",             "Measure initial relative arc shifts",              sets=("EVEN", "ODD"), args_template="--traceset {set}"),
    Stage("07e", "pipeline/step07_wavecal/step07e_refine_stack_arc.py",             "Refine arc-stack alignment",                       sets=("EVEN", "ODD"), args_template="--set {set}"),    
    Stage("07f",  "pipeline/step07_wavecal/step07f_build_master_arc.py",            "Build aligned master arc"),
    Stage("07g",  "pipeline/step07_wavecal/step07g_solve_wavelength.py",            "Fit global wavelength solution"),
    Stage("07h",  "pipeline/step07_wavecal/step07h_propagate_wavesol.py",           "Propagate wavelength solution to all slit arcs"),
    Stage("08a",  "pipeline/step08_extract1d/step08a_extract_1d.py",                "Ridge-guided optimal extraction",                 sets=("EVEN", "ODD"), args_template="--set {set}"),
    Stage("08b",  "pipeline/step08_extract1d/step08b_merge_even_odd.py",            "Merge EVEN and ODD extracted spectra"),
    Stage("08c",  "pipeline/step08_extract1d/step08c_attach_wavelength.py",         "Attach wavelength vectors to extracted spectra",  args_template="--overwrite"),
    Stage("09a", "pipeline/step09_oh_refine/step09a_measure_oh_shifts.py",          "Measure OH wavelength shifts"),
    Stage("09b", "pipeline/step09_oh_refine/step09b_apply_oh_shifts.py",            "Apply OH wavelength refinement"),
    Stage("10a", "pipeline/step10_telluric/step10a_build_telluric_template.py",     "Build empirical O2 telluric template"),
    Stage("10b", "pipeline/step10_telluric/step10b_apply_telluric.py",              "Apply O2 telluric correction"),
    Stage("11a",  "pipeline/step11_fluxcal/step11a_extract_header_radec_resilient.py", "Extract RA/DEC and slit metadata"),
    Stage("11b",  "pipeline/step11_fluxcal/step11b_query_skymapper.py",             "Query SkyMapper photometry"),
    Stage("11c",  "pipeline/step11_fluxcal/step11c_fluxcal.py",                     "Apply photometric flux calibration"),
)


# -----------------------------------------------------------------------------
# QC companions
#
# Paths are relative to the repository root.
# QC output products should be written by the QC scripts into the canonical
# reduced/qc/<step>/ tree defined in the target config.
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
    "09b": ("qc/step10/qc_step10_oh.py",),
    "10b": ("qc/step09/qc_step09_telluric.py",),
    "11c": ("qc/step11/qc_step11_grid.py", "qc/step11/qc_step11_summary.py"),
}


# -----------------------------------------------------------------------------
# Optional output checks keyed by stage.
#
# Values are variable names from the active target config module.
# These are lightweight contract checks to catch broken file flow early.
# -----------------------------------------------------------------------------
OUTPUT_CHECKS: dict[str, tuple[str, ...]] = {
    "06c": ("SCI_EVEN_TRACECOORDS", "SCI_ODD_TRACECOORDS"),
    "07a": ("MASTER_ARC_DIFF",),
    "07f": ("MASTER_ARC_FITS",),
    "07h": ("ARC_1D_WAVELENGTH_ALL",),
    "08a": ("EXTRACT1D_EVEN", "EXTRACT1D_ODD"),
    "08b": ("EXTRACT1D_ALL",),
    "08c": ("EXTRACT1D_WAV",),
    "09b": ("EXTRACT1D_OHREF",),
    "10a": ("TELLURIC_TEMPLATE",),
    "10b": ("EXTRACT1D_TELLCOR",),
    "11c": ("EXTRACT1D_FLUXCAL", "FLUXCAL_SUMMARY_CSV"),
}


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
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


def format_stage_args(stage: Stage, set_name: str | None, args: argparse.Namespace) -> list[str]:
    if stage.key == "11a":
        vals: list[str] = []
        if args.step11a_infile:
            vals.extend(["--infile", args.step11a_infile])
        if args.step11a_outcsv:
            vals.extend(["--out", args.step11a_outcsv])
        return vals

    if stage.key == "11c":
        vals: list[str] = []
        if args.step11c_extract:
            vals.append(args.step11c_extract)
        if args.step11c_photcsv:
            vals.append(args.step11c_photcsv)
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


def build_command(script_path: Path, extra_args: Sequence[str]) -> list[str]:
    return [sys.executable, str(script_path), *extra_args]


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


def validate_stage_outputs(stage_key: str, cfg_module, selected_sets: tuple[str, ...]) -> list[Path]:
    required_names = OUTPUT_CHECKS.get(stage_key, ())
    paths: list[Path] = []

    for name in required_names:
        if not hasattr(cfg_module, name):
            raise AttributeError(f"Config module is missing required product variable: {name}")
        p = Path(getattr(cfg_module, name))

        # For single-set runs, avoid requiring the other parity product.
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


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Run the SAMOS pipeline sequentially.")
    ap.add_argument(
        "--repo-root",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Repository root containing pipeline/, qc/, config/, etc. (default: parent of drivers/)",
    )
    ap.add_argument(
        "--config",
        type=str,
        default="config.reductions.run8_dolidze25",
        help="Importable target-config module providing canonical products",
    )
    ap.add_argument("--from-step", type=str, default="04", help="First stage to run, e.g. 04, 07a, 09a")
    ap.add_argument("--to-step", type=str, default="11c", help="Last stage to run, e.g. 08c, 10b, 11c")
    ap.add_argument("--only", nargs="*", default=None, help="Explicit stage list, e.g. --only 08a 08b 08c")
    ap.add_argument("--set", type=str, default="ALL", help="EVEN, ODD, or ALL for set-based stages")
    ap.add_argument("--run-qc", action="store_true", help="Run QC companions when registered and present")
    ap.add_argument("--dry-run", action="store_true", help="Print commands without executing them")
    ap.add_argument("--keep-going", action="store_true", help="Continue after failures")
    ap.add_argument("--verbose", action="store_true", help="Print every command before execution")
    ap.add_argument("--skip-checks", action="store_true", help="Skip canonical output-existence checks after key stages")
    ap.add_argument("--step11a-infile", type=str, default="", help="Emergency override for Step11a input FITS")
    ap.add_argument("--step11a-outcsv", type=str, default="", help="Emergency override for Step11a output CSV")
    ap.add_argument("--step11c-extract", type=str, default="", help="Emergency override for Step11c extracted FITS")
    ap.add_argument("--step11c-photcsv", type=str, default="", help="Emergency override for Step11c photometric CSV")
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

            cmd = build_command(script_path, format_stage_args(stage_obj, set_name, args))
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

            # Run stage-level output checks only once after the full stage completes.
            # For set-based stages, defer checks until the final selected parity.
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

                    qc_args: list[str] = []
                    if set_name is not None:
                        qc_str = str(qc_path)
                    
                        if qc_str.endswith("qc/step04/qc_step04_trace_quicklooks.py"):
                            qc_args = ["--traceset", set_name]
                        else:
                            qc_args = ["--set", set_name]
        
    
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
