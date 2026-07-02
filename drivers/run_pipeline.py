#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SAMOS/SAMI master pipeline driver.

Run the spectroscopic reduction pipeline in canonical stage order using the
active target configuration module. The executable driver contains orchestration
logic only; stage order, QC registration, output contracts, and argument
construction are kept in companion modules under drivers/.
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import os
import shlex
import subprocess
import sys
import time
from typing import Iterable, Sequence

from drivers.registry import QC_REGISTRY, OUTPUT_CHECKS, SCRIPT_REGISTRY, Stage
from drivers.stage_args import format_stage_args
from drivers.qc_args import format_qc_args

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
    ap.add_argument("--config", type=str, default="config.target_config")
    ap.add_argument("--from-step", type=str, default="04")
    ap.add_argument("--to-step", type=str, default="12e")
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
    cfg_module = import_target_config(args.config)


    print("=== SAMOS CONFIG ===")
    print("TARGET:", cfg_module.TARGET_NAME)
    print("NIGHT :", cfg_module.NIGHT_ID)
    print("SCI   :", len(cfg_module.SCIENCE_FILES))
    print("ARC   :", len(cfg_module.ARC_FILES))
    print("QUARTZ:", len(cfg_module.QUARTZ_FILES))
    print("ROOT  :", cfg_module.REDUCED_DIR)  
    print(f"[INFO] repo_root = {repo_root}")
    print(f"[INFO] config    = {args.config}")
    if args.dry_run:
        print("[DRY-RUN] Commands will be printed but not executed.")

    if not repo_root.exists():
        raise FileNotFoundError(repo_root)

    selected_sets = normalize_sets(args.set)
    stages = expand_requested_stages(args.from_step, args.to_step, args.only)

    print_plan(stages, selected_sets, args.run_qc)

    t0 = time.time()
    failures: list[str] = []

    for stage in stages:
        
        for stage_obj, set_name in iter_stage_runs(stage, selected_sets):
            t_stage = time.time()
            label = f"{stage_obj.key}:{set_name}" if set_name else stage_obj.key
            verb = "Would run" if args.dry_run else "Running"
            print(f"\n=== {verb} {label} — {stage_obj.description} ===")

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

            if args.dry_run:
                print(f"[DRY-RUN] {label} command not executed")
            else:
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
            
                    qc_args = format_qc_args(qc_path, set_name, cfg_module, repo_root)

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
                        if args.dry_run:
                            print(f"[DRY-RUN] {qlabel} command not executed")
                        else:
                            print(f"[OK] {qlabel}")

        if failures and not args.keep_going:
            break

    dt = time.time() - t0
    print("\n=== Pipeline run complete ===")
    print(f"Elapsed time: {dt:.1f} s")
    print("Active SAMOS reduction:")
    print("  TARGET_NAME =", cfg_module.TARGET_NAME)
    print("  NIGHT_ID    =", cfg_module.NIGHT_ID)
    print("  RUN_ROOT    =", cfg_module.RUN_ROOT)
    print("  REDUCED_DIR =", cfg_module.REDUCED_DIR)

    if failures:
        print("Failures:")
        for f in failures:
            print("  -", f)
        return 1

    print("All requested stages completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
