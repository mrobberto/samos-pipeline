#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

repo = Path(__file__).resolve().parents[1]
cmd = [
    sys.executable,
    str(repo / "drivers" / "run_pipeline.py"),
    "--from-step", "09a",
    "--to-step", "09c",
    "--run-qc",
]
raise SystemExit(subprocess.run(cmd, cwd=repo).returncode)