#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC for Step10 — OH wavelength shifts

PURPOSE
-------
Provide diagnostic plots to validate OH-based wavelength corrections.

GENERATES
---------
- Histogram of shifts across slits
- Shift vs slit index
- Used vs rejected slits

INTERPRETATION
--------------
Good behavior:
    - narrow distribution (~0.1–0.3 nm)
    - few rejected slits
    - smooth variation vs slit ID

Warning signs:
    - bimodal distribution → reference issue
    - large scatter → poor sky extraction
    - many rejected slits → weak OH signal

OUTPUT
------
PNG files in:
    config.ST10_OH / "qc_step10"

NOTES
-----
This script does not modify data.
Use it to validate Step10 before proceeding to Step11.
"""

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
import config

CSV = Path(config.ST09_OH_REFINE) / "oh_shift_table.csv"
OUTDIR = Path(getattr(config, "QC09_DIR", Path(config.ST09_OH_REFINE) / "qc")) / "oh"
OUTDIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(CSV)

# ---- histogram ----
plt.figure()
plt.hist(df["shift_nm"], bins=20)
plt.xlabel("Shift (nm)")
plt.ylabel("N")
plt.title("OH shift distribution")
plt.savefig(OUTDIR / "hist_shift.png")
plt.close()

# ---- vs slit ----
plt.figure()
plt.plot(df["shift_nm"], "o")
plt.xlabel("Slit index")
plt.ylabel("Shift (nm)")
plt.title("Shift per slit")
plt.savefig(OUTDIR / "shift_vs_slit.png")
plt.close()

# ---- used vs rejected ----
plt.figure()
plt.scatter(df.index, df["shift_nm"], c=df["use"])
plt.title("Used (True=1) vs rejected (False=0)")
plt.savefig(OUTDIR / "used_flags.png")
plt.close()

print("QC written to", OUTDIR)