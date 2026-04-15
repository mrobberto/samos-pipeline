#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
QC script: compare OLD vs NEW Even_traces.fits and gap-cut behavior.

Purpose
-------
Compare the old and new Step04 EVEN quartz-trace products and visualize
why the gap-cut CSVs differ.

Inputs
------
- OLD Even_traces.fits
- NEW Even_traces.fits
- optionally:
    Even_traces_gap_cuts.csv in the same directories

Outputs
-------
- summary CSV of slit-by-slit old/new y_cut values
- PNG figures for slits where old/new y_cut differ

Notes
-----
This script does not require slit masks or slitid maps.
It estimates slit centers from the X-profile of each Even_traces image,
assigns EVEN slit IDs in RA order (0,2,4,...) and then extracts a local
Y-profile around each slit center.

If the old/new images have slightly different slit-center positions, the
script matches slits by EVEN slit ID derived from RA order.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


# ============================================================================
# USER PATHS
# ============================================================================
OLD_TRACES = Path(
    "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/"
    "@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SAMI/"
    "20260113_dev20250321/reduced/04_pixflat/Even_traces.fits"
)

NEW_TRACES = Path(
    "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/"
    "@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SAMI/"
    "Dolidze25/reduced/04_traces/Even_traces.fits"
)

# If present, these will be used automatically.
OLD_GAPCSV = OLD_TRACES.parent / "Even_traces_gap_cuts.csv"
NEW_GAPCSV = NEW_TRACES.parent / "Even_traces_gap_cuts.csv"

# Output directory for QC products
OUTDIR = NEW_TRACES.parent / "qc_compare_old_new_even_gapcuts"

# Detection / extraction settings
PROFILE_SMOOTH = 3.0
MIN_PEAK_DIST = 18
PEAK_PROMINENCE = 0.15
PEAK_HEIGHT_FRAC = 0.10
ACTIVE_FRAC = 0.03
ACTIVE_PAD = 20

X_HALF_WINDOW = 8         # half-width of extraction window around slit center
Y_ZOOM_PAD = 120          # rows shown around relevant region
IMAGE_CLIP_PCT = (5, 99)  # display stretch percentiles

# ============================================================================
# Y-WINDOW CONTROL
# ============================================================================
USE_Y_WINDOW = True
YMIN = 1800
YMAX = 2100


# ============================================================================
# HELPERS
# ============================================================================
def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_fits_image(path: Path) -> tuple[np.ndarray, fits.Header]:
    with fits.open(path) as hdul:
        data = np.asarray(hdul[0].data, dtype=np.float32)
        hdr = hdul[0].header.copy()
    if data.ndim != 2:
        raise ValueError(f"{path} is not a 2D image; shape={data.shape}")
    return data, hdr


def load_gap_csv(path: Path) -> dict[int, int | None]:
    """
    Read CSV with columns slit_id,y_cut
    Blank y_cut -> None
    """
    out: dict[int, int | None] = {}
    if not path.exists():
        return out

    with open(path, "r", newline="") as f:
        rdr = csv.DictReader(f)
        for row in rdr:
            sid = int(row["slit_id"])
            raw = (row.get("y_cut", "") or "").strip()
            out[sid] = None if raw == "" else int(raw)
    return out


def find_active_band(img: np.ndarray) -> tuple[int, int]:
    prof = np.nanmedian(np.clip(img, 0, None), axis=0).astype(float)
    mx = np.nanmax(prof)
    if not np.isfinite(mx) or mx <= 0:
        return 0, img.shape[1] - 1
    on = prof > (ACTIVE_FRAC * mx)
    if not np.any(on):
        return 0, img.shape[1] - 1
    x0 = int(np.argmax(on))
    x1 = int(len(on) - 1 - np.argmax(on[::-1]))
    x0 = max(0, x0 - ACTIVE_PAD)
    x1 = min(img.shape[1] - 1, x1 + ACTIVE_PAD)
    return x0, x1


def detect_even_slit_centers(img: np.ndarray) -> dict[int, int]:
    """
    Detect slit centers and assign EVEN slit IDs in RA order:
      x descending -> slit IDs 0,2,4,...

    Returns
    -------
    dict: {slit_id: x_center}
    """
    x0, x1 = find_active_band(img)

    prof = np.nanpercentile(np.clip(img[:, x0:x1 + 1], 0, None), 90, axis=0).astype(float)
    prof_s = gaussian_filter1d(prof, sigma=PROFILE_SMOOTH)

    mx = np.nanmax(prof_s)
    if not np.isfinite(mx) or mx <= 0:
        raise RuntimeError("No usable signal in X-profile.")

    prof_n = prof_s / mx

    peaks, props = find_peaks(
        prof_n,
        distance=MIN_PEAK_DIST,
        height=PEAK_HEIGHT_FRAC,
        prominence=PEAK_PROMINENCE,
    )

    centers = (peaks + x0).astype(int)
    if centers.size == 0:
        raise RuntimeError("No slit centers detected.")

    # RA order = x descending
    order = np.argsort(centers)[::-1]
    centers_ra = centers[order]

    sid_to_x = {}
    for rank, xc in enumerate(centers_ra):
        sid = 2 * rank
        sid_to_x[sid] = int(xc)

    return sid_to_x


def robust_y_profile(img: np.ndarray, xc: int, half_window: int = 8) -> np.ndarray:
    """
    Build a per-row slit profile using a narrow extraction window around xc.
    Background is estimated from sidebands in the same row.
    """
    ny, nx = img.shape
    out = np.full(ny, np.nan, dtype=float)

    for y in range(ny):
        xa = max(0, xc - half_window)
        xb = min(nx, xc + half_window + 1)
        row = img[y, xa:xb].astype(float)

        if row.size < 3 or not np.isfinite(row).any():
            continue

        # simple local background from ends of the extraction window
        nside = max(2, min(4, row.size // 4))
        side = np.concatenate([row[:nside], row[-nside:]])
        bkg = np.nanmedian(side)

        val = np.nanmedian(row) - bkg
        out[y] = val

    return out


def choose_zoom_limits(ny, old_cut, new_cut):
    if USE_Y_WINDOW:
        return YMIN, YMAX

    ys = [y for y in [old_cut, new_cut] if y is not None]
    if len(ys) == 0:
        return 0, ny - 1
    yc = int(np.median(ys))
    y0 = max(0, yc - Y_ZOOM_PAD)
    y1 = min(ny - 1, yc + Y_ZOOM_PAD)
    return y0, y1

def image_limits(img: np.ndarray) -> tuple[float, float]:
    good = img[np.isfinite(img)]
    if good.size == 0:
        return 0.0, 1.0
    vmin = np.nanpercentile(good, IMAGE_CLIP_PCT[0])
    vmax = np.nanpercentile(good, IMAGE_CLIP_PCT[1])
    if not np.isfinite(vmin):
        vmin = 0.0
    if not np.isfinite(vmax) or vmax <= vmin:
        vmax = vmin + 1.0
    return float(vmin), float(vmax)


def plot_one_slit(
    sid: int,
    old_img: np.ndarray,
    new_img: np.ndarray,
    old_x: int,
    new_x: int,
    old_cut: int | None,
    new_cut: int | None,
    out_png: Path,
) -> None:
    ny_old, nx_old = old_img.shape
    ny_new, nx_new = new_img.shape

    old_prof = robust_y_profile(old_img, old_x, half_window=X_HALF_WINDOW)
    new_prof = robust_y_profile(new_img, new_x, half_window=X_HALF_WINDOW)

    y0_old, y1_old = choose_zoom_limits(ny_old, old_cut, new_cut)
    y0_new, y1_new = choose_zoom_limits(ny_new, old_cut, new_cut)

    old_x0 = max(0, old_x - 20)
    old_x1 = min(nx_old - 1, old_x + 20)
    new_x0 = max(0, new_x - 20)
    new_x1 = min(nx_new - 1, new_x + 20)

    vmin_old, vmax_old = image_limits(old_img[y0_old:y1_old + 1, old_x0:old_x1 + 1])
    vmin_new, vmax_new = image_limits(new_img[y0_new:y1_new + 1, new_x0:new_x1 + 1])

    fig = plt.figure(figsize=(12, 8))

    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(
        old_img[y0_old:y1_old + 1, old_x0:old_x1 + 1],
        origin="lower",
        aspect="auto",
        vmin=vmin_old,
        vmax=vmax_old,
        extent=[old_x0, old_x1, y0_old, y1_old],
    )
    ax1.axvline(old_x, ls="--")
    if old_cut is not None:
        ax1.axhline(old_cut, ls=":")
    ax1.set_title(f"OLD image — slit {sid} (x={old_x})")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(
        new_img[y0_new:y1_new + 1, new_x0:new_x1 + 1],
        origin="lower",
        aspect="auto",
        vmin=vmin_new,
        vmax=vmax_new,
        extent=[new_x0, new_x1, y0_new, y1_new],
    )
    ax2.axvline(new_x, ls="--")
    if new_cut is not None:
        ax2.axhline(new_cut, ls=":")
    ax2.set_title(f"NEW image — slit {sid} (x={new_x})")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    ax3 = fig.add_subplot(2, 1, 2)
    y_old = np.arange(old_prof.size)
    y_new = np.arange(new_prof.size)
    ax3.plot(y_old, old_prof, label=f"OLD slit {sid}", lw=1.5)
    ax3.plot(y_new, new_prof, label=f"NEW slit {sid}", lw=1.5)
    if old_cut is not None:
        ax3.axvline(old_cut, ls="--", label=f"OLD y_cut={old_cut}")
    if new_cut is not None:
        ax3.axvline(new_cut, ls="--", label=f"NEW y_cut={new_cut}")
    ax3.set_xlabel("Y")
    ax3.set_ylabel("Local quartz profile")
    ax3.set_title(f"Y-profile comparison — slit {sid}")
    ax3.legend(loc="best")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(f"OLD vs NEW gap-trim comparison — EVEN slit {sid}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================
def main():
    ensure_dir(OUTDIR)

    if not OLD_TRACES.exists():
        raise FileNotFoundError(OLD_TRACES)
    if not NEW_TRACES.exists():
        raise FileNotFoundError(NEW_TRACES)

    print(f"[INFO] OLD_TRACES = {OLD_TRACES}")
    print(f"[INFO] NEW_TRACES = {NEW_TRACES}")
    print(f"[INFO] OLD_GAPCSV = {OLD_GAPCSV}  exists={OLD_GAPCSV.exists()}")
    print(f"[INFO] NEW_GAPCSV = {NEW_GAPCSV}  exists={NEW_GAPCSV.exists()}")
    print(f"[INFO] OUTDIR     = {OUTDIR}")

    old_img, old_hdr = load_fits_image(OLD_TRACES)
    new_img, new_hdr = load_fits_image(NEW_TRACES)

    print(f"[INFO] OLD shape = {old_img.shape}")
    print(f"[INFO] NEW shape = {new_img.shape}")

    old_sid_to_x = detect_even_slit_centers(old_img)
    new_sid_to_x = detect_even_slit_centers(new_img)

    print(f"[INFO] OLD detected slits = {len(old_sid_to_x)}")
    print(f"[INFO] NEW detected slits = {len(new_sid_to_x)}")

    old_gaps = load_gap_csv(OLD_GAPCSV)
    new_gaps = load_gap_csv(NEW_GAPCSV)

    all_sids = sorted(set(old_sid_to_x.keys()) | set(new_sid_to_x.keys()) | set(old_gaps.keys()) | set(new_gaps.keys()))

    summary_rows = []
    diff_sids = []

    for sid in all_sids:
        old_cut = old_gaps.get(sid, None)
        new_cut = new_gaps.get(sid, None)
        old_x = old_sid_to_x.get(sid, None)
        new_x = new_sid_to_x.get(sid, None)

        changed = old_cut != new_cut
        if changed:
            diff_sids.append(sid)

        summary_rows.append({
            "slit_id": sid,
            "old_x": "" if old_x is None else old_x,
            "new_x": "" if new_x is None else new_x,
            "old_y_cut": "" if old_cut is None else old_cut,
            "new_y_cut": "" if new_cut is None else new_cut,
            "changed": int(changed),
        })

    summary_csv = OUTDIR / "compare_old_new_gapcuts_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["slit_id", "old_x", "new_x", "old_y_cut", "new_y_cut", "changed"]
        )
        w.writeheader()
        w.writerows(summary_rows)

    print(f"[INFO] Wrote summary CSV: {summary_csv}")
    print(f"[INFO] Number of differing slits: {len(diff_sids)}")
    print(f"[INFO] Differing slit IDs: {diff_sids}")

    # Make figures only for slits where old/new differ and both centers exist
    made = 0
    for sid in diff_sids:
        if sid not in old_sid_to_x or sid not in new_sid_to_x:
            print(f"[WARN] slit {sid}: missing center in old or new detection; skipping figure")
            continue

        tag = f"_Y{YMIN}_{YMAX}" if USE_Y_WINDOW else ""
        out_png = OUTDIR / f"compare_old_new_slit_{sid:03d}{tag}.png"
        plot_one_slit(
            sid=sid,
            old_img=old_img,
            new_img=new_img,
            old_x=old_sid_to_x[sid],
            new_x=new_sid_to_x[sid],
            old_cut=old_gaps.get(sid, None),
            new_cut=new_gaps.get(sid, None),
            out_png=out_png,
        )
        made += 1

    print(f"[INFO] Wrote {made} slit comparison PNGs")

    # Also make a quick histogram-like text summary
    n_old_cut = sum(v is not None for v in old_gaps.values())
    n_new_cut = sum(v is not None for v in new_gaps.values())
    print(f"[INFO] OLD cuts present: {n_old_cut}")
    print(f"[INFO] NEW cuts present: {n_new_cut}")


if __name__ == "__main__":
    main()