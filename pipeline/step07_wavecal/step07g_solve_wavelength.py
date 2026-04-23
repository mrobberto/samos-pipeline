#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step07g — global wavelength solution on MASTER arc + per-slit propagated solutions.

Plumbing-updated version that preserves the original Step07.4 science logic but
uses the restored pipeline naming and products.

Inputs
------
From config.ST07_WAVECAL:
- config.MASTER_ARC_FITS
- ArcDiff_*_pixflatcorr_clipped_1D_slitid_EVEN.fits
- ArcDiff_*_pixflatcorr_clipped_1D_slitid_ODD.fits

From calibration/reference_tables/nist_list:
- SOAR_air_strong_lines_Ar.txt
- SOAR_air_strong_lines_Hg.txt

Outputs
-------
To config.ST07_WAVECAL:
- arc_master_wavesol.fits
- arc_wavesol_per_slit.fits
- step07g_master_peaks_used.png
- step07g_master_residuals.png
- step07g_slit_spectra_wavelength.png

Key convention
--------------
Panel B:
    lambda_slit(ywin) = p( ywin + SHIFT_TO_MASTER[slit] )

Behavior notes
--------------
- Preserves the original working science logic.
- Window definition is fixed to the original working values:
      YWIN0    = 1225
      FIRSTLEN = 2875
- Plots are saved to disk by default and are NOT shown interactively.
- Use --show-plots only when you explicitly want windows on screen.
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--show-plots",
        action="store_true",
        help="Display QC plots interactively in addition to saving PNG files.",
    )
    return ap.parse_args()


ARGS = parse_args()

# Force a non-interactive backend unless explicitly requested.
import matplotlib
if not ARGS.show_plots:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config
from config.pipeline_config import REFERENCE_TABLES_DIR

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("step07g_solve_wavelength")


def maybe_show_or_close() -> None:
    if ARGS.show_plots:
        plt.show()
    plt.close()


def default_arc1d_path(trace_set: str) -> Path:
    wavecal_dir = Path(config.ST07_WAVECAL).expanduser()
    if hasattr(config, "MASTER_ARC_DIFF"):
        stem = Path(config.MASTER_ARC_DIFF).stem
        p = wavecal_dir / f"{stem}_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"
        if p.exists():
            return p
    hits = sorted(wavecal_dir.glob(f"*_1D_slitid_{trace_set}.fits"))
    if hits:
        return hits[-1]
    return wavecal_dir / f"ArcDiff_pixflatcorr_clipped_1D_slitid_{trace_set}.fits"


MASTER_FITS = Path(config.MASTER_ARC_FITS).expanduser()
ARC1D_EVEN = default_arc1d_path("EVEN")
ARC1D_ODD = default_arc1d_path("ODD")

OUT_MASTER = Path(config.ST07_WAVECAL) / "arc_master_wavesol.fits"
OUT_GLOBAL = Path(config.ST07_WAVECAL) / "arc_wavesol_per_slit.fits"

NIST_DIR = REFERENCE_TABLES_DIR / "nist_list"

YWIN0 = int(config.WAVECAL_YWIN0)
FIRSTLEN = int(config.WAVECAL_FIRSTLEN)

ywin = np.arange(FIRSTLEN, dtype=float)

L1 = 763.51060
L2 = 811.53110

POLY_ORDER = 3
SMOOTH_SIGMA = 2.0
PEAK_DISTANCE = 6
PEAK_PROM = None

N_PEAKS_REFINE = 12
N_LINES_KEEP = 80

REFINE_TOL_NM_START = 1.5
REFINE_TOL_NM_END = 0.10
MAX_ITERS = 8

QC_SLITS = ["SLIT000", "SLIT024", "SLIT052", "SLIT001", "SLIT029", "SLIT051"]
QC_XLIM = (740, 830)

LAMBDA_MIN = 575
LAMBDA_MAX = 950


def _as_float_prefix(s: str) -> float:
    s = str(s).strip()
    m = re.match(r"^\s*([0-9]*\.?[0-9]+)", s)
    return float(m.group(1)) if m else np.nan


def parse_nist_strong_txt(path: Path):
    wA, I = [], []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if (not line) or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            inten = _as_float_prefix(parts[0])
            wavA = _as_float_prefix(parts[1])
            if np.isfinite(inten) and np.isfinite(wavA):
                I.append(inten)
                wA.append(wavA)
    wA = np.array(wA, float)
    I = np.array(I, float)
    return wA / 10.0, I


def load_arc1d_flux_map(*fits_paths):
    arc1d = {}
    for fp in fits_paths:
        if fp is None:
            continue
        if not fp.exists():
            log.warning("Missing arc1d file: %s", fp)
            continue
        with fits.open(fp) as h:
            for ext in h[1:]:
                name = str(ext.header.get("EXTNAME", "")).strip().upper()
                if not name.startswith("SLIT"):
                    continue
                arc1d[name] = ext.data[0].astype(float)
    return arc1d


def load_nist_bright_lines(nist_dir: Path, nkeep=80, lam_min=LAMBDA_MIN, lam_max=LAMBDA_MAX):
    files = [
        nist_dir / "SOAR_air_strong_lines_Ar.txt",
        nist_dir / "SOAR_air_strong_lines_Hg.txt",
    ]
    all_w, all_I = [], []

    log.info("Using NIST files:")
    for p in files:
        if not p.exists():
            raise FileNotFoundError(p)
        w, I = parse_nist_strong_txt(p)
        m = (w >= lam_min) & (w <= lam_max) & np.isfinite(I)
        log.info("  %s: parsed=%d kept_in_band=%d", p.name, len(w), int(m.sum()))
        all_w.append(w[m])
        all_I.append(I[m])

    w = np.concatenate(all_w) if all_w else np.array([])
    I = np.concatenate(all_I) if all_I else np.array([])
    if w.size == 0:
        raise RuntimeError("No NIST lines in requested band.")

    idx = np.argsort(I)[::-1]
    w = w[idx]
    I = I[idx]

    w_keep, I_keep = [], []
    for ww, ii in zip(w, I):
        if (not w_keep) or abs(ww - w_keep[-1]) > 0.02:
            w_keep.append(ww)
            I_keep.append(ii)
        if len(w_keep) >= nkeep:
            break

    w_keep = np.array(w_keep, float)
    I_keep = np.array(I_keep, float)
    srt = np.argsort(w_keep)
    return w_keep[srt], I_keep[srt]


def nearest_lines(lam, lines_nm):
    lam = np.asarray(lam, float)
    L = np.asarray(lines_nm, float)
    idx = np.searchsorted(L, lam)
    idx = np.clip(idx, 1, len(L) - 1)
    left = L[idx - 1]
    right = L[idx]
    return np.where(np.abs(lam - left) <= np.abs(lam - right), left, right)


def polyfit_with_sigma_clip(x, y, order=3, tol_nm=0.10, niter=5):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < order + 2:
        return None, None

    mask = np.ones_like(x, dtype=bool)
    p = None
    for _ in range(niter):
        coeff = np.polyfit(x[mask], y[mask], order)
        p = np.poly1d(coeff)
        resid = y - p(x)
        mask = np.abs(resid) < tol_nm
        if mask.sum() < order + 2:
            break
    return p, mask


def norm(f):
    m = np.isfinite(f)
    if not m.any():
        return f
    s = np.nanpercentile(f[m], 99)
    if not np.isfinite(s) or s == 0:
        s = 1.0
    return f / s


def load_slit_metadata():
    meta = {}
    for fn in [ARC1D_EVEN, ARC1D_ODD]:
        if not fn.exists():
            continue
        with fits.open(fn) as h:
            for ext in h[1:]:
                name = ext.name.strip().upper()
                if not name.startswith("SLIT"):
                    continue
                meta[name] = ext.header
    return meta


def read_shift_to_master(master_fits: Path) -> dict[str, int]:
    with fits.open(master_fits) as h:
        if "SLITLIST" not in h:
            raise KeyError(f"{master_fits} missing SLITLIST extension")
        slitlist = h["SLITLIST"].data
    out = {}
    for row in slitlist:
        slit = str(row["SLIT"]).strip().upper()
        if slit.startswith("SLIT"):
            out[slit] = int(row["SHIFT_TO_MASTER"])
    if not out:
        raise RuntimeError(f"No SHIFT_TO_MASTER entries found in {master_fits}")
    return out


def main():
    log.info("MASTER_FITS = %s", MASTER_FITS)
    log.info("ARC1D_EVEN  = %s", ARC1D_EVEN)
    log.info("ARC1D_ODD   = %s", ARC1D_ODD)
    log.info("OUT_MASTER  = %s", OUT_MASTER)
    log.info("OUT_GLOBAL  = %s", OUT_GLOBAL)
    log.info("NIST_DIR    = %s", NIST_DIR)
    log.info("Using MASTER window: y[%d:%d] (FIRSTLEN=%d)", YWIN0, YWIN0 + FIRSTLEN, FIRSTLEN)
    log.info("show_plots  = %s", ARGS.show_plots)

    if not MASTER_FITS.exists():
        raise FileNotFoundError(MASTER_FITS)
    if not ARC1D_EVEN.exists():
        raise FileNotFoundError(ARC1D_EVEN)
    if not ARC1D_ODD.exists():
        raise FileNotFoundError(ARC1D_ODD)

    master_full = fits.getdata(MASTER_FITS, ext=0).astype(float)
    master_win = master_full[YWIN0:YWIN0 + FIRSTLEN].astype(float)

    m_s = gaussian_filter1d(master_win, SMOOTH_SIGMA)

    peak_prom = PEAK_PROM
    if peak_prom is None:
        valid = np.isfinite(m_s)
        p95 = np.nanpercentile(m_s[valid], 95)
        p50 = np.nanpercentile(m_s[valid], 50)
        peak_prom = 0.05 * (p95 - p50)

    peaks, props = find_peaks(m_s, prominence=peak_prom, distance=PEAK_DISTANCE)
    prom = props.get("prominences", np.zeros_like(peaks, float))
    idxp = np.argsort(prom)[::-1]
    peaks = peaks[idxp]

    if len(peaks) < 2:
        raise RuntimeError("Not enough peaks detected in MASTER window.")

    y1, y2 = float(peaks[0]), float(peaks[1])
    a0 = (L2 - L1) / (y2 - y1)
    b0 = L1 - a0 * y1

    log.info("Two brightest peaks (window coords): y=%d, %d", int(y1), int(y2))
    log.info("Hard-anchor seed: lambda = %.6f*y + %.3f nm", a0, b0)
    log.info("  lambda(y=0)=%.3f, lambda(y=end)=%.3f", b0, a0 * (FIRSTLEN - 1) + b0)

    lines_nm, _lines_I = load_nist_bright_lines(NIST_DIR, nkeep=N_LINES_KEEP, lam_min=LAMBDA_MIN, lam_max=LAMBDA_MAX)
    log.info("Total NIST bright lines used: %d", len(lines_nm))

    peaks_use = np.sort(peaks[: min(N_PEAKS_REFINE, len(peaks))]).astype(float)    
    # --- ADD RED-END PEAKS (manual extension) ---
    extra_red_pixels = np.array([1119, 1038, 692], dtype=float)  # from your identification
    #    
    # convert to working coordinates
    extra_red_y = extra_red_pixels - YWIN0
    #
    # append
    peaks_use = np.concatenate([peaks_use, extra_red_y])
    
    
    p = None
    tol = REFINE_TOL_NM_START

    for it in range(MAX_ITERS):
        lam_pred = (a0 * peaks_use + b0) if p is None else p(peaks_use)
        pick = nearest_lines(lam_pred, lines_nm)
        # --- OVERRIDE red-end peak matches ---
        red_lines_nm = np.array([912.2967, 922.4499, 965.7786])
        #       
        # find indices of the added peaks
        for i, y in enumerate(peaks_use):
            if y < 0:  # red extrapolation region
                # match manually based on order (safe here)
                idx = np.where(peaks_use == y)[0][0]
                if idx < len(red_lines_nm):
                    pick[idx] = red_lines_nm[idx]
            
        
        resid = lam_pred - pick

        ok = np.abs(resid) < tol
        y_m = peaks_use[ok]
        lam_m = pick[ok]

        log.info("Iter %d: tol=%.3f nm, matches=%d", it, tol, len(y_m))

        if len(y_m) < POLY_ORDER + 4:
            tol *= 1.25
            continue

        p_new, _ = polyfit_with_sigma_clip(
            y_m, lam_m, order=POLY_ORDER,
            tol_nm=max(REFINE_TOL_NM_END, 0.6 * tol), niter=5
        )
        if p_new is None:
            tol *= 1.25
            continue

        p = p_new
        tol = max(REFINE_TOL_NM_END, tol * 0.75)

    if p is None:
        raise RuntimeError("Polynomial refinement failed.")

    lam_fit = p(peaks_use)
    pick = nearest_lines(lam_fit, lines_nm)
    resid_nm = lam_fit - pick
    good = np.abs(resid_nm) < REFINE_TOL_NM_END
    rms_nm = np.sqrt(np.mean(resid_nm[good] ** 2)) if np.any(good) else np.nan
    nmatch = int(np.sum(good))

    log.info("Final: order=%d, NMATCH=%d, RMS~%.4f nm", POLY_ORDER, nmatch, rms_nm)

    ph = fits.PrimaryHDU()
    ph.header["YWIN0"] = YWIN0
    ph.header["FIRSTLEN"] = FIRSTLEN
    ph.header["ORDER"] = POLY_ORDER
    ph.header["NMATCH"] = nmatch
    ph.header["RMSNM"] = float(rms_nm) if np.isfinite(rms_nm) else -1.0
    ph.header["ANCH1NM"] = float(L1)
    ph.header["ANCH2NM"] = float(L2)
    ph.header["ANCHY1"] = int(y1)
    ph.header["ANCHY2"] = int(y2)
    ph.header["PROPCONV"] = "B"
    for k, c in enumerate(p.c[::-1]):
        ph.header[f"WVC{k}"] = float(c)

    tab = fits.BinTableHDU.from_columns([
        fits.Column(name="Y_PEAK", format="D", array=peaks_use),
        fits.Column(name="LAM_FIT_NM", format="D", array=lam_fit),
        fits.Column(name="LAM_MATCH_NM", format="D", array=pick),
        fits.Column(name="RESID_NM", format="D", array=resid_nm),
        fits.Column(name="GOOD", format="L", array=good.astype(bool)),
    ], name="PEAK_MATCHES")

    fits.HDUList([ph, tab]).writeto(OUT_MASTER, overwrite=True)
    log.info("Wrote: %s", OUT_MASTER)

    slit_meta = load_slit_metadata()
    shift_to_master = read_shift_to_master(MASTER_FITS)

    hdus = [fits.PrimaryHDU()]
    hdus[0].header["REFSLIT"] = "MASTER_FRAME"
    hdus[0].header["SHIFTSRC"] = "SHIFT_TO_MASTER from arc_master.fits SLITLIST"
    hdus[0].header["YWIN0"] = YWIN0
    hdus[0].header["FIRSTLEN"] = FIRSTLEN
    hdus[0].header["ORDER"] = POLY_ORDER
    hdus[0].header["NMATCH"] = nmatch
    hdus[0].header["RMSNM"] = float(rms_nm) if np.isfinite(rms_nm) else -1.0
    hdus[0].header["PROPCONV"] = "B"
    for k, c in enumerate(p.c[::-1]):
        hdus[0].header[f"WVC{k}"] = float(c)

    yy = np.poly1d([1.0, 0.0])
    for s in sorted(shift_to_master.keys()):
        sh = int(shift_to_master[s])
        q = p(yy + sh)

        hdu = fits.ImageHDU(np.zeros((1,), dtype=np.float32), name=s)
        if s in slit_meta:
            src_hdr = slit_meta[s]
            for k in ("SLITID", "INDEX", "RA", "DEC", "TRACESET"):
                if k in src_hdr:
                    hdu.header[k] = src_hdr[k]
        hdu.header["SHIFT_P"] = sh
        hdu.header["ORDER"] = POLY_ORDER
        for k, c in enumerate(q.c[::-1]):
            hdu.header[f"WVC{k}"] = float(c)
        hdus.append(hdu)

    fits.HDUList(hdus).writeto(OUT_GLOBAL, overwrite=True)
    log.info("Wrote: %s", OUT_GLOBAL)

    qc_dir = Path(config.ST07_WAVECAL)
    qc_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(10, 4))
    plt.plot(m_s, lw=1)
    plt.scatter(peaks_use, m_s[peaks_use.astype(int)], s=25, label="peaks used")
    plt.title("MASTER window (smoothed) with peaks used")
    plt.xlabel("Window Y (px)")
    plt.ylabel("Flux")
    plt.legend()
    plt.tight_layout()
    plt.savefig(qc_dir / "step07g_master_peaks_used.png", dpi=150, bbox_inches="tight")
    maybe_show_or_close()

    plt.figure(figsize=(6, 4))
    plt.hist(resid_nm[good], bins=20)
    plt.title(f"MASTER residuals (|resid|<{REFINE_TOL_NM_END:.2f} nm), RMS={rms_nm:.4f} nm")
    plt.xlabel("Residual (nm)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(qc_dir / "step07g_master_residuals.png", dpi=150, bbox_inches="tight")
    maybe_show_or_close()

    plt.figure(figsize=(10, 5))
    plotted = 0
    arc1d = load_arc1d_flux_map(ARC1D_EVEN, ARC1D_ODD)
    for s in QC_SLITS:
        s = s.strip().upper()
        if s not in arc1d or s not in shift_to_master:
            continue
        f_full = arc1d[s]
        f = norm(f_full[YWIN0:YWIN0 + FIRSTLEN])
        sh = int(shift_to_master[s])
        lam = p(ywin + sh)
        plt.plot(lam, f, lw=1, label=s)
        plotted += 1

    plt.title(f"QC: slit spectra in wavelength space (panel B, plotted {plotted})")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Flux (normalized)")
    plt.xlim(QC_XLIM[0], QC_XLIM[1])
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(qc_dir / "step07g_slit_spectra_wavelength.png", dpi=150, bbox_inches="tight")
    maybe_show_or_close()

    log.info("Step07g complete.")


if __name__ == "__main__":
    main()
