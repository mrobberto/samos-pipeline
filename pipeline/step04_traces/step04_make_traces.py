#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step 04 — Trace determination (Level2-style, robust for dense packed slitlets)

1) Build quartz trace image
   Even_traces = quartzB - quartzA (mosaicked frames)

2) Detect slit centers
   - collapse image into robust X-profile
   - find peaks corresponding to slit traces

3) Assign slit IDs in RA order
   - RA increases right → left on detector
   - centers sorted by X descending
   - EVEN slits labeled 0,2,4,...
   - ODD slits labeled 1,3,5,...

4) Build slit mask via row-wise local segmentation
   - within a window around each center
   - estimate background from sidebands
   - detect contiguous pixels above threshold
   - cap width and optionally shrink edges

5) Construct slit ID map
   - assign each mask pixel to nearest slit center

6) Optional: identify second spectral order
   - quartz traces may show faint higher-order features
     at the long-wavelength end of some slitlets
   - these features are separated from the first-order
     spectrum by a gap in detector space
   - when a clear gap is detected, an empirical cutoff
     is placed within the gap and pixels beyond it are excluded
   - this ensures that geometry products describe only
     the uncontaminated first-order spectrum

7) Derive slit geometry model
   - compute per-row centroid and edges
   - smooth and fit polynomials:
        x_center(y)
        x_left(y)
        x_right(y)

Outputs (reduced/04_traces)

Trace products
  Even_traces.fits

Segmentation products
  Even_traces_mask.fits
  Even_traces_slitid.fits

Slit catalog
  Even_traces_slit_table.csv

Trace geometry reference
  Even_traces_geometry.fits

Diagnostics
  Even_traces_traces.reg
  Even_traces_gap_cuts.csv


Run commands
------------
python pipeline/step04_traces/step04_make_traces.py --set EVEN
python pipeline/step04_traces/step04_make_traces.py --set ODD

or Spyder:

runfile("pipeline/step04_traces/step04_make_traces.py", args="--set EVEN")
runfile("pipeline/step04_traces/step04_make_traces.py", args="--set ODD")
"""

import logging
from pathlib import Path
import sys
import csv
import argparse



def load_radec_table(path: Path):
    """
    Read a RA/Dec table sorted by RA increasing.

    Supported columns (case-insensitive, tolerant):
      - RA:  ra, ra_deg, radeg, ra_hours, rah
      - DEC: dec, dec_deg, decdeg, decd
      - Optional slit label/id:
          label, slit, slit_id, slitid, id
      - Optional target index:
          index, idx, targidx, target_index

    Returns list of dicts (strings preserved):
      [{'sid': int|None, 'idx': int|None, 'ra': str, 'dec': str}, ...]
    """
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"No header found in {path}")

        fieldmap = {k.strip().lower(): k for k in reader.fieldnames}

        def pick(*cands):
            for c in cands:
                if c in fieldmap:
                    return fieldmap[c]
            return None

        ra_key  = pick("ra", "ra_deg", "radeg", "ra_hours", "rah")
        dec_key = pick("dec", "dec_deg", "decdeg", "decd")
        # Your radec tables use: label, index, RA, DEC (tab/CSV). We treat:
        #   label -> slit label / global SID (e.g. 0,2,4,...)
        #   index -> science proposal target index (integer)
        sid_key = pick("label", "slit", "slit_id", "slitid", "id")
        idx_key = pick("index", "idx", "targidx", "target_index", "targetid")

        if ra_key is None or dec_key is None:
            raise ValueError(f"{path} must contain RA/Dec columns. Found: {reader.fieldnames}")

        for r in reader:
            ra = (r.get(ra_key, "") or "").strip()
            dec = (r.get(dec_key, "") or "").strip()
            sid = None
            if sid_key is not None:
                raw = (r.get(sid_key, "") or "").strip()
                if raw != "":
                    try:
                        sid = int(float(raw))
                    except Exception:
                        sid = None

            idx = None
            if 'idx_key' in locals() and idx_key is not None:
                raw = (r.get(idx_key, "") or "").strip()
                if raw != "":
                    try:
                        idx = int(float(raw))
                    except Exception:
                        idx = None

            rows.append({"sid": sid, "idx": idx, "ra": ra, "dec": dec})

    return rows
THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parents[1]   # .../samos-pipeline
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import config  # noqa: E402

import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter
from scipy.signal import find_peaks



# ---------------------------------------------------------------------
# Command line arguments
# ---------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Step04 trace maker")
parser.add_argument(
    "--set",
    dest="trace_set",
    choices=["EVEN", "ODD"],
    default="EVEN",
    help="Trace set to process (EVEN or ODD)"
)

args, unknown = parser.parse_known_args()
TRACE_SET = args.trace_set.upper()




# -----------------------------------------------------------------------------
# USER SETTINGS
# -----------------------------------------------------------------------------
#FILE_A = "038.quartz.fits"
#FILE_B = "039.quartz.fits"
# Choose trace set: "EVEN" or "ODD"
# EVEN uses QUARTZ_B_EVEN (default 039) with QUARTZ_A (038): diff = B - A
# ODD  uses QUARTZ_B_ODD  (040) with QUARTZ_B_EVEN (039): diff = 040 - 039

if TRACE_SET.upper() == "EVEN":
    quartz_a_name = config.QUARTZ_SLITS_OFF
    quartz_b_name = config.QUARTZ_SLITS_ON_EVEN
    TRACE_TAG = "Even"
    TRACE_BASE = "Even_traces"
    TRACE_BASE_CAP = "EVEN"
    SID_BASE = 0   # 0,2,4,...
    radec_path = Path(config.RADEC_EVEN_CSV)
elif TRACE_SET.upper() == "ODD":
    quartz_a_name = config.QUARTZ_SLITS_OFF
    quartz_b_name = config.QUARTZ_SLITS_ON_ODD
    TRACE_TAG = "Odd"
    TRACE_BASE = "Odd_traces"
    TRACE_BASE_CAP = "ODD"
    SID_BASE = 1   # 1,3,5,...
    radec_path = Path(config.RADEC_ODD_CSV)
else:
    raise ValueError("TRACE_SET must be 'EVEN' or 'ODD'")

in_dir = Path(config.ST03P5_ROWSTRIPE)
FILE_A = in_dir / quartz_a_name
FILE_B = in_dir / quartz_b_name

OUTDIR = Path(config.ST04_TRACES)
OUTDIR.mkdir(parents=True, exist_ok=True)

EXPECTED_NSLITS = 32  # informational only (never used to stop anything)

YWIN0 = int(config.WAVECAL_YWIN0)
FIRSTLEN = int(config.WAVECAL_FIRSTLEN)

# SlitID map background label (must not collide with real slit IDs)
BKGID = -1

# Active band detection
ACTIVE_FRAC = 0.03
ACTIVE_PAD = 20

# Center finding (X profile)
PROFILE_SMOOTH = 3.0          # smooth 1D profile
MIN_PEAK_DIST = 18            # px (>= slit pitch; prevents duplicates)
PEAK_PROMINENCE = 0.15        # fraction of max(profile) after normalization
PEAK_HEIGHT_FRAC = 0.10       # fraction of max(profile)

# Local mask building per slit
# Trace geometry fit (derived from quartz mask/slitid; saved for later steps)
TRACE_PORDER = 5            # polynomial order for x_center(y)
TRACE_SMOOTH = 9            # median filter size along y (odd int)
TRACE_WEIGHTED = True       # use flux-weighted centroid within mask

# Edge geometry fit (left/right edges) to stabilize slit boundaries in rectification
EDGE_PORDER = TRACE_PORDER   # polynomial order for x_left(y), x_right(y)
EDGE_SMOOTH = TRACE_SMOOTH   # median filter size along y (odd int)
EDGE_PAD_PIX = 0.5          # padding added to modeled edges when masking/rectifying (pixels)

TRACE_CENTER_PORDER = 2
TRACE_EDGE_PORDER   = 2
CENTER_SMOOTH_WIN   = 31
EDGE_SMOOTH_WIN     = 31
CENTER_OUTLIER_PIX  = 1.5
EDGE_OUTLIER_PIX    = 2.0
MIN_ROWS_FIT        = 20

HALF_WINDOW = 18              # window half-size arofgeometrrund each center to inspect
SIDEBAND = 6                  # sideband width (pixels) for local background
LOCAL_NSIG = 5.0              # threshold = bkg + LOCAL_NSIG * sigma
USE_MAD = True                # robust sigma estimate from sidebands
MAX_WIDTH = 13                # cap final width (px) so it doesn’t bloat
EDGE_SHRINK = 1               # remove 1 px on each side after segmentation (0 disables)

TRACE_CENTER_HW = 10          # half-window for tracing center on each row
TRACE_CENTER_JUMP = 2.0       # max allowed row-to-row center motion
TRACE_CENTER_SMOOTH = 31      # smoothing window for traced centers
TRACE_CENTER_SEED_Y = None    # None => ny//2

TRACE_WIDTH_PORDER = 1      # 0 = constant width, 1 = slowly varying width
WIDTH_SMOOTH_WIN   = 31
WIDTH_OUTLIER_PIX  = 1.5

# --- First-order trimming based on quartz gap (per slit) ---
# The 2nd order is best identified in the quartz continuum (Even_traces), not in sparse arc lines.
# If enabled, we detect a per-slit "gap" along the dispersion (Y) direction and zero-out mask/slitid
# beyond the gap, so downstream wavecal/extraction operate on first order only.
TRIM_SECOND_ORDER = True

# Solid by-design first-order length (pixels along dispersion axis)
FIRST_ORDER_LEN = FIRSTLEN

GAP_SMOOTH = 51              # odd smoothing window along y
GAP_DROP_FRAC = 0.35         # threshold = GAP_DROP_FRAC * local median baseline
GAP_MIN_RUN = 25             # min contiguous low-run length (px) to call it a gap
MASK_COV_FRAC = 0.25         # fraction of slit width that must be "on" to consider the row active

# Row selection: use only rows where slit is “present” to avoid noise-only rows
ROW_MIN_SIGNAL = 3.0          # require peak - bkg >= this to accept row segment
TRACE_PAD = 1.0             # pixels added to the left/right of the traces to make them a bit fatter
# -----------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
log = logging.getLogger("_04_make_even_traces_and_slitid")


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


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 10:
        return float(np.nanstd(x)) if x.size else 0.0
    if USE_MAD:
        med = np.nanmedian(x)
        mad = np.nanmedian(np.abs(x - med))
        return float(1.4826 * mad) if mad > 0 else float(np.nanstd(x))
    return float(np.nanstd(x))


def contiguous_around_center(above: np.ndarray, c: int) -> tuple[int, int] | None:
    """Return (l,r) inclusive indices of the contiguous True-run containing c."""
    if c < 0 or c >= above.size or not above[c]:
        return None
    l = c
    while l - 1 >= 0 and above[l - 1]:
        l -= 1
    r = c
    while r + 1 < above.size and above[r + 1]:
        r += 1
    return l, r



# -----------------------------------------------------------------------------
# Gap / first-order trimming helpers (use quartz continuum)
# -----------------------------------------------------------------------------
def _smooth_1d(y, w=51):
    w = max(3, int(w) | 1)
    k = np.ones(w, dtype=float) / w
    yy = np.nan_to_num(y, nan=0.0)
    return np.convolve(yy, k, mode="same")


def detect_first_order_start_y(profile_y: np.ndarray,
                               y0: int, y1: int,
                               search_band: int = 1800,
                               smooth_w: int = 101,
                               frac: float = 0.30,
                               run: int = 150) -> int | None:
    """Return y_start where first order begins (sustained rise), else None."""
    sm = _smooth_1d(profile_y, w=smooth_w)
    span = y1 - y0 + 1
    m0 = y0 + int(0.60 * span)
    m1 = y0 + int(0.90 * span)
    main_level = float(np.nanmedian(sm[m0:m1]))
    if not np.isfinite(main_level) or main_level <= 0:
        return None
    thr = frac * main_level

    lo = y0
    hi = min(y1, y0 + int(search_band))

    above = np.nan_to_num(sm[lo:hi+1], nan=0.0) > thr
    count = 0
    for i, ok in enumerate(above):
        count = count + 1 if ok else 0
        if count >= int(run):
            return int(lo + i - run + 1)
    return None


def should_trim_second_order(profile_y: np.ndarray,
                            y0: int, y1: int,
                            y_start: int,
                            smooth_w: int = 101,
                            leak_level_frac: float = 0.12) -> bool:
    """Decide whether there is meaningful 2nd-order leak below y_start."""
    sm = _smooth_1d(profile_y, w=smooth_w)
    span = y1 - y0 + 1
    m0 = y0 + int(0.60 * span)
    m1 = y0 + int(0.90 * span)
    main_level = float(np.nanmedian(sm[m0:m1]))
    if not np.isfinite(main_level) or main_level <= 0:
        return False

    y_start = int(np.clip(y_start, y0+1, y1))
    bottom_level = float(np.nanpercentile(sm[y0:y_start], 85))
    if not np.isfinite(bottom_level):
        return False

    return (bottom_level / main_level) >= float(leak_level_frac)

def detect_gap_end_y(profile_y: np.ndarray,
                     y0: int, y1: int,
                     smooth_w: int = 101,
                     low_frac: float = 0.12,
                     min_run: int = 120,
                     search_band: int = 2200) -> int | None:
    """
    Find the first long 'low' run (gap) scanning from the high-Y end downward.
    Return the midpoint of the first significant low-signal gap encountered
    when scanning from the high-Y end downward. Rows above this cut are
    treated as second-order contamination and removed.
    The trim cut should then remove rows ABOVE this value.
    """
    sm = _smooth_1d(profile_y, w=smooth_w)

    span = y1 - y0 + 1
    m0 = y0 + int(0.60 * span)
    m1 = y0 + int(0.90 * span)
    main_level = float(np.nanmedian(sm[m0:m1]))
    if not np.isfinite(main_level) or main_level <= 0:
        return None

    low_thr = float(low_frac) * main_level

    # Search near the high-Y end (where the 2nd order appears in your data)
    hi = y1
    lo = max(y0, y1 - int(search_band))

    seg = np.nan_to_num(sm[lo:hi+1], nan=np.inf)

    # Reverse so we scan from high Y downward
    seg_rev = seg[::-1]
    is_low = seg_rev < low_thr

    count = 0
    start = None

    for i, ok in enumerate(is_low):
        if ok:
            if count == 0:
                start = i
            count += 1
        else:
            if count >= min_run:
                end = i - 1
                gap_len = end - start + 1
                if gap_len < 200:
                    return None
            
                # Convert reversed indices back to original Y coordinates
                # start/end of gap in original indexing
                y_gap_start = hi - end
                y_gap_end   = hi - start
            
                # Cut at the midpoint of the detected gap
                y_gap_mid = 0.5 * (y_gap_start + y_gap_end)
                return int(round(y_gap_mid))

            count = 0
            start = None

    # Handle run reaching the end
    if count >= min_run:
        end = len(is_low) - 1
        gap_len = end - start + 1
        if gap_len >= 200:
            y_gap_start = hi - end
            y_gap_end   = hi - start
            y_gap_mid = 0.5 * (y_gap_start + y_gap_end)
            return int(round(y_gap_mid))
    

    return None


def trim_second_order_from_mask(diff: np.ndarray, mask: np.ndarray, slitid: np.ndarray, slit_rows: list) -> tuple[np.ndarray, np.ndarray, list]:
    """
    Given quartz diff image, boolean mask and slitid map, detect per-slit gap positions and
    zero-out mask/slitid on the high-Y side of the gap, where the second-order contamination is present

    Returns (mask, slitid, updated_slit_rows).

    Also writes OUTDIR/Even_traces_gap_cuts.csv for provenance.
    """
    
    ny, nx = diff.shape
    gap_cuts: dict[int, int | None] = {}

    for row in slit_rows:
        sid = int(row[0])
        if sid < 0:
            gap_cuts[sid] = None
            continue

        comp = (slitid == sid) & mask
        if not np.any(comp):
            gap_cuts[sid] = None
            continue

        # Build per-row quartz profile inside the footprint
        prof_y = np.full(ny, np.nan, dtype=float)
        for y in range(ny):
            xs_row = np.where(comp[y])[0]
            if xs_row.size < 3:
                continue
            prof_y[y] = float(np.nanmedian(diff[y, xs_row]))

        y0_hint = None
        try:
            # slit_rows can be (sid, cx, ..., nrows, y0, y1) [len=11]
            # or (sid, old_sid, cx, ..., nrows, y0, y1) [len=12]
            y0_idx = 10 if len(row) >= 12 else 9
            y0_hint = int(row[y0_idx])
        except Exception:
            y0_hint = None
        if y0_hint is not None and y0_hint < 0:
            y0_hint = None

        # Build prof_y as you already do
        ys_present = np.where(np.any(comp, axis=1))[0]
        if ys_present.size < 200:
            continue
        y0 = int(ys_present.min())
        y1 = int(ys_present.max())
        
        """
        y_start = detect_first_order_start_y(prof_y, y0=y0, y1=y1)
        if y_start is None:
            y_cut = None
        else:
            if should_trim_second_order(prof_y, y0=y0, y1=y1, y_start=y_start):
                y_cut = int(y_start - 1)
            else:
                y_cut = None
        """
        y_gap_start = detect_gap_end_y(prof_y, y0=y0, y1=y1)
        y_cut = y_gap_start  # remove everything at/above this cut (upper 2nd-order side)
        
        if y_cut is not None:
            # remove top rows >= y_cut
            kill = comp.copy()
            kill[:y_cut, :] = False   # keep rows below the cut
            mask[kill] = False
            slitid[kill] = BKGID
    
        gap_cuts[sid] = y_cut

    n_found = sum(1 for v in gap_cuts.values() if v is not None)
    log.info("Gap-based trims applied to %d slits (gap inside FOV).", n_found)

    # Update slit_rows (nrows, y0, y1) after trimming so geometry fits are consistent
    new_slit_rows = []
    for row in slit_rows:
        sid = int(row[0])
        if sid < 0:
            new_slit_rows.append(row)
            continue

        comp = (slitid == sid)
        if not np.any(comp):
            # keep same columns but zero nrows; y0/y1 unchanged
            row = list(row)
            if len(row) >= 12:
                # (sid, old_sid, cx, ..., w84, nrows, y0, y1)
                row[9] = 0
            else:
                if len(row) < 11:
                    row = row + [np.nan] * (11 - len(row))
                row[8] = 0
            new_slit_rows.append(tuple(row))
            continue
        
        ys_present = np.where(np.any(comp, axis=1))[0]
        y0_new = int(ys_present.min())
        y1_new = int(ys_present.max())
        nrows_new = int(ys_present.size)

        row = list(row)
        if len(row) >= 12:
            row[9] = nrows_new
            row[10] = y0_new
            row[11] = y1_new
        else:
            if len(row) < 11:
                row = row + [np.nan] * (11 - len(row))
            row[8] = nrows_new
            row[9] = y0_new
            row[10] = y1_new
        new_slit_rows.append(tuple(row))

    # Write CSV with cut locations for downstream debugging
    try:
        cut_path = OUTDIR / f"{TRACE_BASE}_gap_cuts.csv"
        lines = ["slit_id,y_cut\n"]
        for sid in sorted(gap_cuts.keys()):
            yc = gap_cuts[sid]
            lines.append(f"{sid},{'' if yc is None else int(yc)}\n")
        cut_path.write_text("".join(lines))
        log.info("Wrote %s", cut_path)
    except Exception as e:
        log.warning("Could not write gap cuts CSV: %s", e)

    return mask, slitid, new_slit_rows

def medfilt1d_nan(x, win):
    x = np.asarray(x, float)
    n = x.size
    if win % 2 == 0:
        win += 1
    h = win // 2
    out = np.full(n, np.nan, float)
    for i in range(n):
        a = max(0, i - h)
        b = min(n, i + h + 1)
        vals = x[a:b]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i] = np.median(vals)
    return out


def robust_polyfit(y, x, order=3, clip_pix=1.5, niter=3):
    y = np.asarray(y, float)
    x = np.asarray(x, float)

    ok = np.isfinite(y) & np.isfinite(x)
    if ok.sum() < max(MIN_ROWS_FIT, order + 2):
        return None, np.full_like(x, np.nan, dtype=float), np.nan

    yy = y[ok]
    xx = x[ok]

    mask = np.ones_like(xx, dtype=bool)

    for _ in range(niter):
        if mask.sum() < max(MIN_ROWS_FIT, order + 2):
            break
        coeff = np.polyfit(yy[mask], xx[mask], order)
        model = np.polyval(coeff, yy)
        resid = xx - model
        newmask = np.abs(resid) <= clip_pix
        if newmask.sum() == mask.sum():
            mask = newmask
            break
        mask = newmask

    if mask.sum() < max(MIN_ROWS_FIT, order + 2):
        coeff = np.polyfit(yy, xx, min(order, 1))
        model = np.polyval(coeff, yy)
        resid = xx - model
    else:
        coeff = np.polyfit(yy[mask], xx[mask], order)
        model = np.polyval(coeff, yy)
        resid = xx - model

    rms = np.sqrt(np.nanmean(resid**2)) if resid.size else np.nan

    full_model = np.full_like(x, np.nan, dtype=float)
    full_model[ok] = np.polyval(coeff, yy)

    # return in ascending power order to match PC0, PC1, ...
    return coeff[::-1], full_model, rms


def trace_center_and_edges_from_slit(diff, slitid, sid, ymin=None, ymax=None):
    """
    Measure raw center/edges for one slit while preserving detector Y.
    Uses the existing slitid footprint only to identify candidate pixels,
    but derives center from quartz flux in each row.
    """
    ny, nx = diff.shape

    if ymin is None:
        ymin = 0
    if ymax is None:
        ymax = ny - 1

    y = np.arange(ny, dtype=int)

    xcen_raw = np.full(ny, np.nan, float)
    xleft_raw = np.full(ny, np.nan, float)
    xright_raw = np.full(ny, np.nan, float)
    width_raw = np.full(ny, np.nan, float)

    for yy in range(max(0, ymin), min(ny - 1, ymax) + 1):
        xs = np.where(slitid[yy] == sid)[0]
        if xs.size < 3:
            continue

        # Raw edges directly from current slit footprint
        xleft_raw[yy] = float(xs.min())
        xright_raw[yy] = float(xs.max())
        width_raw[yy] = float(xs.max() - xs.min())

        # Flux-weighted center from the actual quartz signal
        vals = diff[yy, xs].astype(float)
        good = np.isfinite(vals)
        if good.sum() < 3:
            xcen_raw[yy] = float(np.mean(xs))
            continue

        vals = vals[good]
        xuse = xs[good].astype(float)

        base = np.nanmedian(vals)
        w = vals - base
        w[w < 0] = 0.0

        if np.nansum(w) > 0:
            xcen_raw[yy] = float(np.nansum(xuse * w) / np.nansum(w))
        else:
            xcen_raw[yy] = float(np.mean(xs))

    return y, xcen_raw, xleft_raw, xright_raw, width_raw


def recenter_edges_around_model(diff, slitid, sid, xcen_model, xleft_raw, xright_raw, ymin, ymax):
    """
    Optional refinement: keep Y fixed, but stabilize edges relative to the
    smooth center model using the original slitid extent as a guide.
    """
    ny, nx = diff.shape
    xleft2 = np.array(xleft_raw, copy=True)
    xright2 = np.array(xright_raw, copy=True)

    for yy in range(max(0, ymin), min(ny - 1, ymax) + 1):
        if not np.isfinite(xcen_model[yy]):
            continue

        xs = np.where(slitid[yy] == sid)[0]
        if xs.size < 3:
            continue

        # preserve measured width, but allow symmetric stabilization
        if np.isfinite(xleft_raw[yy]) and np.isfinite(xright_raw[yy]):
            hw = 0.5 * (xright_raw[yy] - xleft_raw[yy])
            xleft2[yy] = xcen_model[yy] - hw
            xright2[yy] = xcen_model[yy] + hw

    return xleft2, xright2

def smooth_nan_1d(x, win=31):
    x = np.asarray(x, float)
    n = x.size
    if n < 3:
        return x.copy()
    if win % 2 == 0:
        win += 1
    h = win // 2
    out = np.full(n, np.nan, float)
    for i in range(n):
        a = max(0, i - h)
        b = min(n, i + h + 1)
        vals = x[a:b]
        vals = vals[np.isfinite(vals)]
        if vals.size:
            out[i] = np.median(vals)
    ok = np.isfinite(out)
    if ok.sum() >= 2:
        yy = np.arange(n, dtype=float)
        out[~ok] = np.interp(yy[~ok], yy[ok], out[ok])
    return out


def robust_row_centroid(row, x_seed, half_window=10):
    """
    Flux-weighted centroid near a seed X position.
    Uses only local positive signal above a local background.
    """
    nx = row.size
    if not np.isfinite(x_seed):
        return np.nan

    xc = int(round(float(x_seed)))
    xa = max(0, xc - half_window)
    xb = min(nx, xc + half_window + 1)
    if xb - xa < 3:
        return np.nan

    prof = np.asarray(row[xa:xb], float)
    if not np.isfinite(prof).any():
        return np.nan

    base = np.nanmedian(prof)
    w = prof - base
    w[~np.isfinite(w)] = 0.0
    w[w < 0] = 0.0

    if np.sum(w) <= 0:
        return np.nan

    xx = np.arange(xa, xb, dtype=float)
    return float(np.sum(xx * w) / np.sum(w))


def trace_one_slit_center(diff, x_init, y_start, y_stop, step=1, half_window=10,
                          jump_max=2.0):
    """
    Trace one slit center through rows, preserving Y.
    Propagates from a starting row upward or downward.
    """
    ny, nx = diff.shape
    xtrace = np.full(ny, np.nan, float)

    if not np.isfinite(x_init):
        return xtrace

    xtrace[y_start] = float(x_init)
    prev = float(x_init)

    if y_stop >= y_start:
        yrange = range(y_start + step, y_stop + 1, step)
    else:
        yrange = range(y_start - step, y_stop - 1, -step)

    for y in yrange:
        xnew = robust_row_centroid(diff[y], prev, half_window=half_window)

        if not np.isfinite(xnew):
            xtrace[y] = prev
            continue

        if abs(xnew - prev) > jump_max:
            # do not allow abrupt local jumps
            xnew = prev + np.sign(xnew - prev) * jump_max

        xtrace[y] = xnew
        prev = xnew

    return xtrace


def build_row_dependent_centers(diff, centers, y_seed=None,
                                half_window=10, jump_max=2.0, smooth_win=31):
    """
    Returns xcenter_rows with shape (ny, nslits), where each column is the
    traced center X(y) for one slit.
    """
    ny, nx = diff.shape
    centers = np.asarray(centers, float)
    nslits = centers.size

    if y_seed is None:
        y_seed = ny // 2

    xcenter_rows = np.full((ny, nslits), np.nan, float)

    for i, cx in enumerate(centers):
        down = trace_one_slit_center(diff, cx, y_seed, 0,
                                     half_window=half_window, jump_max=jump_max)
        up = trace_one_slit_center(diff, cx, y_seed, ny - 1,
                                   half_window=half_window, jump_max=jump_max)

        xt = np.full(ny, np.nan, float)
        xt[:y_seed + 1] = down[:y_seed + 1]
        xt[y_seed:] = up[y_seed:]

        xt = smooth_nan_1d(xt, win=smooth_win)
        xcenter_rows[:, i] = xt

    return xcenter_rows



def segment_around_moving_center(diff, xcenter_rows, half_window, sideband,
                                 local_nsig, row_min_signal, max_width,
                                 edge_shrink):
    """
    Build mask using row-dependent traced centers.
    """
    ny, nx = diff.shape
    nslits = xcenter_rows.shape[1]

    mask = np.zeros((ny, nx), dtype=bool)
    slit_rows_tmp = []

    for old_sid, _ in enumerate(np.arange(nslits), start=1):
        i = old_sid - 1
        widths = []
        y_used = []

        for y in range(ny):
            cx = xcenter_rows[y, i]
            if not np.isfinite(cx):
                continue

            cxi = int(round(cx))
            xa = max(0, cxi - half_window)
            xb = min(nx, cxi + half_window + 1)
            row = diff[y, xa:xb].astype(float)

            if row.size < 3 or not np.isfinite(row).any():
                continue

            c_local = int(round(cx - xa))
            c_local = max(0, min(c_local, row.size - 1))

            l1 = max(0, c_local - half_window)
            l2 = max(0, c_local - sideband)
            r1 = min(row.size, c_local + sideband + 1)
            r2 = min(row.size, c_local + half_window + 1)

            side = np.concatenate([row[l1:l2], row[r1:r2]]) if (l2 > l1 or r2 > r1) else row
            bkg = float(np.nanmedian(side))
            sig = robust_sigma(side)
            if sig <= 0:
                sig_fallback = np.nanstd(side)
                sig = float(sig_fallback) if np.isfinite(sig_fallback) and sig_fallback > 0 else 1.0

            thr = bkg + local_nsig * sig

            peak = float(np.nanmax(row))
            if (peak - bkg) < row_min_signal:
                continue

            above = row > thr
            seg = contiguous_around_center(above, c_local)
            if seg is None:
                continue

            l, r = seg
            w = (r - l + 1)

            if w > max_width:
                half = max_width // 2
                l = max(0, c_local - half)
                r = min(row.size - 1, l + max_width - 1)

            if edge_shrink > 0 and (r - l + 1) > (2 * edge_shrink + 2):
                l += edge_shrink
                r -= edge_shrink

            mask[y, xa + l: xa + r + 1] = True
            widths.append(float(r - l + 1))
            y_used.append(y)

        slit_rows_tmp.append((old_sid, widths, y_used))

    return mask, slit_rows_tmp


def assign_slitid_from_row_centers(mask, xcenter_rows, new_sid_for_oldidx, bkgid):
    """
    Assign slit IDs row by row using the traced row-dependent centers.
    """
    ny, nx = mask.shape
    slitid = np.full(mask.shape, bkgid, dtype=np.int16)

    ys, xs = np.where(mask)
    if xs.size == 0:
        return slitid

    # row by row so each row uses its own traced centers
    for y in range(ny):
        xx = np.where(mask[y])[0]
        if xx.size == 0:
            continue

        xrow = xcenter_rows[y, :]
        ok = np.isfinite(xrow)
        if ok.sum() == 0:
            continue

        d = np.abs(xx[:, None] - xrow[None, :])
        d[:, ~ok] = np.inf
        nearest = np.argmin(d, axis=1)

        slitid[y, xx] = new_sid_for_oldidx[nearest].astype(np.int16)

    return slitid


def main():
    fileA = FILE_A
    fileB = FILE_B
    if not fileA.exists():
        raise FileNotFoundError(fileA)
    if not fileB.exists():
        raise FileNotFoundError(fileB)
    if not radec_path.exists():
        raise FileNotFoundError(radec_path)

    log.info("TRACE_SET   = %s", TRACE_SET)
    log.info("Quartz OFF  = %s", fileA)
    log.info("Quartz ON   = %s", fileB)
    log.info("RADEC table = %s", radec_path)
    log.info("Output dir  = %s", OUTDIR)

    # -------------------------------------------------------------------------
    # 1) Even_traces
    # -------------------------------------------------------------------------
    from astropy.io import fits
    import matplotlib.pyplot as plt
    
    # 1. Open the FITS file
    with fits.open(fileA) as hduA:
        imgA = hduA[0].data
    with fits.open(fileB) as hduB:
        hdr = hduB[0].header
        imgB = hduB[0].data
    
#    hduA = samos.read_SAMI_mosaic(str(fileA))
#    hduB = samos.read_SAMI_mosaic(str(fileB))
    
#    imgA = np.asarray(hduA.data, dtype=np.float32)
#    imgB = np.asarray(hduB.data, dtype=np.float32)
    if imgA.shape != imgB.shape:
        raise ValueError(f"Shape mismatch: {imgA.shape} vs {imgB.shape}")

    diff = imgB - imgA
    hdr = hduB[0].header.copy()
    hdr.add_history(f"Quartz subtraction: {FILE_B.name} - {FILE_A.name}")

    traces_path = OUTDIR / f"{TRACE_BASE}.fits"
    fits.PrimaryHDU(diff.astype(np.float32), header=hdr).writeto(traces_path, overwrite=True)
    log.info("Wrote %s", traces_path)

    # -------------------------------------------------------------------------
    # 2) Find active band and slit centers from 1D profile
    # -------------------------------------------------------------------------
    img_s = gaussian_filter(diff, sigma=(2.0, 0.8))

    x0, x1 = find_active_band(img_s)
    log.info("Active X band: x_active=(%d,%d)", x0, x1)

    prof = np.nanpercentile(np.clip(img_s[:, x0:x1 + 1], 0, None), 90, axis=0).astype(float)

    prof_s = gaussian_filter(prof, sigma=PROFILE_SMOOTH)
    mx = np.nanmax(prof_s)
    if not np.isfinite(mx) or mx <= 0:
        raise RuntimeError("Profile has no signal; cannot find slit centers.")
    prof_n = prof_s / mx

    peaks, props = find_peaks(
        prof_n,
        distance=MIN_PEAK_DIST,
        height=PEAK_HEIGHT_FRAC,
        prominence=PEAK_PROMINENCE,
    )
    centers = (peaks + x0).astype(int)
    
    # ---------------------------------------------------------------------
    # 2b) Load RA/Dec table and map to detected slits (RA order = x DESC)
    # ---------------------------------------------------------------------
    radec_rows = load_radec_table(radec_path)

    if len(radec_rows) != len(centers):
        log.warning(
            "%s rows (%d) != detected centers (%d). Will match by min length.",
            radec_path.name, len(radec_rows), len(centers),
        )

    # radec table sorted by RA increasing; RA increases right->left on detector.
    # Pixel x increases left->right, so RA order corresponds to centers sorted by x DESC.
    idx_x_desc = np.argsort(centers)[::-1]
    nmap = min(len(radec_rows), len(centers))

    # Map RA/Dec to ORIGINAL slit index (old_sid = 1..N in centers order)
    # and also (optionally) define the FINAL slit IDs from the table's 'sid' column.
    slit_radec = {}            # old_sid -> {'ra':..., 'dec':...}
    sid_from_old_sid = {}      # old_sid -> final sid (odd/even labeling)

    # If the table provides explicit sid labels, use them; otherwise use generated labels.
    table_has_sid = any(r.get("sid") is not None for r in radec_rows[:nmap])

    for j in range(nmap):
        old_sid = int(idx_x_desc[j] + 1)  # old IDs are 1..N in the original centers order (centers array)
        slit_radec[old_sid] = {"ra": radec_rows[j]["ra"], "dec": radec_rows[j]["dec"], "idx": radec_rows[j].get("idx", None)}

        if table_has_sid and (radec_rows[j].get("sid") is not None):
            sid_from_old_sid[old_sid] = int(radec_rows[j]["sid"])

    log.info(
        "Loaded %d RA/Dec rows from %s and mapped to %d slits.",
        len(radec_rows), radec_path, nmap,
    )

    # ---------------------------------------------------------------------
    # Relabel slit IDs by RA order.
    # Assign GLOBAL slit IDs (no conflicts between sets)
    # use EVEN (2,4,...) or ODD (1,3,...) generated labels in RA order.
    # ---------------------------------------------------------------------

    nslits = len(centers)

    # RA order: right -> left
    order_ra = np.argsort(centers)[::-1]
    
    # Determine parity for this run
    if TRACE_SET.upper() == "EVEN":
        start_id = 0
    elif TRACE_SET.upper() == "ODD":
        start_id = 1
    else:
        raise ValueError("TRACE_SET must be EVEN or ODD")
    
    # Map old detection index -> new global slit ID
    new_sid_for_oldidx = np.zeros(nslits, dtype=int)
    
    for rank, old_idx in enumerate(order_ra):
        new_sid_for_oldidx[old_idx] = start_id + 2 * rank
    
    # Convenience: old SID (1..N) -> new global SID
    new_sid_for_oldsid = {
        old_sid: int(new_sid_for_oldidx[old_sid - 1])
        for old_sid in range(1, nslits + 1)
    }
    
    log.info(
            "Detected %d slit centers (EXPECTED_NSLITS=%s informational).",
            len(centers),
            str(EXPECTED_NSLITS),
        )
    if EXPECTED_NSLITS is not None and len(centers) != int(EXPECTED_NSLITS):
        log.warning(
            "Expected %d, found %d slit centers (EXPECTED_NSLITS is informational only).",
            int(EXPECTED_NSLITS),
            len(centers),
        )

    if len(centers) == 0:
        raise RuntimeError("No slit centers found. Lower PEAK_PROMINENCE/PEAK_HEIGHT_FRAC.")


    # -------------------------------------------------------------------------
    # 3) Build row-dependent center traces, then per-slit mask
    # -------------------------------------------------------------------------
    ny, nx = diff.shape

    seed_y = TRACE_CENTER_SEED_Y if TRACE_CENTER_SEED_Y is not None else (ny // 2)

    # xcenter_rows[y, i] = traced X center of slit i at row y
    xcenter_rows = build_row_dependent_centers(
        diff,
        centers,
        y_seed=seed_y,
        half_window=TRACE_CENTER_HW,
        jump_max=TRACE_CENTER_JUMP,
        smooth_win=TRACE_CENTER_SMOOTH,
    )

    mask, slit_rows_tmp = segment_around_moving_center(
        diff,
        xcenter_rows,
        half_window=HALF_WINDOW,
        sideband=SIDEBAND,
        local_nsig=LOCAL_NSIG,
        row_min_signal=ROW_MIN_SIGNAL,
        max_width=MAX_WIDTH,
        edge_shrink=EDGE_SHRINK,
    )

    slit_rows = []

    for old_sid, cx in enumerate(centers, start=1):
        sid = new_sid_for_oldsid[old_sid]

        widths = slit_rows_tmp[old_sid - 1][1]
        y_used = slit_rows_tmp[old_sid - 1][2]

        if len(widths) == 0:
            slit_rows.append((sid, old_sid, cx, np.nan, np.nan, np.nan,
                              np.nan, np.nan, np.nan, 0, -1, -1))
            continue

        widths = np.asarray(widths, float)
        wmin = float(np.nanmin(widths))
        wmean = float(np.nanmean(widths))
        wmax = float(np.nanmax(widths))
        w16 = float(np.nanpercentile(widths, 16))
        w50 = float(np.nanpercentile(widths, 50))
        w84 = float(np.nanpercentile(widths, 84))
        y0 = int(np.min(y_used))
        y1 = int(np.max(y_used))

        slit_rows.append((sid, old_sid, cx, wmin, wmean, wmax,
                          w16, w50, w84, int(widths.size), y0, y1))

    # -------------------------------------------------------------------------
    # 4) SlitID map: row-dependent nearest-center assignment inside mask
    # -------------------------------------------------------------------------
    slitid = assign_slitid_from_row_centers(
        mask,
        xcenter_rows,
        new_sid_for_oldidx,
        BKGID,
    )


    # -------------------------------------------------------------------------
    # --- SAVE PRE-TRIM STATE ---
    mask_pretrim_path = OUTDIR / f"{TRACE_BASE}_mask_pretrim.fits"
    slitid_pretrim_path = OUTDIR / f"{TRACE_BASE}_slitid_pretrim.fits"
    
    fits.PrimaryHDU(mask.astype(np.uint8)).writeto(mask_pretrim_path, overwrite=True)
    fits.PrimaryHDU(slitid.astype(np.int16)).writeto(slitid_pretrim_path, overwrite=True)
    
    log.info("Wrote %s", mask_pretrim_path)
    log.info("Wrote %s", slitid_pretrim_path)
    # -------------------------------------------------------------------------
    # 4b) OPTIONAL: trim 2nd order using quartz gap, per slit (from continuum)
    # -------------------------------------------------------------------------
    if TRIM_SECOND_ORDER:
        log.info("Trimming 2nd order using quartz gap (per slit)...")
        mask, slitid, slit_rows = trim_second_order_from_mask(diff, mask, slitid, slit_rows)
# 5) Write outputs
    # -------------------------------------------------------------------------
    mhdr = hdr.copy()
    mhdr.add_history(f"{TRACE_BASE}_mask: row-dependent traced centers + per-row local segmentation")
    mhdr["SLITSCH"] = ("EVEN0_ODD1", "Slit labeling: EVEN starts at 0, ODD starts at 1")
    mhdr["TRCSET"] = (TRACE_BASE_CAP, "Trace set used to define slit geometry")
    mhdr["NSLITS"] = (int(len(centers)), "Number of slit centers detected")
    mhdr["BKGID"] = (int(BKGID), "Background label in slitid map")
    mhdr["HALFWIN"] = (int(HALF_WINDOW), "Half window for per-row segmentation")
    mhdr["LNSIG"] = (float(LOCAL_NSIG), "Local threshold sigma for segmentation")
    mhdr["MAXW"] = (int(MAX_WIDTH), "Maximum mask width (px)")
    mhdr["SHRINK"] = (int(EDGE_SHRINK), "Edge shrink (px) after segmentation")
    mhdr["EXPN"] = (str(EXPECTED_NSLITS), "Expected slit count (informational only)")

    mask_path = OUTDIR / f"{TRACE_BASE}_mask.fits"
    fits.PrimaryHDU(mask.astype(np.uint8), header=mhdr).writeto(mask_path, overwrite=True)
    log.info("Wrote %s", mask_path)

    shdr = hdr.copy()
    shdr.add_history(f"{TRACE_BASE}_slitid: row-dependent nearest-center assignment inside mask")
    shdr["SLITSCH"] = ("EVEN0_ODD1", "Slit labeling: EVEN starts at 0, ODD starts at 1")
    shdr["TRCSET"] = (TRACE_BASE_CAP, "Trace set used to define slit geometry")
    shdr["NSLITS"] = (int(len(centers)), "Number of slit centers detected")
    shdr["BKGID"] = (int(BKGID), "Background label in slitid map")

    slitid_path = OUTDIR / f"{TRACE_BASE}_slitid.fits"
    fits.PrimaryHDU(slitid, header=shdr).writeto(slitid_path, overwrite=True)
    log.info("Wrote %s", slitid_path)
    
    """
    table_path = OUTDIR / f"{TRACE_BASE}_slit_table.csv"
    lines = ["slit_id,old_slit_id,ra,dec,xc,width_min,width_mean,width_max,width_p16,width_med,width_p84,nrows_used,y0,y1\\n"]
    for row in slit_rows:
        (sid, old_sid, xc, wmin, wmean, wmax, w16, w50, w84, nrows, y0, y1) = row
        radec = slit_radec.get(old_sid, {"ra": "", "dec": ""})
        lines.append(            f"{sid},{old_sid},{radec['ra']},{radec['dec']},{xc},{wmin},{wmean},{wmax},{w16},{w50},{w84},{nrows},{y0},{y1}\\n")
    table_path.write_text("".join(lines))
    """
    # --- write slit table CSV (REAL newlines) ---
    table_path = OUTDIR / f"{TRACE_BASE}_slit_table.csv"
    
    lines = []
    lines.append("slit_id,old_slit_id,index,ra,dec,xc,width_min,width_mean,width_max,width_p16,width_med,width_p84,nrows_used,y0,y1\n")
    
    for row in slit_rows:
        # unpack according to your slit_rows structure
        sid, old_sid, xc, wmin, wmean, wmax, w16, w50, w84, nrows, y0, y1 = row
    
        radec = slit_radec.get(old_sid, {"ra": "", "dec": ""})  # RA/Dec tied to OLD id
        lines.append(
            f"{int(sid)},{int(old_sid)},{'' if radec.get('idx', None) is None else int(radec.get('idx'))},{radec['ra']},{radec['dec']},{int(xc)},"
            f"{wmin},{wmean},{wmax},{w16},{w50},{w84},{int(nrows)},{int(y0)},{int(y1)}\n"
        )

    # write as text with correct newlines
    table_path.write_text("".join(lines), encoding="utf-8")
    log.info("Wrote %s", table_path)
    
    # -------------------------------------------------------------------------
    # 5b) ALSO write *_mask_reg / *_slitid_reg as ID-preserving copies
    #     (Downstream QC + some steps prefer *_reg if present.)
    # -------------------------------------------------------------------------
    mask_reg_path = OUTDIR / f"{TRACE_BASE}_mask_reg.fits"
    slitid_reg_path = OUTDIR / f"{TRACE_BASE}_slitid_reg.fits"
    
    # check so the pipeline cannot silently regress to 1..N again:
    u = np.unique(slitid[slitid != BKGID]).astype(int)
    u = u[u != BKGID]
    u = np.sort(u)
    exp = None
    # Prefer expected ID set from radec table 'label' column if present
    radec_sids = [r.get('sid', None) for r in radec_rows]
    radec_sids = [int(s) for s in radec_sids if s is not None]
    if len(radec_sids) > 0:
        exp = np.sort(np.unique(np.array(radec_sids, dtype=int)))
    else:
        # fallback: legacy expectations
        if TRACE_SET.upper() == "EVEN":
            exp = np.arange(0, 66, 2)
        else:
            exp = np.arange(1, 65, 2)
    
    if not (len(u) == len(exp) and np.all(u == exp)):
        log.warning("GLOBAL ID SET NOT CANONICAL")
        log.warning("Found: %s", u)
        log.warning("Expected: %s", exp)
    
    fits.PrimaryHDU(mask.astype(np.uint8), header=mhdr).writeto(mask_reg_path, overwrite=True)
    fits.PrimaryHDU(slitid.astype(np.int16), header=shdr).writeto(slitid_reg_path, overwrite=True)
    
    log.info("Wrote %s (copy; preserves IDs)", mask_reg_path)
    log.info("Wrote %s (copy; preserves IDs)", slitid_reg_path)
    
    
    # -------------------------------------------------------------------------
    # 6) Write a geometry reference MEF (per-slit polynomial x_center(y))
    #     This is the "fixed geometry" product to be used by later steps (e.g. 06c)
    # -------------------------------------------------------------------------
    geom_path = OUTDIR / f"{TRACE_BASE}_geometry.fits"

    ghdr0 = hdr.copy()
    ghdr0.add_history(f"{TRACE_BASE}_geometry: trace model x_center(y) derived from quartz mask/slitid")
    ghdr0["TRCSET"] = (TRACE_BASE_CAP, "Trace set used to define slit geometry")
    ghdr0["NSLITS"] = (int(len(centers)), "Number of slit centers detected")
    ghdr0["PORDER"] = (int(TRACE_PORDER), "Polynomial order for x_center(y) in each slit")
    ghdr0["TSMOOTH"] = (int(TRACE_SMOOTH), "Median filter size along y for x_center")
    ghdr0["TWEIGHT"] = (bool(TRACE_WEIGHTED), "Use flux-weighted centroid within mask")
    ghdr0["EPORDER"] = (int(TRACE_EDGE_PORDER), "Polynomial order for edges xL(y), xR(y) in each slit")
    ghdr0["ESMOOTH"] = (int(EDGE_SMOOTH), "Median filter size along y for edges")
    ghdr0["EPAD"] = (float(EDGE_PAD_PIX), "Recommended edge pad (px) for masking")

    geom_hdus = [fits.PrimaryHDU(header=ghdr0)]

    for sid, old_sid, cx, wmin, wmean, wmax, w16, w50, w84, nrows, y0, y1 in slit_rows:
        sid = int(sid)
        if sid < 0:
            continue

        comp = (slitid == sid)
        if not np.any(comp):
            # still write an extension so downstream knows slit exists
            hh = fits.Header()
            hh["EXTNAME"] = f"SLIT{sid:03d}"
            hh["SLITID"] = sid
            hh["OLDSID"] = (int(old_sid), "Original slit index before RA relabel")
            hh["YMIN"] = int(y0) if int(y0) >= 0 else 0
            hh["YMAX"] = int(y1) if int(y1) >= 0 else 0
            hh["XREF"] = float(cx)
            hh["PORDER"] = int(TRACE_PORDER)
            radec = slit_radec.get(old_sid, {"ra": "", "dec": ""})  # <-- RA/Dec still tied to OLD sid mapping
            if radec is not None:
                hh["RA"]  = (str(radec["ra"]),  "Slit RA from radec.csv (sorted by RA)")
                hh["DEC"] = (str(radec["dec"]), "Slit Dec from radec.csv (sorted by RA)")
                if radec.get("idx", None) is not None:
                    hh["INDEX"] = (int(radec["idx"]), "Target index from input radec table")
            hh.add_history("No pixels for this slit in slitid; geometry unavailable.")
            geom_hdus.append(fits.ImageHDU(data=np.full((ny,), np.nan, dtype=np.float32), header=hh))
            continue

        ny, nx = diff.shape

        # preserve the existing Y limits from slit_rows
        y_min = int(y0) if int(y0) >= 0 else 0
        y_max = int(y1) if int(y1) >= 0 else (ny - 1)

        yall, xcen_raw, xleft_raw, xright_raw, width_raw = trace_center_and_edges_from_slit(
            diff, slitid, sid, ymin=y_min, ymax=y_max
        )

        # smooth raw measurements before fitting
        xcen_s = medfilt1d_nan(xcen_raw, CENTER_SMOOTH_WIN)
        xleft_s = medfilt1d_nan(xleft_raw, EDGE_SMOOTH_WIN)
        xright_s = medfilt1d_nan(xright_raw, EDGE_SMOOTH_WIN)

        # fit center first
        coeff_c, xcen_fit, rms_c = robust_polyfit(
            yall, xcen_s, order=TRACE_CENTER_PORDER, clip_pix=CENTER_OUTLIER_PIX
        )

        if coeff_c is None:
            # safe fallback: use old fixed center
            xcen_fit = np.full(ny, float(cx), dtype=float)
            coeff_c = np.zeros(TRACE_CENTER_PORDER + 1, dtype=float)
            coeff_c[0] = float(cx)
            rms_c = np.nan

        # ------------------------------------------------------------
        # Fit width instead of left/right edges independently
        # This keeps the two edges parallel by construction.
        # ------------------------------------------------------------

        # raw slit width from current footprint
        width_raw = xright_raw - xleft_raw
        width_s = medfilt1d_nan(width_raw, WIDTH_SMOOTH_WIN)

        coeff_w, width_fit, rms_w = robust_polyfit(
            yall, width_s, order=TRACE_WIDTH_PORDER, clip_pix=WIDTH_OUTLIER_PIX
        )

        if coeff_w is None:
            medw = np.nanmedian(width_raw[y_min:y_max+1])
            if not np.isfinite(medw):
                medw = float(w50) if np.isfinite(w50) else 6.0

            width_fit = np.full(ny, float(medw), dtype=float)
            coeff_w = np.zeros(TRACE_WIDTH_PORDER + 1, dtype=float)
            coeff_w[0] = float(medw)
            rms_w = np.nan

        # protect against pathological widths
        badw = (~np.isfinite(width_fit)) | (width_fit <= 1.0)
        if np.any(badw):
            medw = np.nanmedian(width_fit[np.isfinite(width_fit) & (width_fit > 1.0)])
            if not np.isfinite(medw):
                medw = np.nanmedian(width_raw[y_min:y_max+1])
            if not np.isfinite(medw):
                medw = float(w50) if np.isfinite(w50) else 6.0
            width_fit[badw] = float(medw)

        # reconstruct edges from center + width
        xL_model = xcen_fit - 0.5 * width_fit
        xR_model = xcen_fit + 0.5 * width_fit

        # expand traces slightly to avoid clipping
        xL_model -= TRACE_PAD
        xR_model += TRACE_PAD

        # keep compatibility with downstream header-writing code
        # LC/RC still need polynomial coefficient arrays.
        # We derive them from center and width:
        #
        #   left  = center - 0.5*width
        #   right = center + 0.5*width
        #
        ncl = max(len(coeff_c), len(coeff_w))
        coeff_l = np.zeros(ncl, dtype=float)
        coeff_r = np.zeros(ncl, dtype=float)

        coeff_l[:len(coeff_c)] += coeff_c
        coeff_r[:len(coeff_c)] += coeff_c

        coeff_l[:len(coeff_w)] -= 0.5 * coeff_w
        coeff_r[:len(coeff_w)] += 0.5 * coeff_w

        # optional RMS estimates relative to raw measured edges
        rms_l = np.sqrt(np.nanmean((xleft_raw[y_min:y_max+1] - xL_model[y_min:y_max+1])**2))
        rms_r = np.sqrt(np.nanmean((xright_raw[y_min:y_max+1] - xR_model[y_min:y_max+1])**2))

        bad = np.isfinite(xL_model) & np.isfinite(xR_model) & (xR_model <= xL_model)
        if np.any(bad):
            mid = 0.5 * (xL_model[bad] + xR_model[bad])
            xL_model[bad] = mid - 0.5
            xR_model[bad] = mid + 0.5

        xMID_model = xcen_fit

        # preserve the existing XREF convention
        x_ref = float(np.nanmedian(xMID_model[y_min:y_max + 1])) if y_max >= y_min else float(cx)

        n_good_c = int(np.sum(np.isfinite(xcen_s[y_min:y_max+1])))
        n_good_lr = int(np.sum(np.isfinite(width_s[y_min:y_max+1])))




        hh = fits.Header()
        hh["EXTNAME"] = f"SLIT{sid:03d}"
        hh["SLITID"] = sid
        hh["XC0"] = (float(cx), "Center from 1D profile (pixels)")
        hh["YMIN"] = (int(y_min), "Valid y min for fit")
        hh["YMAX"] = (int(y_max), "Valid y max for fit")
        hh["XREF"] = (float(x_ref), "Reference x for rectification (pixels, mid-edge)")
        hh["PORDER"] = int(TRACE_CENTER_PORDER)
        hh["EPORDER"] = int(TRACE_EDGE_PORDER)
        hh["EPAD"] = (float(EDGE_PAD_PIX), "Edge pad (px) recommended for masking")
        hh["NROWC"] = (int(n_good_c), "Rows used in center fit")
        hh["NROWLR"] = (int(n_good_lr), "Rows used in edge fits")
        hh["RMSC"] = (float(rms_c) if np.isfinite(rms_c) else -1.0, "RMS residual center fit (px)")
        hh["RMSL"] = (float(rms_l) if np.isfinite(rms_l) else -1.0, "RMS residual left edge fit (px)")
        hh["RMSR"] = (float(rms_r) if np.isfinite(rms_r) else -1.0, "RMS residual right edge fit (px)")

        for i, c in enumerate(coeff_c):
            hh[f"PC{i}"] = (float(c), f"center x(y) coeff c{i} in pixels, power basis")
        for i, c in enumerate(coeff_l):
            hh[f"LC{i}"] = (float(c), f"left edge xL(y) coeff c{i} in pixels, power basis")
        for i, c in enumerate(coeff_r):
            hh[f"RC{i}"] = (float(c), f"right edge xR(y) coeff c{i} in pixels, power basis")

        # --- Attach RA/Dec from radec.csv mapping (lookup uses OLD slit id)
        radec = slit_radec.get(old_sid, None)
        if radec is not None:
            hh["RA"]  = (str(radec.get("ra", "")),  "Slit RA from radec.csv")
            hh["DEC"] = (str(radec.get("dec", "")), "Slit Dec from radec.csv")
            if radec.get("idx", None) is not None:
                hh["INDEX"] = (int(radec["idx"]), "Target index from input radec table")
        
        # Optional but VERY useful for debugging / provenance
        hh["OLDSID"] = (int(old_sid), "Original slit index before RA relabel")
        
        geom_hdus.append(fits.ImageHDU(data=xcen_fit.astype(np.float32), header=hh))

    fits.HDUList(geom_hdus).writeto(geom_path, overwrite=True)
    log.info("Wrote %s", geom_path)

    log.info("Wrote %s", table_path)

    # -------------------------------------------------------------------------
    # 7) Write DS9 region file (trace centers) in DS9 1-based pixel coordinates
    # -------------------------------------------------------------------------
    reg_path = OUTDIR / f"{TRACE_BASE}_traces.reg"
    reg_color = "green" if TRACE_SET.upper() == "EVEN" else "red"

    with open(reg_path, "w") as f:
        f.write("# Region file format: DS9 version 4.1\n")
        f.write(f'global color={reg_color} width=1 font="helvetica 10 normal roman"\n')
        f.write("image\n")
        f.write(f"# SAMOS Step04: {TRACE_BASE_CAP} slits, RA-ordered (x large -> x small), DS9 coords (1-based)\n")

        # slit_rows: (sid, old_sid, cx, wmin, wmean, wmax, w16, w50, w84, nrows, y0, y1)
        for row in slit_rows:
            if len(row) < 12:
                continue
            sid, old_sid, cx, wmin, wmean, wmax, w16, w50, w84, nrows, y0, y1 = row
            if y0 < 0 or y1 < 0:
                continue

            # Convert numpy 0-based -> DS9 1-based
            x_ds9  = float(cx) + 1.0
            y0_ds9 = float(y0) + 1.0
            y1_ds9 = float(y1) + 1.0

            f.write(f"line({x_ds9:.2f},{y0_ds9:.2f},{x_ds9:.2f},{y1_ds9:.2f})\n")

            # Label: separate text() region (more reliable than attaching text to line())
            y_lab = max(y0_ds9, min(y1_ds9, y1_ds9 - 20.0))
            f.write(f'text({x_ds9:.2f},{y_lab:.2f}) # text={{SLIT{int(sid):03d}}}\n')

    log.info("Wrote %s", reg_path)

    widths_med = [r[7] for r in slit_rows if len(r) > 7 and np.isfinite(r[7])]
    if widths_med:
        log.info(
            "Measured widths (px): min=%.1f  med=%.1f  max=%.1f",
            float(np.min(widths_med)),
            float(np.median(widths_med)),
            float(np.max(widths_med)),
        )


if __name__ == "__main__":
    main()
