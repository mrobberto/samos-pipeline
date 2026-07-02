#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 04:31:23 2026

@author: robberto

to run:
from /samos-pipeline

PYTHONPATH=. python tools/build_fits_overview.py   "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SAMI/20260113/preprocessed/03p5_rowstripe"   --thumbs
  
"""
#!/usr/bin/env python3
from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
from astropy.io import fits
import config

import sys

REPO_ROOT = Path(
    "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/samos-pipeline"
)

if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


USER_NOTES = {
    # Use either full filename or file number.
    # Filename wins over file number.
    "001": "001-002: Bias test",
    "036": "Arc slits on.",
    "037": "Arc slits off.",
}


    
def get(h, keys, default="unavailable"):
    for k in keys:
        if k in h and h[k] not in ("", "unavail", "unavailable"):
            return h[k]
    return default

def filenumber(name):
    m = re.match(r"(\d+)", name)
    return m.group(1) if m else ""

def ccdsize(h, data):
    if data is not None and hasattr(data, "shape"):
        return list(data.shape[::-1])
    n1 = h.get("NAXIS1", None)
    n2 = h.get("NAXIS2", None)
    return [n1, n2] if n1 and n2 else "unavailable"

def make_thumb(fits_path, jpg_path, stretch=(1, 99), binning=8):

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with fits.open(fits_path, memmap=True) as hdul:

        data = None

        for h in hdul:
            if h.data is not None and getattr(h.data, "ndim", 0) == 2:
                data = np.asarray(h.data, dtype=np.float32)
                break

    if data is None:
        return False

    # -------------------------------------------------
    # aggressive downsampling for speed
    # -------------------------------------------------
    ny, nx = data.shape

    by = max(1, ny // 512)
    bx = max(1, nx // 512)

    data = data[::by, ::bx]

    # robust scaling
    finite = np.isfinite(data)

    if finite.sum() == 0:
        return False

    vals = data[finite]

    vmin, vmax = np.percentile(vals, stretch)

    # -------------------------------------------------
    # save lightweight JPEG
    # -------------------------------------------------
    fig = plt.figure(figsize=(3, 3))

    ax = fig.add_axes([0, 0, 1, 1])

    ax.imshow(
        data,
        origin="lower",
        cmap="gray",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )

    ax.set_axis_off()

    fig.savefig(
        jpg_path,
        dpi=70,
        #quality=60,
    )

    plt.close(fig)

    return True

def user_note_for(path):
    name = path.name
    num = filenumber(name)

    if name in USER_NOTES:
        return USER_NOTES[name]
    if num in USER_NOTES:
        return USER_NOTES[num]

    return ""


def main():
    ap = argparse.ArgumentParser(description="Build HTML overview table for FITS files.")
    ap.add_argument("folder", type=Path, help="Folder containing FITS files")
    ap.add_argument("--out", type=Path, default=None, help="Output HTML file")
    ap.add_argument("--csv", type=Path, default=None, help="Optional output CSV file")
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--thumbs", action="store_true", help="Create JPG thumbnails")
    ap.add_argument("--thumb-dir", type=Path, default=None)
    args = ap.parse_args()

    folder = args.folder.expanduser().resolve()
    pattern = "**/*.fits" if args.recursive else "*.fits"
    files = sorted(folder.glob(pattern))

    out_html = args.out or folder / "overview.html"
    out_csv = args.csv or folder / "overview.csv"
    thumb_dir = args.thumb_dir or out_html.parent
    thumb_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for p in files:
        try:
            with fits.open(p, memmap=False) as hdul:
                h = hdul[0].header
                data = hdul[0].data
        except Exception as e:
            rows.append(dict(
                Filename=p.name, Filenumber=filenumber(p.name),
                Date="unreadable", Time="unreadable", Object=str(e),
                RA="", DEC="", Note="", ExpTime="", CCDSize="", image=""
            ))
            continue

        jpg_name = p.with_suffix(".jpg").name
        img_html = ""

        if args.thumbs:
            jpg_path = thumb_dir / jpg_name
            try:
                ok = make_thumb(p, jpg_path)
                if ok:
#                    img_html = f'<img src="{rel_jpg.as_posix()}" 
                    img_html = f'<img src="{jpg_path.name}" width="180" loading="eager">'
            except Exception:
                img_html = ""
        else:
            img_html = f'<img src="{jpg_name}">'

        rows.append(dict(
            Filename=p.name,
            Filenumber=filenumber(p.name),
            Date=get(h, ["DATE-OBS", "DATE"]),
            Time=get(h, ["TIME-OBS", "UT", "UTC"]),
            Object=get(h, ["OBJECT", "OBSTYPE"]),
            RA=get(h, ["RA", "OBSRA", "TELRA"]),
            DEC=get(h, ["DEC", "OBSDEC", "TELDEC"]),
#            Note=get(h, ["NOTES", "COMMENT"], default=""),
            Note=user_note_for(p),
            ExpTime=float(get(h, ["EXPTIME", "EXPOSURE"], default=np.nan)),
            CCDSize=ccdsize(h, data),
            image=img_html,
        ))

    df = pd.DataFrame(rows)

    if "Filenumber" in df:
        df = df.sort_values("Filenumber")

    df.to_csv(out_csv, index=False)

    html = df.to_html(escape=False, border=1)
    out_html.write_text(html, encoding="utf-8")

    print("Wrote:", out_html)
    print("Wrote:", out_csv)
    print("Rows :", len(df))

if __name__ == "__main__":
    main()
