#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May 16 07:43:24 2026

@author: robberto
PYTHONPATH=. python tools/SISI_build_fits_overview.py \
  "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01/SISI/SISI_2026-01-13/fits" \
  --thumbs
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from pathlib import Path
import argparse
import re
import numpy as np
import pandas as pd
from astropy.io import fits


USER_NOTES = {
    # Use filename or file number.
    # "078.fits": "short note",
    "002": "bias test",
    "003": "bias test",
    "004": "start 20 bias frames, readout set at 800KHz",
    "024": "start 10 bias frames, Readount at 50KHz",
    "034": "10 exposures, no light from SAM, Readount at 50KHz",
    "044": "5 exposures, TO error sets NO light from SAM; Readout at 50KHz",
    "048": "partial illumination. Operator reports ice on the dome, going to interrupt",
    "049": "resuming after break, target check, 50KHz",
    "062": "last Target Acq.",
    "063": "063-072: 10 x10 s , r-band",
    "073": "073-077: 5x100 s , r-band, back to 800KHz",
    "078": "078-082: 5x100 s , i-band",
    "083": "Test 50KHz; 1 x 1 s, r-band",
    "084": "084-093: 10x10 s , r-band",
    "094": "094-103: 10x100 s , z-band",
    "103a": "Bias check",
    "103b": "103b-103d: T01 target acq. check, 50KHz",
    "104": "104-113: 10x10s, z-band",
    "114": "114-118: 5x100s, g-band",
    "119": "119-123: 5x100s, r-band, back to 800KHz",
    "124": "124-128: 5x100s, i-band",
    "129": "129-133: 5x100s, i-band",
    "134": "134-138: 5x100s, z-band",
    "139": "139-140: 2x30s, z-band; MACS_J1105.7; no ligh ",
    "141": "141-143: 3x30s r-band, no light",
    "144": "144-148: i-band, 5 exposures for Target Acq.",    
    "149": "149: 1x36s, slits on; check OK",
    "150": "150-159: 10x60s, z-band; MACS_J1105.7; bias ",
    "160-165": "CAL: Commanded Quartz, no ligh",

}


def get(h, keys, default="unavailable"):
    for k in keys:
        if k in h:
            v = h[k]
            if v not in ("", " ", "unavail", "unavailable", "UNKNOWN", "NONE"):
                return v
    return default


def filenumber(name):
    m = re.match(r"(\d+)", name)
    return m.group(1) if m else ""


def user_note_for(path):
    return USER_NOTES.get(path.name, USER_NOTES.get(filenumber(path.name), ""))


def ccdsize(h, data):
    if data is not None and hasattr(data, "shape"):
        return list(data.shape[::-1])
    n1 = h.get("NAXIS1", None)
    n2 = h.get("NAXIS2", None)
    return [n1, n2] if n1 and n2 else "unavailable"


def make_thumb(fits_path, jpg_path, stretch=(1, 99)):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with fits.open(fits_path, memmap=False, do_not_scale_image_data=False) as hdul:
        data = None
        for h in hdul:
            if h.data is not None and getattr(h.data, "ndim", 0) == 2:
                data = np.asarray(h.data, dtype=np.float32)
                break

    if data is None:
        return False

    ny, nx = data.shape
    by = max(1, ny // 700)
    bx = max(1, nx // 700)
    data = data[::by, ::bx]
    # SISI raw frames are mirrored horizontally
    data = np.fliplr(data)

    good = np.isfinite(data)
    if good.sum() == 0:
        return False

    vmin, vmax = np.percentile(data[good], stretch)

    fig = plt.figure(figsize=(3.2, 3.2))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(data, origin="lower", cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_axis_off()
    fig.savefig(jpg_path, dpi=80)
    plt.close(fig)
    return True


def main():
    ap = argparse.ArgumentParser(description="Build SISI/SAMI imaging FITS overview.")
    ap.add_argument("folder", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None)
    ap.add_argument("--recursive", action="store_true")
    ap.add_argument("--thumbs", action="store_true")
    ap.add_argument("--thumb-dir", type=Path, default=None)
    args = ap.parse_args()

    folder = args.folder.expanduser().resolve()
    out_html = args.out or folder / "overview_sisi.html"
    out_csv = args.csv or folder / "overview_sisi.csv"
    thumb_dir = args.thumb_dir or out_html.parent
    thumb_dir.mkdir(parents=True, exist_ok=True)

    pattern = "**/*.fits" if args.recursive else "*.fits"
    files = sorted(folder.glob(pattern))

    rows = []

    for p in files:
        try:
            with fits.open(p, memmap=False) as hdul:
                h = hdul[0].header
                data = hdul[0].data
        except Exception as e:
            rows.append(dict(Filename=p.name, Error=str(e)))
            continue

        jpg_name = p.with_suffix(".jpg").name
        img_html = ""

        if args.thumbs:
            jpg_path = thumb_dir / jpg_name
            try:
                if make_thumb(p, jpg_path):
                    rel = jpg_path.relative_to(out_html.parent)
                    img_html = f'<img src="{rel.as_posix()}" width="180" loading="eager">'
            except Exception as e:
                print(f"[WARN] thumbnail failed for {p.name}: {e}")
        else:
            img_html = f'<img src="{jpg_name}" width="180" loading="eager">'

        rows.append(dict(
            Filename=p.name,
            Filenumber=filenumber(p.name),
            Date=get(h, ["DATE-OBS", "DATE"]),
            Time=get(h, ["TIME-OBS", "UT", "UTC"]),
            Object=get(h, ["OBJECT", "OBJNAME", "TARGET", "OBSTYPE"]),
            Filter=get(h, ["FILTER", "FILTER1", "FILTER2", "FILTERS"]),
            RA=get(h, ["RA", "OBSRA", "TELRA", "OBJRA"]),
            DEC=get(h, ["DEC", "OBSDEC", "TELDEC", "OBJDEC"]),
            ExpTime=get(h, ["EXPTIME", "EXPOSURE", "ITIME"], default=np.nan),
            Airmass=get(h, ["AIRMASS", "SECZ"], default=""),
            Rotator=get(h, ["ROTATOR", "ROTPOSN", "PA"], default=""),
            CCDSize=ccdsize(h, data),
            Note=user_note_for(p),
            image=img_html,
        ))

    df = pd.DataFrame(rows)
    if "Filenumber" in df.columns:
        df = df.sort_values("Filenumber")

    df.to_csv(out_csv, index=False)
    out_html.write_text(df.to_html(escape=False, border=1), encoding="utf-8")

    print("Wrote:", out_html)
    print("Wrote:", out_csv)
    print("Rows :", len(df))


if __name__ == "__main__":
    main()