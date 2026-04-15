#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step00 — Standardize image orientation

Read raw SAMOS multi-extension FITS files using the SAMOS mosaic reader,
assemble each file into a single 2D full-frame image, then rotate the
assembled image by 180 degrees to enforce the SAMOS pipeline convention:

- wavelength increases bottom → top
- RA increases right → left

Output products are single-extension FITS files for downstream pipeline use.
"""

import logging
from pathlib import Path

import config
import numpy as np
from astropy.io import fits

from samos.class_samos import SAMOS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(Path(__file__).stem)


def rot180(data: np.ndarray) -> np.ndarray:
    """Rotate a 2D image by 180 degrees."""
    return np.rot90(data, 2)


def main():
    config.ensure_directories()

    in_dir = Path(config.RAW_DIR)
    out_dir = Path(config.ST00_ORIENT)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("RAW_DIR      = %s", in_dir)
    logger.info("ST00_ORIENT  = %s", out_dir)
    logger.info("config file  = %s", Path(config.__file__).resolve())

    files = sorted(in_dir.glob("*.fits"))
    if not files:
        raise FileNotFoundError(f"No FITS files found in {in_dir}")

    wrote = 0
    samos_reader = SAMOS(str(in_dir))

    for p in files:
        root = p.stem
        out = out_dir / f"{root}_oriented.fits"

        if out.exists():
            logger.info("Skipping existing: %s", out.name)
            continue

        logger.info("Reading raw MEF mosaic: %s", p.name)

        # Assemble the raw SAMOS MEF into a single full-frame image.
        mosaic_hdu = samos_reader.read_SAMI_mosaic(str(p))
        data = np.asarray(mosaic_hdu.data)
        hdr = mosaic_hdu.header.copy()

        if data.ndim != 2:
            raise ValueError(f"{p.name}: assembled SAMOS mosaic is not 2D (shape={data.shape})")

        # Rotate the assembled full frame, not the individual extensions.
        data_rot = rot180(data)

        hdr["ROT180"] = (True, "Image rotated by 180 deg (Step00)")
        hdr["ORIGFILE"] = (p.name, "Original raw MEF filename")
        hdr["PIPESTEP"] = ("STEP00", "Pipeline step")
        hdr.add_history("Step00: raw MEF assembled to full frame using SAMOS.read_SAMI_mosaic().")
        hdr.add_history("Step00: full assembled image rotated by 180 deg to SAMOS standard orientation.")

        fits.PrimaryHDU(
            data=data_rot.astype(np.float32),
            header=hdr,
        ).writeto(out, overwrite=True)

        wrote += 1
        logger.info("Wrote: %s", out.name)

    logger.info("Done. Wrote %d files -> %s", wrote, out_dir)


if __name__ == "__main__":
    main()