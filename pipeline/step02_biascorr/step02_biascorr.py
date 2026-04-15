from __future__ import annotations

import logging
from pathlib import Path

import config
from astropy.io import fits
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(Path(__file__).stem)


def main():
    config.ensure_directories()

    in_dir = Path(config.ST00_ORIENT)
    out_dir = Path(config.ST02_BIASCORR)
    out_dir.mkdir(parents=True, exist_ok=True)

    masterbias = Path(config.ST01_BIAS) / "MasterBias.fits"
    if not masterbias.exists():
        raise FileNotFoundError(f"Missing {masterbias}. Run Step01 first.")

    logger.info("Input dir   = %s", in_dir)
    logger.info("Output dir  = %s", out_dir)
    logger.info("MasterBias  = %s", masterbias)

    # Load master bias once
    with fits.open(masterbias) as hdul:
        bias = np.array(hdul[0].data, dtype=np.float32)

    files = sorted(in_dir.glob("*.fits"))
    if not files:
        raise FileNotFoundError(f"No input files found in {in_dir}")

    wrote = 0

    for p in files:
        root = p.stem
        out = out_dir / f"{root}_biascorr.fits"

        if out.exists():
            logger.info("Skipping existing: %s", out.name)
            continue

        with fits.open(p) as hdul:
            img = np.array(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header.copy()

        if img.shape != bias.shape:
            raise ValueError(
                f"{p.name}: shape mismatch {img.shape} vs bias {bias.shape}"
            )

        img_corr = img - bias

        hdr["BIASCOR"] = (True, "Bias subtraction applied (Step02)")
        hdr["BIASFILE"] = (masterbias.name, "Master bias used")
        hdr.add_history("Step02: image - MasterBias")

        fits.PrimaryHDU(
            data=img_corr.astype(np.float32),
            header=hdr,
        ).writeto(out, overwrite=True)

        wrote += 1
        logger.info("Wrote: %s", out.name)

    logger.info("Done. Wrote %d files -> %s", wrote, out_dir)


if __name__ == "__main__":
    main()