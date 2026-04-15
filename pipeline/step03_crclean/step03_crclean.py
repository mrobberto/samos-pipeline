from __future__ import annotations

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


def main():
    config.ensure_directories()

    in_dir = Path(config.ST02_BIASCORR)
    out_dir = Path(config.ST03_CRCLEAN)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(in_dir.glob("*_biascorr.fits"))
    if not files:
        raise FileNotFoundError(f"No *_biascorr.fits found in {in_dir}. Run Step02 first.")

    logger.info("Input dir  = %s", in_dir)
    logger.info("Output dir = %s", out_dir)
    logger.info("Found %d files", len(files))

    samos = SAMOS()  # only used for CR_correct

    cr_params = dict(
        contrast=5,
        cr_threshold=15,
        neighbor_threshold=5,
        readnoise=float(config.READNOISE_E),
        effective_gain=float(config.GAIN_E_PER_ADU),
    )

    wrote = 0

    for p in files:
        with fits.open(p) as hdul:
            img = np.array(hdul[0].data, dtype=np.float32)
            hdr = hdul[0].header.copy()

        if bool(hdr.get("CRCLEAN", False)):
            logger.info("Skipping already CRCLEAN=True: %s", p.name)
            continue

        clean, mask = samos.CR_correct(img, return_mask=True, **cr_params)

        root = p.stem.replace("_biascorr", "")
        out = out_dir / f"{root}_biascorr_cr.fits"

        ohdr = hdr.copy()
        ohdr["CRCLEAN"] = (True, "Cosmic rays removed")
        ohdr["CRALG"] = ("LACOSMIC", "Cosmic ray algorithm")
        ohdr["CRTHR"] = (float(cr_params["cr_threshold"]), "LAcosmic cr_threshold")
        ohdr["CRRN_E"] = (float(cr_params["readnoise"]), "Readnoise (e-)")
        ohdr["CRGAIN"] = (float(cr_params["effective_gain"]), "Gain (e-/ADU)")
        ohdr["INSTEP"] = ("ST02", "Input stage")
        ohdr["OUTSTEP"] = ("ST03", "Output stage")

        hdul_out = fits.HDUList([
            fits.PrimaryHDU(data=clean.astype(np.float32), header=ohdr),
            fits.ImageHDU(data=mask.astype(np.uint8), name="CRMASK"),
        ])

        hdul_out.writeto(out, overwrite=True)
        wrote += 1

        logger.info("Wrote: %s", out.name)

    logger.info("Done. Wrote %d CR-cleaned files -> %s", wrote, out_dir)


if __name__ == "__main__":
    main()