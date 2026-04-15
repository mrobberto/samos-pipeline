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


def is_bias_file(path: Path) -> bool:
    # Prefer filename heuristic first
    low = path.name.lower()
    if "bias" in low:
        return True

    # Fallback to FITS header
    try:
        with fits.open(path) as hdul:
            hdr = hdul[0].header
        for k in ("IMAGETYP", "OBSTYPE", "TYPE", "OBJECT"):
            v = str(hdr.get(k, "")).strip().upper()
            if any(x in v for x in ("BIAS", "ZERO", "ZEROS", "BIAS FRAME")):
                return True
    except Exception:
        pass

    return False


def main():
    config.ensure_directories()

    in_dir = Path(config.ST00_ORIENT)
    out_dir = Path(config.ST01_BIAS)
    out_dir.mkdir(parents=True, exist_ok=True)

    out_file = out_dir / "MasterBias.fits"

    bias_files = [p for p in sorted(in_dir.glob("*.fits")) if is_bias_file(p)]
    if not bias_files:
        raise FileNotFoundError(f"No bias frames found in {in_dir}")

    logger.info("Found %d bias frames", len(bias_files))
    logger.info("Input dir  = %s", in_dir)
    logger.info("Output file = %s", out_file)

    stack = []
    hdr0 = None

    for p in bias_files:
        with fits.open(p) as hdul:
            img = np.array(hdul[0].data, dtype=np.float32)
            if hdr0 is None:
                hdr0 = hdul[0].header.copy()

        if img.ndim != 2:
            raise ValueError(f"{p.name}: expected 2D single-extension image, got shape {img.shape}")

        stack.append(img)

    cube = np.stack(stack, axis=0)

    # Median combine is appropriate for master bias
    master = np.nanmedian(cube, axis=0).astype(np.float32)

    hdr = hdr0 if hdr0 is not None else fits.Header()
    hdr["NCOMBINE"] = (len(bias_files), "Number of bias frames combined")
    hdr.add_history("Step01: MasterBias = median of Step00 oriented bias frames.")
    hdr.add_history(
        "Inputs: " + ",".join([p.name for p in bias_files[:10]]) +
        ("..." if len(bias_files) > 10 else "")
    )

    fits.PrimaryHDU(data=master, header=hdr).writeto(out_file, overwrite=True)
    logger.info("Wrote %s", out_file)


if __name__ == "__main__":
    main()