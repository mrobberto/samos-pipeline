#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import logging
from pathlib import Path
import numpy as np
from astropy.io import fits
import lacosmic
from astropy.stats import sigma_clip

logger = logging.getLogger(__name__)


class SAMOS:
    """
    SAMOS reduction utilities.

    Conventions:
      - raw_dir points to .../fits  (raw inputs)
      - reduced/ is created next to fits/
      - stages are subfolders under reduced/
    """

    def __init__(self, raw_dir: str | None = None):
        self.raw_dir = Path(raw_dir).resolve() if raw_dir is not None else None
        if self.raw_dir is not None:
            logger.info("Raw dir: %s", self.raw_dir)
            
    # -------------------------------------------------------------------------
    # Stage folder helper
    # -------------------------------------------------------------------------
    def stage(self, name: str) -> Path:
        p = self.reduced_dir / name
        p.mkdir(parents=True, exist_ok=True)
        return p

    # -------------------------------------------------------------------------
    # Read and rearrange the 4 SAMI CCDs into a mosaic (expects SAMI MEF)
    # -------------------------------------------------------------------------
    def read_SAMI_mosaic(self, file: str) -> fits.PrimaryHDU:
        hdul = fits.open(file)
        hdr = hdul[0].header

        # will throw IndexError if not MEF; callers should guard
        data1 = hdul[1].data[:, 54:2101]   # top left
        data2 = hdul[2].data[:, 64:2111]   # top right
        data3 = hdul[3].data[:, 54:2101]   # bottom left
        data4 = hdul[4].data[:, 64:2111]   # bottom right

        mosaic_data = np.zeros((data1.shape[0] * 2, data1.shape[1] + data2.shape[1]), dtype=np.float32)
        mosaic_data[data1.shape[0]:, :data1.shape[1]] = np.flip(data1, axis=0)
        mosaic_data[data1.shape[0]:, data1.shape[1]:] = np.flip(data2, axis=0)
        mosaic_data[:data3.shape[0], :data3.shape[1]] = np.flip(data3, axis=0)
        mosaic_data[:data3.shape[0], data3.shape[1]:] = np.flip(data4, axis=0)

        mosaic = np.flip(mosaic_data, axis=0)

        # Fix column 2046
        mosaic[:, 2046] = (mosaic[:, 2045] + mosaic[:, 2047]) / 2.0

        return fits.PrimaryHDU(data=mosaic.astype(np.float32), header=hdr)

    # -------------------------------------------------------------------------
    # Robust frame loader (mosaic if MEF, else primary)
    # -------------------------------------------------------------------------
    def _load_frame(self, file: str, use_mosaic: bool = True) -> np.ndarray:
        file = str(file)

        with fits.open(file) as hdul:
            nhdu = len(hdul)

        if use_mosaic and nhdu >= 5:
            hdu = self.read_SAMI_mosaic(file)
            return np.array(hdu.data, dtype=np.float32)

        with fits.open(file) as hdul:
            if hdul[0].data is None:
                raise ValueError(f"{file}: primary HDU has no image data.")
            return np.array(hdul[0].data, dtype=np.float32)

    # -------------------------------------------------------------------------
    # Cosmic ray correction (LAcosmic)
    # -------------------------------------------------------------------------
    def CR_correct(
        self,
        image,
        contrast=5,
        cr_threshold=20,
        neighbor_threshold=5,
        readnoise=3.0,
        effective_gain=2.0,
        return_mask=False,
    ):
        cleaned, mask = lacosmic.lacosmic(
            image,
            contrast=contrast,
            cr_threshold=cr_threshold,
            neighbor_threshold=neighbor_threshold,
            readnoise=readnoise,
            effective_gain=effective_gain,
        )
        if return_mask:
            return cleaned, mask
        return cleaned

    # -------------------------------------------------------------------------
    # Coadd frames (unweighted)
    # -------------------------------------------------------------------------
    def coadd_frames(
        self,
        files,
        method: str = "median",
        sigma_clip_on: bool = False,
        sigma: float = 3.0,
        maxiters: int = 5,
        use_mosaic: bool = True,
        output_header_from: str = "first",
    ) -> fits.PrimaryHDU:

        if not files:
            raise ValueError("coadd_frames: files is empty.")

        data_list = []
        ref_shape = None

        for f in files:
            img = self._load_frame(f, use_mosaic=use_mosaic)

            if ref_shape is None:
                ref_shape = img.shape
            elif img.shape != ref_shape:
                raise ValueError(f"Shape mismatch: {f} {img.shape} != {ref_shape}")

            data_list.append(img)

        cube = np.stack(data_list, axis=0)

        if sigma_clip_on:
            clipped = sigma_clip(cube, sigma=sigma, maxiters=maxiters, axis=0)
            cube_use = clipped
        else:
            cube_use = cube

        m = method.lower()
        if m == "median":
            master = np.nanmedian(cube_use, axis=0)
        elif m == "mean":
            master = np.nanmean(cube_use, axis=0)
        else:
            raise ValueError("method must be 'median' or 'mean'.")

        # header
        if output_header_from == "first":
            with fits.open(files[0]) as hdul:
                hdr = hdul[0].header.copy()
        else:
            hdr = fits.Header()

        hdr["NCOMBINE"] = (len(files), "Number of frames combined")
        hdr["COMBMETH"] = (m, "Combine method")
        hdr["SCLIP"] = (bool(sigma_clip_on), "Sigma clipping applied")
        if sigma_clip_on:
            hdr["SCLIPSIG"] = (float(sigma), "Sigma clip threshold")
            hdr["SCLIPIT"] = (int(maxiters), "Sigma clip maxiters")

        return fits.PrimaryHDU(data=np.array(master, dtype=np.float32), header=hdr)

    # -------------------------------------------------------------------------
    # Coadd frames (weighted mean; ADU/s option; correct EXPTIME bookkeeping)
    # -------------------------------------------------------------------------
    def coadd_frames_weighted(
        self,
        files,
        weight_mode: str = "exptime",     # "exptime" or "equal"
        exptime_key: str = "EXPTIME",
        normalize_to_rate: bool = True,  # convert to ADU/s before combining
        sigma_clip_on: bool = False,
        sigma: float = 3.0,
        maxiters: int = 5,
        use_mosaic: bool = True,
        output_header_from: str = "first",
        add_keywords: bool = True,
    ) -> fits.PrimaryHDU:

        if not files:
            raise ValueError("coadd_frames_weighted: files is empty.")

        data_list = []
        weights = []
        exptimes = []
        ref_shape = None

        for f in files:
            img = self._load_frame(f, use_mosaic=use_mosaic).astype(np.float32)

            with fits.open(f) as hdul:
                t = hdul[0].header.get(exptime_key, None)

            if weight_mode.lower() == "exptime":
                if t is None:
                    raise KeyError(f"{f}: missing {exptime_key} (needed for weight_mode='exptime').")
                t = float(t)
                if t <= 0:
                    raise ValueError(f"{f}: {exptime_key} must be >0.")
            else:
                t = float(t) if t is not None else np.nan

            if ref_shape is None:
                ref_shape = img.shape
            elif img.shape != ref_shape:
                raise ValueError(f"Shape mismatch: {f} {img.shape} != {ref_shape}")

            if normalize_to_rate:
                if not np.isfinite(t) or t <= 0:
                    raise ValueError(f"{f}: cannot normalize_to_rate=True without valid {exptime_key}.")
                img = img / t  # ADU/s

            if weight_mode.lower() == "exptime":
                w = float(t)
            elif weight_mode.lower() == "equal":
                w = 1.0
            else:
                raise ValueError("weight_mode must be 'exptime' or 'equal'.")

            data_list.append(img)
            weights.append(w)
            exptimes.append(t)

        exptimes = np.array(exptimes, dtype=float)
        weights = np.array(weights, dtype=float)

        finite_t = exptimes[np.isfinite(exptimes) & (exptimes > 0)]
        t_sum = float(np.sum(finite_t)) if finite_t.size else np.nan
        t_min = float(np.min(finite_t)) if finite_t.size else np.nan
        t_max = float(np.max(finite_t)) if finite_t.size else np.nan
        t_mean = float(np.mean(finite_t)) if finite_t.size else np.nan

        cube = np.stack(data_list, axis=0)            # (N,Y,X)
        w3 = weights[:, None, None].astype(np.float32)

        if sigma_clip_on:
            clipped = sigma_clip(cube, sigma=sigma, maxiters=maxiters, axis=0)
            cube_use = np.array(clipped.filled(np.nan), dtype=np.float32)
            mask_use = np.array(clipped.mask, dtype=bool)
            w_eff = np.where(mask_use, 0.0, w3).astype(np.float32)
        else:
            cube_use = cube
            w_eff = w3

        num = np.nansum(w_eff * cube_use, axis=0)
        den = np.nansum(w_eff, axis=0)
        master = np.divide(num, den, out=np.full_like(num, np.nan, dtype=np.float32), where=den > 0)

        # header
        if output_header_from == "first":
            with fits.open(files[0]) as hdul:
                hdr0 = hdul[0].header.copy()
        else:
            hdr0 = fits.Header()

        if add_keywords:
            hdr0["NCOMBINE"] = (len(files), "Number of frames combined")
            hdr0["COMBMETH"] = ("weighted_mean", "Combine method")
            hdr0["WTMODE"] = (weight_mode.lower(), "Weighting mode")
            hdr0["NRM2RATE"] = (bool(normalize_to_rate), "Normalized to ADU/s before combining")
            hdr0["SCLIP"] = (bool(sigma_clip_on), "Sigma clipping applied")
            if sigma_clip_on:
                hdr0["SCLIPSIG"] = (float(sigma), "Sigma clip threshold")
                hdr0["SCLIPIT"] = (int(maxiters), "Sigma clip maxiters")

            if finite_t.size:
                hdr0["TUNIT"] = ("s", "Exposure time unit")
                hdr0["TMIN"] = (t_min, "Min input EXPTIME (s)")
                hdr0["TMAX"] = (t_max, "Max input EXPTIME (s)")
                hdr0["TMEAN"] = (t_mean, "Mean input EXPTIME (s)")
                hdr0["TEXPTIME"] = (t_sum, "Total integration time (s)")

                # key choice for later Poisson+RN on ADU/s coadd:
                if normalize_to_rate and weight_mode.lower() == "exptime":
                    hdr0["EXPTIME"] = (t_sum, "Effective exposure time for noise in ADU/s (sum inputs)")
                    hdr0["BUNIT"] = ("ADU/s", "Rate image")
                    hdr0.add_history(f"ADU/s weighted coadd; set EXPTIME=sum(EXPTIME_i)={t_sum:.3f}s for variance.")
                else:
                    hdr0["EXPTIME"] = (t_sum, "Total integration time (s)")
                    if normalize_to_rate:
                        hdr0["BUNIT"] = ("ADU/s", "Rate image")

        return fits.PrimaryHDU(data=np.array(master, dtype=np.float32), header=hdr0)

    # -------------------------------------------------------------------------
    # Stage 1: subtract superbias (NO cosmic ray cleaning here)
    # -------------------------------------------------------------------------
    def subtract_superbias_from_directory(
        self,
        directory: str,
        superbias,
        file_type: str = ".fits",
        use_mosaic: bool = True,
        output_suffix: str = "_biascorr",
        out_dir: str | None = None,
        overwrite: bool = False,
        # robust skipping / detection
        skip_if_header_biascor: bool = True,
        skip_if_name_contains_suffix: bool = True,
        bias_name_keywords: tuple[str, ...] = ("bias", "masterbias", "superbias"),
        header_bias_keys: tuple[str, ...] = ("IMAGETYP", "OBSTYPE", "TYPE", "OBJECT"),
        header_bias_values: tuple[str, ...] = ("BIAS", "ZERO", "ZEROS", "BIAS FRAME"),
    ):
        """
        Bias-subtract raw data into out_dir, without CR cleaning.
        """

        import os

        # load superbias image
        if isinstance(superbias, str):
            with fits.open(superbias) as hdul:
                sb = np.array(hdul[0].data, dtype=np.float32)
            sb_name = os.path.basename(superbias)
        elif isinstance(superbias, fits.PrimaryHDU):
            sb = np.array(superbias.data, dtype=np.float32)
            sb_name = "HDU"
        else:
            sb = np.array(superbias, dtype=np.float32)
            sb_name = "ARRAY"

        if sb.ndim != 2:
            raise ValueError("Superbias must be 2D.")

        directory = str(directory)
        outdir = str(out_dir) if out_dir is not None else directory
        os.makedirs(outdir, exist_ok=True)
        written = []

        def _is_bias_from_header(hdr) -> bool:
            for k in header_bias_keys:
                if k in hdr:
                    v = str(hdr.get(k, "")).strip().upper()
                    if any(bv in v for bv in header_bias_values):
                        return True
            return False

        def _is_bias_from_name(fname: str) -> bool:
            low = fname.lower()
            return any(kw in low for kw in bias_name_keywords)

        for fname in sorted(os.listdir(directory)):
            low = fname.lower()
            if not low.endswith(file_type.lower()):
                continue

            if _is_bias_from_name(fname):
                continue

            if skip_if_name_contains_suffix and output_suffix.lower() in low:
                continue

            inpath = os.path.join(directory, fname)

            # robust load: mosaic only if MEF
            with fits.open(inpath) as hdul:
                nhdu = len(hdul)
            if use_mosaic and nhdu >= 5:
                hdu_in = self.read_SAMI_mosaic(inpath)
                img = np.array(hdu_in.data, dtype=np.float32)
                hdr = hdu_in.header.copy()
            else:
                with fits.open(inpath) as hdul:
                    if hdul[0].data is None:
                        continue
                    img = np.array(hdul[0].data, dtype=np.float32)
                    hdr = hdul[0].header.copy()

            if _is_bias_from_header(hdr):
                continue

            if skip_if_header_biascor and bool(hdr.get("BIASCOR", False)):
                continue

            if img.shape != sb.shape:
                raise ValueError(f"Shape mismatch {fname}: image={img.shape} superbias={sb.shape}")

            corr = img - sb

            root, ext = os.path.splitext(fname)
            outname = f"{root}{output_suffix}{ext}"
            outpath = os.path.join(outdir, outname)

            if os.path.exists(outpath) and not overwrite:
                continue

            ohdr = hdr.copy()
            ohdr["BIASCOR"] = (True, "Superbias subtracted")
            ohdr["BIASREF"] = (sb_name, "Superbias reference")
            ohdr["HISTORY"] = "Bias subtraction stage"

            fits.PrimaryHDU(corr.astype(np.float32), header=ohdr).writeto(outpath, overwrite=overwrite)
            written.append(outpath)

        logger.info("Bias subtraction complete: wrote %d files -> %s", len(written), outdir)
        return written

    # -------------------------------------------------------------------------
    # Stage 2: cosmic ray cleaning on bias-corrected products (NO bias subtraction)
    # -------------------------------------------------------------------------
    def clean_cosmics_in_directory(
        self,
        directory: str,
        file_type: str = ".fits",
        pattern_contains: str = "_biascorr",
        exclude_keywords: tuple[str, ...] = ("master", "super", "bias"),
        out_dir: str | None = None,
        cr_suffix: str = "_cr",
        overwrite: bool = False,
        write_crmask: bool = True,
        crmask_extname: str = "CRMASK",
        # LAcosmic params
        cr_params: dict | None = None,
        # for reading images:
        use_mosaic: bool = False,   # biascorr outputs are usually single-HDU mosaics
    ):
        """
        Apply LAcosmic to bias-corrected products in 'directory', writing to out_dir.
        """
        import os

        directory = str(directory)
        outdir = str(out_dir) if out_dir is not None else directory
        os.makedirs(outdir, exist_ok=True)

        default_cr_params = dict(
            contrast=5,
            cr_threshold=20,
            neighbor_threshold=5,
            readnoise=3.0,
            effective_gain=2.0,
        )
        if cr_params:
            default_cr_params.update(cr_params)

        written = []

        for fname in sorted(os.listdir(directory)):
            low = fname.lower()
            if not low.endswith(file_type.lower()):
                continue
            if pattern_contains.lower() not in low:
                continue
            if any(k in low for k in exclude_keywords):
                continue
            if cr_suffix.lower() in low:
                continue  # already CR cleaned

            inpath = os.path.join(directory, fname)

            # load image (single-HDU normally here)
            with fits.open(inpath) as hdul:
                hdr = hdul[0].header.copy()
                img = np.array(hdul[0].data, dtype=np.float32)

            if img is None:
                continue

            cleaned, crmask = self.CR_correct(img, return_mask=True, **default_cr_params)

            root, ext = os.path.splitext(fname)
            outname = f"{root}{cr_suffix}{ext}"
            outpath = os.path.join(outdir, outname)
            if os.path.exists(outpath) and not overwrite:
                continue

            ohdr = hdr.copy()
            ohdr["CRCLEAN"] = (True, "Cosmic rays removed")
            ohdr["CRALG"] = ("LACOSMIC", "Cosmic ray algorithm")
            ohdr["CRTHR"] = (float(default_cr_params["cr_threshold"]), "LAcosmic cr_threshold")
            ohdr["CRRN_E"] = (float(default_cr_params["readnoise"]), "Readnoise (e-) used by LAcosmic")
            ohdr["CRGAIN"] = (float(default_cr_params["effective_gain"]), "Gain (e-/ADU) used by LAcosmic")
            ohdr["HISTORY"] = "Cosmic ray cleaning stage"

            hdul_out = fits.HDUList([fits.PrimaryHDU(cleaned.astype(np.float32), header=ohdr)])
            if write_crmask and crmask is not None:
                mh = fits.ImageHDU(data=crmask.astype(np.uint8), name=crmask_extname)
                mh.header["COMMENT"] = "1 = pixel flagged as cosmic ray by LAcosmic"
                hdul_out.append(mh)

            hdul_out.writeto(outpath, overwrite=overwrite)
            written.append(outpath)

        logger.info("CR cleaning complete: wrote %d files -> %s", len(written), outdir)
        return written
