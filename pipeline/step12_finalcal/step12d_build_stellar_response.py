#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Step12d — build the adopted ensemble spectrophotometric response
================================================================

This stage converts the validated Step11 ensemble shape correction into the
production Step12 response.

The response shape is supplied by the validated quadratic ensemble solution
(normalized at the SkyMapper i-band pivot).  Step12d then derives ONE global
normalization factor from the Step11 spectra themselves so that, on average,
the correction preserves the synthetic SkyMapper i-band flux.

Thus the production response is

    R_prod(lambda) = C_i * R_shape(lambda)

where C_i is the median, over usable spectra, of

    <f_nu>_i,Step11 / <f_nu>_i,Step11*R_shape .

No per-object catalog normalization is performed here or in Step12e.

Outputs
-------
step12d_stellar_response_master.fits
step12d_stellar_response_summary.csv
step12d_stellar_response_metadata.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits

import config

log = logging.getLogger("step12d_build_stellar_response")

C_A_S = 2.99792458e18  # Angstrom/s

PIVOT_R_NM = 613.8440330950057
PIVOT_I_NM = 776.79762950059
PIVOT_Z_NM = 914.5992987637427


def _default_response_csv() -> Path:
    name = "qc_step11_ensemble_response.csv"
    candidates = [
        Path(getattr(config, "ST11_FLUXCAL", ".")) / name,
        Path(getattr(config, "ST11_FLUXCAL", ".")) / "qc_step11" / name,
        Path(getattr(config, "ST10_TELLURIC", ".")) / name,
        Path(getattr(config, "ST12_FINALCAL", ".")) / name,
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find qc_step11_ensemble_response.csv. Tried:\n  "
        + "\n  ".join(str(p) for p in candidates)
    )


def _companion_loo(path: Path) -> Path:
    return path.with_name(path.stem + "_loo.csv")


def _validated_table(df: pd.DataFrame, trust_min: float, trust_max: float) -> pd.DataFrame:
    required = ["lambda_nm", "response_quadratic", "loo_p16", "loo_p84"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Response CSV missing required columns: {missing}")

    q = df[required].copy()
    for c in required:
        q[c] = pd.to_numeric(q[c], errors="coerce")

    q = q.replace([np.inf, -np.inf], np.nan).dropna()
    q = q.sort_values("lambda_nm").drop_duplicates("lambda_nm")

    if len(q) < 3:
        raise RuntimeError("Too few finite response samples.")

    lam = q["lambda_nm"].to_numpy(float)
    if trust_min < lam.min() or trust_max > lam.max():
        raise ValueError(
            f"Requested trusted interval [{trust_min:.3f}, {trust_max:.3f}] nm "
            f"is outside response grid [{lam.min():.3f}, {lam.max():.3f}] nm."
        )
    if trust_max <= trust_min:
        raise ValueError("--trust-max must be greater than --trust-min")

    rows = []
    for w in (trust_min, trust_max):
        rows.append(
            {
                "lambda_nm": w,
                "response_quadratic": float(np.interp(w, lam, q["response_quadratic"])),
                "loo_p16": float(np.interp(w, lam, q["loo_p16"])),
                "loo_p84": float(np.interp(w, lam, q["loo_p84"])),
            }
        )

    inside = q[(q["lambda_nm"] >= trust_min) & (q["lambda_nm"] <= trust_max)]
    out = pd.concat([inside, pd.DataFrame(rows)], ignore_index=True)
    out = out.sort_values("lambda_nm").drop_duplicates("lambda_nm").reset_index(drop=True)

    for c in ("response_quadratic", "loo_p16", "loo_p84"):
        if not np.all(np.isfinite(out[c])):
            raise RuntimeError(f"Non-finite values remain in {c}")
        if np.any(out[c].to_numpy(float) <= 0):
            raise RuntimeError(f"Non-positive values found in {c}")

    return out


def _loo_stats(loo_path: Path) -> dict:
    stats = {
        "n_stars": np.nan,
        "loo_gray_median_mag": np.nan,
        "loo_linear_median_mag": np.nan,
        "loo_quadratic_median_mag": np.nan,
        "quadratic_better_than_gray_n": np.nan,
    }
    if not loo_path.exists():
        return stats

    d = pd.read_csv(loo_path)
    stats["n_stars"] = int(len(d))

    pairs = [
        ("loo_rms_gray_mag", "loo_gray_median_mag"),
        ("loo_rms_linear_mag", "loo_linear_median_mag"),
        ("loo_rms_quadratic_mag", "loo_quadratic_median_mag"),
    ]
    for col, key in pairs:
        if col in d:
            x = pd.to_numeric(d[col], errors="coerce").to_numpy(float)
            x = x[np.isfinite(x)]
            if x.size:
                stats[key] = float(np.median(x))

    if {"loo_rms_quadratic_mag", "loo_rms_gray_mag"} <= set(d.columns):
        a = pd.to_numeric(d["loo_rms_quadratic_mag"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(d["loo_rms_gray_mag"], errors="coerce").to_numpy(float)
        good = np.isfinite(a) & np.isfinite(b)
        stats["quadratic_better_than_gray_n"] = int(np.sum(a[good] < b[good]))

    return stats


def load_filter_curve(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            s = line.strip()
            if not s or s.startswith("#"):
                continue
            vals = []
            for p in s.replace(",", " ").split():
                try:
                    vals.append(float(p))
                except ValueError:
                    continue
            if len(vals) >= 2:
                rows.append((vals[0], vals[1]))

    if len(rows) < 3:
        raise RuntimeError(f"Could not read two-column filter curve: {path}")

    a = np.asarray(rows, float)
    w = a[:, 0]
    t = a[:, 1]

    med = float(np.nanmedian(w))
    if med > 3000:      # Angstrom -> nm
        w = w / 10.0
    elif med < 10:      # micron -> nm
        w = w * 1000.0

    good = np.isfinite(w) & np.isfinite(t) & (t >= 0)
    w, t = w[good], t[good]
    order = np.argsort(w)
    return w[order], t[order]


def synthetic_fnu_and_coverage(lam_nm, flam_A, filt_nm, filt_t):
    lam = np.asarray(lam_nm, float)
    flam = np.asarray(flam_A, float)

    m = np.isfinite(lam) & np.isfinite(flam)
    if m.sum() < 2:
        return np.nan, 0.0

    lam = lam[m]
    flam = flam[m]
    order = np.argsort(lam)
    lam, flam = lam[order], flam[order]
    lam, idx = np.unique(lam, return_index=True)
    flam = flam[idx]

    fw = np.asarray(filt_nm, float)
    ft = np.asarray(filt_t, float)
    positive = np.isfinite(fw) & np.isfinite(ft) & (ft > 0)
    fw, ft = fw[positive], ft[positive]

    if fw.size < 3 or lam.size < 2:
        return np.nan, 0.0

    lamA_f = fw * 10.0
    denom_full = np.trapezoid(ft / lamA_f, lamA_f)
    if not np.isfinite(denom_full) or denom_full <= 0:
        return np.nan, 0.0

    overlap = (fw >= lam.min()) & (fw <= lam.max())
    if overlap.sum() < 3:
        return np.nan, 0.0

    fw_o = fw[overlap]
    ft_o = ft[overlap]
    lamA_o = fw_o * 10.0
    denom_cov = np.trapezoid(ft_o / lamA_o, lamA_o)
    coverage = float(denom_cov / denom_full)

    f_interp = np.interp(fw_o, lam, flam)
    numerator = np.trapezoid(f_interp * ft_o * lamA_o, lamA_o)

    if not np.isfinite(numerator) or denom_cov <= 0:
        return np.nan, coverage

    return float(numerator / (C_A_S * denom_cov)), coverage


def _response_on_grid(lam_nm, master_lam, master_resp):
    lam_nm = np.asarray(lam_nm, float)
    out = np.full(lam_nm.shape, np.nan, dtype=float)
    good = np.isfinite(lam_nm)
    if good.any():
        out[good] = np.interp(
            lam_nm[good],
            master_lam,
            master_resp,
            left=float(master_resp[0]),
            right=float(master_resp[-1]),
        )
    return out


def derive_global_i_normalization(
    spectra_path: Path,
    master_lam: np.ndarray,
    shape_resp: np.ndarray,
    filter_i_path: Path,
    min_coverage: float = 0.90,
):
    """
    Derive one global scalar that preserves Step11 synthetic i-band flux,
    in the median over usable slit spectra.
    """
    fw, ft = load_filter_curve(filter_i_path)
    rows = []

    with fits.open(spectra_path, memmap=False) as h:
        for ext in h[1:]:
            if not (ext.name or "").upper().startswith("SLIT"):
                continue
            if ext.data is None:
                continue

            # Independent source-validity gate. Step12d remains protected
            # even if an upstream file contains rejected slit HDUs.
            s08use = int(ext.header.get("S08USE", 0))
            s08good = int(ext.header.get("S08GOOD", s08use))
            s08clas = str(ext.header.get("S08CLAS", "")).strip().upper()
            if (
                s08use != 1
                or s08good != 1
                or s08clas in {"EMPTY", "NOSEED"}
            ):
                continue

            names = set(ext.columns.names)
            if "LAMBDA_NM" not in names or "FLUX_FLAM" not in names:
                continue

            lam = np.asarray(ext.data["LAMBDA_NM"], float)
            flux = np.asarray(ext.data["FLUX_FLAM"], float)
            resp = _response_on_grid(lam, master_lam, shape_resp)
            shaped = flux * resp

            fpre, c0 = synthetic_fnu_and_coverage(lam, flux, fw, ft)
            fshp, c1 = synthetic_fnu_and_coverage(lam, shaped, fw, ft)

            if (
                c0 >= min_coverage
                and c1 >= min_coverage
                and np.isfinite(fpre)
                and np.isfinite(fshp)
                and fpre > 0
                and fshp > 0
            ):
                rows.append((ext.name.upper(), float(fpre / fshp)))

    if not rows:
        raise RuntimeError("No spectra usable for global i-band response normalization.")

    ratios = np.asarray([x[1] for x in rows], float)
    factor = float(np.median(ratios))
    p16, p84 = np.percentile(ratios, [16, 84])

    return factor, float(p16), float(p84), rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Build Step12d production response from validated ensemble shape."
    )
    ap.add_argument("--response-csv", type=str, default="")
    ap.add_argument("--spectra", type=str, default="")
    ap.add_argument("--out-fits", type=str, default="")
    ap.add_argument("--summary-csv", type=str, default="")
    ap.add_argument("--metadata-json", type=str, default="")
    ap.add_argument("--trust-min", type=float, default=PIVOT_R_NM)
    ap.add_argument("--trust-max", type=float, default=PIVOT_Z_NM)
    ap.add_argument("--min-i-coverage", type=float, default=0.90)
    return ap.parse_args()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    )
    args = parse_args()

    response_csv = Path(args.response_csv) if args.response_csv else _default_response_csv()
    spectra_path = Path(args.spectra) if args.spectra else Path(config.EXTRACT1D_STEP12_INPUT)
    out_fits = Path(args.out_fits) if args.out_fits else Path(config.STEP12D_MASTER_FITS)
    summary_csv = Path(args.summary_csv) if args.summary_csv else Path(config.STEP12D_SUMMARY_CSV)
    metadata_json = Path(args.metadata_json) if args.metadata_json else Path(config.STEP12D_METADATA_JSON)

    for p in (out_fits, summary_csv, metadata_json):
        p.parent.mkdir(parents=True, exist_ok=True)

    log.info("Validated ensemble response: %s", response_csv)
    log.info("Step11 spectra for global i normalization: %s", spectra_path)
    log.info("Trusted interval: %.6f .. %.6f nm", args.trust_min, args.trust_max)

    raw = pd.read_csv(response_csv)
    master = _validated_table(raw, args.trust_min, args.trust_max)
    stats = _loo_stats(_companion_loo(response_csv))

    lam = master["lambda_nm"].to_numpy(np.float64)
    shape_resp = master["response_quadratic"].to_numpy(np.float64)
    shape_p16 = master["loo_p16"].to_numpy(np.float64)
    shape_p84 = master["loo_p84"].to_numpy(np.float64)

    global_i_norm, norm_p16, norm_p84, norm_rows = derive_global_i_normalization(
        spectra_path,
        lam,
        shape_resp,
        Path(config.FILTER_I),
        min_coverage=float(args.min_i_coverage),
    )

    resp = shape_resp * global_i_norm
    p16 = shape_p16 * global_i_norm
    p84 = shape_p84 * global_i_norm

    phdr = fits.Header()
    phdr["PIPESTEP"] = ("STEP12", "SAMOS pipeline step")
    phdr["STAGE"] = ("12d", "Pipeline stage")
    phdr["METHOD"] = ("ENS_QUAD", "Validated ensemble quadratic response")
    phdr["RESPCSV"] = (response_csv.name, "Input ensemble-response CSV")
    phdr["TRUSTLO"] = (float(args.trust_min), "Trusted response lower wavelength [nm]")
    phdr["TRUSTHI"] = (float(args.trust_max), "Trusted response upper wavelength [nm]")
    phdr["SHAPENM"] = (PIVOT_I_NM, "Shape-response pivot normalization [nm]")
    phdr["GINORM"] = (global_i_norm, "Global i-band normalization folded into response")
    phdr["GINP16"] = (norm_p16, "16th percentile individual i-band norm")
    phdr["GINP84"] = (norm_p84, "84th percentile individual i-band norm")
    phdr["NGINORM"] = (len(norm_rows), "Spectra used for global i normalization")
    phdr["PIVOTR"] = (PIVOT_R_NM, "SkyMapper r pivot [nm]")
    phdr["PIVOTI"] = (PIVOT_I_NM, "SkyMapper i pivot [nm]")
    phdr["PIVOTZ"] = (PIVOT_Z_NM, "SkyMapper z pivot [nm]")
    if np.isfinite(stats["n_stars"]):
        phdr["NSTARS"] = (int(stats["n_stars"]), "Stars in ensemble LOO validation")
    phdr.add_history("Step12d shape response from validated ensemble quadratic.")
    phdr.add_history("One global factor preserves median Step11 synthetic i-band flux.")
    phdr.add_history("No per-object photometric normalization is part of production Step12.")

    cols = fits.ColDefs(
        [
            fits.Column(name="LAMBDA_NM", format="D", array=lam),
            fits.Column(name="RESP_MASTER", format="D", array=resp),
            fits.Column(name="RESP_P16", format="D", array=p16),
            fits.Column(name="RESP_P84", format="D", array=p84),
            fits.Column(name="RESP_SHAPE", format="D", array=shape_resp),
        ]
    )
    hdu = fits.BinTableHDU.from_columns(cols, name="MASTER_RESPONSE")
    hdu.header["TRUSTLO"] = float(args.trust_min)
    hdu.header["TRUSTHI"] = float(args.trust_max)
    hdu.header["SHAPENM"] = PIVOT_I_NM
    hdu.header["GINORM"] = global_i_norm

    fits.HDUList([fits.PrimaryHDU(header=phdr), hdu]).writeto(out_fits, overwrite=True)

    def interp(w, a):
        return float(np.interp(w, lam, a))

    summary = {
        "method": "ensemble_quadratic_plus_global_i_normalization",
        "source_csv": str(response_csv),
        "spectra_path": str(spectra_path),
        "trust_min_nm": float(args.trust_min),
        "trust_max_nm": float(args.trust_max),
        "pivot_r_nm": PIVOT_R_NM,
        "pivot_i_nm": PIVOT_I_NM,
        "pivot_z_nm": PIVOT_Z_NM,
        "global_i_norm": global_i_norm,
        "global_i_norm_p16": norm_p16,
        "global_i_norm_p84": norm_p84,
        "global_i_norm_n": len(norm_rows),
        "shape_response_r": interp(PIVOT_R_NM, shape_resp),
        "shape_response_i": interp(PIVOT_I_NM, shape_resp),
        "shape_response_z": interp(PIVOT_Z_NM, shape_resp),
        "response_r": interp(PIVOT_R_NM, resp),
        "response_i": interp(PIVOT_I_NM, resp),
        "response_z": interp(PIVOT_Z_NM, resp),
        **stats,
    }
    pd.DataFrame([summary]).to_csv(summary_csv, index=False)

    metadata = {
        **summary,
        "output_fits": str(out_fits),
        "edge_policy_for_step12e": "hold nearest trusted boundary; never polynomial-extrapolate",
        "global_i_normalization_rows": [
            {"slit": slit, "factor": fac} for slit, fac in norm_rows
        ],
        "notes": [
            "RESP_SHAPE is normalized at the i-band pivot.",
            "RESP_MASTER = global_i_norm * RESP_SHAPE.",
            "The global factor preserves the median Step11 synthetic i-band flux.",
            "No per-object photometric normalization is used in production Step12.",
        ],
    }
    metadata_json.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    log.info(
        "Global i normalization: %.8f  (N=%d; p16=%.8f p84=%.8f)",
        global_i_norm, len(norm_rows), norm_p16, norm_p84,
    )
    log.info("Wrote: %s", out_fits)
    log.info("Wrote: %s", summary_csv)
    log.info("Wrote: %s", metadata_json)
    log.info(
        "Production response at pivots: r=%.4f i=%.4f z=%.4f",
        summary["response_r"], summary["response_i"], summary["response_z"],
    )
    if np.isfinite(stats["loo_quadratic_median_mag"]):
        log.info(
            "Shape-fit LOO median RMS: gray=%.4f linear=%.4f quadratic=%.4f mag",
            stats["loo_gray_median_mag"],
            stats["loo_linear_median_mag"],
            stats["loo_quadratic_median_mag"],
        )


if __name__ == "__main__":
    main()
