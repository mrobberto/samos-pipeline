#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""QC for the corrected Step08c detector-window wavelength mapping.

Checks, slit by slit:
1. Candidate LAMBDA_NM equals the explicitly supplied Step07h wavelength MEF
   sampled at index = Y0DET + YPIX - YWIN0.
2. All pre-existing Step08b table columns are unchanged by Step08c.
3. Optionally, quantify the wavelength difference relative to a frozen/reference
   wavelength-attached product.

This is a diagnostic only; it does not modify any FITS product.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from astropy.io import fits


def parse_args():
    ap = argparse.ArgumentParser(description="QC corrected Step08c wavelength mapping")
    ap.add_argument("--input", required=True, type=Path,
                    help="Step08b merged pixel-space input used to make candidate")
    ap.add_argument("--candidate", required=True, type=Path,
                    help="Corrected Step08c wavelength-attached candidate")
    ap.add_argument("--wave-mef", required=True, type=Path,
                    help="Exact Step07h wavelength MEF supplied to Step08c")
    ap.add_argument("--reference", type=Path, default=None,
                    help="Optional frozen/old Step08c product for delta-lambda statistics")
    return ap.parse_args()


def slit_names(h):
    return [x.name.upper() for x in h[1:] if (x.name or "").upper().startswith("SLIT")]


def finite_max_abs(a, b):
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        return np.inf
    if a.dtype.kind in "SUO" or b.dtype.kind in "SUO":
        return 0.0 if np.array_equal(a, b) else np.inf
    af = np.asarray(a, float)
    bf = np.asarray(b, float)
    same_nan = np.array_equal(np.isnan(af), np.isnan(bf))
    if not same_nan:
        return np.inf
    q = np.isfinite(af) & np.isfinite(bf)
    if not q.any():
        return 0.0
    return float(np.nanmax(np.abs(af[q] - bf[q])))


def main():
    args = parse_args()
    for p in [args.input, args.candidate, args.wave_mef]:
        if not p.exists():
            raise FileNotFoundError(p)
    if args.reference is not None and not args.reference.exists():
        raise FileNotFoundError(args.reference)

    with fits.open(args.input, memmap=False) as hi, \
         fits.open(args.candidate, memmap=False) as hc, \
         fits.open(args.wave_mef, memmap=False) as hw:

        common = sorted(set(slit_names(hi)) & set(slit_names(hc)) & set(slit_names(hw)))
        ywin0 = float(hc[0].header["YWIN0"])
        firstlen = int(hc[0].header["FIRSTLEN"])

        max_map_err = 0.0
        max_invariant = 0.0
        n_map = 0
        n_bad_map = 0

        for slit in common:
            inp = hi[slit]
            cand = hc[slit]
            wav = hw[slit]

            if "LAMBDA_NM" not in cand.columns.names:
                n_bad_map += 1
                continue

            ypix = np.asarray(cand.data["YPIX"], float)
            y0det = float(cand.header.get("Y0DET", cand.header.get("YMIN")))
            shift = float(cand.header["SHIFT2M"])
            ydet = y0det + ypix
            idxf = ydet - ywin0
            idx = np.rint(idxf).astype(int)
            if not np.allclose(idxf, idx, atol=1e-6, rtol=0):
                raise RuntimeError(f"{slit}: non-integer detector-window indices")

            arr = np.asarray(wav.data, float)
            lam_full = arr[1]
            expected = np.full(len(ypix), np.nan, float)
            good = (idx >= 0) & (idx < len(lam_full))
            expected[good] = lam_full[idx[good]]

            y_eff = idxf + shift
            expected[(y_eff < 0) | (y_eff > (firstlen - 1))] = np.nan
            actual = np.asarray(cand.data["LAMBDA_NM"], float)

            if not np.array_equal(np.isnan(expected), np.isnan(actual)):
                n_bad_map += 1
            q = np.isfinite(expected) & np.isfinite(actual)
            if q.any():
                err = float(np.nanmax(np.abs(actual[q] - expected[q])))
                max_map_err = max(max_map_err, err)
                n_map += 1

            # Step08c must leave every pre-existing Step08b column unchanged.
            for col in inp.columns.names:
                if col not in cand.columns.names:
                    max_invariant = np.inf
                    continue
                max_invariant = max(max_invariant, finite_max_abs(inp.data[col], cand.data[col]))

        print("INPUT     :", args.input)
        print("CANDIDATE :", args.candidate)
        print("WAVE MEF  :", args.wave_mef)
        print("candidate WAVMAP =", hc[0].header.get("WAVMAP", "(primary not set)"))
        print("---")
        print("N common slits:", len(common))
        print("N mapped slits:", n_map)
        print("N slits with NaN-mask mismatch:", n_bad_map)
        print("max |candidate - detector-indexed Step07h| [nm] =", max_map_err)
        print("max finite invariant-column difference =", max_invariant)

    if args.reference is not None:
        with fits.open(args.candidate, memmap=False) as hc, fits.open(args.reference, memmap=False) as hr:
            common = sorted(set(slit_names(hc)) & set(slit_names(hr)))
            rows = []
            for slit in common:
                if "LAMBDA_NM" not in hc[slit].columns.names or "LAMBDA_NM" not in hr[slit].columns.names:
                    continue
                a = np.asarray(hc[slit].data["LAMBDA_NM"], float)
                b = np.asarray(hr[slit].data["LAMBDA_NM"], float)
                q = np.isfinite(a) & np.isfinite(b)
                if not q.any():
                    continue
                d = a[q] - b[q]
                rows.append((slit, float(np.nanmedian(d)), float(np.nanmax(np.abs(d)))))

            if rows:
                med = np.array([r[1] for r in rows])
                mx = np.array([r[2] for r in rows])
                print("--- candidate - frozen reference wavelength ---")
                print("N comparable slits:", len(rows))
                print("slit median delta nm min/med/max = %.4f / %.4f / %.4f" %
                      (np.nanmin(med), np.nanmedian(med), np.nanmax(med)))
                print("N |median delta| >= 1/2/5/10 nm = %d / %d / %d / %d" %
                      tuple(np.sum(np.abs(med) >= x) for x in [1, 2, 5, 10]))
                worst = sorted(rows, key=lambda r: abs(r[1]), reverse=True)[:10]
                print("largest slit median deltas:")
                for slit, dmed, dmax in worst:
                    print("  %-8s median=%+9.4f nm  maxabs=%9.4f nm" % (slit, dmed, dmax))

    if max_invariant != 0.0:
        raise SystemExit("FAIL: Step08c changed one or more pre-existing Step08b columns")
    if n_bad_map != 0 or max_map_err > 1e-5:
        raise SystemExit("FAIL: candidate wavelength does not match detector-indexed Step07h MEF")
    print("PASS: detector-window wavelength mapping and extraction-column invariance verified")


if __name__ == "__main__":
    main()
