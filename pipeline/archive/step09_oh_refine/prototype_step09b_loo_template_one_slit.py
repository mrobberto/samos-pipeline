#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, List

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from astropy.io import fits
from scipy.interpolate import UnivariateSpline
from scipy.signal import correlate


def norm_slit(s: str) -> str:
    s = str(s).strip().upper()
    digits = "".join(ch for ch in s if ch.isdigit())
    return f"SLIT{int(digits):03d}" if digits else s


def robust_sigma(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    mad = np.nanmedian(np.abs(x - med))
    sig = 1.4826 * mad if np.isfinite(mad) and mad > 0 else np.nanstd(x)
    return float(sig) if np.isfinite(sig) and sig > 0 else float(np.nanstd(x))


def robust_rms(x: np.ndarray) -> float:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan
    med = np.nanmedian(x)
    return float(np.sqrt(np.nanmedian((x - med) ** 2)))


def robust_ylim(y, q=(1, 99), pad=0.08):
    y = np.asarray(y, float)
    y = y[np.isfinite(y)]
    if y.size == 0:
        return (-1.0, 1.0)
    lo, hi = np.nanpercentile(y, q)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        m = np.nanmedian(y) if y.size else 0.0
        return (m - 1.0, m + 1.0)
    d = hi - lo
    return (lo - pad * d, hi + pad * d)


def choose_lambda_column(names: Iterable[str]) -> str:
    cols = list(names)
    low = {c.lower(): c for c in cols}
    for cand in ["LAMBDA_NM", "lambda_nm", "WAVELENGTH_NM", "WAVELENGTH", "LAMBDA"]:
        if cand in cols:
            return cand
        if cand.lower() in low:
            return low[cand.lower()]
    raise KeyError("No wavelength column found")


def choose_unsubtracted_signal(tab):
    cols = list(tab.columns.names)
    low = {c.lower(): c for c in cols}

    def get(name: str):
        if name in cols:
            return np.asarray(tab[name], float)
        if name.lower() in low:
            return np.asarray(tab[low[name.lower()]], float)
        return None

    for c in ["OBJ_PRESKY", "OBJ_RAW"]:
        arr = get(c)
        if arr is not None:
            return c, np.ravel(arr)

    flux = get("FLUX")
    sky = get("SKY")
    if flux is not None and sky is not None:
        return "FLUX+SKY", np.ravel(flux + sky)

    for c in ["SKY", "FLUX"]:
        arr = get(c)
        if arr is not None:
            return c, np.ravel(arr)

    raise KeyError("No suitable signal column found")


def list_slit_names(extract_fits: Path) -> List[str]:
    out = []
    with fits.open(extract_fits) as h:
        for hdu in h[1:]:
            nm = str(getattr(hdu, "name", "")).strip().upper()
            if nm.startswith("SLIT"):
                out.append(norm_slit(nm))
            else:
                sid = hdu.header.get("SLITID")
                if sid is not None:
                    try:
                        out.append(f"SLIT{int(sid):03d}")
                    except Exception:
                        pass
    return sorted(set(out))


def read_one_slit(extract_fits: Path, slit: str):
    slit = norm_slit(slit)
    with fits.open(extract_fits) as h:
        hdu = None
        if slit in h:
            hdu = h[slit]
        else:
            sid = int("".join(ch for ch in slit if ch.isdigit()))
            for ext in h[1:]:
                try:
                    if int(ext.header.get("SLITID", -999)) == sid:
                        hdu = ext
                        break
                except Exception:
                    pass
        if hdu is None:
            raise KeyError(f"Could not find {slit}")
        tab = hdu.data
        lam_col = choose_lambda_column(tab.columns.names)
        sig_name, sig = choose_unsubtracted_signal(tab)
        lam = np.ravel(np.asarray(tab[lam_col], float))
    m = np.isfinite(lam) & np.isfinite(sig)
    lam, sig = lam[m], sig[m]
    o = np.argsort(lam)
    return lam[o], sig[o], sig_name


def bin_anchors(lam, y, mask, bin_nm=10.0, stat='median'):
    lam = np.asarray(lam, float)
    y = np.asarray(y, float)
    good = np.isfinite(lam) & np.isfinite(y) & np.asarray(mask, bool)
    lamg, yg = lam[good], y[good]
    if lamg.size < 8:
        return np.array([]), np.array([])
    bins = np.arange(np.nanmin(lamg), np.nanmax(lamg) + bin_nm, bin_nm)
    xb, yb = [], []
    for i in range(len(bins) - 1):
        m = (lamg >= bins[i]) & (lamg < bins[i + 1])
        if m.sum() < 5:
            continue
        xb.append(float(np.nanmedian(lamg[m])))
        if stat == 'median':
            yv = float(np.nanmedian(yg[m]))
        elif stat == 'p60':
            yv = float(np.nanpercentile(yg[m], 60))
        elif stat == 'p70':
            yv = float(np.nanpercentile(yg[m], 70))
        else:
            yv = float(np.nanmedian(yg[m]))
        yb.append(yv)
    return np.asarray(xb), np.asarray(yb)


def build_continuum(lam, y, mask, bin_nm=10.0, stat='median', spline_s=0.25):
    xb, yb = bin_anchors(lam, y, mask, bin_nm=bin_nm, stat=stat)
    if len(xb) < 4:
        return np.full_like(lam, np.nanmedian(y)), xb, yb
    k = min(3, len(xb) - 1)
    try:
        spl = UnivariateSpline(xb, yb, s=float(spline_s) * len(xb), k=k)
        cont = spl(lam)
    except Exception:
        cont = np.interp(lam, xb, yb, left=yb[0], right=yb[-1])
    return np.asarray(cont, float), xb, yb


def grow_narrow_feature_mask(lam, resid, base_mask, clip_sigma=3.5, pad_nm=0.8, max_feature_nm=3.0):
    lam = np.asarray(lam, float)
    resid = np.asarray(resid, float)
    m = np.asarray(base_mask, bool).copy()
    sig = robust_sigma(resid[m & np.isfinite(resid)])
    if not np.isfinite(sig) or sig <= 0:
        return m
    out = np.isfinite(resid) & (np.abs(resid) > clip_sigma * sig)
    if not np.any(out):
        return m
    idx = np.where(out)[0]
    groups = []
    start = idx[0]
    prev = idx[0]
    for ii in idx[1:]:
        if ii == prev + 1:
            prev = ii
        else:
            groups.append((start, prev))
            start, prev = ii, ii
    groups.append((start, prev))
    for i0, i1 in groups:
        width = float(lam[i1] - lam[i0]) if i1 > i0 else 0.0
        if width > max_feature_nm:
            continue
        lo = lam[i0] - pad_nm
        hi = lam[i1] + pad_nm
        m[(lam >= lo) & (lam <= hi)] = False
    return m


def fit_smooth_numerical_continuum(lam, y, bin_nm=10.0, anchor_stat='median',
                                   spline_s=0.25, clip_sigma=3.5, iterations=4,
                                   pad_nm=0.8, max_feature_nm=3.0):
    mask = np.isfinite(y)
    xb, yb = np.array([]), np.array([])
    cont = np.full_like(y, np.nanmedian(y), dtype=float)
    for _ in range(int(iterations)):
        cont, xb, yb = build_continuum(lam, y, mask, bin_nm=bin_nm, stat=anchor_stat, spline_s=spline_s)
        resid = y - cont
        new_mask = grow_narrow_feature_mask(lam, resid, mask, clip_sigma=clip_sigma, pad_nm=pad_nm, max_feature_nm=max_feature_nm)
        if np.all(new_mask == mask):
            mask = new_mask
            break
        mask = new_mask
    cont, xb, yb = build_continuum(lam, y, mask, bin_nm=bin_nm, stat=anchor_stat, spline_s=spline_s)
    resid = y - cont
    return cont, xb, yb, mask, resid


def estimate_shift_pixels(ref: np.ndarray, arr: np.ndarray, max_lag_pix: int = 6) -> int:
    ref = np.asarray(ref, float)
    arr = np.asarray(arr, float)
    m = np.isfinite(ref) & np.isfinite(arr)
    if m.sum() < 20:
        return 0
    a = ref[m] - np.nanmedian(ref[m])
    b = arr[m] - np.nanmedian(arr[m])
    cc = correlate(b, a, mode='full')
    lags = np.arange(-len(a) + 1, len(a))
    keep = np.abs(lags) <= int(max_lag_pix)
    if not np.any(keep):
        return 0
    lag = int(lags[keep][np.argmax(cc[keep])])
    return lag


def shift_array_nan(arr: np.ndarray, lag: int) -> np.ndarray:
    arr = np.asarray(arr, float)
    out = np.full_like(arr, np.nan)
    if lag == 0:
        out[:] = arr
    elif lag > 0:
        out[lag:] = arr[:-lag]
    else:
        out[:lag] = arr[-lag:]
    return out


def robust_scale_to_target(target: np.ndarray, donor: np.ndarray) -> float:
    t = np.asarray(target, float)
    d = np.asarray(donor, float)
    m = np.isfinite(t) & np.isfinite(d)
    if m.sum() < 20:
        return 1.0
    num = np.nansum(t[m] * d[m])
    den = np.nansum(d[m] * d[m])
    if not np.isfinite(den) or den <= 0:
        return 1.0
    scale = float(num / den)
    if not np.isfinite(scale):
        scale = 1.0
    return float(np.clip(scale, 0.2, 5.0))


def build_leave_one_out_template(target_slit: str, lam_ref: np.ndarray, narrow_by_slit: dict,
                                 max_lag_pix: int = 6, min_valid_fraction: float = 0.6):
    target = np.asarray(narrow_by_slit[target_slit], float)
    stack = []
    rows = []
    for slit, arr in narrow_by_slit.items():
        if slit == target_slit:
            continue
        arr = np.asarray(arr, float)
        lag = estimate_shift_pixels(target, arr, max_lag_pix=max_lag_pix)
        arr_shift = shift_array_nan(arr, lag)
        scale = robust_scale_to_target(target, arr_shift)
        donor = scale * arr_shift
        valid_frac = float(np.isfinite(donor).sum() / len(donor))
        if valid_frac < min_valid_fraction:
            continue
        stack.append(donor)
        rows.append({
            'slit': slit,
            'lag_pix': int(lag),
            'scale_to_target': float(scale),
            'valid_fraction': valid_frac,
        })
    if len(stack) == 0:
        return np.full_like(lam_ref, np.nan), pd.DataFrame(rows)
    stack = np.vstack(stack)
    template = np.nanmedian(stack, axis=0)
    return template, pd.DataFrame(rows)


def fit_template_amplitude(target: np.ndarray, template: np.ndarray) -> float:
    t = np.asarray(target, float)
    u = np.asarray(template, float)
    m = np.isfinite(t) & np.isfinite(u)
    if m.sum() < 20:
        return 1.0
    num = np.nansum(t[m] * u[m])
    den = np.nansum(u[m] * u[m])
    if not np.isfinite(den) or den <= 0:
        return 1.0
    scale = float(num / den)
    return float(np.clip(scale, 0.0, 5.0))


def parse_args():
    p = argparse.ArgumentParser(description="Leave-one-out empirical OH template for one target slit")
    p.add_argument('--extract', type=Path, required=True)
    p.add_argument('--target-slit', type=str, required=True)
    p.add_argument('--slits', nargs='*', default=None)
    p.add_argument('--bin-nm', type=float, default=10.0)
    p.add_argument('--anchor-stat', type=str, default='median')
    p.add_argument('--spline-s', type=float, default=0.25)
    p.add_argument('--clip-sigma', type=float, default=3.5)
    p.add_argument('--iterations', type=int, default=4)
    p.add_argument('--pad-nm', type=float, default=0.8)
    p.add_argument('--max-feature-nm', type=float, default=3.0)
    p.add_argument('--max-lag-pix', type=int, default=6)
    p.add_argument('--min-valid-fraction', type=float, default=0.6)
    p.add_argument('--out-apply-csv', type=Path, default=None)
    p.add_argument('--out-all-cont-csv', type=Path, default=None)
    p.add_argument('--out-meta-csv', type=Path, default=None)
    p.add_argument('--out-json', type=Path, default=None)
    p.add_argument('--qc-png', type=Path, default=None)
    p.add_argument('--show', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    target_slit = norm_slit(args.target_slit)
    slit_list = [norm_slit(s) for s in args.slits] if args.slits else list_slit_names(args.extract)
    if target_slit not in slit_list:
        slit_list.append(target_slit)
    slit_list = sorted(set(slit_list))

    lam_ref = None
    signal_name = None
    narrow_by_slit = {}
    cont_rows = []

    for slit in slit_list:
        try:
            lam, spec, sig_name = read_one_slit(args.extract, slit)
        except Exception:
            continue
        if lam_ref is None:
            lam_ref = lam
            signal_name = sig_name
        if len(lam) != len(lam_ref) or np.nanmax(np.abs(lam - lam_ref)) > 1e-6:
            spec = np.interp(lam_ref, lam, spec, left=np.nan, right=np.nan)
            lam = lam_ref
        cont, xb, yb, mask, narrow = fit_smooth_numerical_continuum(
            lam, spec,
            bin_nm=args.bin_nm,
            anchor_stat=args.anchor_stat,
            spline_s=args.spline_s,
            clip_sigma=args.clip_sigma,
            iterations=args.iterations,
            pad_nm=args.pad_nm,
            max_feature_nm=args.max_feature_nm,
        )
        narrow_by_slit[slit] = narrow
        cont_rows.append(pd.DataFrame({
            'SLIT': slit,
            'LAMBDA_NM': lam,
            'SPECTRUM': spec,
            'CONTINUUM': cont,
            'NARROW': narrow,
            'MASK_USED': mask.astype(int),
        }))

    if target_slit not in narrow_by_slit:
        raise RuntimeError(f"Target slit {target_slit} not available after reading spectra")

    target_narrow = np.asarray(narrow_by_slit[target_slit], float)
    template, meta_df = build_leave_one_out_template(
        target_slit, lam_ref, narrow_by_slit,
        max_lag_pix=args.max_lag_pix,
        min_valid_fraction=args.min_valid_fraction,
    )
    baseline, _, _, _, _ = fit_smooth_numerical_continuum(
        lam_ref, template,
        bin_nm=args.bin_nm,
        anchor_stat=args.anchor_stat,
        spline_s=args.spline_s,
        clip_sigma=args.clip_sigma,
        iterations=args.iterations,
        pad_nm=args.pad_nm,
        max_feature_nm=args.max_feature_nm,
    )
    template = template - baseline
    template -= np.nanmedian(template)   # optional but recommended
    amp = fit_template_amplitude(target_narrow, template)
    oh_model = amp * template
    resid = target_narrow - oh_model

    base = Path(args.extract).resolve().parent
    out_dir = base.parent / '09_oh_refine'
    out_dir.mkdir(parents=True, exist_ok=True)

    out_apply_csv = Path(args.out_apply_csv) if args.out_apply_csv else (out_dir / f'{target_slit}_loo_template_apply.csv')
    out_all_cont_csv = Path(args.out_all_cont_csv) if args.out_all_cont_csv else (out_dir / f'{target_slit}_loo_all_contsub.csv')
    out_meta_csv = Path(args.out_meta_csv) if args.out_meta_csv else (out_dir / f'{target_slit}_loo_template_members.csv')
    out_json = Path(args.out_json) if args.out_json else (out_dir / f'{target_slit}_loo_template_summary.json')
    qc_png = Path(args.qc_png) if args.qc_png else (out_dir / f'qc_{target_slit}_loo_template.png')

    pd.DataFrame({
        'LAMBDA_NM': lam_ref,
        'TARGET_NARROW': target_narrow,
        'TEMPLATE': template,
        'OH_MODEL': oh_model,
        'RESIDUAL': resid,
    }).to_csv(out_apply_csv, index=False)
    pd.concat(cont_rows, ignore_index=True).to_csv(out_all_cont_csv, index=False)
    meta_df.to_csv(out_meta_csv, index=False)

    rms_raw = float(np.sqrt(np.nanmean(resid**2))) if np.isfinite(resid).any() else np.nan
    rms_rob = robust_rms(resid)
    summary = {
        'target_slit': target_slit,
        'extract': str(Path(args.extract).resolve()),
        'signal_column': signal_name,
        'n_input_slits': int(len(slit_list)),
        'n_template_members': int(len(meta_df)),
        'template_amplitude': float(amp),
        'rms_raw': rms_raw,
        'rms_robust': rms_rob,
        'max_lag_pix': int(args.max_lag_pix),
        'min_valid_fraction': float(args.min_valid_fraction),
    }
    out_json.write_text(json.dumps(summary, indent=2))

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(f'Leave-one-out empirical OH template — {target_slit}', fontsize=14)

    ax1 = fig.add_subplot(221)
    ax1.plot(lam_ref, target_narrow, lw=0.8, color='k', label='target narrow')
    ax1.plot(lam_ref, template, lw=0.8, color='tab:blue', label='LOO template')
    ax1.plot(lam_ref, oh_model, lw=0.8, color='tab:orange', label='scaled OH model')
    ax1.set_title('Target narrow residual vs leave-one-out template')
    ax1.set_xlabel('Wavelength (nm)')
    ax1.set_ylabel('Signal')
    ax1.set_ylim(*robust_ylim(np.r_[target_narrow, oh_model], q=(0.5, 99.5)))
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(222)
    ax2.plot(lam_ref, resid, lw=0.8, color='k')
    ax2.axhline(0.0, lw=0.8, ls='--', color='0.5')
    ax2.set_title(f'Residual after scaled template subtraction (RMSrob={rms_rob:.5f}, RMSraw={rms_raw:.5f})')
    ax2.set_xlabel('Wavelength (nm)')
    ax2.set_ylabel('Residual')
    ax2.set_ylim(*robust_ylim(resid, q=(0.5, 99.5)))

    ax3 = fig.add_subplot(223)
    for _, row in meta_df.iterrows():
        slit = row['slit']
        donor = narrow_by_slit[slit]
        lag = int(row['lag_pix'])
        donor = shift_array_nan(donor, lag) * float(row['scale_to_target'])
        ax3.plot(lam_ref, donor, lw=0.35, alpha=0.25, color='0.4')
    ax3.plot(lam_ref, template, lw=1.1, color='tab:blue', label='median template')
    ax3.set_title('Aligned donor narrow spectra used in template')
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Signal')
    ax3.set_ylim(*robust_ylim(template, q=(0.5, 99.5)))
    ax3.legend(fontsize=8)

    ax4 = fig.add_subplot(224)
    txt = [
        f'target slit = {target_slit}',
        f'signal column = {signal_name}',
        f'n input slits = {len(slit_list)}',
        f'n template members = {len(meta_df)}',
        f'template amplitude = {amp:.4f}',
        f'RMS raw = {rms_raw:.6f}',
        f'RMS robust = {rms_rob:.6f}',
        f'max lag = {args.max_lag_pix} pix',
        f'min valid frac = {args.min_valid_fraction:.2f}',
        '',
        'members:',
    ]
    for _, row in meta_df.head(12).iterrows():
        txt.append(f'  {row["slit"]}: lag={int(row["lag_pix"]):+d}, scale={float(row["scale_to_target"]):.3f}')
    ax4.axis('off')
    ax4.text(0.02, 0.98, '\n'.join(txt), va='top', ha='left', family='monospace', fontsize=10)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(qc_png, dpi=150)
    if args.show:
        plt.show()
    else:
        plt.close(fig)

    print('TARGET    =', target_slit)
    print('SIGNAL    =', signal_name)
    print('N_SLITS   =', len(slit_list))
    print('N_MEMBERS =', len(meta_df))
    print('AMP       =', amp)
    print('RMSRAW    =', rms_raw)
    print('RMSROB    =', rms_rob)
    print('OUT APPLY =', out_apply_csv)
    print('OUT CONT  =', out_all_cont_csv)
    print('OUT META  =', out_meta_csv)
    print('QC        =', qc_png)


if __name__ == '__main__':
    main()
