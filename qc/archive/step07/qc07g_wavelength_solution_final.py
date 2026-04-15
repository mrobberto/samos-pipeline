#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Step07g QC — inspect the global wavelength solution and peak matches.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from astropy.io import fits
import config

def robust_sigma(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size < 5:
        return np.nan
    med = np.median(x)
    return 1.4826 * np.median(np.abs(x - med))

def poly_from_header(hdr):
    coeff = []
    for i in range(50):
        k = f'WVC{i}'
        if k in hdr:
            coeff.append(float(hdr[k]))
        else:
            break
    if not coeff:
        raise RuntimeError('No WVC* coefficients found.')
    return np.poly1d(coeff[::-1])

def first_existing(paths):
    for p in paths:
        if p is None:
            continue
        p = Path(p)
        if p.exists():
            return p
    return None

def main():
    ap = argparse.ArgumentParser(description='Step07g QC')
    ap.add_argument('--fits', default=None)
    ap.add_argument('--outdir', default=None)
    args = ap.parse_args()

    st07 = Path(config.ST07_WAVECAL)
    infile = first_existing([Path(args.fits) if args.fits else None, st07 / 'arc_master_wavesol.fits'])
    if infile is None:
        raise FileNotFoundError('Could not find arc_master_wavesol.fits')

    outdir = Path(args.outdir) if args.outdir else (st07 / 'qc_step07g')
    outdir.mkdir(parents=True, exist_ok=True)
    png_path = outdir / 'qc07g_wavesol_summary.png'
    txt_path = outdir / 'qc07g_wavesol_report.txt'

    with fits.open(infile) as h:
        hdr = h[0].header.copy()
        poly = poly_from_header(hdr)
        tab = h['PEAK_MATCHES'].data
        y_peak = np.asarray(tab['Y_PEAK'], float)
        lam_match = np.asarray(tab['LAM_MATCH_NM'], float)
        resid = np.asarray(tab['RESID_NM'], float)
        good = np.asarray(tab['GOOD'], bool)

    firstlen = int(hdr.get('FIRSTLEN', 2875))
    y_eval = np.arange(firstlen, dtype=float)
    lam_eval = poly(y_eval)
    dlam = np.diff(lam_eval)

    resid_good = resid[np.isfinite(resid) & good]
    rms_good = np.sqrt(np.nanmean(resid_good**2)) if resid_good.size else np.nan
    med_dlam = np.nanmedian(dlam[np.isfinite(dlam)]) if np.any(np.isfinite(dlam)) else np.nan
    mono_inc = bool(np.all(dlam[np.isfinite(dlam)] > 0)) if np.any(np.isfinite(dlam)) else False
    mono_dec = bool(np.all(dlam[np.isfinite(dlam)] < 0)) if np.any(np.isfinite(dlam)) else False

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    ax[0, 0].plot(y_eval, lam_eval, color='black', lw=1.2, label='poly')
    ax[0, 0].plot(y_peak, lam_match, 'o', ms=4, label='matched')
    ax[0, 0].set_title('Global wavelength solution')
    ax[0, 0].set_xlabel('Master window pixel y')
    ax[0, 0].set_ylabel('Wavelength (nm)')
    ax[0, 0].legend()

    ax[0, 1].axhline(0, color='0.6', lw=0.8)
    ax[0, 1].plot(y_peak[~good], resid[~good], 'o', ms=4, alpha=0.7, label='rejected')
    ax[0, 1].plot(y_peak[good], resid[good], 'o', ms=4, label='good')
    ax[0, 1].set_title('Residuals vs Y')
    ax[0, 1].set_xlabel('Master window pixel y')
    ax[0, 1].set_ylabel('Residual (nm)')
    ax[0, 1].legend()

    if resid[np.isfinite(resid)].size:
        ax[1, 0].hist(resid[np.isfinite(resid)], bins=30, histtype='step', label='all')
    if resid_good.size:
        ax[1, 0].hist(resid_good, bins=30, histtype='step', label='good')
    ax[1, 0].axvline(0, color='0.6', lw=0.8)
    ax[1, 0].set_title('Residual histogram')
    ax[1, 0].set_xlabel('Residual (nm)')
    ax[1, 0].set_ylabel('N')
    ax[1, 0].legend()

    ax[1, 1].plot(y_eval[:-1], dlam, lw=1.0)
    ax[1, 1].set_title('Dispersion dλ/dy')
    ax[1, 1].set_xlabel('Master window pixel y')
    ax[1, 1].set_ylabel('nm / pix')
    text = f'mono_inc={mono_inc}\nmono_dec={mono_dec}\nmedian dλ/dy={med_dlam:.6f} nm/pix\nRMS(good)={rms_good:.6f} nm'
    ax[1, 1].text(0.03, 0.97, text, transform=ax[1, 1].transAxes, va='top', ha='left', fontsize=9, bbox=dict(boxstyle='round', fc='white', ec='0.7'))

    fig.suptitle(f'Step07g QC — {infile.name}', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(png_path, dpi=160)
    plt.close(fig)

    with open(txt_path, 'w') as f:
        f.write('Step07g QC — global wavelength solution\n')
        for k in ['STAGE', 'YWIN0', 'FIRSTLEN', 'ORDER', 'NMATCH', 'RMSNM']:
            if k in hdr:
                f.write(f'{k} = {hdr[k]}\n')
        f.write(f'RMS_good = {rms_good}\n')
        f.write(f'mono_inc = {mono_inc}\n')
        f.write(f'mono_dec = {mono_dec}\n')
        f.write(f'median_dlam = {med_dlam}\n')
        f.write(f'robust_sigma_good = {robust_sigma(resid_good)}\n')

    print('[DONE] Wrote', png_path)
    print('[DONE] Wrote', txt_path)

if __name__ == '__main__':
    main()