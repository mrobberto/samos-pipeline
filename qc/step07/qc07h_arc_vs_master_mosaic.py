#!/usr/bin/env python3

from pathlib import Path
import argparse
import numpy as np
from astropy.io import fits
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import config

QC_DIR = Path(config.PRODUCT_ROOT) / "qc" / "07_wavecal" / "07h"
QC_DIR.mkdir(parents=True, exist_ok=True)

def norm(x):
    x = np.asarray(x, float)
    m = np.isfinite(x)
    if m.sum() == 0:
        return x
    p = np.nanpercentile(x[m], 99)
    return x / p if np.isfinite(p) and p != 0 else x

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arc", default=str(Path(config.ST07_WAVECAL) / "arc_1d_wavelength_all.fits"))
    ap.add_argument("--master", default=str(Path(config.ST07_WAVECAL) / "arc_master_wavesol.fits"))
    ap.add_argument("--out", default=str(QC_DIR / "qc07h_arc_vs_master_mosaic.pdf"))
    ap.add_argument("--xmin", type=float, default=575)
    ap.add_argument("--xmax", type=float, default=975)
    ap.add_argument("--ncol", type=int, default=6)
    args = ap.parse_args()

    arc_path = Path(args.arc)
    master_path = Path(args.master)
    out = Path(args.out)

    master_flux_path = Path(config.ST07_WAVECAL) / "arc_master.fits"
    
    with fits.open(master_flux_path) as hf:
        master_flux = np.asarray(hf[0].data, float)
    
    with fits.open(master_path) as hm:
        mh = hm[0].header
    
        coeff = []
        i = 0
        while f"WVC{i}" in mh:
            coeff.append(float(mh[f"WVC{i}"]))
            i += 1

        poly = np.poly1d(coeff[::-1])
        y = np.arange(master_flux.size, dtype=float)
        master_lam = poly(y)

    with fits.open(arc_path) as ha:
        slits = [h.name for h in ha[1:] if h.name.startswith("SLIT")]

        n = len(slits)
        ncol = args.ncol
        nrow = int(np.ceil(n / ncol))

        fig, axes = plt.subplots(
            nrow, ncol,
            figsize=(3.2 * ncol, 2.0 * nrow),
            sharex=True,
            sharey=True,
        )
        axes = np.ravel(axes)

        mm = np.isfinite(master_lam) & np.isfinite(master_flux)
        mm &= (master_lam >= args.xmin) & (master_lam <= args.xmax)

        for ax, slit in zip(axes, slits):
            d = np.asarray(ha[slit].data, float)

            flux = d[0]
            lam = d[1]

            m = np.isfinite(lam) & np.isfinite(flux)
            m &= (lam >= args.xmin) & (lam <= args.xmax)

            ax.plot(
                master_lam[mm],
                norm(master_flux[mm]),
                color="red",
                lw=1.0,
                alpha=0.8,
                label="master",
            )
            
            ax.plot(
                lam[m],
                norm(flux[m]),
                color="black",
                lw=0.8,
                label=slit,
            )


            ax.set_title(slit, fontsize=7)
            ax.grid(alpha=0.2)

        for ax in axes[n:]:
            ax.axis("off")

        fig.suptitle("Step07h wavelength propagation QC: each slit vs master arc", fontsize=14)
        fig.supxlabel("Wavelength (nm)")
        fig.supylabel("Normalized arc flux")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        fig.savefig(out)
        plt.close(fig)

    print("[OK] wrote", out)

if __name__ == "__main__":
    main()
