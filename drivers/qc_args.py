#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument construction for registered QC scripts."""
from __future__ import annotations

from pathlib import Path


def format_qc_args(qc_path: Path, set_name: str | None, cfg_module, repo_root: Path) -> list[str]:
    """Return command-line arguments for a registered QC companion script."""
    qc_str = str(qc_path)

    # Step09 closeout QC.
    if qc_str.endswith("qc/step09/qc_step09_preferred_all_slits.py"):
        root = cfg_module.ST09
        return [
            "--root", str(root),
            "--out-pdf", str(Path(root) / "qc_step09_preferred_all_slits.pdf"),
        ]

    if qc_str.endswith("qc/step09/qc_step09_final_mosaic.py"):
        root = cfg_module.ST09
        return [
            "--in", str(cfg_module.EXTRACT1D_OHCLEAN),
            "--outdir", str(Path(root) / "qc_step09"),
            "--column", "STELLAR",
            "--show-pref",
        ]

    # Step10 closeout QC.
    if qc_str.endswith("qc/step10/qc_step10_final_mosaic.py"):
        return [
            "--in", str(cfg_module.EXTRACT1D_TELLCOR),
            "--outdir", str(Path(cfg_module.ST10_TELLURIC) / "qc_step10"),
            "--column", "FLUX_TELLCOR_O2",
        ]

    # Step11 summary QC.
    if qc_str.endswith("qc/step11/qc_step11_summary_b.py"):
        return [
            "--extract", str(cfg_module.EXTRACT1D_FLUXCAL),
            "--photcat", str(cfg_module.STEP11_PHOTCAT),
            "--tracecoords", f"{cfg_module.SCI_EVEN_TRACECOORDS}|{cfg_module.SCI_ODD_TRACECOORDS}",
            "--image", str(
                getattr(
                    cfg_module,
                    "SISI_IMAGE_FITS",
                    repo_root / "calibration" / "sisi" / "Coadd_i_median_078-082_ff_flipx_wcs_manual.fits",
                )
            ),
            "--outpdf", str(Path(cfg_module.ST11_FLUXCAL) / "qc_step11" / "qc_step11_summary_pages.pdf"),
        ]

    # Step11 grid QC auto-discovers its inputs from config/defaults.
    if qc_str.endswith("qc/step11/qc_step11_grid_patched_v2.py"):
        return []

    # Generic fallback for set-based QC.
    if set_name is not None:
        if (
            qc_str.endswith("qc/step04/qc_step04_trace_quicklooks.py")
            or qc_str.endswith("qc/step06/qc_step06b_inspector_final.py")
            or qc_str.endswith("qc/step06/qc_step06c_quicklooks_final.py")
        ):
            return ["--traceset", set_name]
        return ["--set", set_name]

    return []
