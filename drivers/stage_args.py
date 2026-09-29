#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Argument construction for pipeline stages.

Most stages use static templates stored in the stage registry. A few stages
need config-derived arguments or explicit command-line override support; those
exceptions live here rather than in the main driver.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import shlex

from drivers.registry import Stage

def _pick_first_existing(*vals):
    for v in vals:
        if not v:
            continue
        p = Path(v)
        if p.exists():
            return str(p)
    return ""

# -----------------------------------------------------------------------------
# Per-stage argument formatting
# -----------------------------------------------------------------------------
# Most stages use static args_template values from SCRIPT_REGISTRY.
# Stages with config-driven inputs/outputs or backward-compatible overrides are
# handled explicitly here.
# -----------------------------------------------------------------------------
def format_stage_args(stage: Stage, set_name: str | None, args: argparse.Namespace, cfg_module) -> list[str]:

    if stage.key == "08c":
        # Science extraction uses the baseline Step07h wavelength MEF.
        # Optional Step07i manual/tweaked wavelength products must never be
        # selected implicitly merely because a trial-tweak file exists.
        return [
            "--in", str(cfg_module.EXTRACT1D_ALL),
            "--out", str(cfg_module.EXTRACT1D_WAV),
            "--wave-mef", str(cfg_module.ARC_WAVELENGTH_BASE),
        ]

    if stage.key == "08e":
        return [
            "--infile", str(cfg_module.EXTRACT1D_WAV),
            "--table", str(cfg_module.SCIENCE_WAVELENGTH_OFFSET_TABLE),
            "--outfile", str(cfg_module.EXTRACT1D_ABSWAV),
        ]

    if stage.key == "09":
        # Step09 performs wavelength registration only.
        return [
            "--in-fits", str(cfg_module.EXTRACT1D_ABSWAV),
            "--outdir", str(cfg_module.ST09),
        ]

    if stage.key == "10a":
        return [
            "--infile", str(cfg_module.EXTRACT1D_OHREF),
            "--outfile", str(cfg_module.TELLURIC_TEMPLATE),
        ]

    if stage.key == "10b":
        return [
            "--infile", str(cfg_module.EXTRACT1D_OHREF),
            "--template", str(cfg_module.TELLURIC_TEMPLATE),
            "--outfile", str(cfg_module.EXTRACT1D_TELLCOR),
        ]

    if stage.key == "10c":
        return [
            "--infile", str(cfg_module.EXTRACT1D_TELLCOR),
            "--outfile", str(cfg_module.EXTRACT1D_TELLCOR_ILLUMREL),
            "--reference-csv", str(cfg_module.ILLUMREL_REFERENCE_CSV),
        ]

    if stage.key == "11a":
        vals: list[str] = []
        infile = (
            args.step11a_infile
            or str(cfg_module.STEP11_INPUT_SPECTRA)
            or str(cfg_module.EXTRACT1D_TELLCOR)
        )
        outcsv = (
            args.step11a_outcsv
            or str(getattr(cfg_module, "STEP11_RADEC", ""))
            or str(Path(getattr(cfg_module, "ST11_FLUXCAL")) / "slit_trace_radec_all.csv")
        )
        if not infile:
            raise ValueError("Step11a requires a config EXTRACT1D_TELLCOR/STEP11_INPUT_SPECTRA or --step11a-infile")
        vals.extend(["--infile", infile, "--out", outcsv])

        even_geom = (
            getattr(cfg_module, "EVEN_TRACES_GEOM", None)
            or getattr(cfg_module, "EVEN_TRACE_GEOM", None)
            or getattr(cfg_module, "EVEN_TRACES_GEOMETRY", None)
        )
        odd_geom = (
            getattr(cfg_module, "ODD_TRACES_GEOM", None)
            or getattr(cfg_module, "ODD_TRACE_GEOM", None)
            or getattr(cfg_module, "ODD_TRACES_GEOMETRY", None)
        )
        if even_geom:
            vals.extend(["--even-geom", str(even_geom)])
        if odd_geom:
            vals.extend(["--odd-geom", str(odd_geom)])
        return vals

    if stage.key == "11b":
        return [
            "--in", str(cfg_module.STEP11_RADEC),
            "--out", str(cfg_module.STEP11_PHOTCAT),
        ]

    if stage.key == "11c":
        # Keep override support, but default to config-driven discovery if available.
        extract = (
            args.step11c_extract
            or str(getattr(cfg_module, "STEP11_INPUT_SPECTRA", ""))
            or str(getattr(cfg_module, "EXTRACT1D_TELLCOR", ""))
        )
        phot = (
            args.step11c_photcsv
            or str(getattr(cfg_module, "STEP11_PHOTCAT", ""))
            or str(getattr(cfg_module, "SKYMAPPER_CSV", ""))
            or str(getattr(cfg_module, "PHOTCSV", ""))
            or _pick_first_existing(
                Path(getattr(cfg_module, "ST11_FLUXCAL", "")) / "slit_trace_radec_skymapper_all.csv",
                Path(getattr(cfg_module, "ST11_FLUXCAL", "")) / "skymapper_photometry.csv",
                Path(getattr(cfg_module, "ST11_FLUXCAL", "")) / "skymapper.csv",
                Path(getattr(cfg_module, "ST11_FLUXCAL", "")) / "step11b_skymapper.csv",
            )
        )
        
        vals: list[str] = []
        if extract:
            vals.append(extract)
        if phot:
            vals.append(phot)
        return vals
    
    if stage.key == "12e":
        return [
            "--in", str(cfg_module.EXTRACT1D_FLUXCAL),
            "--master", str(cfg_module.STEP12D_MASTER_FITS),
            "--out", str(cfg_module.EXTRACT1D_FINALCAL),
        ]

    if stage.key == "12d":
        return [
            "--response-csv", str(cfg_module.STEP12_RESPONSE_CSV),
            "--spectra", str(cfg_module.EXTRACT1D_STEP12_INPUT),
            "--out-fits", str(cfg_module.STEP12D_MASTER_FITS),
            "--summary-csv", str(cfg_module.STEP12D_SUMMARY_CSV),
            "--metadata-json", str(cfg_module.STEP12D_METADATA_JSON),
            "--trust-min", "600.0",
            "--trust-max", "1000.0",
            "--min-i-coverage", "0.90",
        ]

    if stage.key == "12c":
        return [
            "--id-col", "slit",
            "--r-col", "r_mag",
            "--i-col", "i_mag",
            "--z-col", "z_mag",
            "--mode", "perstar",
            "--bandpass-mode", "edge_matched",
        ]

    if not stage.args_template:
        return []
    s = stage.args_template.format(set=set_name) if set_name else stage.args_template
    return shlex.split(s)

