#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Declarative execution registry for the SAMI/SAMOS spectroscopic pipeline.

This module contains pipeline knowledge only: stage order, QC companions,
and lightweight output contracts. The driver imports these definitions and
keeps the execution logic separate from the pipeline table of contents.
"""
from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class Stage:
    """One pipeline stage entry used by the master driver registry."""
    key: str
    script: str
    description: str
    sets: tuple[str, ...] = ()
    args_template: str = ""

    @property
    def is_set_based(self) -> bool:
        return len(self.sets) > 0
    

# -----------------------------------------------------------------------------
# Stage registry
# -----------------------------------------------------------------------------
# The order here is the authoritative pipeline execution order.
#
# IMPORTANT:
# - Step09 is a single ABAB OH-clean stage.
# - The historical 09a/09b/09bm/09c subdivision is no longer operational.
# - Step11c should point to the currently adopted production script.
# -----------------------------------------------------------------------------
SCRIPT_REGISTRY: tuple[Stage, ...] = (
    Stage("00",   "pipeline/step00_orient/step00_rotate.py",                        "Rotate frames to the standard orientation"),
    Stage("01",   "pipeline/step01_bias/step01_masterbias.py",                      "Build master bias"),
    Stage("02",   "pipeline/step02_biascorr/step02_biascorr.py",                    "Apply bias correction"),
    Stage("03",   "pipeline/step03_crclean/step03_crclean.py",                      "Cosmic-ray cleaning"),
    Stage("03.5", "pipeline/step03p5_rowstripe/step03p5_remove_rowstripe.py",       "Remove row-wise striping and match quadrant pedestals"),
    Stage("04",   "pipeline/step04_traces/step04_make_traces.py",                   "Build slit traces and geometry",                  sets=("EVEN", "ODD"), args_template="--set {set}"),
    Stage("05",   "pipeline/step05_pixflat/step05_build_pixflat.py",                "Build pixel flat from quartz differences",        sets=("EVEN", "ODD"), args_template="--set {set}"),
    Stage("06a",  "pipeline/step06_science_rectify/step06a_make_final_science.py",  "Build FinalScience mosaic"),
    Stage("06b",  "pipeline/step06_science_rectify/step06b_apply_pixflat_clip.py",  "Apply pixel flat to FinalScience",                sets=("EVEN", "ODD"), args_template="--traceset {set}"),
    Stage("06c",  "pipeline/step06_science_rectify/step06c_TRACECOORDS_representation.py", "Rectify slitlets into TRACECOORDS",        sets=("EVEN", "ODD"), args_template="--traceset {set}"),
    Stage("07a",  "pipeline/step07_wavecal/step07a_make_arc_diff.py",               "Build arc-difference frame"),
    Stage("07b",  "pipeline/step07_wavecal/step07b_apply_pixflat_arc.py",           "Apply pixel flat to arc frame"),
    Stage("07c",  "pipeline/step07_wavecal/step07c_extract_arc_1d.py",              "Extract rectified 1D arc slit spectra",           sets=("EVEN", "ODD"), args_template="--traceset {set}"),
    Stage("07d",  "pipeline/step07_wavecal/step07d_find_line_shifts.py",            "Measure initial relative arc shifts",             sets=("EVEN", "ODD"), args_template="--traceset {set}"),
    Stage("07e",  "pipeline/step07_wavecal/step07e_refine_stack_arc.py",            "Refine arc-stack alignment",                      sets=("EVEN", "ODD"), args_template="--set {set}"),
    Stage("07f",  "pipeline/step07_wavecal/step07f_build_master_arc.py",            "Build aligned master arc"),
    Stage("07g",  "pipeline/step07_wavecal/step07g_solve_wavelength.py",            "Fit global wavelength solution"),
    Stage("07h",  "pipeline/step07_wavecal/step07h_propagate_wavesol.py",           "Propagate wavelength solution to all slit arcs"),
    Stage("07i",   "pipeline/step07_wavecal/step07i_apply_manual_waveshifts.py",    "Apply optional manual slit wavelength zero-point shifts" ),
    Stage("08a1", "pipeline/step08_extract1d/step08a1_trace_analysis.py",
          "Trace analysis, ridge detection, and slit classification",
          sets=("EVEN", "ODD"), args_template="--set {set}"),
    
    Stage("08a2", "pipeline/step08_extract1d/step08a2_extract_1d.py",
          "Ridge-guided optimal extraction",
          sets=("EVEN", "ODD"), args_template="--set {set}"),
    Stage("08b",  "pipeline/step08_extract1d/step08b_merge_even_odd.py",            "Merge EVEN and ODD extracted spectra"),
    Stage("08c",  "pipeline/step08_extract1d/step08c_attach_wavelength.py",         "Attach wavelength vectors to extracted spectra"),
    Stage("09",   "pipeline/step09_oh_refine/step09_abab_driver.py",                "Full OH cleanup and preferred-spectrum selection (A/B/A/B)"),
    Stage("10a",  "pipeline/step10_telluric/step10a_build_telluric_template.py",    "Build empirical O2 telluric template"),
    Stage("10b",  "pipeline/step10_telluric/step10b_apply_telluric.py",             "Apply O2 telluric correction"),
    Stage("11a",  "pipeline/step11_fluxcal/step11a_extract_header_radec_resilient.py", "Extract RA/DEC and slit metadata"),
    Stage("11b",  "pipeline/step11_fluxcal/step11b_query_skymapper.py",             "Query SkyMapper photometry"),
    Stage("11c",  "pipeline/step11_fluxcal/step11c_fluxcal.py",                     "Apply photometric flux calibration"),
#    Stage("12a", "pipeline/step12_finalcal/step12a_build_illum_profile.py",         "Build 1D illumination profiles"),
#    Stage("12b", "pipeline/step12_finalcal/step12b_apply_illum_profile.py",         "Apply 1D illumination correction"),
#
    Stage("12d", "pipeline/step12_finalcal/step12d_build_stellar_response.py",      "Build ensemble stellar-response correction"),
    Stage("12e", "pipeline/step12_finalcal/step12e_apply_stellar_response.py",      "Apply ensemble stellar-response correction"),
)

# -----------------------------------------------------------------------------
# QC registry
# -----------------------------------------------------------------------------
# These are the preferred QC companions for the current operational pipeline.
# Keep this list aligned with the canonical QC scripts actually used in the
# notebooks and science validation workflow.
# -----------------------------------------------------------------------------   
QC_REGISTRY: dict[str, tuple[str, ...]] = {
    "04":  ("qc/step04/qc_step04_trace_quicklooks.py",),
    "05":  ("qc/step05/qc_step05_pixflat.py",),
    "06a": ("qc/step06/qc_step06a_mosaic_final.py",),
    "06b": ("qc/step06/qc_step06b_inspector_final.py",),
    "06c": ("qc/step06/qc_step06c_quicklooks_final.py",),
    "07g": ("qc/step07/qc07g_inspect_wavelength_solution.py",),
    "07h": ("qc/step07/qc07h_arc_wavelength_products.py",),
    "08a": ("qc/step08/qc_step08_extract.py",),
    "08c": ("qc/step08/qc_step08c_wavelength_alignment.py",),
    "09":  ("qc/step09/qc_step09_preferred_all_slits.py", "qc/step09/qc_step09_final_mosaic.py"),
    "10b": ("qc/step10/qc_step10_final_mosaic.py",),
    "11c": ("qc/step11/qc_step11_grid_patched_v2.py", "qc/step11/qc_step11_summary_b.py"),
}

# -----------------------------------------------------------------------------
# Output contract checks
# -----------------------------------------------------------------------------
# These are lightweight post-stage assertions using canonical variables from the
# active target config. They are intended to catch broken file flow early.
# -----------------------------------------------------------------------------
OUTPUT_CHECKS: dict[str, tuple[str, ...]] = {
    "06c": ("SCI_EVEN_TRACECOORDS", "SCI_ODD_TRACECOORDS"),
    "07a": ("MASTER_ARC_DIFF",),
    "07g": ("WAVESOL_ALL_FITS",),
    "07h": ("ARC_1D_WAVELENGTH_ALL",),
    "07i": ("ARC_WAVELENGTH_ACTIVE",),
    "08a2": ("EXTRACT1D_EVEN", "EXTRACT1D_ODD"),
    "08b": ("EXTRACT1D_ALL",),
    "08c": ("EXTRACT1D_WAV",),
    "09":  ("EXTRACT1D_OHCLEAN",),
    "10a": ("TELLURIC_TEMPLATE",),
    "10b": ("EXTRACT1D_TELLCOR",),
    "11c": ("EXTRACT1D_FLUXCAL", "FLUXCAL_SUMMARY_CSV"),
#    "12a": ("ILLUM1D_PROFILE_EVEN", "ILLUM1D_PROFILE_ODD"),
#    "12b": ("EXTRACT1D_ILLUMCORR",),
#    "12c": ("EXTRACT1D_FINALCAL", "STEP12C_SUMMARY_CSV"),
    "12d": ("STEP12D_MASTER_FITS", "STEP12D_SUMMARY_CSV"),
    "12e": ("QC_STEP12D_RESPONSE_PDF",),
}

