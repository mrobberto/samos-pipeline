# config/reductions/template_target_config.py
#
# PURPOSE
# -------
# Template for a target-specific reduction profile.
#
# Copy this file to something like:
#   run8_dolidze26.py
#
# Then edit only the target-specific values:
# - NIGHT_ID
# - TARGET_NAME
# - TARGET_FILE_STEM
# - SAMI_ROOT / RUN_ROOT
# - file lists and ON/OFF pairings
#
# IMPORTANT
# ---------
# Do NOT duplicate generic pipeline conventions here.
# Generic helpers, stage directory naming, and canonical output filenames live
# in config.pipeline_config.
#
# This file should only define:
# - target-specific roots
# - target-specific file selections
# - target-specific full paths obtained by combining local step directories with
#   generic canonical filenames from pipeline_config
#
from __future__ import annotations
from pathlib import Path
from config.pipeline_config import *

NIGHT_ID = "20260113"
TARGET_NAME = "TARGET_PLACEHOLDER"
TARGET_FILE_STEM = "targetstem"

SAMI_ROOT = Path("/path/to/run8_science_2026_01/SAMI").resolve()

NIGHT_ROOT = SAMI_ROOT / NIGHT_ID
TARGET_ROOT = SAMI_ROOT / TARGET_NAME

RAW_DIR = NIGHT_ROOT / "raw"
PREPROC_DIR = NIGHT_ROOT / "preprocessed"
REGIONS_DIR = NIGHT_ROOT / "regions"
NIGHT_LOGDIR = NIGHT_ROOT / "logs"

INPUT_DIR = TARGET_ROOT / "input"
REDUCED_DIR = TARGET_ROOT / "reduced"
TABLES_DIR = TARGET_ROOT / "tables"
TARGET_LOGDIR = TARGET_ROOT / "logs"

SCIENCE_FILES = []
ARC_FILES = []
QUARTZ_FILES = []

ARC_A = ""
ARC_B = ""
QUARTZ_A = ""
QUARTZ_B_EVEN = ""
QUARTZ_B_ODD = ""

# Example step directories
ST08_EXTRACT1D = reduced_step_dir(REDUCED_DIR, "08")
ST09_OH_REFINE = reduced_step_dir(REDUCED_DIR, "09")
ST10_TELLURIC = reduced_step_dir(REDUCED_DIR, "10")
ST11_FLUXCAL = reduced_step_dir(REDUCED_DIR, "11")

# Example canonical products using generic filenames from pipeline_config
EXTRACT1D_WAV = ST08_EXTRACT1D / NAME_EXTRACT1D_WAV
EXTRACT1D_OHCLEAN = ST09_OH_REFINE / NAME_EXTRACT1D_OHCLEAN
TELLURIC_TEMPLATE = ST10_TELLURIC / NAME_TELLURIC_TEMPLATE
EXTRACT1D_TELLCOR = ST10_TELLURIC / NAME_EXTRACT1D_TELLCOR
EXTRACT1D_FLUXCAL = ST11_FLUXCAL / NAME_EXTRACT1D_FLUXCAL