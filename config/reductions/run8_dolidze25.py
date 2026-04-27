from pathlib import Path

# -----------------------------------------------------------------------------
# Reduction identity
# -----------------------------------------------------------------------------
NIGHT_ID = "20260113"
TARGET_NAME = "Dolidze25"
TARGET_FILE_STEM = "dolidze"

RUN_ROOT = Path(
    "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS/_Run8_Science_2026_01"
).resolve()

# -----------------------------------------------------------------------------
# Instrument / detector constants, only if target/run dependent
# -----------------------------------------------------------------------------
GAIN_E_PER_ADU = 2.1
READNOISE_E = 3.8

# -----------------------------------------------------------------------------
# Target-specific external inputs
# -----------------------------------------------------------------------------
BIAS_FILES = [f"{i:03d}.bias.fits" for i in range(1, 21)]

SCIENCE_FILES = [
    "021.dolidze_biascorr_cr_rowcorr.fits",
    "022.dolidze_biascorr_cr_rowcorr.fits",
    "023.dolidze_biascorr_cr_rowcorr.fits",
    "024.dolidze_biascorr_cr_rowcorr.fits",
]

ARC_FILES = [
    "035.arc_biascorr_cr_rowcorr.fits",
    "036.arc_biascorr_cr_rowcorr.fits",
    "037.arc_biascorr_cr_rowcorr.fits",
    "044.arc_biascorr_cr_rowcorr.fits",
    "048.arc_biascorr_cr_rowcorr.fits",
    "049.arc_biascorr_cr_rowcorr.fits",
]

QUARTZ_FILES = [
    "029.quartz_biascorr_cr_rowcorr.fits",
    "030.quartz_biascorr_cr_rowcorr.fits",
    "031.quartz_biascorr_cr_rowcorr.fits",
    "032.quartz_biascorr_cr_rowcorr.fits",
    "033.quartz_biascorr_cr_rowcorr.fits",
    "034.quartz_biascorr_cr_rowcorr.fits",
    "038.quartz_biascorr_cr_rowcorr.fits",
    "039.quartz_biascorr_cr_rowcorr.fits",
    "040.quartz_biascorr_cr_rowcorr.fits",
    "041.quartz_biascorr_cr_rowcorr.fits",
    "042.quartz_biascorr_cr_rowcorr.fits",
    "043.quartz_biascorr_cr_rowcorr.fits",
    "050.quartz_biascorr_cr_rowcorr.fits",
    "051.quartz_biascorr_cr_rowcorr.fits",
    "052.quartz_biascorr_cr_rowcorr.fits",
    "053.quartz_biascorr_cr_rowcorr.fits",
]

MASK_FILES = [
    "054.mask_biascorr_cr_rowcorr.fits",
    "055.mask_biascorr_cr_rowcorr.fits",
    "056.mask_biascorr_cr_rowcorr.fits",
    "057.mask_biascorr_cr_rowcorr.fits",
]

ARC_SLITS_ON = "036.arc_biascorr_cr_rowcorr.fits"
ARC_SLITS_OFF = "037.arc_biascorr_cr_rowcorr.fits"

QUARTZ_SLITS_OFF = "038.quartz_biascorr_cr_rowcorr.fits"
QUARTZ_SLITS_ON_EVEN = "039.quartz_biascorr_cr_rowcorr.fits"
QUARTZ_SLITS_ON_ODD = "040.quartz_biascorr_cr_rowcorr.fits"

# External astrometry/reference image
SISI_IMAGE_NAME = "Coadd_i_median_078-082_ff_flipx_wcs_manual.fits"

# Optional wavecal runtime knobs
WAVECAL_YWIN0 = 0
WAVECAL_FIRSTLEN = 4112