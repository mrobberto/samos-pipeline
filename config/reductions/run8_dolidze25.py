from pathlib import Path
from config.pipeline_config import REPO_ROOT

# -----------------------------------------------------------------------------
# Reduction identity
# -----------------------------------------------------------------------------
NIGHT_ID = "20260113"
TARGET_NAME = "Dolidze25"
TARGET_FILE_STEM = "dolidze"

TOP_ROOT = Path(
    "/Users/robberto/Library/CloudStorage/Box-Box/My Documents - Massimo Robberto/@Massimo/_Science/2. Projects_HW/2017.SAMOS"
)

RUN_ROOT = TOP_ROOT / "_Run8_Science_2026_01"

PIPELINE_ROOT = REPO_ROOT
PRODUCT_ROOT = REPO_ROOT / "products" / "Run8_Dolidze25"

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
    "025.dolidze_oriented_biascorr_cr_rowcorr.fits",
    "026.dolidze_oriented_biascorr_cr_rowcorr.fits",
    "027.dolidze_oriented_biascorr_cr_rowcorr.fits",
]

ARC_FILES = [
    "044.arc_oriented_biascorr_cr_rowcorr.fits",
]

QUARTZ_FILES = [
    "029.quartz_oriented_biascorr_cr_rowcorr.fits",
    "030.quartz_oriented_biascorr_cr_rowcorr.fits",
    "031.quartz_oriented_biascorr_cr_rowcorr.fits",
    "032.quartz_oriented_biascorr_cr_rowcorr.fits",
    "033.quartz_oriented_biascorr_cr_rowcorr.fits",
    "034.quartz_oriented_biascorr_cr_rowcorr.fits",
    "038.quartz_oriented_biascorr_cr_rowcorr.fits",
    "039.quartz_oriented_biascorr_cr_rowcorr.fits",
    "040.quartz_oriented_biascorr_cr_rowcorr.fits",
    "041.quartz_oriented_biascorr_cr_rowcorr.fits",
    "042.quartz_oriented_biascorr_cr_rowcorr.fits",
    "043.quartz_oriented_biascorr_cr_rowcorr.fits",
    "050.quartz_oriented_biascorr_cr_rowcorr.fits",
    "051.quartz_oriented_biascorr_cr_rowcorr.fits",
    "052.quartz_oriented_biascorr_cr_rowcorr.fits",
    "053.quartz_oriented_biascorr_cr_rowcorr.fits",
]

#MASK_FILES = [
#    "054.mask_oriented_biascorr_cr_rowcorr.fits",
#    "055.mask_oriented_biascorr_cr_rowcorr.fits",
#    "056.mask_oriented_biascorr_cr_rowcorr.fits",
#    "057.mask_oriented_biascorr_cr_rowcorr.fits",
#]

ARC_SLITS_ON  = "044.arc_oriented_biascorr_cr_rowcorr.fits"
ARC_SLITS_OFF = "037.arc_oriented_biascorr_cr_rowcorr.fits"

QUARTZ_SLITS_OFF = "038.quartz_oriented_biascorr_cr_rowcorr.fits"
QUARTZ_SLITS_ON_EVEN = "043.quartz_oriented_biascorr_cr_rowcorr.fits"
QUARTZ_SLITS_ON_ODD = "042.quartz_oriented_biascorr_cr_rowcorr.fits"

# External astrometry/reference image
SISI_IMAGE_NAME = "Coadd_i_median_078-082_ff_flipx_wcs_manual.fits"

# Optional wavecal runtime knobs
WAVECAL_YWIN0 = 0
WAVECAL_FIRSTLEN = 4112

#to fix the wavelengths, edit and use this table...
WAVESHIFT_TABLE = "run8_dolidze25_manual_waveshifts.csv"

