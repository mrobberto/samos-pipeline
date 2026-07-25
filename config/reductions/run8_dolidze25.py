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
# Dataset parameters
# -----------------------------------------------------------------------------
STEP04_ACTIVE_FRAC = 0.03
STEP04_ACTIVE_PAD = 20
STEP04_PROFILE_SMOOTH = 3.0
STEP04_MIN_PEAK_DIST = 18
STEP04_PEAK_PROMINENCE = 0.15
STEP04_PEAK_HEIGHT_FRAC = 0.10
STEP04_HALF_WINDOW = 18
STEP04_SIDEBAND = 6
STEP04_LOCAL_NSIG = 5.0
STEP04_MAX_WIDTH = 13
STEP04_EDGE_SHRINK = 1
STEP04_TRACE_CENTER_HW = 10
STEP04_TRACE_CENTER_JUMP = 2.0
STEP04_TRACE_CENTER_SMOOTH = 31

STEP05_SIGMA_Y = 75.0
STEP05_SIGMA_X = 12.0
STEP05_MASK_EROSION_ITERS = 2
STEP05_CLIP_LO = 0.5
STEP05_CLIP_HI = 2.0

# -----------------------------------------------------------------------------
# Step06: science combination, flat-fielding, and TRACECOORDS
# -----------------------------------------------------------------------------

# Step06a — science combination
STEP06A_EXPTIME_KEY = "EXPTIME"
STEP06A_NORMALIZE_TO_RATE = True
STEP06A_SIGMA_CLIP = True
STEP06A_SIGMA = 3.0
STEP06A_MAXITERS = 5

# Step06b — pixel-flat application
STEP06B_CLIP_LO = 0.70
STEP06B_CLIP_HI = 1.30
STEP06B_REGISTER_FLAT = False

# Step06c — TRACECOORDS
# Defined now; we will wire these into 06c in the next pass.
STEP06C_PADX = 7
STEP06C_INTERP_ORDER = 1
STEP06C_WIDTH_KEY = "width_med"
STEP06C_PAD_PIX = 0.5
STEP06C_MIN_MASK_PIX_PER_ROW = 10


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

