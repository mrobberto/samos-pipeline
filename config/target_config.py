from . import pipeline_config as pc
from .reductions import run8_dolidze25 as profile

cfg = pc.build_config(profile)

def ensure_directories():
    from config.pipeline_config import ensure_directories as _ensure
    _ensure([
        cfg.RAW_DIR,
        cfg.PREPROC_DIR,
        cfg.REDUCED_DIR,
        cfg.TABLES_DIR,
        cfg.TARGET_LOGDIR,
        cfg.ST00_ORIENT,
        cfg.ST01_BIAS,
        cfg.ST02_BIASCORR,
        cfg.ST03_CRCLEAN,
        cfg.ST03P5_ROWSTRIPE,
        cfg.ST04_TRACES,
        cfg.ST05_PIXFLAT,
        cfg.ST06_SCIENCE,
        cfg.ST07_WAVECAL,
        cfg.ST08_EXTRACT1D,
        cfg.ST09,
        cfg.ST10_TELLURIC,
        cfg.ST11_FLUXCAL,
        cfg.ST12_FINALCAL,
    ])
    
# Export profile + derived names for old scripts:
globals().update({
    k: v for k, v in vars(cfg).items()
    if not k.startswith("_")
})

# Also export generic constants/helpers:
globals().update({
    k: v for k, v in vars(pc).items()
    if not k.startswith("_")
})

