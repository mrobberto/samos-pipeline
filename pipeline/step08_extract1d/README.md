# step08_extract1d

## Purpose

Perform ridge-guided 1D extraction of spectra from TRACECOORDS slit images.

Step08 is organized into modular sub-steps:

08a1 → trace analysis (ridge detection + slit classification)  
08a2 → optimal extraction (flux measurement)  
08b  → merge EVEN/ODD  
08c  → attach wavelength solution  

---

## Input

From Step06:

    config.ST06_SCIENCE

Files:

    *_tracecoords.fits

Each extension contains a rectified slit image in TRACECOORDS.

---

## Step08a1 — Trace analysis

Determines where the object is in each slit.

### Operations

- Detect seed location using:
  - block-based method
  - patch-based method (preferred)
- Build ridge X0(y) from seed using constrained tracking
- Compute slit metrics (FWHM, centering, edge proximity)
- Classify slits (GOOD / USE)

### Output

    trace_analysis_optimal_ridge_{even,odd}.fits
    step08a_slit_quality_{even,odd}.csv

Notes:
- FITS contains ridge and geometry only
- CSV provides editable control of extraction

---

## Step08a2 — Optimal extraction

Performs flux extraction using ridge from Step08a1.

### Key principle

Ridge is fixed — extraction does not modify geometry

### Method

#### Sky estimation
- row-by-row sky
- fallback: pooled sky (±YSKYWIN)
- continuity fallback

#### Extraction

Horne-style optimal extraction:

    flux = Σ [P * (D - sky) / V] / Σ [P² / V]

#### Aperture correction
- estimate PSF truncation
- apply correction when safe

### Output

    extract1d_optimal_ridge_{even,odd}.fits

Each slit contains:

- YPIX         : row index  
- FLUX         : extracted flux  
- VAR          : variance  
- SKY          : sky estimate  
- OBJ_PRESKY   : pre-sky aperture sum  
- X0           : ridge position  
- NOBJ         : object pixels  
- NSKY         : sky pixels  
- SKYSIG       : sky noise  
- APLOSS_FRAC  : PSF fraction recovered  
- FLUX_APCORR  : corrected flux  
- VAR_APCORR   : corrected variance  
- EDGEFLAG     : truncation flag  

---

## Step08b — Merge EVEN/ODD

Combines parity products into a single file.

Output:

    extract1d_optimal_ridge_all.fits

- preserves SLIT### identifiers  
- no modification of spectra  

---

## Step08c — Attach wavelength

Applies Step07 wavelength solution:

    λ(y) = polynomial( YDET - YWIN0 + SHIFT_TO_MASTER )

Output:

    extract1d_optimal_ridge_all_wav.fits

Adds:

    LAMBDA_NM

---

## Key concepts

### TRACECOORDS

- Y = dispersion axis  
- X = spatial axis  
- geometry flattened from detector frame  

---

### Ridge-guided extraction

- ridge follows science signal (not quartz geometry)  
- allows curvature and flexure  
- prevents contamination from neighboring slits  

---

### Decoupled architecture

Detection (08a1) ≠ Extraction (08a2)

Advantages:
- robust trace identification  
- explicit QC via CSV  
- manual override capability  
- improved stability in crowded fields  

---

## Pipeline context

Step06 → TRACECOORDS images  
Step08 → extraction  
Step09 → sky/telluric refinement  
Step10 → telluric correction  
Step11 → flux calibration  

---

## Summary

Step08 implements a modular ridge-guided extraction that:

- separates geometry from photometry  
- uses robust seed detection (patch-based)  
- preserves sky structure  
- corrects aperture losses  
- produces science-ready spectra  