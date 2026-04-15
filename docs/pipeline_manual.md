 # SAMOS Spectroscopic Pipeline Manual

## Overview

The SAMOS pipeline reduces multi-slit spectroscopic data from raw calibration
frames to fully calibrated, science-ready 1D spectra.

The pipeline is organized as a sequence of modular steps:

| Step | Function |
|------|---------|
| Step04 | Slit geometry reconstruction |
| Step05 | Pixel flat |
| Step06 | Science preparation and rectification |
| Step07 | Wavelength calibration |
| Step08 | Optimal extraction |
| Step09 | Telluric correction |
| Step10 | Wavelength refinement (OH lines) |
| Step11 | Flux calibration |

Each step performs a **single, well-defined transformation** and does not
overlap with others.

---

# Step04 — Slit Geometry

## Purpose

Determine the geometric footprint of each slit on the detector using quartz
lamp exposures.

This defines the reference coordinate system used throughout the pipeline.

## Method

1. Quartz difference image:

Quartz_B - Quartz_A


2. Detect slit centers from collapsed X profile

3. Perform row-by-row segmentation:
- local background estimation
- thresholding
- contiguous region selection

4. Assign slit IDs

5. Fit geometry:
- center trace: low-order polynomial
- width model: smooth, low-order
- edges reconstructed from center + width

## Outputs

- `*_geometry.fits`
- `*_mask.fits`
- `*_slitid.fits`
- `*_slit_table.csv`

## Design principles

- robust to noise and cosmic rays
- smooth, low-order geometry
- physically consistent slit edges

---

# Step05 — Pixel Flat

## Purpose

Correct pixel-to-pixel detector sensitivity variations.

## Method

1. Quartz difference image:

Quartz_B - Quartz_A


2. Normalize to unity

3. Clip extreme values

## Output

- `PixelFlat_from_quartz_diff.fits`

## Notes

- Flat is applied in Step06
- Should not introduce large-scale gradients

---

# Step06 — Science Preparation

## Purpose

Prepare science frames and transform slitlets into TRACECOORDS.

## Substeps

### Step06.0 — calibration
- bias-corrected, CR-cleaned
- converted to ADU/s

### Step06b — flat-fielding
- divide by pixel flat

### Step06c — rectification

Transform slitlets into TRACECOORDS:

- Y axis → dispersion direction
- X axis → spatial direction
- preserve curvature

## Outputs

- `*_tracecoords_EVEN.fits`
- `*_tracecoords_ODD.fits`

## Key principle

TRACECOORDS preserves detector geometry while isolating slitlets.

---

# Step07 — Wavelength Calibration

## Purpose

Map detector Y coordinate to physical wavelength.

## Method

1. Extract arc spectra per slit
2. Measure slit-to-slit shifts
3. Build master arc
4. Fit global polynomial:

λ = f(Y)

5. Propagate solution to all slits

## Outputs

- `arc_master.fits`
- `arc_master_wavesol.fits`
- `arc_wavesol_per_slit.fits`

## Notes

- wavelength increases bottom → top
- polynomial is authoritative (WVC*)

---

# Step08 — Optimal Extraction

## Purpose

Extract 1D spectra from TRACECOORDS slit images.

## Substeps

### Step08a — extraction

#### Ridge tracking
- center seeded from bright region
- tracked row-by-row with corridor constraint
- smoothed with polynomial

#### Sky subtraction
- row-based sky preferred (preserves OH lines)
- one-sided sky to avoid contamination
- fallback to pooled sky

#### Optimal extraction
- Gaussian profile weighting
- variance-aware extraction

#### Aperture correction
- correct partial truncation
- flag edge losses

### Step08b — merge
Combine EVEN and ODD slit sets

### Step08c — wavelength attach

Use Step07 solution:


YDET = Y0DET + YPIX
λ = f(YDET + SHIFT_TO_MASTER)


## Output


extract1d_optimal_ridge_all_wav.fits


## Notes

- still in instrumental units
- no telluric or flux correction yet

---

# Step09 — Telluric Correction

## Purpose

Remove atmospheric O₂ absorption bands.

## Substeps

### Step09a — template construction

1. select good slits
2. continuum normalize locally
3. align spectra per band
4. median stack
5. convert to optical depth

Bands:
- B (~687 nm)
- A (~760 nm)

### Step09b — application

Fit model in log space:


log(F) = continuum - a * τ(λ - shift)


Apply:


F_corrected = F / T_model


## Output


extract1d_optimal_ridge_all_wav_tellcorr.fits


## Notes

- multiplicative correction
- wavelength unchanged

---

# Step10 — OH Wavelength Refinement

## Purpose

Correct residual wavelength zero-point errors using OH sky lines.

## Substeps

### Step10a — measure shifts

1. select OH windows
2. high-pass filter
3. cross-correlate with reference slit
4. combine shifts across windows

### Step10b — apply shifts


λ_corrected = λ + shift_nm


- use per-slit shift if valid
- otherwise fallback to median

## Output


extract1d_optimal_ridge_all_wav_tellcorr_OHref.fits


## Notes

- additive correction
- typically small (~0.1–0.3 nm)

---

# Step11 — Flux Calibration

## Purpose

Convert spectra to physical flux units.

## Substeps

### Step11a — RA/DEC extraction
Build slit coordinate table

### Step11b — photometry
Query SkyMapper (r, i, z)

### Step11c — calibration

1. optional throughput correction
2. optional color correction
3. synthetic photometry
4. compute scale factors

Two modes:

#### GRAY
- single scale factor

#### TILT
- scale + spectral slope

Apply:


Fλ = S × counts × corrections


## Output


extract1d_fluxcal.fits


## Final product

Spectra are:

- wavelength calibrated
- telluric corrected
- OH refined
- flux calibrated

---

# Pipeline Design Principles

## 1. Separation of concerns

| Step | Effect |
|------|-------|
| Step08 | extraction |
| Step09 | telluric |
| Step10 | wavelength |
| Step11 | flux |

---

## 2. No resampling

- extraction preserves sampling
- wavelength corrections are additive
- flux calibration is multiplicative

---

## 3. Robustness

- fallback strategies at every step
- NaN-safe operations
- per-slit quality flags

---

## 4. Trace-driven extraction

- ridge defines object position
- quartz geometry defines bounds

---

## Final Output


extract1d_fluxcal.fits