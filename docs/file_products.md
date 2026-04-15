# SAMOS Pipeline — File Products

This document defines the canonical data products of the SAMOS reduction
pipeline from Step06 through Step11.

It serves as the **data contract** for the pipeline:
- which files are produced
- where they live
- what they contain
- which step produces them

Only **stage-boundary products** are listed as canonical.
Intermediate files are documented but considered internal.

---

# 📂 Directory Structure

All reduced data are written under:


<run_root>/reduced/


Stage directories:

| Step | Directory |
|------|----------|
| Step06 | `06_science/` |
| Step07 | `07_wavecal/` |
| Step08 | `08_extract1d/` |
| Step09 | `09_telluric/` |
| Step10 | `10_oh/` |
| Step11 | `11_fluxcal/` |

---

# 🔹 Step06 — Science Products (TRACECOORDS)

### Outputs


06_science/


- `FinalScience_*_ADUperS.fits`
- `FinalScience_*_pixflatcorr_clipped_{EVEN|ODD}.fits`
- `FinalScience_*_tracecoords_{EVEN|ODD}.fits`

### Description

- Full-frame calibrated science image
- Pixel-flat corrected image
- Rectified slitlets (TRACECOORDS)

👉 TRACECOORDS is the reference geometry for all later steps

---

# 🔹 Step07 — Wavelength Calibration

### Outputs


07_wavecal/


- `arc_master.fits`
- `arc_master_wavesol.fits`
- `arc_wavesol_per_slit.fits`
- `arc_1d_wavelength_all.fits`

### Description

- Master arc reference
- Global wavelength solution (WVC*)
- Per-slit wavelength calibration
- Reference wavelength grid

---

# 🔹 Step08 — 1D Extraction

## Internal products (not canonical)


08_extract1d/


- `extract1d_optimal_ridge_even.fits`
- `extract1d_optimal_ridge_odd.fits`
- `extract1d_optimal_ridge_all.fits`

👉 These are intermediate and used only within Step08

---

## Canonical output


extract1d_optimal_ridge_all_wav.fits


### Contents (per slit)

- `YPIX`           — pixel row index
- `FLUX`           — extracted flux
- `VAR`            — variance
- `SKY`            — sky spectrum
- `LAMBDA_NM`      — wavelength (attached in Step08c)

### Description

- Fully extracted 1D spectra
- Pixel → wavelength mapping applied
- Still in **instrumental units**

---

# 🔹 Step09 — Telluric Correction

### Input


extract1d_optimal_ridge_all_wav.fits


### Outputs


09_telluric/


- `telluric_o2_template.fits`
- `extract1d_optimal_ridge_all_wav_tellcorr.fits`

### Contents (per slit)

- `FLUX_TELLCOR_O2`
- `VAR_TELLCOR_O2` (if variance present)

### Description

- Removes O₂ A and B band absorption
- Empirical template derived from data
- Wavelength unchanged

---

# 🔹 Step10 — OH Wavelength Refinement

### Input


extract1d_optimal_ridge_all_wav_tellcorr.fits


### Outputs


10_oh/


- `QC_OH_BG_registration.csv`
- `extract1d_optimal_ridge_all_wav_tellcorr_OHref.fits`

### Description

- Per-slit wavelength zero-point correction
- Based on OH sky emission lines
- Applies small additive shifts to wavelength grid

---

# 🔹 Step11 — Flux Calibration

## Step11a — RA/DEC extraction


slit_trace_radec_all.csv


## Step11b — Photometry


slit_trace_radec_skymapper_all.csv


## Step11c — Final calibrated spectra

### Inputs

- Step10 spectra
- SkyMapper photometry
- Optional throughput / color correction tables

### Outputs


11_fluxcal/


- `extract1d_fluxcal.fits`
- `step11_fluxcal_summary.csv`
- `step11_fluxcal_QA.png`

---

## Final science product


extract1d_fluxcal.fits


### Contents (per slit)

- `FLUX_FLAM`   — calibrated flux  
  [erg s⁻¹ cm⁻² Å⁻¹]

- `VAR_FLAM2`   — variance

- Intermediate columns:
  - `FLUX_TELLCOR_O2`
  - `FLUX_TPUTCOR`
  - `FLUX_COLORCOR`

---

# 🔁 Pipeline Product Flow


Step06 → TRACECOORDS slitlets

Step07 → wavelength solution

Step08 → extract1d_optimal_ridge_all_wav.fits

Step09 → extract1d_optimal_ridge_all_wav_tellcorr.fits

Step10 → extract1d_optimal_ridge_all_wav_tellcorr_OHref.fits

Step11 → extract1d_fluxcal.fits


---

# 🧠 Key Principles

### 1. Only stage outputs are canonical
Intermediate products are internal and may change.

---

### 2. One file per stage
Each step produces a single main product:
- simplifies debugging
- avoids ambiguity

---

### 3. Separation of corrections

| Step | Effect |
|------|-------|
| Step08 | extraction |
| Step09 | telluric (multiplicative) |
| Step10 | wavelength (additive) |
| Step11 | flux scale + continuum |

---

### 4. No step overlaps responsibility
Each step modifies only one aspect of the data.

---

# ✅ Final Product

After Step11:


extract1d_fluxcal.fits


Spectra are:

- wavelength calibrated  
- telluric corrected  
- instrument-response corrected  
- continuum corrected  
- flux calibrated  

👉 Ready for science analysis and publication