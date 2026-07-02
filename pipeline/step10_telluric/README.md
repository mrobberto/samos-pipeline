# Step10 — Telluric correction (O₂ A and B bands)

## Purpose

Apply an **empirical telluric correction** to remove atmospheric O₂ absorption
features from the extracted 1D spectra.

This step:

1. Builds empirical templates for the O₂ **B band** and **A band**
2. Fits each slit independently
3. Allows the two bands to have **independent depth and wavelength shift**
4. Produces telluric-corrected 1D spectra

---

## Input

From **Step09 (OH refinement + consensus)**:

**extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits**


This file contains:

- wavelength-calibrated 1D spectra (`LAMBDA_NM`)
- sky-subtracted spectra (`STELLAR_CONSENSUS`)
- residuals and diagnostic columns

---

## Processing

### Step10a — build telluric template

An empirical telluric template is constructed directly from the data.

For a subset of suitable slits:

- select spectra with valid wavelength coverage and sufficient S/N
- normalize the continuum locally around each band
- extract wavelength windows:

  - **B band**: ~682–692 nm  
  - **A band**: ~752.5–768.5 nm  

- align spectra using cross-correlation
- robustly median-stack aligned spectra

Outputs:

- transmission template (`T_MED`)
- optical depth (`TAU_O2 = -ln T`)

The A and B bands are treated independently because:

- small wavelength shifts may differ between bands  
- absorption depth may not scale identically  

---

### Step10b — apply telluric correction

For each slit:

- read `LAMBDA_NM`
- select science spectrum (default: `STELLAR_CONSENSUS`)
- normalize locally around each band
- fit A and B bands independently:

  - separate amplitude (optical depth scaling)  
  - separate wavelength shift  

- use weighted least squares emphasizing strong absorption cores
- build a **piecewise transmission model**
- divide the spectrum by the transmission:

**FLUX_TELLCOR_O2 = FLUX / T**


If variance is available:

**VAR_TELLCOR_O2 = VAR / T^2**


Spectra without reliable telluric fits are passed through unchanged.

---

## Output

Directory:

**config.ST10_TELLURIC**


Canonical products:

**telluric_O2_template.fits**
**extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus_tellcorr.fits**


---

## Template contents

### `telluric_O2_template.fits`

Extensions:

**O2_BAND**
**O2_ABAND**


Columns:

- `LAMBDA_NM`
- `T_MED`   — median transmission
- `TAU_O2`  — optical depth

---

## Corrected spectrum contents

Each slit extension includes original Step09 columns plus:

- `FLUX_TELLCOR_O2`
- `VAR_TELLCOR_O2` (if available)

Header keywords include:

- `TELL_OK`   — overall success flag  
- `TELL_OKA`  — A-band fit success  
- `TELL_OKB`  — B-band fit success  
- `TELL_SHA` / `TELL_SHB` — wavelength shifts  
- `TELL_AA` / `TELL_AB` — amplitudes  
- `TELLBAND` — bands used (A, B, or both)  

---

## How to run

```bash
python step10a_build_telluric_template.py
python step10b_apply_telluric.py
