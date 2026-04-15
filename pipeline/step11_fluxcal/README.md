# step11_fluxcal

## Purpose

Perform photometric flux calibration of extracted spectra using broadband photometry.

This step:

1. Extracts slit sky coordinates (RA/DEC)
2. Queries external photometric catalogs (SkyMapper)
3. Applies instrument throughput correction
4. Applies global color-response correction
5. Anchors spectra to broadband photometry

The result is a set of **flux-calibrated 1D spectra** suitable for scientific analysis.

---

## Input

### Extracted spectra

From Step10:

```python
config.ST10_OH
```

File:

```text
extract1d_tellcorr_OHref.fits
```

These spectra are:

* extracted (Step08)
* telluric-corrected (Step09)
* wavelength-refined using OH lines (Step10)

---

## Processing

### Step11a — extract RA/DEC

Reads slit metadata from FITS headers and builds:

```text
slit_trace_RA_match_ALL.csv
```

Columns include:

* slit ID (`SLIT###`)
* RA, DEC (degrees)
* trace geometry (X0, XLO, XHI, YMIN)

---

### Step11b — query SkyMapper

Queries SkyMapper DR4 using slit positions.

Output:

```text
slit_trace_RA_match_skymapper_ALL.csv
```

Adds:

* `r_mag`, `i_mag`, `z_mag`
* `r_err`, `i_err`, `z_err`
* `match_sep_arcsec` (match distance)

---

### Step11c — flux calibration

For each slit:

#### 1. Throughput correction

```text
flux → flux / throughput(λ)
```

Removes instrument + telescope response.

---

#### 2. Color-response correction

```text
flux → flux / color_response(λ)
```

Corrects slope bias introduced by quartz flat-fielding.

---

#### 3. Synthetic photometry

Compute band-integrated fluxes:

```text
C_r, C_i, C_z
```

Compare with catalog fluxes:

```text
F_r, F_i, F_z
```

---

#### 4. Calibration fit

Two modes:

* **GRAY**

  ```text
  FLUX_FLAM = S × flux
  ```

* **TILT**

  ```text
  FLUX_FLAM = S × flux × (λ / λ₀)^α
  ```

---

#### 5. Residual diagnostics

```text
Δm = m_syn − m_cat
```

Used for QC and validation.

---

## Output

Directory:

```python
config.ST11_FLUXCAL
```

### Main products

```text
extract1d_fluxcal.fits
```

Each slit contains:

* `FLUX_FLAM` — calibrated flux
* `VAR_FLAM2` — calibrated variance

Intermediate columns (if enabled):

* `FLUX_TPUTCOR`
* `FLUX_COLORCOR`

---

### Summary table

```text
step11_fluxcal_summary.csv
```

Contains per-slit:

* calibration mode (GRAY/TILT)
* scale factor `S`
* slope `alpha`
* Δm residuals (r/i/z)
* QC flags

---

### QA plot

```text
step11_fluxcal_QA.png
```

Shows:

* Δm distributions (r, i, z)
* distribution of scale factors

---

## Key concepts

### Photometric anchoring

* Spectra are scaled to match broadband photometry
* Matching is done via **synthetic photometry**, not direct fitting

---

### Throughput correction

* Removes instrumental spectral response
* Based on telescope + instrument + detector

---

### Color correction

* Corrects continuum slope bias from quartz flat
* Derived globally across all slits

---

### Calibration modes

* **GRAY** → single scale factor
* **TILT** → scale + spectral slope

---

## Pipeline context

```text
Step08 → 1D extraction
Step09 → telluric correction
Step10 → wavelength refinement (OH)
Step11 → flux calibration
```

---

## Notes

* Broadband photometry defines the **continuum normalization and slope**
* Spectra are **not forced** to pass exactly through photometric points
* Residuals (Δm) provide calibration quality diagnostics
* Not all slits will be calibratable (missing photometry, low S/N)

---

## Design choices

* Empirical photometric anchoring instead of standard stars
* Separation of throughput and color corrections
* Use of synthetic photometry for consistency
* Robust handling of missing or poor-quality data

---

## Future improvements

* Joint calibration across multiple slits
* Inclusion of spectrophotometric standard stars
* Improved filter transmission modeling
* Per-slit quality weighting in global corrections

---

## Summary

Step11 produces final science-ready spectra by combining:

* instrumental corrections
* photometric anchoring
* robust calibration diagnostics

The output `extract1d_fluxcal.fits` is the **final product of the pipeline**.
