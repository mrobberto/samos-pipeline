# step09_oh_refine

Files at 2026.04.04
step09_continuum_moving_population.py → continuum engine
step09_twopass_continuum_driver.py → experimental refinement driver
step09a_measure_oh_shifts.py → main OH shift stage
step09b_apply_oh_shifts.py → apply stage

## Purpose

Refine wavelength calibration using OH night-sky emission lines.

This step:

1. Measures per-slit wavelength zero-point shifts
2. Applies corrections to align spectra across slits

---

## Input

From Step08:

```python
```

File:

```text
```

---

## Processing

### Step09a — measure OH shifts

* Use SKY spectrum from each slit
* Select OH-rich wavelength windows (~780–930 nm)
* Interpolate onto common grid
* Remove continuum (high-pass filtering)
* Normalize via robust z-score
* Cross-correlate with reference slit
* Combine shifts across windows

Output:

```text
QC_OH_BG_registration.csv
```

---

### Step09b — apply shifts

For each slit:

```text
LAMBDA_NM_new = LAMBDA_NM_old + shift_nm
```

* Use per-slit shift if valid
* Otherwise fallback to median of good shifts

Outputs:

```text
extract1d_tellcorr_OHref.fits
```

---

## Output

Directory:

```python
config.ST10_OH
```

Products:

```text
QC_OH_BG_registration.csv
extract1d_tellcorr_OHref.fits
```

---

## Key concepts

### OH-based refinement

* OH lines provide precise wavelength reference
* independent of arc calibration
* corrects small zero-point errors

---

### Constant shift model

* assumes rigid offset per slit
* no change in dispersion solution

---

### Robust fitting

* multiple wavelength windows
* weighted median combination
* rejection of poor solutions

---
## Pipeline context

```text
Step09 → telluric correction
Step10 → wavelength refinement (OH)
Step11 → flux calibration
```

---

## Design choices

* use sky emission instead of arc lines
* per-slit correction
* robust statistics (median + MAD)
* fallback for bad slits

---

## Future improvements

* wavelength-dependent correction (if needed)
* 2D wavelength solution refinement
* adaptive window selection
