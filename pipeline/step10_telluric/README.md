# step09_telluric

## Purpose

Build and apply an empirical telluric correction for atmospheric O2 absorption.

This step:

1. Builds empirical templates for the O2 **B band** and **A band**
2. Fits each slit independently
3. Allows the two bands to have different depth and wavelength shift
4. Produces telluric-corrected 1D spectra

---

## Input

From Step08:

```python
config.ST08_EXTRACT1D
```

Canonical input:

```text
Extract1D_optimal_ridgeguided_POOLSKY_ALL_WAV_OHref.fits
```

This file contains:

* extracted 1D spectra
* `LAMBDA_NM`
* wavelength-refined solution
* flux and variance columns

---

## Processing

### Step09a — build telluric template

For each candidate slit:

* select spectra with valid wavelength coverage
* normalize continuum locally in each band
* build B-band and A-band vectors independently
* align contributors separately for:

  * B band: ~682–692 nm
  * A band: ~752.5–768.5 nm
* robustly median-stack accepted contributors

Outputs:

* transmission template
* optical-depth template

The B and A bands are treated independently because:

* wavelength registration may differ slightly
* absorption depth may not scale identically

---

### Step09b — apply telluric correction

For each slit:

* read `LAMBDA_NM`
* select science flux column
* normalize locally around each telluric band
* fit A band and B band independently:

  * separate amplitude
  * separate wavelength shift
* use weighted least squares emphasizing stronger absorption
* build a piecewise transmission correction
* divide flux by the derived telluric transmission

If variance is available, propagate:

```text
VAR_TELLCOR_O2 = VAR / T^2
```

---

## Output

Directory:

```python
config.ST09_TELLURIC
```

Recommended canonical products:

```text
telluric_template.fits
extract1d_tellcorr.fits
```

---

## Template contents

### `telluric_template.fits`

Extensions:

```text
O2_BAND
O2_ABAND
```

Columns:

* `LAMBDA_NM`
* `T_MED`
* `TAU_O2`

---

## Corrected spectrum contents

### `extract1d_tellcorr.fits`

Each slit extension includes the original extraction columns plus:

* `FLUX_TELLCOR_O2`
* `VAR_TELLCOR_O2` (if variance exists)

Header keywords include:

* `TELL_OK`
* `TELL_OKA`
* `TELL_OKB`
* `TELL_SHA`
* `TELL_SHB`
* `TELL_AA`
* `TELL_AB`
* `TELLBAND`

---

## How to run

```python
runfile("step09_telluric/step09a_build_telluric_template.py")
runfile("step09_telluric/step09b_apply_telluric.py")
```

---

## Notes

* The correction is empirical, built from the data themselves
* A-band and B-band are intentionally decoupled
* This avoids residuals caused by a single shared shift or amplitude
* Spectra with weak or unusable telluric information are passed through safely

---

## Pipeline context

```text
Step08 → extract 1D spectra
Step09 → telluric correction
Step10 → OH wavelength refinement
Step11 → flux calibration
```

---

## Design choices

* empirical template instead of external atmosphere model
* independent A/B fitting
* weighted fitting to prioritize real absorption cores
* conservative pass-through when no reliable fit is possible

---

## Future considerations

* optional joint fit with regularization between A and B
* slit quality ranking for template contributors
* extension to H2O or other telluric bands
