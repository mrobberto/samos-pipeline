# Step 11 — Flux Calibration

This step performs the **absolute flux calibration** of the extracted 1D spectra and a subsequent **empirical refinement** using external photometry (SkyMapper).

---

## Overview

### Step11a — Slit → Sky coordinates extraction

Script:
`step11a_extract_header_radec_resilient.py`

* Extracts RA/DEC for each slit from the science products
* Produces:

  * `slit_trace_radec_all.csv`

---

### Step11b — Photometric crossmatch (SkyMapper)

Script:
`step11b_query_skymapper.py`

* Queries SkyMapper for matched sources
* Produces:

  * `slit_trace_radec_skymapper_all.csv`

---

### Step11c — First-pass absolute flux calibration

Script:
`step11c_fluxcal.py`

* Converts extracted spectra into **physical flux-density units**
* Anchors spectra to SkyMapper photometry (r/i/z bands)

Output:

* `Extract1D_fluxcal.fits`

Key columns:

* `FLUX_FLAM` — flux (erg s⁻¹ cm⁻² Å⁻¹)
* `VAR_FLAM2` — propagated variance

This step establishes the **absolute photometric scale**.

---

### Step11d — Empirical refinement

Script:
`step11d_refine_fluxcal.py`

Applies a smooth multiplicative correction:

[
F_{\lambda,\mathrm{refined}} = R_{\mathrm{11d}}(\lambda),F_{\lambda,\mathrm{11c}}
]

#### Method

* A **quadratic response function** is fit per slit using synthetic photometry
* Two bandpass modes are supported:

**full**

* Uses standard r/i/z bandpasses

**edge_matched**

* Uses truncated bandpasses to reduce edge biases:

  * r_short: **λ ≥ 600 nm**
  * i: unchanged
  * z_short: **λ ≤ 930 nm**

* Synthetic magnitudes for `r_short` and `z_short` are derived from the full r/i/z photometry

#### Outputs

* `Extract1D_fluxcal_refined_perstar.fits`
* `Extract1D_fluxcal_step11d_summary.csv`
* `Extract1D_fluxcal_step11d_debug.csv`
* `Extract1D_fluxcal_step11d_metadata.json`

Key columns:

* `RESP_STEP11D` — multiplicative response
* `FLUX_FLAM_REFINED` — refined spectrum

---

## Philosophy

* **Step11c** sets the absolute scale
* **Step11d** applies a smooth broadband correction

The refinement:

* is driven by **integrated band fluxes**
* corrects large-scale response mismatches (e.g. grating edges)
* preserves spectral features
* remains close to unity unless required by photometry

---

## Expected behavior

* `FLUX_FLAM` and `FLUX_FLAM_REFINED` are similar in scale
* `RESP_STEP11D` is smooth and positive
* Typical response variation:

  * ~0.5–3 across wavelength range
* Refined spectra reproduce SkyMapper photometry in synthetic band flux

---

## Validation

The Step11d implementation has been validated against a reference single-slit solver:

* `dev/1slittester.py`

This script reproduces the per-slit solution and is used for debugging and verification.

---

## Additional tools

### Signal-to-noise estimation

Script:
`step11c_part2_continuum_snr.py`

* Computes continuum signal-to-noise per slit
* Useful for:

  * assessing spectral quality
  * selecting reliable calibration stars

Often used together with:

* `step11c_part3_rank_calibrators.py`

These tools are diagnostic and not part of the core pipeline execution.

---

## Notes

* Inputs:

  * Step08 (extraction)
  * Step10 (telluric correction)

* All scripts are designed to run from the repository root:

```bash
PYTHONPATH=. python ...
```

---

## Development scripts

Reference and experimental tools are stored in:

```
dev/
```

These are not part of the production pipeline.
