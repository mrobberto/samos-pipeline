# step08_extract1d

## Purpose

Perform optimal 1D extraction of spectra from rectified TRACECOORDS slit images.

This step:

1. Tracks the object ridge in each slit
2. Estimates sky background per row
3. Performs optimal extraction
4. Corrects for aperture losses at slit edges
5. Produces calibrated 1D spectra with QC diagnostics

---

## Input

From Step06:

```python
config.ST06_SCIENCE
```

Files:

```text
*_tracecoords.fits
```

Each extension contains a rectified slit image in TRACECOORDS.

---

## Processing

### Ridge tracking

* Build coarse Y-binned profiles
* Find seed near slit center
* Track ridge upward and downward
* Fit global polynomial ridge

Fallback:

* If tracking is weak → use stiff ridge fit

Output:

```text
x0(y): ridge center per row
```

---

### Sky estimation

Hybrid method:

1. **Primary:** row-by-row sky
2. **Fallback:** pooled sky (±YSKYWIN rows)

Features:

* asymmetric sky selection (left/right)
* sigma clipping
* low-tail rejection
* interpolation for gaps

---

### Optimal extraction

For each row:

```text
flux = Σ [P * (D - sky) / V] / Σ [P² / V]
```

Where:

* P = Gaussian profile
* V = variance (Poisson + read noise + sky)

Outputs:

```text
FLUX
VAR
```

---

### Aperture-loss correction

* Detect truncation at trace edges
* Estimate captured PSF fraction
* Apply correction if safe

Outputs:

```text
FLUX_APCORR
VAR_APCORR
APLOSS_FRAC
EDGEFLAG
```

---

### QC diagnostics

Per slit:

* number of object pixels (NOBJ)
* sky samples (NSKY)
* sky noise (SKYSIG)
* flux and sky consistency checks

Flags:

```text
S08BAD = sky/flux inconsistency
S08EMP = empty extraction
```

---

## Output

Directory:

```python
config.ST08_EXTRACT1D
```

Files:

```text
Extract1D_optimal_ridgeguided_POOLSKY_{EVEN,ODD}.fits
```

Each extension (SLIT###) contains:

| Column      | Description            |
| ----------- | ---------------------- |
| YPIX        | row index              |
| FLUX        | extracted flux         |
| VAR         | variance               |
| SKY         | sky estimate           |
| X0          | ridge position         |
| NOBJ        | object pixels          |
| NSKY        | sky pixels             |
| SKYSIG      | sky noise              |
| APLOSS_FRAC | PSF fraction recovered |
| FLUX_APCORR | corrected flux         |
| VAR_APCORR  | corrected variance     |
| EDGEFLAG    | truncation flag        |

---

## Step08.5 — merge EVEN/ODD

Combine outputs into one file:

```text
Extract1D_optimal_ridgeguided_POOLSKY_ALL.fits
```

* preserves global SLIT### IDs
* ensures no duplication

---

## Step08.6 — attach wavelength

Using Step07 outputs:

```text
λ(y) = λ_master(y + SHIFT_TO_MASTER)
```

Adds:

```text
LAMBDA_NM
```

to each slit.

---

## Key concepts

### TRACECOORDS

* dispersion axis = Y
* spatial axis = X
* ridge follows object, not slit geometry

---

### Ridge-guided extraction

* allows flexure between quartz and science
* prevents sky truncation
* robust to curvature

---

### Sky strategy

* preserve skyline structure
* fallback only when necessary

---

### Aperture correction

* compensates edge clipping
* ensures photometric accuracy

---

## Pipeline context

```text
Step06 → rectified slit images
Step08 → 1D extraction
Step09 → sky/telluric refinement
Step10 → telluric correction
Step11 → flux calibration
```

---

## Notes

* Typical aperture width: ~6 pixels
* Ridge smoothing ensures stability over ~300 rows
* Extraction robust to moderate misalignment

---

## Future improvements

* adaptive PSF width
* 2D sky modeling
* cosmic-ray masking in TRACECOORDS
* multi-object deblending

---

## Summary

Step08 performs a physically motivated, ridge-guided optimal extraction that:

* tracks real object position
* preserves sky structure
* corrects aperture losses
* produces science-ready spectra

---
