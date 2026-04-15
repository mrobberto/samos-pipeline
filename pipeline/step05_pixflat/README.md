# step05_pixflat

## Purpose

Build pixel-to-pixel flat-field correction images (**pixflat**) using quartz lamp illumination.

This corrects small-scale detector variations while preserving large-scale illumination structure.

---

## Input

### Quartz frames

Directory:

```python
config.ST03_CRCLEAN
```

Inputs:

```python
config.QUARTZ_A
config.QUARTZ_B_EVEN
config.QUARTZ_B_ODD
```

These are **cosmic-ray-cleaned, orientation-corrected quartz frames**.

---

### Trace masks

From Step04:

```python
config.ST04_PIXFLAT
```

Preferred:

```text
*_mask_reg.fits
```

Fallback:

```text
*_mask.fits
```

---

## Processing

For each trace set (EVEN, ODD):

---

### 1. Quartz difference

```text
quartz_diff = quartzB - quartzA
```

---

### 2. Illumination model

* Apply strong Gaussian smoothing to quartz_diff
* Use only pixels inside slit mask
* Ignore NaNs via weighted filtering

Result:

```text
illum2d
```

---

### 3. Pixel flat

```text
pixflat = quartz_diff / illum2d
```

Then:

* clip to safe range
* renormalize to median ≈ 1 inside mask

---

### 4. Mask handling

* mask edges eroded before smoothing
* prevents edge artifacts in illum2d

---

## Output

Directory:

```python
config.ST05_FLATCORR
```

For each set (Even / Odd):

---

### Quartz difference

```text
Even_traces_QuartzDiff_for_pixflat.fits
Odd_traces_QuartzDiff_for_pixflat.fits
```

---

### Illumination model

```text
Even_traces_Illum2D_quartzdiff.fits
Odd_traces_Illum2D_quartzdiff.fits
```

---

### Pixel flat (main product)

```text
Even_traces_PixelFlat_from_quartz_diff.fits
Odd_traces_PixelFlat_from_quartz_diff.fits
```

---

## Header updates

Headers include:

```text
TRACESET = EVEN / ODD
QZ_A     = quartz A filename
QZ_B     = quartz B filename
MASKFILE = mask used
SY, SX   = smoothing scales
ERODE    = mask erosion iterations
CLIPLO   = clipping lower bound
CLIPHI   = clipping upper bound
```

and provenance from Step04 masks.

---

## How to run

```python
runfile("step05_pixflat/step05_build_pixflat.py")
```

or:

```bash
python step05_pixflat/step05_build_pixflat.py
```

---

## Notes

* Must be run **after Step04**
* Produces separate pixflats for EVEN and ODD
* Flat is applied later in Step06

---

## Design choices

* Uses quartz difference to isolate slit illumination
* Removes large-scale structure via heavy smoothing
* Preserves only pixel-to-pixel variations
* Uses trace masks to avoid background contamination
* Applies robust normalization within slit regions

---

## Pipeline context

```text
Step04 → trace masks and geometry
Step05 → pixel flat construction
Step06 → apply flat + rectify spectra
```

---

## Critical dependencies

* Accuracy of Step04 masks directly affects pixflat quality
* Misaligned masks → incorrect flat correction

---

## Future considerations

* per-slit normalization (optional)
* improved handling of edge pixels
* wavelength-dependent flat modeling (if needed)
