# step06_science_rectify

## Purpose

Prepare science data for spectral extraction by:

1. Combining individual exposures into a science mosaic
2. Applying pixel-flat correction
3. OPTIONAL: Rectify slitlets into a common coordinate system (**TRACECOORDS**)

This step provides an optional rectified representation (TRACECOORDS) 
useful for visualization, quality control, and some analysis tasks.
The core pipeline operates in detector coordinates.

---

## Input

### Science frames

From:

```python
config.ST03P5_ROWSTRIPE
```

Files:

```text
Science files listed in

config.SCIENCE_FILES
```

These frames are already:

* orientation-corrected (Step00)
* bias-corrected (Step02)
* cosmic-ray cleaned (Step03)

---

### Pixel flat

From Step05:

```python
config.ST05_PIXFLAT
```

Files:

```text
Even_traces_PixelFlat_from_quartz_diff.fits
Odd_traces_PixelFlat_from_quartz_diff.fits
```

---

### Trace geometry and masks

From Step04:

```python
config.ST04_TRACES
```

Files:

```text
Even_traces_geometry.fits
Odd_traces_geometry.fits

Even_traces_mask.fits / Even_traces_mask_reg.fits
Odd_traces_mask.fits  / Odd_traces_mask_reg.fits

Even_traces_slitid.fits
Odd_traces_slitid.fits
```

---

## Processing

### Step06a — build science mosaic

* Combine all CR-cleaned science frames
* Exposure-time-weighted mean
* Optional sigma clipping
* Optional normalization to ADU/s (default)

Output:

```text
FinalScience_<target>_ADUperS.fits
```

---

### Step06b — pixel flat correction

* Divide science frame by pixel flat
* Apply correction only within trace mask
* Clip flat values to safe range:

```text
FLCLIPLO = 0.70
FLCLIPHI = 1.30
```

* Optional flat registration (integer X shift)

Outputs:

```text
FinalScience_<target>_ADUperS_pixflatcorr_EVEN.fits
FinalScience_<target>_ADUperS_pixflatcorr_ODD.fits
```

---

### OPTIONAL: Step06c — slit rectification (TRACECOORDS)

* Use Step04 geometry (polynomial slit model)
* Map curved slit traces → rectangular slit frames
* Preserve spectral curvature (no centroid shifting)

Output:

```text
FinalScience_<target>_ADUperS_pixflatcorr_EVEN_tracecoords.fits

FinalScience_<target>_ADUperS_pixflatcorr_ODD_tracecoords.fits
```
One TRACECOORDS file is produced for each trace set (EVEN and ODD), each containing one FITS extension per slitlet.

Multi-extension FITS file:

```text
EXTNAME = SLIT###
```

---

## Output

Directory:

```python
config.ST06_SCIENCE
```

### Products

```text
FinalScience_<target>_ADUperS.fits

FinalScience_<target>_ADUperS_pixflatcorr_EVEN.fits
FinalScience_<target>_ADUperS_pixflatcorr_ODD.fits

FinalScience_<target>_ADUperS_pixflatcorr_EVEN_tracecoords.fits
FinalScience_<target>_ADUperS_pixflatcorr_ODD_tracecoords.fits
```

---

## TRACECOORDS definition

* Y axis → dispersion (wavelength direction)
* X axis → spatial direction within slit
* Slits are rectangular
* Pixels outside slit → NaN

Typical slit width: ~10–20 pixels (X)

---

## Header provenance

### Step06a

```text
INSTEP   = ST03          / input stage
NRATE    = True/False    / normalized to ADU/s
SIGCLIP  = True/False
SIGMA    = sigma threshold
EXPTKEY  = exposure time keyword
```

---

### Step06b

```text
SCIENCE  = input science file
PIXFLAT  = pixel flat used
MASKFILE = trace mask used
TRACESET = EVEN / ODD
FLCLIPLO = lower clip
FLCLIPHI = upper clip
REGFLAT  = flat registration flag
FLATDX   = applied shift
```

---

### Step06c

```text
GEOM     = TRACECOORDS
DISPAXIS = Y
CROSSAX  = X
SLITID   = slit identifier
XREF     = slit center reference
YMIN/YMAX = valid row range
```

---

## How to run

```python
runfile("step06_science_rectify/step06a_make_final_science.py")

runfile("step06_science_rectify/step06b_apply_pixflat_clip.py", args="--traceset EVEN")
runfile("step06_science_rectify/step06b_apply_pixflat_clip.py", args="--traceset ODD")

runfile("step06_science_rectify/step06c_TRACECOORDS_representation.py")
```

---

## Notes

* Step06c produces a rectified TRACECOORDS representation intended primarily for visualization, quality control, and extraction algorithms that benefit from rectangular slit coordinates. The science mosaic in detector coordinates remains the authoritative science image.
* No spectral centering or shifting is performed here
* Geometry from Step04 is strictly preserved

---

## Design choices

* Flat applied only within trace mask
* No resampling during flat correction
* Rectification preserves spectral structure
* Uses quartz-derived geometry (Step04)
* TRACECOORDS retains physical curvature information

---

## Pipeline context

```text
Step04 → Trace geometry and slit definitions

Step05 → Pixel-flat calibration

Step06a → Build final science mosaic

Step06b → Apply pixel-flat correction

Step06c → Generate optional TRACECOORDS representation

Step07 → Wavelength calibration

Step08 → Spectral extraction
```

---

## Critical dependencies

* Step04 geometry must be accurate
* Step05 flat must match trace set (EVEN/ODD)
* Misalignment propagates to extraction and wavelength calibration

---

## Future considerations

* variance propagation
* sub-pixel flat registration
* improved edge masking
* slit-by-slit diagnostics
