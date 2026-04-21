# step03_crclean

## Purpose

Determine the geometric layout of spectral traces on the detector using quartz-lamp exposures.

This step produces the reference trace geometry and slit masks used by all subsequent extraction and calibration steps.
---

## Input

Directory:

```python
config.ST03_CRCLEAN
```

Expected files:

*_biascorr_cr_rowcorr.fits   (quartz frames)

Paired exposures:

quartz with slit pattern (Q_on)
quartz with mirrors off (Q_off)
```

---

##  Processing

1. Build quartz difference image:

Q_diff = Q_on - Q_off

This removes diffuse background and isolates slit traces.

2. Detect slit centers:

collapse image along dispersion

identify peaks corresponding to slit traces

3. Assign slit IDs:

ordered by RA (right → left)

EVEN / ODD numbering scheme

4. Build slit masks:

row-wise segmentation around each trace

local background estimation

contiguous detection above threshold

optional edge trimming

5. Construct slit ID map:

assign each pixel to nearest slit center

6. Identify higher-order features (optional):

faint second-order signal may appear at long wavelengths

separated from first-order trace by a gap

when detected, an empirical cutoff is applied within the gap

pixels beyond this cutoff are excluded from the mask

7. Derive trace geometry:

compute per-row centroid and edges

fit smooth polynomial models:

 x_center(y)
 x_left(y)
 x_right(y)


---

## Output

Directory:

```python
config.ST04_TRACES
```

Output files:

```text
Even_traces.fits
Even_traces_mask.fits
Even_traces_slitid.fits
Even_traces_slit_table.csv
Even_traces_geometry.fits
Even_traces_gap_cuts.csv
```
---
## Notes

    Processing is performed in detector coordinates

    No rectification is applied at this stage

    Second-order trimming is:

        conservative

        applied only when a clear gap is detected

    The primary rejection of out-of-range wavelengths is applied later after wavelength calibration

## Pipeline context
    
    Step03 → cosmic-ray cleaned frames
    Step04 → trace determination
    Step05 → pixel flat
    Step06 → science rectification

---

## How to run

From the pipeline root:

```python
runfile("step03_crclean/step03_crclean.py")
```

or from command line:

```bash
python step03_crclean/step03_crclean.py
```


---

## Pipeline context



---

## Future considerations

* Parameter tuning per dataset (thresholds)
* Optional masking propagation into later steps
* Parallel processing for large datasets
