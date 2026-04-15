# step02_biascorr

## Purpose

Apply bias correction to all frames by subtracting the master bias.

This removes the detector electronic offset from science and calibration data.

---

## Input

### Science and calibration frames

Directory:

```python
config.ST00_ORIENT
```

This contains the **orientation-standardized raw data** produced by Step00.

### Master bias

```python
config.ST01_BIAS / "MasterBias.fits"
```

Produced by Step01.

---

## Processing

Each input frame is:

1. Converted to detector coordinates (mosaic) using:

```python
SAMOS.read_SAMI_mosaic()
```

2. Bias-subtracted:

```text
image_corrected = image - MasterBias
```

3. Written to disk with updated naming.

Processing is performed via:

```python
SAMOS.subtract_superbias_from_directory()
```

---

## Output

Directory:

```python
config.ST02_BIASCORR
```

Output files:

```text
<original_name>_biascorr.fits
```

---

## Header updates

The output FITS headers include provenance information from the subtraction process.

---

## How to run

From the pipeline root:

```python
runfile("step02_biascorr/step02_biascorr.py")
```

or from command line:

```bash
python step02_biascorr/step02_biascorr.py
```

---

## Notes

* This step must be run **after Step01**
* All frames (science, flats, arcs) are bias-corrected
* Output files are mosaicked 2D images (not MEF)

---

## Design choices

* Uses mosaicked detector images for consistency with downstream processing
* Applies a single master bias to all frames
* Preserves original file naming with a `_biascorr` suffix
* Operates on **orientation-corrected data**

---

## Pipeline context

```text
Step00 → standardize orientation
Step01 → build master bias
Step02 → apply bias correction
Step03 → cosmic ray cleaning
```

---

## Future considerations

* Optional variance propagation could be enabled
* Overscan correction could be added if needed
