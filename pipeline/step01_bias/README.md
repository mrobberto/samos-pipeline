# step01_bias

## Purpose

Create a **master bias frame** by combining all bias exposures in the dataset.

The master bias represents the detector electronic offset and is subtracted from all science and calibration frames in the next step.

---

## Input

Directory defined in:

```python
config.ST00_ORIENT
```

This directory contains the **orientation-standardized raw data** produced by Step00.

Bias frames are identified automatically using:

* filename (contains `"bias"`), or
* FITS header keywords:

  * `IMAGETYP`
  * `OBSTYPE`
  * `TYPE`
  * `OBJECT`

matching values such as:

* `BIAS`
* `ZERO`
* `BIAS FRAME`

---

## Processing

1. Each bias file (MEF format) is converted to a 2D image using:

```python
SAMOS.read_SAMI_mosaic()
```

2. All bias images are stacked into a 3D cube.

3. The master bias is computed as:

```text
median(stack)
```

This provides a robust estimate against outliers and cosmic rays.

---

## Output

Directory:

```python
config.ST01_BIAS
```

Output file:

```text
MasterBias.fits
```

---

## Header updates

The output FITS header includes:

```text
NCOMBINE = number of bias frames used
```

and HISTORY entries documenting:

* the combination method (median)
* the list of input files (truncated if long)

---

## How to run

From the pipeline root:

```python
runfile("step01_bias/step01_masterbias.py")
```

or from command line:

```bash
python step01_bias/step01_masterbias.py
```

---

## Notes

* This step should be run **once per dataset**
* All downstream processing assumes a valid `MasterBias.fits`
* The median combine is sufficient for bias frames (no scaling required)

---

## Design choices

* Uses **median combine** for robustness
* Uses **mosaicked frames** to operate in detector coordinates
* Bias identification is **automatic**, minimizing manual selection
* Operates on **orientation-corrected data** to ensure consistency with all downstream steps

---

## Pipeline context

```text
Step00 → standardize orientation
Step01 → build master bias
Step02 → apply bias correction
```

---

## Future considerations

* Optional sigma-clipping could be added if large numbers of frames are used
* Bias stability across nights could be monitored for long campaigns
