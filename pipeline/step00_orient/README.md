# step00_orient

## Purpose

Standardize the orientation of all raw SAMOS images before any calibration or reduction.

This step ensures that all subsequent processing operates on a consistent detector geometry:

* **Wavelength increases bottom → top (Y axis)**
* **RA increases right → left (X axis)**

The transformation applied is a **180° rotation** of the detector frame.

---

## Why this step is necessary

SAMOS raw data may be recorded with an orientation that is inconsistent with the assumptions of the pipeline.

If not corrected:

* trace geometry becomes inconsistent
* wavelength direction may be inverted
* QC plots become misleading
* downstream steps require special-case handling

This step enforces a **single global convention**.

---

## Input

Directory defined in:

```python
config.RAW_DIR
```

Expected contents:

* raw science frames
* bias frames
* flats (quartz)
* arc frames

All FITS files in this directory are processed.

---

## Output

Directory:

```python
config.ST00_ORIENT
```

Output files:

```
<original_name>_oriented.fits
```

Each output file is a 180° rotated version of the input.

---

## Header updates

The primary header is updated with:

```
ROT180 = True    / Image rotated by 180 deg (Step00)
```

and a HISTORY entry documenting the operation.

---

## How to run

From the `scripts/` or pipeline root:

```python
runfile("step00_orient/step00_rotate.py")
```

or from command line:

```bash
python step00_orient/step00_rotate.py
```

---

## Notes

* This is a **pre-reduction normalization step**
* It should be run **once per dataset**
* All subsequent steps assume data are already oriented

---

## Design choice

The rotation is applied as a **data-level transformation** (not just metadata), so that:

* all downstream algorithms operate in a consistent coordinate system
* no conditional logic is required later in the pipeline

---

## Future considerations

* If future instrument configurations change orientation, this step can be adapted without modifying downstream code
* Alternative implementations (e.g. header-based orientation flags) are intentionally avoided to keep the pipeline simple and robust
