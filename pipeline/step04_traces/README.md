# step03_crclean

## Purpose

Remove cosmic rays from all bias-corrected frames using a LAcosmic-style algorithm.

This step produces clean science and calibration images suitable for trace finding, flat-fielding, and spectral extraction.

---

## Input

Directory:

```python
config.ST02_BIASCORR
```

This contains bias-corrected images produced by Step02.

Expected files:

```text
*_biascorr.fits
```

---

## Processing

For each input frame:

1. The image is read as a 2D detector frame.

2. Cosmic rays are identified and removed using:

```python
SAMOS.CR_correct()
```

based on a LAcosmic-style algorithm.

3. Parameters used include:

* `cr_threshold`
* `neighbor_threshold`
* `readnoise`
* `gain`

4. A cosmic ray mask is also generated.

---

## Output

Directory:

```python
config.ST03_CRCLEAN
```

Output files:

```text
<original_name>_biascorr_cr.fits
```

Each file contains:

* Primary HDU: cleaned image
* Extension `CRMASK`: cosmic ray mask

---

## Header updates

The primary header includes:

```text
CRCLEAN = True    / Cosmic rays removed
CRALG   = LACOSMIC
CRTHR   = threshold used
CRRN_E  = readnoise used
CRGAIN  = gain used
```

The `CRMASK` extension contains:

```text
1 = pixel flagged as cosmic ray
```

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

## Notes

* This step must be run **after Step02**
* All frames (science, flats, arcs) are processed
* Output images are clean 2D detector frames

---

## Design choices

* Uses LAcosmic-style detection for robust CR removal
* Produces a **separate mask extension** for diagnostics
* Avoids reprocessing files already marked with `CRCLEAN=True`
* Operates exclusively on bias-corrected data

---

## Pipeline context

```text
Step00 → standardize orientation
Step01 → build master bias
Step02 → apply bias correction
Step03 → remove cosmic rays
Step04 → trace identification
```

---

## Future considerations

* Parameter tuning per dataset (thresholds)
* Optional masking propagation into later steps
* Parallel processing for large datasets
