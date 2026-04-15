# step07_wavecal

## Purpose

Perform wavelength calibration using arc lamp exposures.

This step:

1. Builds a cleaned arc image (ArcDiff)
2. Applies pixel-flat correction
3. Extracts 1D arc spectra per slit
4. Aligns spectra using inter-slit shifts
5. Builds a master arc spectrum
6. Derives a global wavelength solution
7. Propagates the solution to all slits

---

## Input

### Arc frames

From:

```python
config.ST03_CRCLEAN
```

Files:

```text
*.arc_biascorr_cr.fits
```

---

### Pixel flat

From Step05:

```python
config.ST05_FLATCORR
```

---

### Trace masks and slit IDs

From Step04:

```python
config.ST04_PIXFLAT
```

---

### Rectified slit geometry

From Step06:

```python
config.ST06_SCIENCE
```

(used implicitly through geometry and slit mapping)

---

## Processing

### Step07a — arc difference

```text
ArcDiff = Arc_B − Arc_A
```

Removes background and isolates emission lines.

Output:

```text
ArcDiff_*.fits
```

---

### Step07b — pixel flat correction

* Apply EVEN/ODD pixel flats inside masks only
* Clip flat values to avoid noise amplification

Output:

```text
ArcDiff_*_pixflatcorr_clipped.fits
```

---

### Step07c — 1D extraction

* Use Step04 slitid maps
* Extract per-slit spectra along dispersion axis

Output:

```text
ArcDiff_*_1D_slitid_{EVEN,ODD}.fits
```

Each extension:

```text
SLIT###
data[0] = flux(y)
data[1] = npix_per_row(y)
```

---

### Step07d — initial shifts

* Identify brightest emission line per slit
* Compute shift relative to reference slit

Output:

```text
Arc_shifts_initial_{EVEN,ODD}.csv
```

---

### Step07e — refined shifts

* Refine shifts using cross-correlation
* Restrict to window around bright line

Output:

```text
Arc_shifts_final_{EVEN,ODD}.csv
Arc_stack_aligned_final_{EVEN,ODD}.fits
```

---

### Step07f — master arc

* Align all slits using final shifts
* Merge EVEN + ODD
* Build stacked master spectrum

Output:

```text
Arc_master_aligned_ALL.fits
```

Contents:

* MASTER_MEDIAN
* MASTER_MEAN
* ALIGNED_STACK
* COVERAGE
* SLITLIST (with SHIFT_TO_MASTER)

---

### Step07g — wavelength solution

* Detect emission lines in master arc
* Match to NIST line lists
* Fit polynomial:

```text
λ(y) = polynomial(y)
```

Output:

```text
Arc_master_wavelength_solution.fits
Arc_wavelength_solutions_global.fits
```

---

### Step07h — propagate solution

For each slit:

```text
λ_slit(y) = λ_master(y + SHIFT_TO_MASTER)
```

Output:

```text
*_1D_wavelength_ALL.fits
```

Each extension:

```text
data[0] = flux
data[1] = wavelength (nm)
```

---

## Output

Directory:

```python
config.ST07_WAVECAL
```

---

## Key concepts

### Global solution

* One wavelength solution derived from MASTER arc
* Applied to all slits via shifts

---

### SHIFT_TO_MASTER

* Encodes slit-to-slit spectral offsets
* Includes EVEN/ODD global alignment

---

### Panel B propagation

```text
λ(y_slit) = λ_master(y + SHIFT)
```

This ensures:

* consistent wavelength scale
* minimal per-slit fitting

---

## How to run

```python
runfile("step07_wavecal/step07a_make_arc_diff.py")

runfile("step07_wavecal/step07b_apply_pixflat_arc.py")

runfile("step07_wavecal/step07c_extract_arc_1d.py", args="--set EVEN")
runfile("step07_wavecal/step07c_extract_arc_1d.py", args="--set ODD")

runfile("step07_wavecal/step07d_find_line_shifts.py", args="--set EVEN")
runfile("step07_wavecal/step07d_find_line_shifts.py", args="--set ODD")

runfile("step07_wavecal/step07e_refine_stack_arc.py", args="--set EVEN")
runfile("step07_wavecal/step07e_refine_stack_arc.py", args="--set ODD")

runfile("step07_wavecal/step07f_build_master_arc.py")

runfile("step07_wavecal/step07g_solve_wavelength.py")

runfile("step07_wavecal/step07h_propagate_wavesol.py")
```

---

## Notes

* Uses first-order spectrum only (length ≈ 2875 px)
* Second order must be excluded upstream (Step04)
* Accuracy depends on:

  * slit geometry (Step04)
  * flat correction (Step05)
  * alignment (Step07d/e)

---

## Pipeline context

```text
Step06 → rectified slit spectra
Step07 → wavelength calibration
Step08 → optimal extraction
```

---

## Critical dependencies

* Correct slit geometry (Step04)
* Consistent TRACECOORDS (Step06)
* Reliable shift estimation

---

## Future improvements

* global 2D wavelength solution (optional)
* slit-dependent distortion terms
* automated outlier rejection
* improved NIST matching
