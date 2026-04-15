## Step04 QC — Physical Geometry Validation

### Purpose

Step04 defines the slit geometry model used by all later spectroscopic steps.
Its QC must verify that the fitted slit centers and edges follow the actual
quartz trace illumination and that the mask/slit ID maps remain continuous and
physically plausible.

This is the main geometry validation step of the pipeline.

---

### Inputs

Expected Step04 products in:

```text
config.ST04_PIXFLAT

Primary inputs:

Even_traces.fits / Odd_traces.fits
Even_traces_geometry.fits / Odd_traces_geometry.fits

Preferred mask / slit-ID products:

Even_traces_mask_reg.fits / Odd_traces_mask_reg.fits
Even_traces_slitid_reg.fits / Odd_traces_slitid_reg.fits

Fallback products:

Even_traces_mask.fits / Odd_traces_mask.fits
Even_traces_slitid.fits / Odd_traces_slitid.fits
QC script

Recommended QC script:

runfile("qc_step04_physical_geometry_validation_final.py")

Output folders:

04_pixflat/qc_step04_even/
04_pixflat/qc_step04_odd/

Products written:

per-slit PNG figures
summary CSV
summary text report
What the QC checks

For each slit, the QC script checks:

Quartz overlay
fitted center trace overlaid on the quartz trace image
fitted left/right edges overlaid on the slit illumination
Raw vs fitted geometry
center residuals
left-edge residuals
right-edge residuals
Width stability
slit width as a function of detector Y
smoothness of the width model
Mask / slit-ID continuity
fragmented rows
disconnected slit segments
rows missing from the slit-ID map
Worst-case ranking
identifies slits with the largest residuals or strongest fragmentation
What good results look like

A good Step04 solution should show:

fitted center line following the quartz ridge closely
left and right edges approximately parallel
smooth center/edge curves with no abrupt jumps
slit width varying slowly with Y
little or no fragmentation in the mask
very few missing rows in the slit-ID map

In the PNG overlays, the fitted geometry should visually sit on top of the
bright quartz trace for most detector rows.

Warning signs

Investigate a slit if you see any of the following:

center or edge fits visibly detached from the quartz trace
abrupt wiggles or kinks in the fitted curves
width changing sharply from row to row
many fragmented rows
many missing rows in the slit-ID map
trace splitting into disconnected islands
edges crossing or diverging unphysically

These usually indicate one of:

weak local signal
bad thresholding in row-wise segmentation
slit contamination
unstable edge detection
overfitting / poor polynomial behavior
Typical failure modes
1. Wrong slit center seed

The slit is detected, but the center is biased left or right.
This propagates into edge placement and later rectification.

Symptom: overlay shifted systematically relative to the quartz ridge.

2. Fragmented segmentation

The mask breaks into multiple islands along Y.

Symptom: slit-ID map has disconnected regions or frequent missing rows.

3. Unstable edge detection

Edges fluctuate more than the center.

Symptom: width(Y) is noisy or edges visibly wander.

4. Overly aggressive trimming

Useful slit rows are removed.

Symptom: shortened valid Y-range or large gaps in the mask.

Pass / fail guidance

A slit is usually acceptable if:

center residuals are small and smooth
edge residuals are modest and do not drift systematically
fragmentation is minimal
width behavior is smooth and physically plausible

A slit should be reviewed manually if:

residuals are large over extended Y ranges
fragmentation is persistent
geometry is clearly inconsistent with the quartz trace
the slit appears truncated or broken
Why this step matters downstream

Step04 is foundational.

Its outputs are used by:

Step05 to define illuminated regions for the pixel flat
Step06 to rectify slitlets into TRACECOORDS
Step07 to define valid first-order arc regions
Step08 to constrain extraction geometry

If Step04 geometry is wrong, later steps may still run, but with:

distorted rectification
truncated sky regions
poor ridge behavior
degraded wavelength calibration
unstable extraction near slit edges
Recommended operator workflow

After running Step04 for EVEN and ODD:

run the Step04 QC script
inspect the summary CSV/report
open the worst few slits by residual / fragmentation
confirm that overlays are physically sensible
only then proceed to Step05 and Step06
One-line acceptance criterion

Proceed only if the fitted slit geometry tracks the quartz illumination
smoothly and the mask/slit-ID maps are continuous for the great majority of slits.



## Step05 QC — Pixel Flat Validation

### Purpose

Validate the pixel-to-pixel flat-field correction derived from quartz
illumination.

The goal is to ensure that the flat:

- removes pixel-scale variations
- does not introduce large-scale gradients
- remains close to unity
- does not contain extreme or unstable values

---

### Inputs

Expected Step05 products in:

```text
config.ST05_FLATCORR

For each trace set:

quartz_diff_even.fits / quartz_diff_odd.fits
illum2d_even.fits / illum2d_odd.fits
pixflat_even.fits / pixflat_odd.fits

Mask reference (from Step04):

Even_traces_mask_reg.fits / Odd_traces_mask_reg.fits
fallback: *_mask.fits
What to check
1. Pixflat distribution

Inside the slit mask:

median ≈ 1
narrow distribution
no strong skew

Typical acceptable range:

0.7 – 1.3 (most pixels)
hard limits enforced by clipping:
CLIP_LO
CLIP_HI
2. Visual inspection

Display:

quartz_diff
illum2d
pixflat

Check:

quartz_diff → shows slit illumination
illum2d → smooth version of quartz
pixflat → no large-scale structure

👉 The pixflat should look structureless, except for small pixel variations.

3. Large-scale residuals

The pixflat should NOT show:

gradients along Y (dispersion)
gradients along X (spatial)
banding or striping

These indicate:

insufficient smoothing (SIGMA_Y, SIGMA_X)
mask problems
incorrect illumination model
4. Edge behavior

Check near slit edges:

no strong spikes or ringing
erosion of mask should prevent edge artifacts

If present:

increase MASK_EROSION_ITERS
5. Bad pixels / extremes

Check for:

large clusters at clip limits
isolated hot/cold pixels

Too many clipped pixels may indicate:

bad quartz subtraction
low S/N
mask misalignment
Quick diagnostic (recommended)

In Python:

import numpy as np
from astropy.io import fits

pf = fits.getdata("pixflat_even.fits")
mask = fits.getdata("Even_traces_mask_reg.fits") > 0

vals = pf[mask]

print("median =", np.nanmedian(vals))
print("std    =", np.nanstd(vals))
print("min/max=", np.nanmin(vals), np.nanmax(vals))
Expected behavior

A good pixflat:

median ≈ 1.0
std small (few percent)
no visible large-scale patterns
clean inside slit mask
stable between EVEN and ODD
Warning signs

Investigate if you see:

strong gradients in pixflat
large number of clipped pixels
pixflat deviating significantly from unity
edge artifacts or ringing
differences between EVEN and ODD flats
Why this matters

Step05 feeds directly into Step06:

bad pixflat → distorted spectra
introduces artificial features
biases sky subtraction
propagates into extraction and calibration
Acceptance criterion

Proceed only if the pixflat is close to unity, free of large-scale structure,
and stable across the slit mask.



📄 Step06 section for qc_guide.md
## Step06 QC — Science Preparation and TRACECOORDS Validation

### Purpose

Step06 prepares the science data for spectral extraction by:

- building the full-frame science mosaic (Step06a)
- applying the pixel flat within slit regions (Step06b)
- rectifying slitlets into TRACECOORDS (Step06c)

QC at this stage ensures that:

- the science mosaic is well-formed
- flat-field correction is applied correctly
- slitlets are properly extracted and not truncated

---

# Step06a QC — FinalScience Mosaic

### Script

```python
runfile("qc_step06a_mosaic.py")
Output
config.ST06_SCIENCE / qc_step06a/
step06a_mosaic_summary.png
step06a_mosaic_report.txt
What is checked
full-frame image structure
histogram of pixel values
row and column median profiles
finite-pixel fraction
header keywords (EXPTIME, NCOMBINE, BUNIT)
Good behavior
image is continuous with no seams
no large NaN regions
smooth row/column medians
reasonable normalization (ADU/s)
Warning signs
large empty regions
discontinuities from bad stacking
extreme pixel values
inconsistent header bookkeeping
Step06b QC — Pixel-Flat Correction
Script 1 — Inspector
runfile("qc_step06b_inspector.py", args="--traceset EVEN")
runfile("qc_step06b_inspector.py", args="--traceset ODD")
Output
config.ST06_SCIENCE / qc_step06b_even/
config.ST06_SCIENCE / qc_step06b_odd/
What is checked
raw science vs corrected science
pixel flat consistency
raw/corrected ratio inside mask
histogram of values
header provenance:
TRACESET
REGFLAT
FLATDX
clipping limits
Good behavior
raw/corrected ≈ flat inside mask
flat median ≈ 1
no structure outside mask
narrow distribution of ratio
Warning signs
ratio deviates strongly from flat
artifacts outside slit regions
excessive clipping
incorrect trace set or mask usage
Script 2 — Reg vs Non-Reg Comparison
runfile("qc_step06b_compare_reg.py", args="--traceset EVEN")
What is checked
difference between registered and non-registered correction
ratio reg / non-reg
header consistency
Good behavior
reg ≈ non-reg
difference is small and structureless
ratio ≈ 1
Warning signs
structured residuals along slit direction
edge distortions
large FLATDX shifts
Step06c QC — TRACECOORDS Slitlets
Script 1 — Quicklooks
runfile("qc_step06c_quicklooks.py", args="--traceset EVEN")
runfile("qc_step06c_quicklooks.py", args="--traceset ODD")
Output
config.ST06_SCIENCE / qc_step06c_even/
config.ST06_SCIENCE / qc_step06c_odd/
one JPG per slit
one montage image
What is checked
visual integrity of slitlets
slit width and continuity
presence of signal across Y
Good behavior
slitlets are continuous
width is stable (~10–20 px typical)
spectra visible across most rows
Warning signs
truncated slitlets
empty or near-empty slits
irregular shapes
edge slits partially missing
Script 2 — Single-Slit Diagnostics
runfile("qc_step06c_single.py", args="--traceset EVEN --slit SLIT018")
What is checked
finite footprint per row:
leftmost X
rightmost X
width
blank-row fraction
width stability
footprint smoothness
Good behavior
few blank rows
width stable across Y
left/right boundaries vary smoothly
Warning signs
many blank rows
abrupt jumps in footprint
strong narrowing toward one end
irregular or fragmented footprint
Why Step06 QC is critical

Step06 defines the input to extraction (Step08).

Problems here propagate directly into:

ridge tracking errors
sky subtraction issues
flux losses at slit edges
distorted spectra
Recommended workflow

After running Step06:

run Step06a QC → verify science mosaic
run Step06b inspector → verify flat correction
optionally compare reg vs non-reg
run Step06c quicklooks → scan all slits
inspect suspicious slits with single-slit QC
Acceptance criterion

Proceed only if:

the science mosaic is clean and well-formed
flat correction behaves as expected inside the slit mask
TRACECOORDS slitlets are continuous and not truncated



## Step07 QC — Wavelength Calibration

### Purpose

Step07 performs the wavelength calibration of the SAMOS spectra by:

- extracting arc spectra per slit (Step07c)
- aligning slitlets to a common reference (Step07d–07f)
- deriving a global wavelength solution (Step07g)
- propagating the solution to all slits (Step07h)

QC at this stage ensures that:

- arc lines are correctly identified and aligned
- relative slit shifts are accurate
- the global wavelength solution is stable and monotonic
- the final wavelength assignment is consistent across all slits

---

# Step07d QC — Initial Alignment

### Script

```python
runfile("QC07d.py")
What is checked
cross-correlation shifts relative to reference slit
shift distribution across slits
Good behavior
shifts are smooth across slit ID
no large outliers
Warning signs
erratic shifts
failed correlations
large scatter between neighboring slits
Step07.2 QC — Peak Detection
Script
runfile("QC07_2_unified.py", args="--traceset EVEN")
What is checked
detected arc peaks per slit
consistency of strong emission lines
Good behavior
clear detection of major lines
consistent peak positions across slits
Warning signs
missing peaks
spurious detections
large positional scatter
Step07e QC — Final Shifts
Script
runfile("QC07_25_unified.py", args="--traceset EVEN")
What is checked
refined slit-to-slit shifts
consistency of alignment after correction
Good behavior
small residual shifts
smooth behavior vs slit ID
Warning signs
systematic offsets
residual trends
inconsistent refinement
Step07f QC — Master Arc
Script
runfile("QC07f.py")
What is checked
stacked master arc spectrum
line sharpness and alignment
Good behavior
narrow, well-defined emission lines
no broadening due to misalignment
Warning signs
blurred lines
double peaks
uneven stacking
Step07g QC — Global Wavelength Solution
Script
runfile("qc07g_wavelength_solution.py")
Output
config.ST07_WAVECAL / qc_step07g/
qc07g_wavesol_summary.png
qc07g_wavesol_report.txt
What is checked
polynomial wavelength solution λ(y)
matched arc lines vs fit
residuals:
RESID_NM = λ_fit − λ_match
residuals vs detector Y
residual histogram
dispersion dλ/dy
monotonicity of solution
Good behavior
residuals tightly clustered around 0
no systematic trend with Y
λ(y) strictly monotonic
stable dispersion across domain
RMS consistent with expectations
Warning signs
large residual scatter
residual drift with Y
non-monotonic solution
inconsistent RMS
Step07h QC — Per-Slit Wavelength Assignment
Script
runfile("QC07h.py")
What is checked
λ vs flux for each slit
monotonicity per slit
wavelength range consistency
Good behavior
smooth spectra in λ-space
monotonic λ arrays
consistent wavelength coverage across slits
Warning signs
non-monotonic λ
truncated wavelength ranges
distorted spectra
Global QC — Scientific Validation
Script 1 — Vertical Lines
runfile("QC75_VerticalLines_wl_All.py")
What is checked
alignment of arc lines in wavelength space across slits
Good behavior
vertical alignment of emission lines
Warning signs
tilted or curved lines → wavelength errors
Script 2 — Companion Global QC
runfile("QC75_companion_global.py")
What is checked
global wavelength consistency
residuals across all slits
line centroid statistics
worst-slit ranking
Good behavior
small residuals across all slits
no systematic trends
consistent λ calibration across field
Warning signs
outlier slits
systematic shifts vs slit position
large centroid scatter
Why Step07 QC is critical

Step07 defines the wavelength scale for the entire pipeline.

Errors here propagate directly into:

incorrect sky subtraction (Step08)
wrong telluric correction (Step09)
distorted OH alignment (Step10)
incorrect flux calibration (Step11)
Recommended workflow

After running Step07:

run QC07d → verify initial alignment
run QC07.2 → verify peak detection
run QC07.25 → verify shift refinement
run QC07f → inspect master arc
run qc07g → validate wavelength solution
run QC07h → inspect per-slit spectra
run QC75 → confirm global consistency
Acceptance criterion

Proceed only if:

arc lines are correctly aligned across slits
wavelength solution is monotonic and stable
residuals are small and structureless
vertical-line QC shows proper alignment
no pathological slits dominate the solution



## Step08 QC — Optimal Extraction and Sky Subtraction

### Purpose

Step08 performs optimal extraction of 1D spectra from TRACECOORDS slitlets.

It includes:
- ridge-guided extraction (object tracing)
- aperture definition and correction
- sky estimation (row-by-row or pooled)
- quality flags for edge effects and extraction failures

QC at this stage ensures that:

- the ridge correctly follows the source
- the aperture captures the object flux
- sky subtraction is unbiased and stable
- no systematic flux loss or contamination is present

---

# Data products

Step08 produces:


extract1d_optimal_ridge_even.fits
extract1d_optimal_ridge_odd.fits
extract1d_optimal_ridge_all.fits
extract1d_optimal_ridge_all_wav.fits


QC for Step08 focuses primarily on the **Step08a outputs**:


extract1d_optimal_ridge_{even,odd}.fits


These contain:
- FLUX, SKY
- FLUX_APCORR (preferred science product)
- X0 (ridge)
- NSKY, SKYSIG
- APLOSS_FRAC
- EDGEFLAG

---

# QC suite overview

The recommended QC scripts are:

| Script | Purpose |
|------|--------|
| `qc_step08_extract.py` | batch overview |
| `qc_step08_single.py` | detailed slit inspection |
| `qc_step08_global_summary.py` | ranking worst slits |
| `qc_step08_scan_all_slits.py` | detect sky/extraction pathologies |
| `qc_step08_masks_overlay_batch.py` | visualize apertures and sky |

---

# 1. Batch overview — extraction sanity

### Script

```python
runfile("qc_step08_extract.py", args="--set EVEN")
What is checked
ridge tracking vs image
extracted flux vs sky
aperture placement
general behavior across slits
Good behavior
ridge follows object signal
flux significantly above sky where expected
smooth variation along Y
Warning signs
ridge jumping between peaks
flux tracking sky instead of object
discontinuities along Y
2. Single-slit inspection (most important)
Script
runfile("qc_step08_single.py", args="--set EVEN --slit SLIT018")
What is checked
2D TRACECOORDS image
ridge position (X0)
aperture boundaries
trace edges
extracted quantities:
FLUX
SKY
FLUX_APCORR
QC diagnostics:
APLOSS_FRAC
NSKY
SKYSIG
EDGEFLAG
Good behavior
ridge centered on object
aperture encloses full flux
SKY tracks background, not object
APLOSS_FRAC ≈ 1
NSKY sufficiently large
EDGEFLAG mostly zero
Warning signs
ridge drifting to secondary peak
aperture clipping flux
SKY contaminated by object
APLOSS_FRAC < 0.8
large EDGEFLAG regions
noisy or unstable SKY
3. Global summary — worst slits
Script
runfile("qc_step08_global_summary.py", args="--set EVEN")
What is checked

Per slit:

fraction of valid flux rows
edge contamination
median APLOSS_FRAC
NSKY / SKYSIG
flags (S08BAD, S08EMP)
Output
ranked list of worst slits
quick identification of problematic extractions
Good behavior
most slits have:
high valid fraction
low edge fraction
APLOSS_FRAC ≈ 1
Warning signs
many slits flagged as bad/empty
systematic low APLOSS_FRAC
large fraction of edge pixels
4. Scan all slits — sky vs flux diagnostics
Script
runfile("qc_step08_scan_all_slits.py", args="--set EVEN")
What is checked
SKY variation along slit
FLUX variation along slit
ratio:
R_sky = SKY_high / SKY_low
R_flux = FLUX_high / FLUX_low
noise ratio (optimal vs box extraction)
Good behavior
R_sky ≈ 1 (stable sky)
R_flux reflects real object structure
noise ratio ≈ 1 (or improved)
Warning signs
R_sky >> 1 → sky bias or gradient
R_flux << 1 → flux loss or clipping
abnormal noise ratio → extraction instability
5. Mask overlay — geometry validation
Script
runfile("qc_step08_masks_overlay_batch.py", args="--set EVEN")
What is checked

Overlay on TRACECOORDS image:

ridge (center)
object aperture
sky regions
invalid pixels
Good behavior
aperture centered on ridge
sky regions cleanly separated
no overlap between object and sky
Warning signs
sky region overlapping object
aperture too narrow or too wide
large masked regions inside slit
Why Step08 QC is critical

Step08 is where raw data become science spectra.

Errors here propagate directly into:

sky subtraction residuals
spectral shape distortions
incorrect flux calibration (Step11)
artificial features in spectra
Recommended workflow

After running Step08a:

run batch QC (qc_step08_extract)
inspect worst slits (qc_step08_global_summary)
deep-dive suspicious slits (qc_step08_single)
validate sky behavior (qc_step08_scan_all_slits)
confirm geometry (qc_step08_masks_overlay_batch)
Acceptance criterion

Proceed only if:

ridge tracks the object reliably
sky subtraction is stable and unbiased
aperture losses are small (APLOSS_FRAC ≈ 1)
no systematic edge contamination
problematic slits are identified and understood



## Step09 QC — Telluric Correction

### Purpose

Step09 removes atmospheric O2 absorption from the extracted spectra using an
empirical telluric template built from the data themselves.

It includes:
- construction of the empirical O2 template (Step09a)
- per-slit application of the A-band and B-band correction (Step09b)

QC at this stage ensures that:

- the empirical template is physically sensible
- the telluric bands are reduced without overcorrection
- continuum shape is preserved outside the bands
- problematic slits are identified and tracked

---

# Data products

Step09 consumes:

```text
extract1d_optimal_ridge_all_wav.fits

and produces:

telluric_o2_template.fits
extract1d_optimal_ridge_all_wav_tellcorr.fits

The main corrected science column is:

FLUX_TELLCOR_O2

and, when available:

VAR_TELLCOR_O2
QC suite overview

The recommended QC scripts are:

Script	Purpose
qc_step09_telluric.py	batch QC overview
qc_step09_single.py	detailed single-slit before/after inspection
qc_step09_summary_pdf.py	multi-slit summary PDF
1. Batch overview — telluric sanity check
Script
runfile("qc_step09_telluric.py")
What is checked
pre- vs post-correction spectra
behavior in the O2 B band (~687 nm)
behavior in the O2 A band (~760 nm)
template consistency
slit-by-slit correction metadata
Good behavior
telluric troughs are reduced substantially
no strong overcorrection below the continuum
continuum is stable outside the correction windows
most slits show successful telluric flags
Warning signs
residual deep absorption after correction
corrected spectrum rising artificially above continuum
correction applied where no telluric feature exists
many slits failing telluric fits
2. Single-slit inspection (most useful)
Script
runfile("qc_step09_single.py", args="--slit SLIT018")
What is checked

For one slit:

pre-telluric spectrum
post-telluric spectrum
post/pre ratio
empirical A-band and B-band template shape
telluric fit metadata:
TELL_OK
TELL_OKA
TELL_OKB
TELL_AA
TELL_AB
TELL_SHA
TELL_SHB
TELLBAND
Good behavior
ratio departs from 1 mainly inside the A/B bands
correction is smooth
fitted A/B amplitudes are reasonable
fitted shifts are small
TELL_OK = True
Warning signs
ratio strongly oscillatory outside bands
very large fitted amplitudes
edge-of-grid fitted shifts
TELL_OK = False or only one band repeatedly succeeding
3. Summary PDF — multi-slit review
Script
runfile("qc_step09_summary_pdf.py")
Output
config.ST09_TELLURIC / qc_step09 / qc_step09_telluric_summary.pdf
What is checked
many slits at once
before/after telluric correction
per-slit A/B success flags
template comparison
Good behavior
consistent correction quality across slits
no systematic over- or under-correction
similar template response for comparable spectra
Warning signs
systematic bias in one band only
repeated failures in one region of the detector
groups of slits with abnormal amplitudes or shifts
Why Step09 QC is critical

Step09 removes broad atmospheric absorption features that directly affect:

continuum shape
equivalent width measurements
comparison between slits
later OH refinement and flux calibration

If Step09 is wrong, Step10 and Step11 may still run, but on spectra with
biased shape and residual atmospheric structure.

Recommended workflow

After running Step09a and Step09b:

run qc_step09_telluric
inspect several representative slits with qc_step09_single
generate qc_step09_summary_pdf
verify that A and B bands are corrected without damaging the continuum
Acceptance criterion

Proceed only if:

the A and B telluric bands are visibly reduced
correction is localized to the telluric regions
continuum outside the bands remains stable
fitted amplitudes and shifts are physically reasonable
failures are rare and understood


## Step10 QC — OH Wavelength Refinement

### Purpose

Step10 refines the wavelength calibration using OH sky emission features.

It includes:
- measurement of slit-by-slit wavelength shifts (Step10a)
- application of additive wavelength correction (Step10b)

QC at this stage ensures that:

- wavelength zero-point offsets are corrected
- OH sky lines align across slits
- fallback behavior is robust
- no systematic wavelength distortions remain

---

# Data products

Step10 consumes:

extract1d_optimal_ridge_all_wav_tellcorr.fits

and produces:

oh_shifts.csv
extract1d_optimal_ridge_all_wav_tellcorr_OHref.fits

---

# QC suite overview

| Script | Purpose |
|------|--------|
| qc_step10_shifts.py | inspect shift distribution |
| qc_step10_summary.py | full validation (recommended) |

---

# 1. Shift distribution

### Script

runfile("qc_step10_shifts.py")

### What is checked

- histogram of shift_nm
- shift vs slit
- used vs rejected slits

---

### Good behavior

- narrow distribution (~0.1–0.3 nm)
- smooth variation across slits
- most slits used

---

### Warning signs

- bimodal distribution
- large scatter
- many rejected slits

---

# 2. Full validation (authoritative QC)

### Script

runfile("qc_step10_summary.py")

---

### What is checked

- measured vs applied shifts
- GOOD vs FALLBACK behavior
- shift statistics:
  - median
  - robust scatter
- consistency with Step08 flags
- alignment of OH sky features (pre vs post)

---

### Good behavior

- applied shift ≈ measured shift
- fallback shifts ≈ global median
- OH lines align after correction
- small residual scatter

---

### Warning signs

- mismatch between measured and applied shifts
- fallback shifts inconsistent
- OH lines still misaligned
- large residual scatter

---

# Why Step10 QC is critical

Step10 defines the final wavelength scale before flux calibration.

Errors here propagate into:

- incorrect line centroids
- velocity measurements
- line ratios
- scientific interpretation

---

# Acceptance criterion

Proceed only if:

- OH lines align across slits
- shift distribution is narrow and well-behaved
- fallback behavior is consistent
- no systematic wavelength offsets remain




## Step11 QC — Flux Calibration

### Purpose

Step11 converts extracted spectra into physically calibrated flux units
using photometric anchoring (SkyMapper r/i/z).

It includes:
- coordinate association (Step11a)
- photometric matching (Step11b)
- flux calibration (Step11c)

---

# Data products

Input:
extract1d_optimal_ridge_all_wav_tellcorr_OHref.fits

Outputs:
slit_trace_radec_all.csv
slit_trace_radec_skymapper_all.csv
extract1d_fluxcal.fits

---

# QC suite overview

| Script | Purpose |
|------|--------|
| qc_step11_summary.py | detailed per-slit validation |
| qc_step11_grid.py | global overview |

---

# 1. Detailed validation (primary QC)

### Script

runfile("qc_step11_summary.py")

---

### What is checked

- slit image and spectrum alignment
- RA/DEC correctness via imaging stamp
- photometric match (r/i/z)
- flux calibration quality
- spectral shape and continuum

---

### Good behavior

- source centered in imaging cutout
- photometric points align with spectrum
- smooth continuum
- consistent flux scale across slits

---

### Warning signs

- source offset from slit center
- photometry inconsistent with spectrum
- discontinuities or slopes
- missing or noisy calibration

---

# 2. Global overview

### Script

runfile("qc_step11_grid.py")

---

### What is checked

- all spectra at once
- consistency of flux scale
- detection of outliers

---

### Good behavior

- similar overall scaling
- consistent shapes
- few outliers

---

### Warning signs

- large scatter between slits
- failed calibrations
- anomalous spectra

---

# Advanced diagnostics (not routine QC)

The following tools are used only for debugging:

- RA/DEC reconstruction from detector geometry
- A-band vs DEC correlation
- slit coordinate corrections

These are not part of the standard pipeline QC workflow.

---

# Why Step11 QC is critical

Step11 defines the final science product.

Errors here directly affect:
- flux measurements
- spectral energy distributions
- comparisons with photometry
- all downstream science

---

# Acceptance criterion

Proceed only if:

- photometric anchoring is consistent
- spectra match imaging sources
- continuum shape is physical
- no systematic calibration errors are present