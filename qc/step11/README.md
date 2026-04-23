# QC — Step 11 Flux Calibration

This directory contains **quality control (QC) scripts** for validating the flux calibration (Step11c) and its refinement (Step11d).

All scripts are designed to run independently and produce diagnostic plots or PDFs.

---

## QC scripts

### 1. Grid overview of flux-calibrated spectra

qc_step11_grid_patched_v2.py


- Displays all calibrated spectra in a grid layout
- Useful for:
  - global inspection
  - identifying outliers or failures

---

### 2. Summary diagnostics

qc_step11_summary.py


- Produces summary plots and statistics for Step11 outputs
- Includes:
  - flux distributions
  - calibration diagnostics

---

### 3. Response function diagnostics

qc_step11_response_summary.py


- Visualizes:
  - per-star response curves
  - master response
  - acceptance criteria

---

### 4. Final validation — refined spectra vs SkyMapper

qc_step11d_refined_vs_skymapper.py


- Multipage QC (PDF output)
- For each slit:
  - plots `FLUX_FLAM` and `FLUX_FLAM_REFINED`
  - overlays SkyMapper r/i/z photometric points

Key features:
- fixed physical subplot size (for consistency)
- per-slit y-axis scaling (for readability)
- displays median refinement response per slit

Output:
11_fluxcal/qc_step11/qc_step11d_refined_vs_skymapper.pdf


---

## Validation strategy

Step11 calibration is validated in two ways:

### Visual validation
- Refined spectra should pass through SkyMapper photometric points
- Continuum shape should be smooth and consistent

### Quantitative validation
- Synthetic photometry (band integration) should reproduce SkyMapper fluxes
- Residuals should be small and unbiased

---

## Folder structure
qc/step11/
qc_step11_grid_patched_v2.py
qc_step11_summary.py
qc_step11_response_summary.py
qc_step11d_refined_vs_skymapper.py
legacy/


---

## Legacy scripts

Older versions and experiments are stored in:
legacy/


These are not part of the standard QC workflow.

---

## Notes

- QC scripts are intended to be run after Step11 completion
- They rely on consistent file naming from the pipeline
- Outputs are suitable for both debugging and publication-quality figures
