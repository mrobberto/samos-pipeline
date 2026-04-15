# SAMOS Pipeline — Run Order

This document describes the execution order for the SAMOS pipeline from
raw calibration frames to final flux-calibrated spectra.

It is an **operator guide**:
- what to run
- in what order
- what outputs to expect

For scientific details, see `pipeline_manual.md`.

---

# 1. Initialization

Ensure:

- raw data are organized (science, quartz, arc)
- `config.py` points to the correct run
- directories exist:

```python
import config
config.ensure_dirs()
2. Step04 — Slit Geometry (Quartz Traces)
Purpose

Reconstruct slit geometry (center + edges) from quartz lamp images.

Input
Quartz_A / Quartz_B exposures (bias-corrected, CR-cleaned)
RA/DEC slit table
Run

(implementation-specific, typically one script per parity)

Outputs
04_pixflat/
Even_traces_geometry.fits
Odd_traces_geometry.fits
Even_traces_mask.fits
Odd_traces_mask.fits
Even_traces_slitid.fits
Odd_traces_slitid.fits
Even_traces_slit_table.csv
Odd_traces_slit_table.csv
Checkpoints
number of slits correct
traces smooth and parallel
no broken or merged slits
3. Step05 — Pixel Flat
Purpose

Derive pixel-to-pixel sensitivity correction from quartz differences.

Input
Quartz_A / Quartz_B (same as Step04)
Run
runfile("Step05_pixel_flat.py")
Output
04_pixflat/
PixelFlat_from_quartz_diff.fits
Checkpoints
values ~1
no large-scale gradients
no dead columns left uncorrected
4. Step06 — Science Preparation and Rectification
Step06.0 — science frame

Output:

FinalScience_*_ADUperS.fits
Step06b — flat-field correction

Outputs:

FinalScience_*_pixflatcorr_clipped_EVEN.fits
FinalScience_*_pixflatcorr_clipped_ODD.fits
Step06c — TRACECOORDS rectification

Outputs:

FinalScience_*_tracecoords_EVEN.fits
FinalScience_*_tracecoords_ODD.fits
Checkpoints
slit width ~10–20 px
spectra continuous
no edge truncation
5. Step07 — Wavelength Calibration

Run full modular chain:

arc extraction
slit shifts
master arc
wavelength solution
propagation
Outputs
07_wavecal/
arc_master.fits
arc_master_wavesol.fits
arc_wavesol_per_slit.fits
arc_1d_wavelength_all.fits
Checkpoints
monotonic wavelength
consistent solution across slits
correct A-band location (~760 nm)
6. Step08 — Extraction
Step08a — extraction
runfile("Step08a_extract_1d_final.py", args="--set EVEN")
runfile("Step08a_extract_1d_final.py", args="--set ODD")

Outputs:

extract1d_optimal_ridge_even.fits
extract1d_optimal_ridge_odd.fits
Step08b — merge
runfile("step08b_merge_even_odd_final.py")

Output:

extract1d_optimal_ridge_all.fits
Step08c — wavelength attach
runfile("step08c_attach_wavelength_final.py", args="--overwrite")

Output:

extract1d_optimal_ridge_all_wav.fits
Step08 QC
runfile("qc_step08_extract.py", args="--set EVEN")
runfile("qc_step08_extract.py", args="--set ODD")

Check:

ridge follows source
sky looks clean
no edge clipping
7. Step09 — Telluric Correction
Step09a — template
runfile("step09a_build_o2_template_final.py")

Output:

telluric_o2_template.fits
Step09b — apply
runfile("step09b_apply_telluric_final.py")

Output:

extract1d_optimal_ridge_all_wav_tellcorr.fits
Step09 QC
runfile("qc_step09_telluric.py")

Check:

A/B bands corrected
no overcorrection
8. Step10 — OH Refinement
Step10a — measure
runfile("step10a_measure_oh_shifts_final.py")

Output:

QC_OH_BG_registration.csv
Step10b — apply
runfile("step10b_apply_oh_shift_final.py")

Output:

extract1d_optimal_ridge_all_wav_tellcorr_OHref.fits
Step10 QC
runfile("qc_step10_oh.py")

Check:

shifts small (~0.1–0.3 nm)
few fallbacks
9. Step11 — Flux Calibration
Step11a — RA/DEC
runfile("step11a_extract_header_radec_final.py")

Output:

slit_trace_radec_all.csv
Step11b — photometry
runfile("step11b_query_skymapper_final.py")

Output:

slit_trace_radec_skymapper_all.csv
Step11c — calibration
runfile("step11c_fluxcal_final.py")

Outputs:

extract1d_fluxcal.fits
step11_fluxcal_summary.csv
step11_fluxcal_QA.png
10. Final Product
extract1d_fluxcal.fits

Spectra are:

extracted
wavelength calibrated
telluric corrected
OH refined
flux calibrated

👉 ready for science analysis