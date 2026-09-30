# Step 11 — Flux Calibration and Ensemble Response

Step 11 prepares the spectrophotometric calibration used by the final Step 12 calibration.

## Production flow

Step10 telluric + relative-illumination spectra feed two parallel products:

- Step11c: absolute flux calibration
- Step11d: ensemble broadband response shape

These are combined by Step12d to build the production master response, which
Step12e applies to the final spectra.

## Step11a — Slit sky coordinates

`step11a_extract_header_radec_resilient.py`

Coordinate priority:

1. coordinates already present in the extracted-spectrum header
2. Step04 slit geometry
3. legacy CSV fallback

Output: `slit_trace_radec_all.csv`

## Step11b — SkyMapper crossmatch

`step11b_query_skymapper.py`

Matches slit coordinates to SkyMapper r/i/z photometry.

Output: `slit_trace_radec_skymapper_all.csv`

## Step11c — Absolute flux calibration

`step11c_fluxcal.py`

Default spectral input: `config.STEP11_INPUT_SPECTRA`

Default photometric catalog: `config.STEP11_PHOTCAT`

Primary output: `config.EXTRACT1D_FLUXCAL`

Currently: `extract1d_fluxcal.fits`

Step11c establishes the absolute flux-density scale.

## Step11d — Ensemble broadband response

`step11d_ensemble_response.py`

Derives one common smooth multiplicative response shape from the ensemble of
usable calibration stars. It does not modify the spectra.

Default spectral input: `config.STEP11_INPUT_SPECTRA`

Default photometric catalog: `config.STEP11_PHOTCAT`

The response is parameterized as

    ln C(lambda) = a1 x + a2 x^2
    x = (lambda - lambda_i,pivot) / scale_nm

and is normalized to unity at the SkyMapper i-band pivot.

Gray, linear, and quadratic models are evaluated. The production quadratic
solution is validated with leave-one-star-out cross-validation.

Canonical outputs:

- `config.STEP11_ENSEMBLE_RESPONSE_CSV`
- `config.STEP11_ENSEMBLE_RESPONSE_LOO_CSV`

Currently:

- `ensemble_response.csv`
- `ensemble_response_loo.csv`

## Step 12 handoff

`step12d_build_stellar_response.py` combines the Step11d response shape with a
single global i-band normalization derived from the Step11c calibrated spectra.

The production trusted wavelength interval is 600–1000 nm.

`step12e_apply_stellar_response.py` applies the resulting master response to
all spectra. Outside the trusted interval, the nearest boundary response is
held fixed; the quadratic response is not extrapolated.

No per-object photometric normalization is applied in production Step12.

## Diagnostic tools

`step11c_part2_continuum_snr.py`

`step11c_part3_rank_calibrators.py`

These are diagnostic tools and are not required by the production chain.

## Running

From the repository root:

    PYTHONPATH=. python pipeline/step11_fluxcal/step11c_fluxcal.py
    PYTHONPATH=. python pipeline/step11_fluxcal/step11d_ensemble_response.py

Production paths are resolved through `config`.
