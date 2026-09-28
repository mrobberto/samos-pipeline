#!/usr/bin/env bash
set -euo pipefail

# Reproduce the Step09 spectra mosaic used for the SAMOS pipeline paper.
#
# IMPORTANT:
# This figure intentionally uses the frozen historical sky-cleaned product
# retained for paper reproducibility. It is NOT the active production pipeline
# product after the ROBMED Step08 revision.

PYTHONPATH=. python qc/step09/qc_step09_final_mosaic.py \
  --in products/Run8_Dolidze25/reduced/08_extract1d/extract1d_optimal_ridge_all_varfix_wavfix_OHref_skyclean099.fits \
  --outdir products/Run8_Dolidze25/reduced/09_abab/qc_step09 \
  --column STELLAR_CONSENSUS \
  --xlo 590 \
  --xhi 960 \
  --ncol 6 \
  --yscale global \
  --ymode log \
  --ylo 0.001
