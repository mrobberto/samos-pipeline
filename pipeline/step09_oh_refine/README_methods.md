
Here is the clean recap.

Step09 current architecture

The final Step09 design is now:

09 ABAB sky-line subtraction→ 09g merge preferred B1/B2 products→ 09j build ABAB component supertable / empirical sky atlas→ 09h  
- consensus restoration safety layer→ final Step09 product for Step10
- The current science product should be:
- 09_abab/extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits
- Do not use the experimental atlas subtraction or local-refinement products downstream.

What we did after step09_abab_driver.py

1. Patched step09_abab_driver.py

We modified the driver so that each ABAB OH-modeling pass asks step09e to save its fitted Gaussian components.

For B1:
    "--comp-table-csv", str(slit_outdir / "step09_pass1b_components.csv"),

For B2/final:
    "--comp-table-csv", str(slit_outdir / "step09_final_components.csv"),

This means every slit directory now contains component catalogs such as:
    09_abab/SLIT025/step09_pass1b_components.csv09_abab/SLIT025/step09_final_components.csv

The full run produced the expected:
    62 final component tables

2. Patched step09e_iterative_oh_line_model.py

step09e already had the LineComponent structure and a components_to_table() helper, but the component table was not actually written.

We added the final writer:

if args.comp_table_csv is not None:
    args.comp_table_csv.parent.mkdir(parents=True, exist_ok=True)
    comp_df = components_to_table(components)
    comp_df.to_csv(args.comp_table_csv, index=False)
    print(f"WROTE COMPONENT TABLE: {args.comp_table_csv}    (n={len(comp_df)})")

This turned the hidden ABAB fit information into a usable line catalog.

3. Ran ABAB again for all slits

Command:
    PYTHONPATH=. python pipeline/step09_oh_refine/step09_abab_driver.py \
    --in-fits ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/08_extract1d/extract1d_optimal_ridge_all_wav.fits \
    --outdir ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab

This regenerated per-slit ABAB products and component CSVs.

Important outputs per slit:

    step09_pass1a_continuum.fits
    step09_pass1b_oh.fits
    step09_pass1b_components.csv
    step09_pass2a_continuum.fits
    step09_final_abab.fits
    step09_final_components.csv
    step09_preferred.fits
    step09_selection.txt

Global output:
    09_abab/step09_summary.csv

4. Merged preferred ABAB products

Script:
    pipeline/step09_oh_refine/step09g_merge_preferred.py

Purpose:
    Collect each slit’s preferred B1/B2 product into one MEF file.

Canonical merged ABAB file:
    09_abab/extract1d_optimal_ridge_all_wav_step09_abab_preferred.fits

Important columns:
    OBJ_PRESKYOH_MODELSTELLARRESID_POSTOHCONTINUUM_STEP09STEP09_PREF

5. Built the component supertable

New script:
    pipeline/step09_oh_refine/step09j_build_component_supertable.py

Purpose:
    Read all per-slit step09_final_components.csv
    files,cluster fitted components by wavelength, 
    and summarize each empirical sky-line family.

Command:
    PYTHONPATH=. python pipeline/step09_oh_refine/step09j_build_component_supertable.py \  
    --root ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab \  
    --pattern "*final_components.csv" \  
    --out-prefix ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/step09_abab_components \  --tol-nm 0.12

Outputs:
    step09_abab_components_all_components.csv
    step09_abab_components_clustered_components.csv
    step09_abab_components_line_supertable.csv

The full table contained:

Input components: 40260
Total slits: 62
Global line families: 2059

The supertable columns include:
    GLOBAL_LINE_ID
    LAMBDA_MED_NM
    LAMBDA_WMED_NM
    LAMBDA_MEAN_NM
    LAMBDA_STD_NM
    LAMBDA_RSIG_N
    MN_DET
    N_SLITS
    DET_FRAC
    SIGMA_MED_NM
    SIGMA_R
    SIG_NM
    FLUX_MED
    FLUX_RSIG
    FLUX_SUM
    PHASES
    QUALITY_FLAGS
    SLITS

Interpretation:
    N_DET   = total fitted components in that wavelength family
    N_SLITS = number of distinct slits where the family appears
    DET_FRAC = N_SLITS / 62
    LAMBDA_RSIG_NM = robust wavelength scatter
    SIGMA_MED_NM = typical fitted width
    FLUX_MED = typical fitted integrated flux

This is now the empirical OH/component atlas for the dataset.

6. Built the robust sky atlas

From the supertable we selected robust sky-line families:
    sky = t[(t.N_SLITS >= 10) & (t.LAMBDA_RSIG_NM < 0.08)]

Output:
    09_abab/step09_abab_components_sky_atlas.csv

This produced:
    437 robust sky-line families

Interpretation:
    These are the statistically reliable, commonly detected sky-line families.

This file is useful for QC and future constrained fitting, but not currently used as the science subtraction product.

7. Added consensus restoration

Script:
    pipeline/step09_oh_refine/step09h_consensus_restore.py

Purpose:
    Use cross-slit agreement to decide whether ABAB-subtracted features are likely real sky or possibly object-specific features.

This is the “elegant safety layer” we designed.
Input:
    extract1d_optimal_ridge_all_wav_step09_abab_preferred.fits
Output:
    extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits
Command:
    PYTHONPATH=. python pipeline/step09_oh_refine/step09h_consensus_restore.py \
    --in-fits ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/extract1d_optimal_ridge_all_wav_step09_abab_preferred.fits \  
    --out-fits ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits

Outputs:
    extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fit
    extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus_peaks.csv
    extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus_summary.csv

Added FITS columns:
    OH_MODEL_CONSENSUS
    RESTORE_MODEL
    STELLAR_CONSENSUS
    RESID_POSTOH_CONSENSUS
    CONSENSUS_FLAG

Interpretation:
    OH_MODEL_CONSENSUS = ABAB OH model after removing non-consensus/objectlike pieces
    RESTORE_MODEL = part of ABAB model restored to the source spectrum
    STELLAR_CONSENSUS = OBJ_PRESKY - OH_MODEL_CONSENSUS

For SLIT025, consensus restored only two low-S/N features:
    628.078 nm
    670.118 nm

with:
    restored_frac ≈ 0.017

So the consensus effect is small, local, and conservative. That is good.

Scripts sorted by Step09 letter

step09_abab_driver.py

Status: production

Role:

    Main Step09 orchestration.
    Runs A1/B1/A2/B2 for each slit.
    Chooses preferred B1 or B2 by robust residual RMS.
    Now also writes component CSVs through step09e.

Keep in notebook: yes

step09d_twopass_continuum_driver.py

Status: production

Role:

    Continuum estimation used by ABAB.
    A1 estimates continuum on OBJ_PRESKY.   
    A2 refines continuum on STELLAR_P1.

Keep in notebook: called indirectly by ABAB driver

step09e_iterative_oh_line_model.py

Status: production, patched

Role:
    Iterative OH component fitting and subtraction.
    Builds OH_MODEL, STELLAR, RESID_POSTOH.
    Now writes component CSVs.

Keep in notebook: called indirectly by ABAB driver

Important patch to keep:
    --comp-table-csv writer

step09g_merge_preferred.py

Status: production

Role:

    Merge each slit’s step09_preferred.fits into a single MEF.
    Creates the canonical ABAB merged product.

Keep in notebook: yes

Output:

extract1d_optimal_ridge_all_wav_step09_abab_preferred.fits

step09h_consensus_restore.py

Status: production safety layer

Role:
    Cross-slit consensus check.
    Restores ABAB-subtracted features that are not common across slits.

Keep in notebook: yes

Output:

    extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits

This is the file to send to Step10.

step09i_local_refine.py
    
Status: experimental / parked

Role attempted:
    Local line-shape refinement of OH_MODEL.

Result:
    Not stable enough.Fitter was underconstrained.Do not use downstream.

Recommendation:
    mv pipeline/step09_oh_refine/step09i_local_refine.py \   
        pipeline/step09_oh_refine/prototype_step09i_local_refine_experimental.py

Keep in notebook: no

step09j_build_component_supertable.py

Status: production QC / atlas builder

Role:
    Build all-components table, clustered components table, and line supertable.

Keep in notebook: yes

Outputs:
    step09_abab_components_all_components.csv
    step09_abab_components_clustered_components.csv
    step09_abab_components_line_supertable.csv
    step09_abab_components_sky_atlas.csv

step09k_atlas_sky_model.py

Status: experimental / diagnostic only

Role attempted:
    Use empirical atlas as direct sky model.

Result:
    Useful diagnostic, but not better than ABAB.
    Global atlas scaling was too rigid.
    Per-line atlas amplitudes overfit or did not improve science product.

Keep in notebook: no, except optional experiment section

Do not use downstream.

QC we should keep

QC 1. ABAB selection summary

Input:
    09_abab/step09_summary.csv

Keep:
    B1 vs B2 selection count
    RMS_B1, RMS_B2
    DELTA_B2_MINUS_B1

Notebook output:
    print number of B1 and B2 preferred slits
    show histogram of RMS_B2 - RMS_B1

Purpose:
    Verify ABAB selection is sane.

QC 2. Component table existence

Check:
    find ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab \  -name "*final_components.csv" | wc -l

Expected:
    62

Purpose:
    Verify component supertable can be rebuilt.

QC 3. Component supertable summary

Read:
    step09_abab_components_line_supertable.csv

Keep plots:
    N_SLITS vs wavelength
    LAMBDA_RSIG_NM vs wavelength
    SIGMA_MED_NM vs wavelength
    FLUX_MED vs wavelength

Keep printed summary:
    number of global line familiesnumber of robust sky-line families

Current values:
    2059 global line families437 robust sky-line families

Purpose:
    Documents the empirical OH atlas and quantifies line-center/width scatter.

QC 4. Robust sky atlas table

Keep:
    step09_abab_components_sky_atlas.csv

Selection:
    N_SLITS >= 10LAMBDA_RSIG_NM < 0.08

Purpose:
    Empirical list of statistically reliable sky lines.

This is valuable for future Step09 refinement and for the paper/pipeline documentation.

QC 5. Consensus summary

Input:
    extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus_summary.csv

Keep:
    restored_frac vs slitn_objectlike_unique vs slit

Interpretation:
    restored_frac ~ 0: ABAB already saferestored_frac 0.01–0.05: small surgical correctionsrestored_frac > 0.05: inspect manually

Purpose:
    Verifies consensus is conservative and not rewriting spectra.

QC 6. Restored-feature zoom plots

Use:
    extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus_peaks.csv

For each slit, inspect peaks with:
    class == OBJECTLIKE_UNIQUE

Plot around restored features:
    ABAB STELLAR
    CONSENSUS STELLAR
    CONSENSUS - ABAB

Example SLIT025 restored features:
    628.078 nm
    670.118 nm

Purpose:
    Demonstrate that consensus restores only small localized features.

Notebook Step09 final cell sequence

This is what the notebook should contain.
    Cell 1 — Run ABAB
        PYTHONPATH=. python pipeline/step09_oh_refine/step09_abab_driver.py \
          --in-fits ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/08_extract1d/extract1d_optimal_ridge_all_wav.fits \  --outdir ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab
    Cell 2 — Merge preferred
        PYTHONPATH=. python pipeline/step09_oh_refine/step09g_merge_preferred.py \
          --root ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab \
          --out ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/extract1d_optimal_ridge_all_wav_step09_abab_preferred.fits
    Cell 3 — Build component supertable
        PYTHONPATH=. python pipeline/step09_oh_refine/step09j_build_component_supertable.py \
          --root ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab \
          --pattern "*final_components.csv" \
          --out-prefix ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/step09_abab_components \
          --tol-nm 0.12
    Cell 4 — Build robust sky atlas
        import pandas as pdbase = "../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/"
        t = pd.read_csv(base + "step09_abab_components_line_supertable.csv")
        sky = t[(t.N_SLITS >= 10) & (t.LAMBDA_RSIG_NM < 0.08)].copy()
        sky = sky.sort_values("LAMBDA_MED_NM")
        sky.to_csv(base + "step09_abab_components_sky_atlas.csv", index=False)
        print("Robust sky atlas lines:", len(sky))
    Cell 5 — Consensus restoration
        PYTHONPATH=. python pipeline/step09_oh_refine/step09h_consensus_restore.py \
          --in-fits ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/extract1d_optimal_ridge_all_wav_step09_abab_preferred.fits \  
          --out-fits ../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits

    Cell 6 — Declare Step09 product
        STEP09_FINAL = "../_Run8_Science_2026_01/SAMI/Dolidze25/reduced/09_abab/extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits"
        print("Step09 final product:", STEP09_FINAL)

Final decision
    Step09 is no longer half-finished if we freeze it this way.

Production path
    ABAB → merge preferred → consensus restore

Production output
    extract1d_optimal_ridge_all_wav_step09_abab_preferred_consensus.fits

Diagnostic/metadata products
    step09_abab_components_line_supertable.csv
    step09_abab_components_sky_atlas.csv
    consensus_summary.csv
    consensus_peaks.csv

Parked experiments
    step09i_local_refine.pystep09k_atlas_sky_model.py
These are valuable, but not part of the current pipeline.