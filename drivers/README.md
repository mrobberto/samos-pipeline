# SAMOS Pipeline Drivers

## Overview

This folder contains the **execution layer** of the SAMOS data reduction pipeline.

The pipeline is organized as a sequence of stages (00 → 11), each implemented
as a standalone script under `pipeline/`. The drivers in this folder orchestrate
their execution.

---

## Main Entry Point

### `run_pipeline.py`

This is the **canonical driver** for the pipeline.

It:
- runs all pipeline stages in the correct order
- uses the active target configuration (`config.target_config`)
- supports partial execution (`--from-step`, `--to-step`, `--only`)
- supports EVEN / ODD / ALL selection for set-based stages
- optionally runs QC scripts after each stage (`--run-qc`)
- validates outputs using config-defined canonical products

### Example usage

Run full pipeline (recommended standard):

```bash
PYTHONPATH=. python drivers/run_pipeline.py --from-step 04 --to-step 11c

Run only the science post-processing:

PYTHONPATH=. python drivers/run_pipeline.py --from-step 09 --to-step 11c

Run flux calibration only:

PYTHONPATH=. python drivers/run_pipeline.py --from-step 11a --to-step 11c


Configuration

The driver relies on the active configuration module:

config.target_config

This selects a target-specific profile, e.g.:

--config config.reductions.run8_dolidze25

The configuration defines:

directory structure
canonical filenames
target-specific inputs and outputs
QC Integration

QC scripts are automatically executed when:

--run-qc

QC scripts are defined in the internal QC_REGISTRY and are expected to:

produce diagnostic plots (PDF/PNG)
validate intermediate and final products
Notebooks

The Jupyter/ subfolder contains interactive notebooks that:

run individual pipeline stages
display intermediate results
provide visual QC

These notebooks are intended for:

development
debugging
science validation

They are not the primary pipeline execution mechanism.

Legacy Scripts

Files such as:

run_pipeline_b.py
run_pipeline_full.py
run_step*_*.py

are legacy helper scripts from earlier pipeline iterations.

They may:

use outdated stage structure
bypass config-driven logic
duplicate functionality now handled by run_pipeline.py

These should be considered deprecated and may be moved to an archive/ folder.

Design Principles
The driver does not implement science logic
All science operations live in pipeline/stepXX_*
The driver only:
orchestrates execution
resolves configuration
enforces file flow
runs QC
Maintenance Notes

When updating the pipeline:

Update SCRIPT_REGISTRY if stage structure changes
Update QC_REGISTRY if QC scripts change
Update OUTPUT_CHECKS if canonical outputs change

Always prefer:

config-defined paths (e.g. EXTRACT1D_OHCLEAN)
over hardcoded filenames