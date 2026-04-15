 1. pipeline_config.py 

Shows clearly separated:

- generic infrastructure
- canonical filenames
- directory conventions

# Canonical pipeline product filenames

This is he single source of truth for naming


2. run8_dolidze25.py 

Key Rule:

target config = directories + composition of generic names

EXTRACT1D_OHCLEAN = ST09_OH_REFINE / NAME_EXTRACT1D_OHCLEAN
EXTRACT1D_TELLCOR = ST10_TELLURIC / NAME_EXTRACT1D_TELLCOR
EXTRACT1D_FLUXCAL = ST11_FLUXCAL / NAME_EXTRACT1D_FLUXCAL


3. Compatibility aliases 

This section is clean

# Temporary compatibility aliases
👉 Leave it for now, but plan to delete later.

4. Potential future cleanup (not urgent)

You still have:

ST09_TELLURIC = ST10_TELLURIC
ST10_OH_REFINE = ST09_OH_REFINE

These are historical artifacts.

Not wrong, but:
- they encode old semantics
- can confuse future readers

Keep for now
❗ Remove when driver + scripts are fully migrated

3. template_target_config.py — excellent

This is now exactly what you want for future targets.
The example:

EXTRACT1D_OHCLEAN = ST09_OH_REFINE / NAME_EXTRACT1D_OHCLEAN

is perfect.

👉 This file is now a real template, not just a stub.


Big-picture validation

You now have a clean 3-layer architecture:

1. Generic pipeline layer

pipeline_config.py

naming conventions
directory structure
helpers

✔ reusable across all targets

2. Target profile layer

run8_dolidze25.py

filesystem roots
file lists
target-specific composition of paths

✔ scalable to Dolidze26, Dolidze27, etc.

3. Template layer

template_target_config.py

enforces discipline
prevents duplication

✔ ensures future consistency

You can now:
✔ Add a new target cleanly
cp template_target_config.py run8_dolidze26.py

Edit only:

TARGET_NAME
TARGET_ROOT
file lists

👉 No touching pipeline logic

✔ Run pipeline unchanged

Driver + scripts use:

config.EXTRACT1D_TELLCOR
config.STEP11_INPUT_SPECTRA

→ automatically correct per target

✔ Keep notebooks clean

Notebook can now do:

INFILE = config.STEP11_INPUT_SPECTRA

No hardcoding.

10_telluric_official → 10_telluric
09_abab_official     → 09_oh_refine

👉 pipeline stays clean

Option B (temporary)

Override in config:

ST10_TELLURIC = REDUCED_DIR / "10_telluric_official"

👉 works, but pollutes config

✅ Final verdict

You now have:

✔ correct abstraction
✔ no duplication
✔ future-proof target configs
✔ canonical naming centralized
✔ compatibility clearly isolated