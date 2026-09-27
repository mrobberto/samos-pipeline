# Region Reference Tables

This directory contains the reference slit-position tables used by the SAMI spectroscopic pipeline.

## Current reference tables

- `radec_Even.csv`
- `radec_Odd.csv`

These are the canonical reference tables used by the pipeline.

## Provenance

The current tables were derived from the original `radec_Even.csv` and `radec_Odd.csv` files.

The original tables were imported into the spreadsheet `Regions_Fixer.xlsx`, combined into a single list, and sorted by Right Ascension (RA). This process revealed errors in the original ordering. The corrected master list was then separated back into EVEN and ODD trace sets, producing the current `radec_Even.csv` and `radec_Odd.csv` files.

## Originals

The `originals/` directory preserves the intermediate region files and source catalogs used during the construction of the corrected reference tables. They are retained for provenance and should normally not be modified.