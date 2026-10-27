# GBM signature measurement manuscript reproducibility

Code supporting **Transcriptomic signatures preserve biological direction without uniquely specifying state in glioblastoma**, synchronized with the final targeted-edit NPJSBA manuscript and final Figures 1–6.

Version 1.1.0 adds corrected Ivy inference, ACT1 overlap sensitivity, Ivy anatomical-profile extraction, final Figure 1/5 assembly, CARE baseline replay and ACT4 resolution-specific validation. This is claim-bounded reproducibility: public aggregate/distribution reaggregation, calculations on external prepared observations, and public-processed-input CRC04 inference are explicitly distinguished. Complete regeneration from redistributed raw databases is not claimed.

See [scientific changes](CHANGELOG.md), [final reproduction commands](docs/FINAL_REPRODUCIBILITY.md), [claim crosswalk](docs/RESULT_CODE_CROSSWALK.tsv), [data access](docs/DATA_ACCESS.md) and [figure map](docs/RESULT_MAP.md). The crosswalk has 81 current entries and four withdrawn historical entries. `provenance/SOURCE_RECOVERY_INVENTORY.tsv` records the previous 51-generator recovery (36 independent reconstructions, 15 author-code replays); that classification is not a claim of independent validation of every file in this release.

```sh
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-lock.txt
MANUSCRIPT_PYTHON=.venv/bin/python Rscript analysis/99_validate_release.R
```

The default suite reaggregates public references and checks numerical kernels; external-input and figure commands are separate. See the executed receipts in `reproducibility/FINAL_VALIDATION.tsv` for the exact scope of each PASS. Output directories and external inputs are ignored by Git.

- `analysis/`: current numerical entry points and final Figure 1/5 builders.
- `audit_tools/`: K02–K05 extraction and verification sources; not new scientific analyses.
- `results_reference/final/`: derived aggregate/distribution and simulated-world reference tables.
- `provenance/recovered_generators/`: curated recovered scientific sources; source-dependent provenance, not a one-command installation.
- `historical/`: superseded references and withdrawn spatial source, excluded from active figures and claims.

The original [v1.0.0](https://github.com/ZichenSheng/GBM-signature-measurement/releases/tag/v1.0.0) remains archived at [version DOI 10.5281/zenodo.22851963](https://doi.org/10.5281/zenodo.22851963); the repository's concept DOI is [10.5281/zenodo.22851962](https://doi.org/10.5281/zenodo.22851962). The new version DOI is recorded after Zenodo archives the immutable release.

[MeasureQual v0.10.0](https://github.com/ZichenSheng/MeasureQual) remains a separate package, archived at [10.5281/zenodo.22846512](https://doi.org/10.5281/zenodo.22846512).
