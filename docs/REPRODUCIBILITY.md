# Final manuscript reproduction

Run Python commands from the repository root. Core entry points work with Python 3.11; some recovered provenance-only sources use Python 3.12+ syntax. R lock and historical receipts are retained. `requirements-lock.txt` pins the tested analysis/plot environment. Recovered complete source chains may require additional source-project R/Numba dependencies and inputs; they are not advertised as immediately runnable kernels.

## Public reference and kernel checks

```sh
MANUSCRIPT_PYTHON=.venv/bin/python Rscript analysis/99_validate_release.R
```

This performs public aggregate and saved-distribution reaggregation, membership checks, exact-support checks, a synthetic CRC04 example, the float precision regression, and the spatial normalization regression. It does not certify external observation preprocessing or replay full Monte Carlo experiments. ACT4 frozen reference worlds are projections of simulation output; patient labels and observation permutation maps are excluded.

## External inputs

`GBM_SOURCE_ROOT` is an external copy of the source project layout; `GBM_DATA_ROOT` contains source datasets; `GBM_VALIDATION_ROOT` contains the previous independent audit and final figure implementation package. These are separate from the public repository. Never point write-producing recovered full source scripts at the protected historical archive. Use a disposable source-project copy and a separate output directory. The curated focused commands below read source inputs and write only explicit outputs.

```sh
python analysis/act1/reproduce_observed.py --expression-dir "$GBM_SOURCE_ROOT/GBM标签数据库/GBM_unified_registry_de_novo_benchmark_AuthorityV2_2026-07-23/09_de_novo_derived/03_preprocessed_expression" --output outputs/act1
python analysis/ivy/reproduce_q2.py --scores "$GBM_SOURCE_ROOT/IVYGAP_PATHOLOGY_ANATOMY_EXTENSION_V1/04_SCORES/IVYGAP_PRIMARY201_MODEL_SCORES.tsv.gz" --output outputs/ivy_q2
python analysis/care/reproduce_baseline.py --source-root "$GBM_SOURCE_ROOT" --output outputs/care
Rscript analysis/06_crc04_exact_inference.R --source-rds /external/Nanostring_GeoMx/exp2/r_object/geomx_exp2_target_Data_norm.RDS
```

ACT1 uses externally prepared, untransformed FPKM matrices (73/108 patients), gene z scores and unsigned/unweighted membership means. The log2 arm is a sensitivity. This is prepared-input reproduction, not full source-data preparation. Q2 must run ALL5, CORE3 and PAN_MVP in the frozen order with one RNG stream. CORE3 is CT/PAN/MVP. CARE uses 56 T1 ALL_AUTHOR_ANNOTATED patients and the recorded 5-fold x100 split rule. CRC04 reads the public processed object from Zenodo 17671259 (access/hash contract in DATA_ACCESS.md); it does not require study data to be redistributed here.

## Final panel source extraction and artwork

```sh
export GBM_PANEL_OUT=outputs/panel_source
python audit_tools/ivy_gene_profiles.py
python audit_tools/ivy_panel_source.py
python audit_tools/overlap_panel_source.py
python audit_tools/act4_resolution_recount.py
export GBM_FIGURE_OUT=outputs/final_figures
python analysis/figures/build_fig1_panels.py
python analysis/figures/build_fig5_panels.py
python analysis/figures/compose_figures.py
python analysis/figures/validate_figures.py
```

K03 recomputes unsigned scores and gene profiles from the external Ivy 122-well gene-z matrix. K05 derives tumour-centred profiles from external frozen scores; its partial R2/BA fields are copied frozen statistics. These scripts output observation-derived intermediates locally; do not publish them without source-specific permission. Figure builders read public aggregate references and the externally retained sample/tumour tables. `GBM_VALIDATION_ROOT/文稿/figure/Figure1.pdf` and `GBM_VALIDATION_ROOT/GBM_NPJSBA_REVISION/02_DELIVERABLES/Figures_revised/Figure5_NPJSBA_revised.pdf` are base artwork dependencies; they are opened read-only and superseded spatial objects are removed. Figure 5's layout JSON is included. Other four final figures are preserved artwork; the release does not promise regeneration of all panels from raw data.

K04 reaggregates the previous independent 500-panel reference; K02 recounts saved 999-draw / exact-blocked / 200,000-draw-naive results. Neither silently launches a new full simulation. See the scientific changelog for historical denominators, precision and resolution distinctions.

## Source recovery

`SOURCE_RECOVERY_INVENTORY.tsv` identifies all 51 recovered source chains with original hashes and prior execution evidence. `ADDED_FILE_SOURCES.tsv` maps curated files to their source and adapted hashes. Recovered originals preserve source-project directory contracts; they are inspection/source-recovery material until the required external layout and dependencies are supplied. Private table/governance builders stay on request. Finding an author generator and rerunning it is not independent replication. The 760-gene restricted pseudobulk inputs are external dependencies, not public full-gene reconstructions.
