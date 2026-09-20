# Minimal environment

Certified: R4.4.1; Python3.11; macOS arm64. Retained R code uses base/recommended methods,stats,utils and digest0.6.39 only. renv.lock contains only digest; renv is a restoration tool. A clean isolated library restore and validation were run.

Retained Python imports:numpy1.26.4,pandas2.2.3,scipy1.14.1,scikit-learn1.5.2. PyArrow18.1.0 is required by pandas.read_parquet in module07. requirements.txt lists direct/actual runtime dependencies; requirements-lock.txt pins their transitive runtime dependencies. No visualization packages or historical framework dependencies are retained.

MeasureQual v0.10.0 (https://github.com/ZichenSheng/MeasureQual; version DOI 10.5281/zenodo.22846512; concept DOI 10.5281/zenodo.22846292) is the current independent software reference. It is not imported or vendored by these manuscript modules.
