# Scientific code changelog

## 1.1.0 — 2026-10-27

This minor release adds final NPJSBA analysis and figure reproduction entry points and corrects the Ivy Q2 precision defect. The public interface gains substantial analysis capabilities, so a minor version is appropriate. Historical v1.0.0, its tag and DOI remain unchanged.

### Corrected implementation

Ivy `q2_identifiability` now stores permutation balanced accuracy in float64, matching observed balanced accuracy. Float32 could round exact ties downward beyond the existing 1e-12 tolerance. The inclusive upper tail, plus-one P value, 2,000 within-tumour permutations, leave-one-tumour-out classifier, seed 2026083102, branch order and branchwise BH correction are unchanged. Production replay yields ALL5=170, CORE3=134 and PAN_MVP=98 at BH FDR<0.05 (historical float32 PAN_MVP=117). A synthetic tie regression detects the old defect. The recovered defective implementation is retained in `historical/ivy_q2_float32.py`; v1.0.0 did not publish this full generator, and its existing kernels and original release remain accessible through Git history.

The spatial per-spot normalization kernel uses explicit column division instead of recycling a library-size vector across a genes-by-spots matrix. An inverse-transform regression detects the original error. The exact binomial alternative is explicitly two-sided; BH covers 600 tests. This is historical diagnostic code, not a return of the withdrawn scientific claim. Corrected niche positivity was also seen in size-matched random sets and was confounded by sequencing depth. No new exploratory rescue is attempted.

CARE's focused WP1C entry point calls the recovered original repeated-fivefold function on external prepared inputs. Its aggregate is read from `mean_CV_R2`, with no literal fallback. The replay gives median 0.0903096094913944 (201 representations; 56 primary CARE patients; 100 repetitions). This is within-CARE held-out-patient prediction, not cross-cohort transfer; inherited transductive normalization and endpoint gene overlap remain limitations.

### Newly public analysis and figure code

ACT1 observations can be regenerated from the external prepared linear-FPKM CGGA matrices, including programme-gene removal and the 201/160/114 sensitivity denominators. The new Figure 1D is post hoc. The original primary null has 1,000 author panels; the new independent overlap sensitivity has 500 panels. Public null tables permit reaggregation of the 500-panel reference (0/500 at the principal removed-gene endpoints, plus-one P=1/501, displayed 0.002). This release does not repeat that full simulation. The observed ecological alignments are 0.939/0.957 before and 0.916/0.933 after removal; removed-gene cross-cohort structure is 0.691.

Ivy K03/K05 source extractors and the final Figure 5D generator retain unsigned, unweighted membership scores. Original predictor coefficients are display annotations, not score weights. External gene-z inputs reproduce the membership scores; tumour-centred means yield Liu MVP>CT in 9/9 tumours, Xiao MVP<CT in 9/9 (PAN>CT in 8/8), and Arimappamagan LE>CT in 8/8. K05 copies frozen partial R2 and balanced-accuracy statistics explicitly; it does not refit those models. Well/tumour-level scores stay external. Final Figures 1 and 5 are rebuilt with the recovered layout configuration; base artwork remains an external figure input. Figures 2, 3, 4 and 6 are preserved final artwork, with source lineage recorded rather than a claim of full plot reconstruction.

### Frozen outputs and Monte Carlo resolution

Historical 999-draw ACT4 results are retained separately from the independent 200,000-draw naive sensitivity and exact donor-blocked calculations (620 enumerated worlds). Exact-support qualification is distinct from enumeration of an exact P value. Formal all-q detection categories are 140/114/4/302 at historical resolution and 142/114/3/301 at higher resolution. q100 naive counts for Bao transcription-17 are 12/20 versus 6/20; adhesion-8, 19/20 versus 20/20; stringent-5, 4/20 versus 3/20. Figure 4 and Figure 5F/G retain 999-draw estimates; SI Table 4 also shows sensitivity results. Per-condition 20 worlds are near-replicates of one alternative. Figure 5G true-null naive/blocked counts are 0/5, 0/8 and 55/3 of 100; the three protagonists' detection decisions do not change at higher resolution. Full expensive experiments are not repeated by the reaggregation validator.

### Provenance and limitations

The evidence inventory confirms all 51 formerly untraced generators: 36 prior independent reconstructions and 15 prior author-code replays. Publishing source is a separate field in the crosswalk. Scientific source chains are curated; private governance/table builders are withheld. Source-dependent originals are identified as such and are not automatically certified end-to-end public workflows. The corrected 34-paper CGGA registry preserves 19/32 documented CGGA users; the former naive32 subset is not proof of development independence. CRC04 retains its real local hash chain, exact 9-patient one-sided sign-flip P=16/512, and four intermediate exact-equality discrepancies; local hash-pinning is not independent external timestamping. Rule-based Figure 6 ledger rows are not MeasureQual executions.

No raw or observation-level study data are redistributed. Simulation-ledger projections omit patient block labels and observation permutation mappings. Restricted inputs, private correspondence and copyrighted paper full texts are excluded. MeasureQual stays v0.10.0, DOI 10.5281/zenodo.22846512.
