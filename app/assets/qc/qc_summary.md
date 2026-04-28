# QC Validation Summary (bundled snapshot)

This file ships with the app so the Quality Control tab can display **example** results when `qc/outputs/` has not been generated yet. Run `python qc/run_qc_experiment.py --mode compare --with-grader` to refresh `qc/outputs/qc_results.csv` and `qc_summary.md` for your environment.

## 1. Absolute validity

**Overall (all rows in bundled snapshot):**
- Report quality score (mean): ~84.8
- Pass rate: ~50% (mixed baseline + grounded)

**By prompt type (bundled):**
- **Baseline Prompt:** mean report quality ~77, pass rate low
- **Grounded Prompt (Production):** mean report quality ~94, pass rate 100% in this slice

## 2. Baseline vs Grounded Prompt (Production)

Paired comparison on the same trial IDs shows higher scores and pass rate for the grounded prompt. For full statistics, run the offline QC batch and open the generated `qc/outputs/qc_summary.md`.

## 3. Note

Replace bundled files by writing your own run into `qc/outputs/`; the dashboard will use those automatically when present.
