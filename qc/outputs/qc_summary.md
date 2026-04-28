# QC Validation Summary

## 1. Absolute validity

**Overall (all outputs):**
- Report quality score (mean): 85.26
- 95% CI: [83.32, 87.21]
- Pass rate: 55.0%

**By prompt mode:**

**Baseline Prompt:**
- Report quality score (mean): 76.27
- Pass rate: 10.0% [95% CI: 4.3–21.4]
- Structural error rate: 0.0%

**Grounded Prompt (Production):**
- Report quality score (mean): 94.25
- Pass rate: 100.0% [95% CI: 92.9–100.0]
- Structural error rate: 0.0%

Grounded Prompt (Production) reflects the deployed configuration; its pass rate is the primary production pass-rate estimate.

Grounded Prompt (Production) achieved higher report quality and pass rates than Baseline Prompt.

- Strict pass criteria are demanding: high report quality scores can still fail pass if required sections, top-state coverage, or numeric fidelity thresholds are not met.
- Structural error signals flag explicit deviations (e.g., unsupported state codes, numeric mismatches); they do not capture all semantic errors.

## Pass-rate comparison

- Baseline Prompt pass rate: 10.0% [95% CI: 4.3–21.4]
- Grounded Prompt (Production) pass rate: 100.0% [95% CI: 92.9–100.0]
- Difference (Grounded − Baseline): 90.0 pp [95% bootstrap CI: 82.0–98.0]
- McNemar (exact binomial): p = 5.684e-14 (discordant pairs: baseline pass / grounded fail = 0, baseline fail / grounded pass = 45).

Pass rate improvement for Grounded Prompt (Production) vs Baseline Prompt is statistically significant.


## 2. Baseline Prompt vs Grounded Prompt (Production)
- Baseline Prompt report quality (mean): 76.27
- Grounded Prompt (Production) report quality (mean): 94.25
- Paired t-statistic: 21.37
- Effect size (Cohen's d): 4.27

## 3. Confidence validation
Confidence validation is limited: nearly all trials share one dominant confidence label, so cross-label calibration is not established.

Summary by dominant confidence label (structured payload):
- High: report quality score 85.26, numeric alignment 81.6%, pass rate 55.0%

Confidence score correlation was not computed (missing or insufficient variation in confidence scores).

## 4. Failure Mode Analysis

The most common issues were minor numeric inconsistencies, missing sections, and confidence wording variations.

| Failure mode | Count | Share |
|---|---:|---:|
| Grader: confidence_misuse non-empty | 56 | 56.0% |
| Missing required sections | 50 | 50.0% |
| Grader: unsupported_claims non-empty | 50 | 50.0% |
| Numeric accuracy below 0.95 | 49 | 49.0% |
| Did not pass absolute validity | 45 | 45.0% |
| Hallucinated state mention(s) | 0 | 0.0% |
| Hallucinated number signal(s) | 0 | 0.0% |
| Incomplete top-state coverage | 0 | 0.0% |
| Confidence label match below 1.0 | 0 | 0.0% |
| Low/moderate confidence disclosure below 1.0 (where applicable) | 0 | 0.0% |
| Unsupported structured confidence claim(s) | 0 | 0.0% |

Confidence Wording Issues (grader) may reflect strict phrasing review rather than contradiction of structured labels.

The most common failure signals were: Grader: confidence_misuse non-empty (56.0% of rows); Missing required sections (50.0% of rows); Grader: unsupported_claims non-empty (50.0% of rows). Counts can overlap (one row may trigger multiple flags). Use the table above and per-row metrics in `qc_results.csv` to prioritize fixes.

## 5. Row-level review
- Inspect `qc_results.csv`: Potential Unsupported Statements, Confidence Wording Issues, grader payload, and report text (column names remain machine-readable).

## 6. Example outputs

```text
    mode  trial_id  validity_score_0_100  hallucinated_state_count  hallucinated_number_count
baseline         1                 79.42                         0                          0
grounded         1                 94.25                         0                          0
baseline         2                 71.08                         0                          0
```