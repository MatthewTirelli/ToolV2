# QC Validation Methodology

## Objective
This framework validates final AI reports from the multi-agent measles pipeline along three dimensions:
1. Absolute validity against structured source-of-truth inputs
2. Comparative improvement (baseline prompt vs grounded prompt)
3. Confidence signal propagation/calibration checks

## Scope
Evaluation targets the final generated report text and compares it against a normalized truth object extracted from pipeline payloads.

## Truth extraction
`qc.validators.extract_ground_truth()` builds a compact schema:
- `top_states`: top three states with risk, case, coverage, wastewater, confidence label/score
- `national_risk_level`

Parsing is flexible and robust to minor payload layout differences.

## Deterministic validation
`qc.validators.validate_report()` computes:
- State coverage and required section presence
- Numeric fidelity and hallucination proxies
- Risk/wastewater/enrichment fidelity
- Confidence usage checks:
  - label alignment
  - low/moderate uncertainty disclosure
  - score fidelity
  - unsupported confidence claims

## Scoring
`qc.scoring.compute_validity_score()` applies weighted scoring:
- Numeric fidelity: 25
- State coverage: 15
- Risk fidelity: 10
- Wastewater fidelity: 10
- Enrichment usage: 10
- Confidence fidelity: 15
- Required sections: 10
- Concision: 5

Hallucination penalty up to -25.

## Experiment execution
`qc/run_qc_experiment.py` supports:
- `baseline`
- `grounded`
- `compare`

Each trial uses the same structured ground truth for both prompt modes, then evaluates and stores trial metrics.

## Statistical analysis
`qc/statistical_analysis.py` computes:
- Mean validity, 95% CI, pass and hallucination rates
- Paired baseline-grounded comparisons (paired t-stat, effect size)
- Confidence validation groups (high/moderate/low)
- Correlation of confidence score vs validity

## Outputs
- `qc/outputs/qc_results.csv`
- `qc/outputs/qc_summary.md`
- `qc/outputs/qc_charts/*.html`

If no experiment has been run, output files remain placeholders until execution.
