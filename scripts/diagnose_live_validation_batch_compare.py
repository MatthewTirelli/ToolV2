#!/usr/bin/env python3
"""
Compare live-style vs batch QC-style validation on a grounded row from qc_results.csv.

Usage (from repo root):
  python scripts/diagnose_live_validation_batch_compare.py

Requires: qc/outputs/qc_results.csv with at least one mode=grounded row.
Does not call OpenAI. Rebuilds trial payload via build_trial_payload(seed=trial_id);
if upstream data changed since the CSV was produced, truth may drift from the original QC run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd

from qc.run_qc_experiment import build_trial_payload
from qc.scoring import compute_validity_score
from qc.validators import GROUNDED_SECTION_HEADERS, extract_ground_truth, validate_report


def main() -> None:
    csv_path = ROOT / "qc" / "outputs" / "qc_results.csv"
    if not csv_path.is_file():
        print("No qc_results.csv; nothing to compare.")
        sys.exit(1)
    df = pd.read_csv(csv_path)
    g = df[df["mode"].astype(str).str.lower() == "grounded"]
    if g.empty:
        print("No grounded rows in qc_results.csv.")
        sys.exit(1)
    row = g.iloc[0]
    trial_id = int(row["trial_id"])
    report_text = str(row.get("report_text", "") or "")
    stored_score = row.get("validity_score_0_100")

    try:
        trial_payload = build_trial_payload(seed=trial_id)
    except RuntimeError as e:
        print(
            json.dumps(
                {
                    "error": str(e),
                    "hint": "Rebuild needs `load_and_model()` (CDC fetches). Run with network from repo root.",
                },
                indent=2,
            )
        )
        sys.exit(2)

    gt = extract_ground_truth(trial_payload, top_n=3)

    m_live = validate_report(report_text=report_text, ground_truth=gt)
    m_batch = validate_report(
        report_text=report_text,
        ground_truth=gt,
        section_headers=GROUNDED_SECTION_HEADERS,
    )
    s_live = compute_validity_score(m_live, concision_score_1_5=None)
    s_batch = compute_validity_score(m_batch, concision_score_1_5=None)

    shape = {
        "has_national_overview": "## National Overview" in report_text,
        "has_executive_summary": "## Executive summary" in report_text,
    }
    out = {
        "trial_id": trial_id,
        "report_section_shape": shape,
        "stored_csv_validity_score_0_100": float(stored_score) if pd.notna(stored_score) else None,
        "replay_live_expander_headers_default": {
            "validity_score_0_100": s_live.get("validity_score_0_100"),
            "required_sections_rate": m_live.get("required_sections_rate"),
        },
        "replay_batch_grounded_headers": {
            "validity_score_0_100": s_batch.get("validity_score_0_100"),
            "required_sections_rate": m_batch.get("required_sections_rate"),
        },
        "interpretation": (
            "Batch QC grounded uses GROUNDED_SECTION_HEADERS; the Streamlit live expander calls validate_report "
            "without section_headers (STATE_SECTION_HEADERS). If the markdown uses grounded-style headers, "
            "live `required_sections_rate` will be low vs batch. If the markdown uses baseline-style headers "
            "on a grounded row, the opposite can happen. Stored CSV scores used the batch header set for that mode "
            "and may not match these replays if `extract_ground_truth` inputs drift from the original QC run."
        ),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
