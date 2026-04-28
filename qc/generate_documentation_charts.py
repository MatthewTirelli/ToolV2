#!/usr/bin/env python3
"""
Generate Plotly HTML used by qc/README.md (summary statistics figure).
Run from repo root: python qc/generate_documentation_charts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "qc" / "outputs" / "qc_results.csv"
OUT = ROOT / "qc" / "outputs" / "qc_charts" / "summary_statistics_by_mode.html"


def main() -> None:
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    if not CSV.is_file():
        print(f"Skip: {CSV} not found")
        return
    df = pd.read_csv(CSV)
    if df.empty or "mode" not in df.columns:
        print("Skip: empty or missing mode")
        return
    d = df.copy()
    d["mode"] = d["mode"].astype(str).str.lower().map(
        {"baseline": "Baseline Prompt", "grounded": "Grounded Prompt (Production)"}
    )
    modes = [m for m in ("Baseline Prompt", "Grounded Prompt (Production)") if m in set(d["mode"])]

    metrics = [
        ("Report quality score (mean)", "validity_score_0_100", 0, 100),
        ("Pass rate (mean)", "passed_absolute_validity", 0, 1),
        ("Numeric alignment (mean)", "numeric_accuracy_rate", 0, 1),
        ("Required sections (mean)", "required_sections_rate", 0, 1),
        ("Top-state coverage (mean)", "top_state_coverage_rate", 0, 1),
        ("Confidence label match (mean)", "confidence_label_match_rate", 0, 1),
    ]

    fig = make_subplots(
        rows=len(metrics),
        cols=1,
        subplot_titles=[m[0] for m in metrics],
        vertical_spacing=0.07,
    )
    for i, (title, col, lo, hi) in enumerate(metrics, start=1):
        if col not in d.columns:
            continue
        sub = d.groupby("mode", sort=False)[col].mean().reindex(modes)
        vals = pd.to_numeric(sub, errors="coerce")
        if col == "passed_absolute_validity":
            vals = vals.astype(float)
        fig.add_trace(
            go.Bar(x=sub.index.astype(str), y=vals, name=title, showlegend=False, marker_color=["#94a3b8", "#2563eb"]),
            row=i,
            col=1,
        )
        fig.update_yaxes(range=[lo, hi * 1.05] if hi <= 1 else [lo, min(100, float(vals.max()) * 1.1 + 5)], row=i, col=1)
    fig.update_layout(
        height=200 * len(metrics) + 80,
        title_text="Summary statistics by prompt type (mean over trials)",
        title_x=0.5,
        font=dict(size=11, color="#334155"),
        paper_bgcolor="#fafafa",
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(OUT), include_plotlyjs="cdn", full_html=True)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
