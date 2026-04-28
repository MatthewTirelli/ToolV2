from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px


def _fmt_pass_rate_wilson(row: Dict[str, Any]) -> str:
    pr = 100.0 * float(row.get("pass_rate", 0.0))
    lo = 100.0 * float(row.get("pass_rate_wilson_low", 0.0))
    hi = 100.0 * float(row.get("pass_rate_wilson_high", 0.0))
    return f"{pr:.1f}% [95% CI: {lo:.1f}–{hi:.1f}]"


def write_summary_markdown(summary: Dict[str, Any], df: pd.DataFrame, out_path: Path) -> None:
    absv = summary.get("absolute_validity", {})
    comp = summary.get("comparative", {})
    conf = summary.get("confidence_validation", {})
    fma = summary.get("failure_mode_analysis") or {}
    by_mode = absv.get("by_mode") or {}
    prc = summary.get("pass_rate_comparison")

    lines = [
        "# QC Validation Summary",
        "",
        "## 1. Absolute validity",
        "",
        "**Overall (all outputs):**",
        f"- Report quality score (mean): {absv.get('mean_score', 0):.2f}" if absv else "- Report quality score (mean): n/a",
        f"- 95% CI: [{absv.get('ci95_low', 0):.2f}, {absv.get('ci95_high', 0):.2f}]" if absv else "- 95% CI: n/a",
        f"- Pass rate: {100*absv.get('pass_rate', 0):.1f}%",
        "",
        "**By prompt mode:**",
        "",
    ]
    if by_mode.get("baseline"):
        b = by_mode["baseline"]
        lines.extend(
            [
                "**Baseline Prompt:**",
                f"- Report quality score (mean): {b['mean_validity']:.2f}",
                f"- Pass rate: {_fmt_pass_rate_wilson(b)}",
                f"- Structural error rate: {100*b['hallucination_rate']:.1f}%",
                "",
            ]
        )
    else:
        lines.extend(["**Baseline Prompt:**", "- _No baseline rows in this run._", ""])

    if by_mode.get("grounded"):
        g = by_mode["grounded"]
        lines.extend(
            [
                "**Grounded Prompt (Production):**",
                f"- Report quality score (mean): {g['mean_validity']:.2f}",
                f"- Pass rate: {_fmt_pass_rate_wilson(g)}",
                f"- Structural error rate: {100*g['hallucination_rate']:.1f}%",
                "",
            ]
        )
    else:
        lines.extend(["**Grounded Prompt (Production):**", "- _No grounded rows in this run._", ""])

    lines.extend(
        [
            "Grounded Prompt (Production) reflects the deployed configuration; its pass rate is the primary production pass-rate estimate.",
            "",
        ]
    )

    if by_mode.get("baseline") and by_mode.get("grounded"):
        b, g = by_mode["baseline"], by_mode["grounded"]
        if g["mean_validity"] > b["mean_validity"] and g["pass_rate"] > b["pass_rate"]:
            lines.extend(
                [
                    "Grounded Prompt (Production) achieved higher report quality and pass rates than Baseline Prompt.",
                    "",
                ]
            )

    lines.extend(
        [
            (
                "- Strict pass criteria are demanding: high report quality scores can still fail pass if required sections, "
                "top-state coverage, or numeric fidelity thresholds are not met."
            ),
            (
                "- Structural error signals flag explicit deviations (e.g., unsupported state codes, numeric mismatches); "
                "they do not capture all semantic errors."
            ),
            "",
            "## Pass-rate comparison",
            "",
        ]
    )

    if prc:
        bpr = 100.0 * float(prc.get("baseline_pass_rate", 0.0))
        gpr = 100.0 * float(prc.get("grounded_pass_rate", 0.0))
        dpr = 100.0 * float(prc.get("pass_rate_difference", 0.0))
        dlo = 100.0 * float(prc.get("difference_ci95_low", 0.0))
        dhi = 100.0 * float(prc.get("difference_ci95_high", 0.0))
        w_b = by_mode.get("baseline") or {}
        w_g = by_mode.get("grounded") or {}
        b_w = _fmt_pass_rate_wilson(w_b) if w_b else f"{bpr:.1f}%"
        g_w = _fmt_pass_rate_wilson(w_g) if w_g else f"{gpr:.1f}%"
        lines.append(f"- Baseline Prompt pass rate: {b_w}")
        lines.append(f"- Grounded Prompt (Production) pass rate: {g_w}")
        lines.append(f"- Difference (Grounded − Baseline): {dpr:.1f} pp [95% bootstrap CI: {dlo:.1f}–{dhi:.1f}]")
        note = prc.get("mcnemar_note")
        pv = prc.get("mcnemar_p_value")
        method = prc.get("mcnemar_method", "")
        if note:
            lines.extend(["", note, ""])
        elif pv is not None:
            sig = float(pv) < 0.05
            lines.append(
                f"- McNemar ({method}): p = {float(pv):.4g} "
                f"(discordant pairs: baseline pass / grounded fail = {prc.get('mcnemar_b')}, "
                f"baseline fail / grounded pass = {prc.get('mcnemar_c')})."
            )
            lines.append("")
            lines.append(
                "Pass rate improvement for Grounded Prompt (Production) vs Baseline Prompt is statistically significant."
                if sig
                else "Pass rate difference was not statistically significant under the paired test."
            )
            lines.append("")
        else:
            lines.append("")
    else:
        lines.extend(
            [
                "_Paired pass-rate comparison unavailable (need both baseline and grounded rows per `trial_id`)._",
                "",
            ]
        )

    lines.extend(["", "## 2. Baseline Prompt vs Grounded Prompt (Production)"])
    if comp:
        pts = comp.get("paired_t_stat")
        d = comp.get("effect_size_cohens_d")
        try:
            pts_f = f"{float(pts):.2f}" if pts is not None else str(pts)
        except (TypeError, ValueError):
            pts_f = str(pts)
        try:
            d_f = f"{float(d):.2f}" if d is not None else str(d)
        except (TypeError, ValueError):
            d_f = str(d)
        lines.extend(
            [
                f"- Baseline Prompt report quality (mean): {comp.get('baseline_mean', 0):.2f}",
                f"- Grounded Prompt (Production) report quality (mean): {comp.get('grounded_mean', 0):.2f}",
                f"- Paired t-statistic: {pts_f}",
                f"- Effect size (Cohen's d): {d_f}",
            ]
        )
    else:
        lines.append("- Comparative results unavailable (run compare mode).")

    lines.extend(["", "## 3. Confidence validation"])
    diverse = bool(conf.get("diverse_labels_for_calibration"))
    by_label = conf.get("by_label_summary") or []
    if not diverse:
        lines.append(
            "Confidence validation is limited: nearly all trials share one dominant confidence label, so cross-label calibration is not established."
        )
    if by_label:
        lines.append("")
        lines.append("Summary by dominant confidence label (structured payload):")
        for g in by_label:
            lines.append(
                f"- {g['confidence_label'].title()}: report quality score {g['mean_validity_score']:.2f}, "
                f"numeric alignment {100*float(g['numeric_accuracy']):.1f}%, pass rate {100*g['pass_rate']:.1f}%"
            )
    groups = conf.get("groups") or []
    if diverse and groups:
        hi = next((g for g in groups if g["confidence_label"] == "high"), None)
        lo = next((g for g in groups if g["confidence_label"] == "low"), None)
        if hi and lo:
            lines.append(
                f"- High-confidence trials: report quality score {hi['mean_validity_score']:.2f}; "
                f"low-confidence trials: {lo['mean_validity_score']:.2f}. Calibration flag: {conf.get('calibration_flag', 'unknown')}."
            )
    elif not by_label:
        lines.append("- Dominant confidence label not available in results.")

    corr = conf.get("confidence_score_validity_correlation")
    lines.append("")
    if corr is not None:
        lines.append(f"Correlation between upstream confidence score and report quality score: r = {corr:.2f}.")
    else:
        lines.append(
            "Confidence score correlation was not computed (missing or insufficient variation in confidence scores)."
        )

    lines.extend(["", "## 4. Failure Mode Analysis", ""])
    fm_rows = fma.get("table") or []
    if fm_rows:
        lines.append(
            "The most common issues were minor numeric inconsistencies, missing sections, and confidence wording variations."
        )
        lines.append("")
        lines.extend(["| Failure mode | Count | Share |", "|---|---:|---:|"])
        for r in fm_rows:
            lines.append(f"| {r['name']} | {r['count']} | {r['percent']:.1f}% |")
        lines.append("")
        lines.append(
            "Confidence Wording Issues (grader) may reflect strict phrasing review rather than contradiction of structured labels."
        )
        lines.append("")
        lines.append(fma.get("interpretation", ""))
    else:
        lines.append("_No failure-mode aggregation (empty dataset)._")

    lines.extend(
        [
            "",
            "## 5. Row-level review",
            "- Inspect `qc_results.csv`: Potential Unsupported Statements, Confidence Wording Issues, grader payload, and report text (column names remain machine-readable).",
            "",
            "## 6. Example outputs",
        ]
    )
    if not df.empty:
        sample = df.head(3)[["mode", "trial_id", "validity_score_0_100", "hallucinated_state_count", "hallucinated_number_count"]]
        lines.append("")
        lines.append("```text")
        lines.append(sample.to_string(index=False))
        lines.append("```")

    out_path.write_text("\n".join(lines))


def write_charts(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    if df.empty:
        return

    dfp = df.copy()
    dfp["Prompt Type"] = dfp["mode"].astype(str).str.lower().map(
        {"baseline": "Baseline Prompt", "grounded": "Grounded Prompt (Production)"}
    ).fillna(dfp["mode"].astype(str))

    fig1 = px.box(
        dfp,
        x="Prompt Type",
        y="validity_score_0_100",
        title="Report Quality Score by Prompt Type",
        labels={"validity_score_0_100": "Report Quality Score (0–100)"},
    )
    fig1.update_layout(
        yaxis_title="Report Quality Score (0–100)",
        xaxis_title="Prompt Type",
    )
    fig1.write_html(str(out_dir / "validity_by_mode.html"))

    fig2 = px.scatter(
        dfp,
        x="dominant_confidence_score",
        y="validity_score_0_100",
        color="Prompt Type",
        title="Confidence and Report Quality",
        labels={
            "dominant_confidence_score": "Confidence score (payload)",
            "validity_score_0_100": "Report Quality Score",
        },
    )
    fig2.update_layout(xaxis_title="Confidence score (payload)", yaxis_title="Report Quality Score")
    fig2.write_html(str(out_dir / "confidence_vs_validity.html"))

    g = (
        dfp.groupby(["Prompt Type", "dominant_confidence_label"], dropna=False)["validity_score_0_100"]
        .mean()
        .reset_index()
    )
    fig3 = px.bar(
        g,
        x="dominant_confidence_label",
        y="validity_score_0_100",
        color="Prompt Type",
        barmode="group",
        title="Report Quality by Confidence Label",
        labels={"validity_score_0_100": "Report Quality Score", "dominant_confidence_label": "Confidence label"},
    )
    fig3.update_layout(yaxis_title="Report Quality Score", xaxis_title="Confidence label")
    fig3.write_html(str(out_dir / "validity_by_confidence_group.html"))
