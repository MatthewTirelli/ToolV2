"""
Streamlit UI: read-only Quality Control Summary tab (offline QC artifacts).
No OpenAI, no qc/run_qc_experiment, no batch regeneration.
"""
from __future__ import annotations

import html
import json
import re
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.utils.logging_config import get_logger

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
QC_OUT = ROOT / "qc" / "outputs"
LIVE_DEBUG_DIR = QC_OUT / "live_validation_debug"
logger = get_logger("qc_panel")
QC_CSV = QC_OUT / "qc_results.csv"
QC_SUMMARY = QC_OUT / "qc_summary.md"
# Shipped with the app so the QC tab has data when `qc/outputs/` is empty (gitignored / not run yet).
BUNDLED_QC_CSV = APP_DIR / "assets" / "qc" / "qc_results.csv"
BUNDLED_QC_SUMMARY = APP_DIR / "assets" / "qc" / "qc_summary.md"


def _active_qc_csv() -> tuple[Path | None, str]:
    """Prefer live `qc/outputs/qc_results.csv`, else bundled snapshot under `app/assets/qc/`."""
    if QC_CSV.is_file():
        return QC_CSV, "local"
    if BUNDLED_QC_CSV.is_file():
        return BUNDLED_QC_CSV, "bundled"
    return None, "none"


def _active_qc_summary() -> tuple[Path | None, str]:
    if QC_SUMMARY.is_file():
        return QC_SUMMARY, "local"
    if BUNDLED_QC_SUMMARY.is_file():
        return BUNDLED_QC_SUMMARY, "bundled"
    return None, "none"


def _format_pass_rate_significance(prc: Dict[str, Any] | None) -> str | None:
    """
    Display string from McNemar p-value (paired pass/fail), same statistic as qc_summary.md
    (`qc.statistical_analysis.analyze_results` → pass_rate_comparison).
    """
    if not prc:
        return None
    note = (prc.get("mcnemar_note") or "").strip()
    if note and "not applicable" in note.lower():
        return None
    pv = prc.get("mcnemar_p_value")
    if pv is None:
        return None
    try:
        p = float(pv)
    except (TypeError, ValueError):
        return None
    if p != p:  # NaN
        return None
    if p >= 0.05:
        return "(not statistically significant)"
    if p < 0.001:
        return "(statistically significant, p < 0.001)"
    return f"(statistically significant, p = {p:.3f})"


_FAILURE_MODE_LABELS: list[tuple[str, str]] = [
    ("Grader: confidence_misuse non-empty", "Confidence Wording Issues"),
    ("Grader: unsupported_claims non-empty", "Potential Unsupported Statements"),
    ("Numeric accuracy below 0.95", "Minor numeric inconsistencies"),
    ("Missing required sections", "Missing report sections"),
    ("Hallucinated state mention(s)", "Unsupported state mentions"),
    ("Hallucinated number signal(s)", "Unsupported number mentions"),
    ("Did not pass absolute validity", "Did not pass strict validation"),
    ("Incomplete top-state coverage", "Incomplete top-state coverage"),
    ("Confidence label match below 1.0", "Confidence alignment below threshold"),
    ("Unsupported structured confidence claim(s)", "Unsupported confidence claims"),
    ("Low/moderate confidence disclosure below 1.0 (where applicable)", "Low-confidence disclosure gaps"),
]


def _section_headers_text_view(detected: list[Any] | None, missing: list[Any] | None) -> str:
    """Plain markdown for section header checklist (avoid dict/JSON appearance in UI)."""
    det = [str(x) for x in (detected or []) if x is not None and str(x).strip()]
    mis = [str(x) for x in (missing or []) if x is not None and str(x).strip()]
    lines: list[str] = ["**Detected in report**"]
    if det:
        lines.extend(f"- `{h}`" for h in det)
    else:
        lines.append("- *(none)*")
    lines.append("")
    lines.append("**Missing**")
    if mis:
        lines.extend(f"- `{h}`" for h in mis)
    else:
        lines.append("- *(none — all required headers found)*")
    return "\n".join(lines)


def _live_validation_checklist_rows(
    metrics: dict[str, Any],
    score: dict[str, Any],
    gh_missing: list[Any] | None,
) -> list[tuple[str, bool, str]]:
    """(label, passed, detail). Mirrors gates in qc.scoring.compute_validity_score (strict pass)."""
    m = metrics
    v = float(score.get("validity_score_0_100") or 0)
    miss = [str(x) for x in (gh_missing or []) if x is not None and str(x).strip()]
    sec_ok = len(miss) == 0
    g1 = v >= 80.0
    g2 = float(m.get("hallucinated_state_count", 0) or 0) == 0.0
    g3 = float(m.get("hallucinated_number_count", 0) or 0) <= 1.0
    g4 = float(m.get("numeric_accuracy_rate", 0) or 0) >= 0.8
    g5 = float(m.get("top_state_coverage_rate", 0) or 0) >= 1.0
    return [
        (
            "Grounded section headers",
            sec_ok,
            "All five required headings found." if sec_ok else f"Missing: {', '.join(miss)}",
        ),
        (
            "Top states mentioned in prose",
            g5,
            f"Coverage {float(m.get('top_state_coverage_rate', 0) or 0):.1%} (required 100%).",
        ),
        (
            "Structured numbers in report text",
            g4,
            f"Match rate {float(m.get('numeric_accuracy_rate', 0) or 0):.1%} (threshold ≥ 80%) for case count, MMR coverage, confidence score.",
        ),
        (
            "State codes vs ground-truth top 3",
            g2,
            f"Hallucinated-state signal count {int(float(m.get('hallucinated_state_count', 0) or 0))} (needs 0).",
        ),
        (
            "Extra numerals (noise) tolerance",
            g3,
            f"Hallucinated-number signal {int(float(m.get('hallucinated_number_count', 0) or 0))} (needs ≤ 1).",
        ),
        (
            "Weighted validity score",
            g1,
            f"{v:.2f} / 100 after penalties (needs ≥ 80).",
        ),
    ]


def _live_validation_context_blurb(expected_states: list[str]) -> str:
    states = ", ".join(html.escape(s) for s in expected_states) if expected_states else "—"
    return (
        "<p class=\"lv-muted\" style=\"margin:0 0 1rem 0;\">"
        f"Compares your latest <strong>Agent&nbsp;3</strong> report to the structured top states "
        f"<strong>{states}</strong> and national fields from this session. "
        "Same validation pipeline as offline <strong>Grounded Prompt (Production)</strong> QC.</p>"
    )


_LIVE_VALIDATION_CSS = """
<style>
.lv-scope { font-size: 0.95rem; line-height: 1.55; color: #334155;
  font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; }
.lv-scope .lv-title { font-size: 1.08rem; font-weight: 600; color: #0f172a; margin: 0 0 0.5rem 0; letter-spacing: -0.01em; }
.lv-scope .lv-muted { color: #64748b; font-size: 0.875rem; line-height: 1.5; margin: 0 0 1rem 0; }
.lv-scope .lv-flex { display: flex; flex-wrap: wrap; gap: 1rem; align-items: stretch; margin-bottom: 1rem; }
.lv-scope .lv-card { flex: 1 1 240px; background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 10px; padding: 1rem 1.1rem; }
.lv-scope .lv-card-highlight { border-color: #cbd5e1; background: #fff; box-shadow: 0 1px 2px rgba(15,23,42,0.06); }
.lv-scope .lv-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; margin-bottom: 0.25rem; }
.lv-scope .lv-score { font-size: 2rem; font-weight: 700; color: #0f172a; line-height: 1.2; }
.lv-scope .lv-badge { display: inline-block; margin-top: 0.65rem; padding: 0.35rem 0.65rem; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }
.lv-scope .lv-badge-yes { background: #dcfce7; color: #166534; border: 1px solid #bbf7d0; }
.lv-scope .lv-badge-no { background: #fee2e2; color: #991b1b; border: 1px solid #fecaca; }
.lv-scope .lv-subchecks { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
.lv-scope .lv-subchecks th { text-align: left; color: #64748b; font-weight: 600; font-size: 0.72rem; text-transform: uppercase;
  letter-spacing: 0.04em; padding: 0.4rem 0.5rem; border-bottom: 1px solid #e2e8f0; background: #f8fafc; }
.lv-scope .lv-subchecks td { padding: 0.5rem 0.5rem; border-bottom: 1px solid #f1f5f9; vertical-align: top; }
.lv-scope .lv-ok { color: #166534; font-weight: 600; white-space: nowrap; }
.lv-scope .lv-bad { color: #b91c1c; font-weight: 600; white-space: nowrap; }
.lv-scope code { font-size: 0.84em; background: #e2e8f0; padding: 0.08em 0.28em; border-radius: 4px; color: #0f172a; }
</style>
"""


def _live_validation_score_and_checks_html(
    *,
    validity: float,
    passed: bool,
    metrics: dict[str, Any],
    score: dict[str, Any],
    gh_missing: list[Any] | None,
    expected_states: list[str],
) -> str:
    badge_cls = "lv-badge-yes" if passed else "lv-badge-no"
    badge_txt = "Strict pass: Yes" if passed else "Strict pass: No"
    rows = _live_validation_checklist_rows(metrics, score, gh_missing)
    tbody = []
    for label, ok, det in rows:
        st_cls = "lv-ok" if ok else "lv-bad"
        word = "Pass" if ok else "Fail"
        tbody.append(
            "<tr>"
            f"<td>{html.escape(label)}</td>"
            f'<td class="{st_cls}">{html.escape(word)}</td>'
            f"<td>{html.escape(det)}</td>"
            "</tr>"
        )
    tbl = (
        '<table class="lv-subchecks"><thead><tr>'
        "<th>Check</th><th>Result</th><th>Notes</th>"
        "</tr></thead><tbody>"
        + "".join(tbody)
        + "</tbody></table>"
    )
    return (
        _live_validation_context_blurb(expected_states)
        + '<div class="lv-flex">'
        '<div class="lv-card lv-card-highlight">'
        '<div class="lv-label">Report quality score (0–100)</div>'
        f'<div class="lv-score">{html.escape(f"{validity:.2f}")}</div>'
        f'<span class="lv-badge {badge_cls}">{html.escape(badge_txt)}</span>'
        '<p class="lv-muted" style="margin-top:0.75rem;margin-bottom:0;">Weighted score after penalties. '
        "Strict pass requires all checklist rows to pass.</p>"
        "</div>"
        f'<div class="lv-card">{tbl}</div>'
        "</div>"
    )


@st.cache_data(show_spinner=False)
def _load_qc_results_csv(path_str: str, mtime: float) -> pd.DataFrame | None:
    p = Path(path_str)
    if mtime < 0 or not p.is_file():
        return None
    return pd.read_csv(p)


@st.cache_data(show_spinner=False)
def _load_qc_summary_md(path_str: str, mtime: float) -> str | None:
    p = Path(path_str)
    if mtime < 0 or not p.is_file():
        return None
    return p.read_text(encoding="utf-8", errors="replace")


def _file_mtime(path: Path) -> float:
    try:
        return float(path.stat().st_mtime)
    except OSError:
        return -1.0


def _coerce_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    sl = s.astype(str).str.lower().str.strip()
    return sl.isin(["true", "1", "yes"])


def _format_qc_results_table(view: pd.DataFrame) -> pd.DataFrame:
    """Display-only formatting: consistent % and score precision (does not alter source CSV)."""
    out = view.copy()
    if "Report quality score" in out.columns:
        v = pd.to_numeric(out["Report quality score"], errors="coerce")
        out["Report quality score"] = v.map(lambda x: f"{float(x):.1f}" if pd.notna(x) else "—")
    for col in ("Numeric alignment", "Required sections", "Confidence alignment"):
        if col in out.columns:
            v = pd.to_numeric(out[col], errors="coerce")
            out[col] = v.map(lambda x: f"{float(x):.1%}" if pd.notna(x) else "—")
    if "Strict validation pass" in out.columns:
        s = out["Strict validation pass"]
        out["Strict validation pass"] = s.apply(
            lambda x: "Yes"
            if str(x).lower() in ("true", "1", "yes")
            else "No"
            if str(x).lower() in ("false", "0", "no")
            else str(x)
        )
    return out


def _prompt_type_label(mode: Any) -> str:
    m = str(mode or "").strip().lower()
    if m == "baseline":
        return "Baseline Prompt"
    if m == "grounded":
        return "Grounded Prompt (Production)"
    return str(mode or "—")


def _grounded_subset(df: pd.DataFrame) -> pd.DataFrame:
    if "mode" not in df.columns:
        return df.iloc[0:0]
    return df.loc[df["mode"].astype(str).str.lower().str.strip() == "grounded"]


def _grounded_story_metrics(df: pd.DataFrame | None) -> dict[str, Any]:
    """Production (grounded) metrics only for top cards."""
    empty: dict[str, Any] = {
        "n": 0,
        "pass_rate": None,
        "mean_quality": None,
        "hallucination_rate": None,
        "baseline_mean_quality": None,
        "improvement_points": None,
    }
    if df is None or df.empty or "mode" not in df.columns:
        return empty
    g = _grounded_subset(df)
    if g.empty:
        return empty
    n = int(len(g))
    pass_r = (
        float(_coerce_bool_series(g["passed_absolute_validity"]).mean())
        if "passed_absolute_validity" in g.columns
        else None
    )
    mq = float(pd.to_numeric(g["validity_score_0_100"], errors="coerce").mean()) if "validity_score_0_100" in g.columns else None
    if "hallucinated_state_count" in g.columns and "hallucinated_number_count" in g.columns:
        hs = pd.to_numeric(g["hallucinated_state_count"], errors="coerce").fillna(0) > 0
        hn = pd.to_numeric(g["hallucinated_number_count"], errors="coerce").fillna(0) > 0
        hall = float((hs | hn).mean())
    else:
        hall = None
    b_mean = None
    imp = None
    b = df.loc[df["mode"].astype(str).str.lower().str.strip() == "baseline"]
    if not b.empty and "validity_score_0_100" in b.columns and mq is not None:
        b_mean = float(pd.to_numeric(b["validity_score_0_100"], errors="coerce").mean())
        imp = mq - b_mean
    return {
        "n": n,
        "pass_rate": pass_r,
        "mean_quality": mq,
        "hallucination_rate": hall,
        "baseline_mean_quality": b_mean,
        "improvement_points": imp,
    }


def _preprocess_detail_markdown(md: str) -> str:
    """Smaller headings in detailed summary + friendlier failure-mode names."""
    text = md
    text = re.sub(
        r"^#\s+QC Validation Summary\s*$",
        "### Quality Control Summary",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    text = re.sub(
        r"^#\s+Quality Control Summary\s*$",
        "### Quality Control Summary",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    lines_out: list[str] = []
    for line in text.splitlines():
        if line.startswith("#######"):
            lines_out.append(line)
        elif line.startswith("######"):
            lines_out.append(line)
        elif line.startswith("#####"):
            lines_out.append(line)
        elif line.startswith("####"):
            lines_out.append(line)
        elif line.startswith("###"):
            lines_out.append("###### " + line[3:].lstrip())
        elif line.startswith("##"):
            lines_out.append("##### " + line[2:].lstrip())
        elif line.startswith("#"):
            lines_out.append("#### " + line[1:].lstrip())
        else:
            lines_out.append(line)
    text = "\n".join(lines_out)
    for old, new in _FAILURE_MODE_LABELS:
        text = text.replace(old, new)
    _legacy_phrases: list[tuple[str, str]] = [
        ("**Baseline:**", "**Baseline Prompt:**"),
        ("**Grounded / production:**", "**Grounded Prompt (Production):**"),
        ("Mean validity score", "Report quality score"),
        ("Hallucination rate", "Structural error rate"),
        ("## 2. Baseline vs grounded comparison", "## 2. Baseline Prompt vs Grounded Prompt (Production)"),
        ("- Baseline mean:", "- Baseline Prompt report quality (mean):"),
        ("- Grounded mean:", "- Grounded Prompt (Production) report quality (mean):"),
        ("| Failure mode | Count | Percent |", "| Failure mode | Count | Share |"),
        ("- Baseline pass rate:", "- Baseline Prompt pass rate:"),
        ("- Grounded pass rate:", "- Grounded Prompt (Production) pass rate:"),
        ("- Difference (grounded − baseline):", "- Difference (Grounded − Baseline):"),
        ("- Overall pass rate:", "- Pass rate:"),
    ]
    for old, new in _legacy_phrases:
        text = text.replace(old, new)
    text = re.sub(
        r"(Paired t-stat(?:istic)?:\s*)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        lambda m: f"{m.group(1)}{float(m.group(2)):.2f}",
        text,
    )
    text = re.sub(
        r"(Effect size \(Cohen's d\):\s*)([-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)",
        lambda m: f"{m.group(1)}{float(m.group(2)):.2f}",
        text,
    )
    _fm_intro = (
        "The most common issues were minor numeric inconsistencies, missing sections, and confidence wording variations."
    )
    _fm_heading = "## 4. Failure Mode Analysis"
    if _fm_heading in text and _fm_intro not in text:
        pre, _, post = text.partition(_fm_heading)
        post_stripped = post.lstrip()
        if post_stripped.startswith("|") or post_stripped.startswith("\n|"):
            text = pre + _fm_heading + "\n\n" + _fm_intro + "\n\n" + post_stripped
        elif post_stripped.startswith("\n\n|"):
            text = pre + _fm_heading + "\n\n" + _fm_intro + post
    return text


def _style_compare_fig(fig: go.Figure, *, title: str) -> go.Figure:
    """Centered title, balanced size for two-column layout (Plotly)."""
    fig.update_layout(
        title=dict(text=title, x=0.5, xanchor="center", font=dict(size=14, color="#0f172a")),
        height=320,
        margin=dict(l=48, r=32, t=52, b=44),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#f8fafc",
        font=dict(size=12, color="#475569"),
        showlegend=False,
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False, tickfont=dict(size=11))
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0", zeroline=False, tickfont=dict(size=11))
    return fig


def _fig_report_quality_by_prompt(df: pd.DataFrame) -> go.Figure | None:
    if "mode" not in df.columns or "validity_score_0_100" not in df.columns:
        return None
    d = df.copy()
    d["prompt_type"] = d["mode"].map(_prompt_type_label)
    d["report_quality"] = pd.to_numeric(d["validity_score_0_100"], errors="coerce")
    d = d.dropna(subset=["report_quality"])
    if d.empty:
        return None
    cat_order = [x for x in ("Baseline Prompt", "Grounded Prompt (Production)") if x in set(d["prompt_type"])]
    d["prompt_type"] = pd.Categorical(d["prompt_type"], categories=cat_order, ordered=True)
    fig = px.box(d, x="prompt_type", y="report_quality", points="outliers")
    fig.update_layout(
        showlegend=False,
        xaxis_title="Prompt Type",
        yaxis_title="Report Quality Score (0–100)",
    )
    return _style_compare_fig(fig, title="Report Quality Score by Prompt Type")


def _fig_pass_rate_by_prompt(df: pd.DataFrame) -> go.Figure | None:
    if "passed_absolute_validity" not in df.columns or "mode" not in df.columns:
        return None
    d = df.copy()
    d["_pass"] = _coerce_bool_series(d["passed_absolute_validity"]).astype(float)
    d["prompt_type"] = d["mode"].map(_prompt_type_label)
    g = d.groupby("prompt_type", as_index=False, sort=False)["_pass"].mean()
    if g.empty:
        return None
    order = [x for x in ("Baseline Prompt", "Grounded Prompt (Production)") if x in set(g["prompt_type"])]
    g["prompt_type"] = pd.Categorical(g["prompt_type"], categories=order, ordered=True)
    g = g.sort_values("prompt_type")
    fig = px.bar(g, x="prompt_type", y="_pass", text=[f"{100 * v:.0f}%" for v in g["_pass"]])
    fig.update_traces(textposition="outside", textfont_size=10)
    fig.update_layout(
        showlegend=False,
        xaxis_title="Prompt Type",
        yaxis_title="Pass Rate",
        yaxis=dict(tickformat=".0%", range=[0, max(0.12, float(g["_pass"].max()) * 1.25)]),
    )
    return _style_compare_fig(fig, title="Pass Rate by Prompt Type")


def render_qc_tab() -> None:
    st.markdown(
        """
<style>
.qc-page-title { font-size: 1.5rem; font-weight: 700; color: #0f172a; margin-bottom: 0.35rem; }
.qc-intro { color: #475569; font-size: 0.95rem; line-height: 1.55; margin-bottom: 0.4rem; }
.qc-trust { color: #64748b; font-size: 0.875rem; line-height: 1.45; margin: 0 0 0.75rem 0; }
.qc-metric-sub { font-size: 0.78rem; color: #64748b; line-height: 1.35; margin: 0.15rem 0 0 0; }
.qc-tagline { color: #475569; font-size: 0.92rem; line-height: 1.5; margin: 0.5rem 0 0.35rem 0; }
.lv-section-title { font-size: 1.05rem; font-weight: 600; color: #0f172a; margin: 0.5rem 0 0.5rem 0; }
</style>
<div class="qc-page-title">Quality Control Summary</div>
""",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="qc-intro">Offline QC results for the <strong>Grounded Prompt (Production)</strong> and '
        "<strong>Baseline Prompt</strong>, compared on the same trial cohort. The batch protocol is run outside the dashboard.</p>",
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="qc-trust">Results are based on repeated evaluation across multiple independent trials.</p>',
        unsafe_allow_html=True,
    )

    active_csv, csv_src = _active_qc_csv()
    active_summary, _summary_src = _active_qc_summary()
    if active_csv is None:
        st.warning("No QC results are available yet.")
        st.caption(
            "Run the offline QC protocol to generate `qc/outputs/qc_results.csv`, "
            "or restore your previous CSV into that path (outputs are gitignored, so they are not in the repo)."
        )
        _downloads_only(None, None)
        _live_validation_expander()
        return

    if csv_src == "bundled":
        st.info(
            "Showing the **bundled sample** QC CSV under `app/assets/qc/` because `qc/outputs/qc_results.csv` "
            "was not found. Your full run is not lost from Git (it was never committed); if you still have the old "
            "file elsewhere, copy it back to `qc/outputs/`. To regenerate: "
            "`python qc/run_qc_experiment.py --n-trials 50 --mode compare --with-grader`."
        )

    csv_mtime = _file_mtime(active_csv)
    summary_mtime = _file_mtime(active_summary) if active_summary else -1.0
    df = _load_qc_results_csv(str(active_csv.resolve()), csv_mtime)
    summary_text = (
        _load_qc_summary_md(str(active_summary.resolve()), summary_mtime) if active_summary is not None else None
    )

    gm = _grounded_story_metrics(df)
    prc: Dict[str, Any] | None = None
    if not df.empty and "mode" in df.columns:
        _modes = set(df["mode"].dropna().astype(str).str.lower().str.strip().unique()) - {"", "nan"}
    else:
        _modes = set()
    if not df.empty and {"baseline", "grounded"} <= _modes:
        from qc.statistical_analysis import analyze_results

        prc = analyze_results(df).get("pass_rate_comparison")

    st.markdown("#### Grounded Prompt (Production) — key metrics")
    if gm["n"] == 0:
        st.info("No grounded rows in QC results. Run offline compare mode including Grounded Prompt (Production).")
    else:
        sig_line = _format_pass_rate_significance(prc)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            pr = gm["pass_rate"]
            st.metric("Grounded accuracy", f"{100*pr:.1f}%" if pr is not None else "—")
        with c2:
            mq = gm["mean_quality"]
            st.metric("Report quality score", f"{mq:.1f} / 100" if mq is not None else "—")
        with c3:
            hr = gm["hallucination_rate"]
            st.metric("Structural errors detected", f"{100*hr:.1f}%" if hr is not None else "—")
        with c4:
            st.metric("Reports evaluated", str(gm["n"]))
        with c5:
            imp = gm["improvement_points"]
            if imp is not None:
                st.metric("Improvement vs baseline", f"{imp:+.1f} pts")
            else:
                st.metric("Improvement vs baseline", "—")
            if sig_line:
                st.markdown(f'<p class="qc-metric-sub">{html.escape(sig_line)}</p>', unsafe_allow_html=True)

        st.markdown("")
        st.markdown(
            '<p class="qc-intro" style="margin-top:0.25rem;">The grounded prompt produces consistently accurate reports '
            "that fully align with structured input data.</p>",
            unsafe_allow_html=True,
        )
        pr = gm["pass_rate"]
        hr = gm["hallucination_rate"]
        p_mcn: float | None = None
        if prc and prc.get("mcnemar_p_value") is not None:
            try:
                p_mcn = float(prc["mcnemar_p_value"])
            except (TypeError, ValueError):
                p_mcn = None
        tags: list[str] = []
        if pr is not None and pr >= 0.9:
            tags.append("High reliability")
        if hr is not None and hr == 0.0:
            tags.append("No structural errors detected")
        if p_mcn is not None and p_mcn < 0.05:
            tags.append("Statistically significant improvement over baseline")
        if tags:
            st.markdown('<p class="qc-tagline"><strong>Summary</strong></p>', unsafe_allow_html=True)
            st.markdown("\n".join(f"- ✔ {t}" for t in tags))

    st.markdown("---")
    st.markdown(
        '<p style="text-align:center;font-size:1.05rem;font-weight:600;color:#0f172a;margin:0.25rem 0 0.75rem 0;">'
        "Prompt performance comparison</p>",
        unsafe_allow_html=True,
    )
    _pcfg = {"displayModeBar": False}
    f1, f2 = _fig_report_quality_by_prompt(df), _fig_pass_rate_by_prompt(df)
    if f1 is None and f2 is None:
        st.info("Charts need more columns in `qc_results.csv`. Run the offline QC protocol after a compare-mode run.")
    else:
        c1, c2 = st.columns(2, gap="medium")
        with c1:
            if f1:
                st.plotly_chart(f1, use_container_width=True, config=_pcfg)
                st.caption("Report quality score distribution by prompt type (same trial IDs).")
            else:
                st.caption("—")
        with c2:
            if f2:
                st.plotly_chart(f2, use_container_width=True, config=_pcfg)
                st.caption("Strict validation pass rate by prompt type (same trial IDs).")
            else:
                st.caption("—")

    _live_validation_expander()

    st.markdown("---")

    with st.expander("View detailed Quality Control Summary", expanded=False):
        if summary_text is None:
            st.warning("Quality control results have not been generated yet. Run the offline QC protocol first.")
        else:
            st.markdown(_preprocess_detail_markdown(summary_text))

    with st.expander("View raw QC results", expanded=False):
        cols = [
            "mode",
            "trial_id",
            "validity_score_0_100",
            "passed_absolute_validity",
            "hallucinated_state_count",
            "hallucinated_number_count",
            "numeric_accuracy_rate",
            "required_sections_rate",
            "confidence_label_match_rate",
        ]
        present = [c for c in cols if c in df.columns]
        if not present:
            st.caption("Expected columns not found in CSV.")
        else:
            view = df[present].copy()
            rename = {
                "mode": "Prompt type",
                "trial_id": "Trial",
                "validity_score_0_100": "Report quality score",
                "passed_absolute_validity": "Strict validation pass",
                "hallucinated_state_count": "Structural state errors",
                "hallucinated_number_count": "Structural number errors",
                "numeric_accuracy_rate": "Numeric alignment",
                "required_sections_rate": "Required sections",
                "confidence_label_match_rate": "Confidence alignment",
            }
            view = view.rename(columns={k: v for k, v in rename.items() if k in view.columns})
            if "Prompt type" in view.columns:
                view["Prompt type"] = view["Prompt type"].map(_prompt_type_label)
            st.dataframe(_format_qc_results_table(view), use_container_width=True, hide_index=True)
        if "report_text" in df.columns:
            keys: list[str] = []
            for _, r in df[["trial_id", "mode"]].drop_duplicates().iterrows():
                if pd.notna(r.get("trial_id")):
                    keys.append(f"{int(r['trial_id'])} | {r['mode']}")
            if keys:
                pick = st.selectbox("Optional: inspect report text for one row", ["—"] + sorted(keys), index=0)
                if pick != "—":
                    tid_s, _, mode_s = pick.partition(" | ")
                    try:
                        tid_i = int(tid_s.strip())
                    except ValueError:
                        tid_i = None
                    mode_s = mode_s.strip()
                    if tid_i is not None:
                        row = df[
                            (pd.to_numeric(df["trial_id"], errors="coerce") == tid_i)
                            & (df["mode"].astype(str) == mode_s)
                        ]
                        if not row.empty:
                            with st.expander("Report text (selected row)", expanded=False):
                                st.markdown(str(row.iloc[0].get("report_text", "") or "—"))

    _downloads_only(active_csv, active_summary)


def _downloads_only(csv_path: Path | None, summary_path: Path | None) -> None:
    st.markdown("#### Downloads")
    dc1, dc2 = st.columns(2)
    with dc1:
        if csv_path is not None and csv_path.is_file():
            st.download_button(
                label="Download qc_results.csv",
                data=csv_path.read_bytes(),
                file_name="qc_results.csv",
                mime="text/csv",
            )
        else:
            st.caption("qc_results.csv not available.")
    with dc2:
        if summary_path is not None and summary_path.is_file():
            st.download_button(
                label="Download qc_summary.md",
                data=summary_path.read_bytes(),
                file_name="qc_summary.md",
                mime="text/markdown",
            )
        else:
            st.caption("qc_summary.md not available.")


def _live_validation_expander() -> None:
    st.markdown(
        '<div class="lv-section-title">Session report validation</div>',
        unsafe_allow_html=True,
    )
    with st.expander("View validation results", expanded=True):
        ma = st.session_state.get("multi_agent_result")
        report = (ma or {}).get("agent3") or {}
        text = (report.get("assistant_text") or "").strip()
        if not ma or not text:
            st.info("Generate a report first to run single-report validation.")
        else:
            try:
                from qc.live_validation_diagnostics import (
                    grounded_prompt_file_info,
                    payloads_align_for_validation,
                    write_live_validation_debug_bundle,
                )
                from qc.scoring import compute_validity_score
                from qc.validators import (
                    GROUNDED_SECTION_HEADERS,
                    extract_ground_truth,
                    validate_report,
                )
            except ImportError as e:
                st.error(f"QC modules unavailable: {e}")
            else:
                prompt_info = grounded_prompt_file_info()
                logger.info(
                    "live_validation prompt_source path=%s exists=%s",
                    prompt_info.get("path"),
                    prompt_info.get("exists"),
                )

                gt = extract_ground_truth(ma, top_n=3)
                alignment = payloads_align_for_validation(ma)

                metrics = validate_report(
                    report_text=text,
                    ground_truth=gt,
                    section_headers=GROUNDED_SECTION_HEADERS,
                )
                score = compute_validity_score(metrics, concision_score_1_5=None)
                passed = bool(score.get("passed_absolute_validity"))
                validity = float(score.get("validity_score_0_100") or 0.0)

                bundle = write_live_validation_debug_bundle(
                    LIVE_DEBUG_DIR,
                    ma=ma,
                    report_text=text,
                    ground_truth=gt,
                    prompt_info=prompt_info,
                    alignment=alignment,
                )

                sec = bundle.get("section_headers") or {}
                gh = sec.get("grounded_prompt_headers") or {}
                exp_states = list((bundle.get("missing_top_states") or {}).get("expected_top_states") or [])

                live_html = (
                    _LIVE_VALIDATION_CSS
                    + '<div class="lv-scope">'
                    + _live_validation_score_and_checks_html(
                        validity=validity,
                        passed=passed,
                        metrics=metrics,
                        score=score,
                        gh_missing=gh.get("missing"),
                        expected_states=exp_states,
                    )
                    + "</div>"
                )
                st.markdown(live_html, unsafe_allow_html=True)

                with st.expander("Advanced diagnostics", expanded=False):
                    st.caption(
                        f"Prompt file: `{prompt_info.get('path')}` — "
                        f"agent2/tool cohort match: {alignment.get('agent2_enriched_block_matches_tool')}"
                    )
                    st.caption(f"Saved under `{LIVE_DEBUG_DIR}`.")
                    st.json(bundle.get("missing_top_states") or {})
                    st.json(bundle.get("state_code_hallucinations_vs_top3") or {})
                    misuse = bundle.get("confidence_misuse_detail") or []
                    st.markdown("**Confidence Wording Issues**")
                    st.json(misuse if misuse else {})
                    st.markdown("**Key rates**")
                    st.json(
                        {
                            k: metrics.get(k)
                            for k in (
                                "hallucinated_state_count",
                                "hallucinated_number_count",
                                "top_state_coverage_rate",
                                "numeric_accuracy_rate",
                                "required_sections_rate",
                                "confidence_label_match_rate",
                                "unsupported_confidence_claim_count",
                            )
                            if k in metrics
                        }
                    )
                    bh = sec.get("batch_baseline_headers") or {}
                    st.markdown("*Contrast: baseline-style header list (not used for live score)*")
                    st.markdown(_section_headers_text_view(bh.get("detected"), bh.get("missing")))
                    comp = bundle.get("header_and_wiring_comparison") or {}
                    v_state = comp.get("validator_state_headers_only") or {}
                    st.caption(
                        f"If this report used only state-style headers, validity would be "
                        f"{v_state.get('score', {}).get('validity_score_0_100', '—')} "
                        f"(required_sections_rate={v_state.get('metrics', {}).get('required_sections_rate')})."
                    )
                    dr = LIVE_DEBUG_DIR / "DIAGNOSTIC_REPORT.md"
                    if dr.is_file():
                        st.markdown(dr.read_text(encoding="utf-8"))
