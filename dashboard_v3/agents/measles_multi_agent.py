"""
Multi-agent measles pipeline: state-risk forecaster (tools) → Census enrichment → grounded report (tools).
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from utils.logging_config import get_logger

from .census_tools import CENSUS_TOOL_NAME, execute_census_tool
from .openai_state_risk_agent import _assistant_message_dict, _load_env
from .openai_state_risk_agent import run_state_risk_forecaster
from .report_context_tools import REPORT_CONTEXT_TOOL_NAME, REPORT_CONTEXT_TOOLS, execute_report_context_tool
from .state_risk_snapshot import DISCLAIMER as SNAPSHOT_DISCLAIMER

logger = get_logger("measles_multi_agent")

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"

AGENT1_USER_MESSAGE = """Summarize current U.S. state-level measles risk using the snapshot from get_state_risk_snapshot.
For the highest-risk states in states_top, explain what is driving total_risk by comparing coverage_points,
case_points, and wastewater_points (which components are largest for each state). Name specific states.
Be concise; this is situational awareness only."""

REPORT_SYSTEM_PROMPT = """You are a concise measles situational-awareness briefer for U.S. public health audiences.
Your first action must be to call the tool get_precomputed_report_inputs exactly once.
Use ONLY data in the tool payload. Do not invent statistics, correlations, or causal claims.
Tone: decisive and operational (use "This indicates…", "This suggests near-term risk…", "This is driven by…")—not academic hedging.
Demographic fields (income_bucket, etc.) are descriptive context only—they are NOT used in risk scoring; do not discuss race or % white; do not run or mention Pearson/correlation/inference."""

REPORT_USER_PROMPT = """Call get_precomputed_report_inputs first. Then write a markdown briefing with these sections IN ORDER:

## Executive summary
3–5 tight bullets. National alarm/baseline/forecast from payload only.

## National outlook and short-horizon forecast
Short paragraph(s). Format numbers: rates to **2 decimals**, percentages to **1 decimal**, risk scores to **1 decimal**.

## Normalized risk context
2–4 bullets comparing the **top cohort in aggregate** (not per-state cards) using **cases_per_100k**, **delta_vs_national**, and **agent2.summary_stats.national_avg_cases_per_100k**. Do **not** repeat per-state risk narratives, tiers, or scores—the dashboard UI already shows compact top-state cards. No correlation language.

## Population context (descriptive only)
At most 3 bullets using **population_bucket** only if helpful. No inference.

## Demographic note
Single sentence: "Demographic data are included for context only and are not used in risk scoring." Do not list % white or income correlations.

## Data gaps and reliability
Bullets: which states in the enriched top cohort have **data_completeness_flag** partial or **Low** confidence—keep brief, no full card-style repeats.

## Low-risk baseline states
One short paragraph describing the lowest-risk **cohort** in aggregate. Do **not** enumerate state codes. Do not restate the dashboard's short bullet summary verbatim.

## What to watch next
Exactly 3 short bullets: operational watch items only. Do not copy the dashboard's fixed bullet text word-for-word; paraphrase the same ideas (wastewater-led signals, above-national burden, low-confidence caveats).

## Limitations and disclaimer
Include payload disclaimer; keep brief.

RULES: No Pearson/correlation/% white. No duplicated stats. No raw long floats—always round as specified. No "prioritization framework" section. No "## Top states" section and no per-state card blocks."""


def _score_col(sr: pd.DataFrame) -> str:
    if "total_risk" in sr.columns:
        return "total_risk"
    if "risk_score" in sr.columns:
        return "risk_score"
    raise ValueError("state_risk_df must include total_risk or risk_score")


def select_top_bottom_states(state_risk_df: pd.DataFrame, n: int = 5) -> Tuple[List[str], List[str], pd.DataFrame]:
    """
    Return (top_n state codes, bottom_n state codes, combined slice with cohort labels).
    """
    if state_risk_df is None or state_risk_df.empty:
        return [], [], pd.DataFrame()

    df = state_risk_df.copy()
    col = _score_col(df)
    if "state" not in df.columns:
        raise ValueError("state_risk_df must include a state column")

    df["_risk"] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["_risk", "state"]).sort_values("_risk", ascending=False).reset_index(drop=True)

    if df.empty:
        return [], [], pd.DataFrame()

    top = df.head(n).copy()
    bottom = df.tail(n).copy()

    top_codes = [str(s).strip().upper() for s in top["state"].tolist()]
    bottom_codes = [str(s).strip().upper() for s in bottom["state"].tolist()]

    top["cohort"] = "top_risk"
    top_set = {str(s).strip().upper() for s in top["state"].tolist()}
    bottom_only = bottom[
        ~bottom["state"].map(lambda x: str(x).strip().upper()).isin(top_set)
    ].copy()
    bottom_only["cohort"] = "bottom_risk"
    combined = pd.concat([top, bottom_only], ignore_index=True)
    return top_codes, bottom_codes, combined


def _forecast_note(forecast_df: Optional[pd.DataFrame]) -> str:
    if forecast_df is None or forecast_df.empty:
        return "National short-horizon forecast: not available (insufficient NNDSS history or projection failed)."
    sub = forecast_df.copy()
    if "forecast" not in sub.columns:
        return "National short-horizon forecast: present but missing forecast column."
    v = pd.to_numeric(sub["forecast"], errors="coerce").dropna()
    if v.empty:
        return "National short-horizon forecast: no numeric values."
    mean_f = float(v.mean())
    return (
        f"National baseline-style weekly forecast (mean over available horizon rows): {mean_f:.2f} cases/week "
        f"(see pipeline get_forecast; not a mechanistic outbreak forecast)."
    )


def _json_safe(v: Any) -> Any:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return float(v) if isinstance(v, np.floating) else int(v)
    if isinstance(v, (int, float, str, bool)):
        return v
    if pd.isna(v):
        return None
    return str(v)


def _state_to_risk_rank(state_risk_df: pd.DataFrame) -> Dict[str, int]:
    """1 = highest total_risk (unchanged ranking; Census does not affect this)."""
    if state_risk_df is None or state_risk_df.empty:
        return {}
    df = state_risk_df.copy()
    col = _score_col(df)
    df = df.dropna(subset=[col, "state"]).sort_values(col, ascending=False, na_position="last")
    return {str(r["state"]).strip().upper(): i + 1 for i, (_, r) in enumerate(df.iterrows())}


def _population_bucket(pop: Optional[int]) -> Optional[str]:
    if pop is None or (isinstance(pop, float) and np.isnan(pop)):
        return None
    try:
        p = int(pop)
    except (TypeError, ValueError):
        return None
    if p < 2_000_000:
        return "small"
    if p <= 10_000_000:
        return "medium"
    return "large"


def _income_bucket(income: Optional[float]) -> Optional[str]:
    if income is None or (isinstance(income, float) and np.isnan(income)):
        return None
    try:
        x = float(income)
    except (TypeError, ValueError):
        return None
    if x < 65_000:
        return "low"
    if x <= 85_000:
        return "medium"
    return "high"


def _safe_float(x: Any) -> Optional[float]:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return None
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    if np.isnan(v) or np.isinf(v):
        return None
    return v


def _cases_per_100k(cases_recent: Any, population: Optional[int]) -> Optional[float]:
    cr = _safe_float(cases_recent)
    if cr is None or population is None or population <= 0:
        return None
    return (cr / float(population)) * 100_000.0


def _vulnerability_flag(coverage_points: Any, median_income: Optional[float]) -> bool:
    cp = _safe_float(coverage_points) or 0.0
    if cp > 10.0:
        return True
    if median_income is not None and median_income < 65_000:
        return True
    return False


def _data_completeness_flag(row: pd.Series) -> str:
    ww = row.get("wastewater_coverage")
    if ww is True or ww == 1 or str(ww).lower() == "true":
        return "full"
    return "partial"


def _risk_row_to_risk_fields(row: pd.Series) -> Dict[str, Any]:
    """Copy risk-model fields only; values pass through unchanged."""
    keys = (
        "state",
        "coverage",
        "cases_recent",
        "ww_recent",
        "wastewater_coverage",
        "coverage_points",
        "case_points",
        "wastewater_points",
        "total_risk",
        "risk_tier",
        "risk_score",
    )
    out: Dict[str, Any] = {}
    for k in keys:
        if k not in row.index:
            continue
        out[k] = _json_safe(row[k])
    return out


def _signal_dominance(row: pd.Series) -> str:
    """Mutually exclusive driver label from point components (risk model unchanged)."""
    wp = _safe_float(row.get("wastewater_points")) or 0.0
    cp = _safe_float(row.get("case_points")) or 0.0
    covp = _safe_float(row.get("coverage_points")) or 0.0
    if wp >= cp and wp >= covp:
        return "Wastewater-driven"
    if cp >= wp and cp >= covp:
        return "Case-driven"
    return "Coverage-driven"


def _risk_narrative(
    *,
    data_completeness_flag: str,
    signal_dominance: str,
    cases_per_100k: Optional[float],
    top_cohort_median_cpk: Optional[float],
) -> str:
    """Single mutually exclusive narrative label."""
    if data_completeness_flag == "partial":
        return "Data-limited"
    if signal_dominance == "Wastewater-driven":
        return "Early signal (wastewater-led)"
    if signal_dominance == "Case-driven":
        if cases_per_100k is not None and top_cohort_median_cpk is not None and cases_per_100k >= top_cohort_median_cpk:
            return "Active spread"
        return "Emerging spread"
    if signal_dominance == "Coverage-driven":
        return "Structural vulnerability"
    return "Mixed signal"


_WHY_IT_MATTERS: Dict[str, str] = {
    "Data-limited": "Interpretation is limited until wastewater and case reporting are more complete.",
    "Early signal (wastewater-led)": "Wastewater is a leading indicator that may precede reported case increases.",
    "Active spread": "Per-capita case burden is elevated versus the top-risk cohort median—monitor for acceleration.",
    "Emerging spread": "Cases drive the score; per-capita burden is below the top-cohort median for now.",
    "Structural vulnerability": "Coverage pressure dominates relative to recent cases and wastewater in this snapshot.",
    "Mixed signal": "Drivers are split across streams—use confidence and data completeness when prioritizing attention.",
}


def _confidence_score_and_label(row: pd.Series, data_completeness_flag: str) -> Tuple[float, str]:
    score = 1.0
    ww = row.get("wastewater_coverage")
    if ww is False or ww == 0:
        ww_missing = True
    elif ww is True or ww == 1:
        ww_missing = False
    else:
        ww_missing = str(ww).lower() not in ("true", "1", "yes")
    if ww_missing:
        score -= 0.3
    if data_completeness_flag == "partial":
        score -= 0.3
    if _safe_float(row.get("cases_recent")) is None:
        score -= 0.2
    score = max(round(score, 2), 0.0)
    if score >= 0.75:
        label = "High"
    elif score >= 0.5:
        label = "Medium"
    else:
        label = "Low"
    return score, label


def compute_national_avg_cases_per_100k(
    state_risk_df: pd.DataFrame,
    census_by_state: Dict[str, Optional[Dict[str, Any]]],
) -> Tuple[Optional[float], int]:
    """Mean cases_per_100k over states with computable values (Census population present)."""
    if state_risk_df is None or state_risk_df.empty:
        return None, 0
    vals: List[float] = []
    for _, rw in state_risk_df.iterrows():
        st = str(rw.get("state", "")).strip().upper()
        dem = census_by_state.get(st)
        pop = dem.get("population") if dem else None
        if pop is not None:
            try:
                pop = int(pop)
            except (TypeError, ValueError):
                pop = None
        cpk = _cases_per_100k(rw.get("cases_recent"), pop)
        if cpk is not None:
            vals.append(cpk)
    if not vals:
        return None, 0
    return round(float(np.mean(vals)), 2), len(vals)


def _census_public(census: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Drop fields not used in reporting (e.g. race proxies)."""
    if not census:
        return census
    return {k: v for k, v in census.items() if k != "pct_white"}


def _format_enriched_numerics(out: Dict[str, Any], national_avg: Optional[float]) -> None:
    """Round display fields in-place for dashboard-friendly JSON."""
    tr = _safe_float(out.get("total_risk"))
    if tr is not None:
        out["total_risk"] = round(tr, 1)
    rs = _safe_float(out.get("risk_score"))
    if rs is not None:
        out["risk_score"] = round(rs, 1)
    cr = _safe_float(out.get("cases_recent"))
    if cr is not None:
        out["cases_recent"] = int(round(cr))
    cpk = _safe_float(out.get("cases_per_100k"))
    if cpk is not None:
        out["cases_per_100k"] = round(cpk, 2)
    if national_avg is not None:
        out["national_avg_cases_per_100k"] = round(float(national_avg), 2)
    dlt = _safe_float(out.get("delta_vs_national"))
    if dlt is not None:
        out["delta_vs_national"] = round(dlt, 2)
    cs = _safe_float(out.get("confidence_score"))
    if cs is not None:
        out["confidence_score"] = round(cs, 2)


def _enrich_one_state(
    row: pd.Series,
    census: Optional[Dict[str, Any]],
    risk_rank: Optional[int],
    cohort: str,
    *,
    national_avg_cpk: Optional[float],
    top_cohort_median_cpk: Optional[float],
) -> Dict[str, Any]:
    risk_fields = _risk_row_to_risk_fields(row)
    st = str(row.get("state", "")).strip().upper()
    pop = None
    income = None
    if census:
        pop = census.get("population")
        if pop is not None:
            try:
                pop = int(pop)
            except (TypeError, ValueError):
                pop = None
        income = _safe_float(census.get("median_household_income"))

    cpk_raw = _cases_per_100k(row.get("cases_recent"), pop)
    delta = (cpk_raw - national_avg_cpk) if (cpk_raw is not None and national_avg_cpk is not None) else None

    dcf = _data_completeness_flag(row)
    sig = _signal_dominance(row)
    narrative = _risk_narrative(
        data_completeness_flag=dcf,
        signal_dominance=sig,
        cases_per_100k=cpk_raw,
        top_cohort_median_cpk=top_cohort_median_cpk,
    )
    conf_score, conf_label = _confidence_score_and_label(row, dcf)

    out: Dict[str, Any] = {
        **risk_fields,
        "state": st,
        "cohort": cohort,
        "risk_rank": risk_rank,
        "census": _census_public(census),
        "cases_per_100k": _json_safe(cpk_raw),
        "population_bucket": _population_bucket(pop),
        "income_bucket": _income_bucket(income),
        "vulnerability_flag": _vulnerability_flag(row.get("coverage_points"), income),
        "data_completeness_flag": dcf,
        "signal_dominance": sig,
        "risk_narrative": narrative,
        "why_it_matters": _WHY_IT_MATTERS.get(narrative, _WHY_IT_MATTERS["Mixed signal"]),
        "national_avg_cases_per_100k": round(float(national_avg_cpk), 2) if national_avg_cpk is not None else None,
        "delta_vs_national": round(delta, 2) if delta is not None else None,
        "confidence_score": conf_score,
        "confidence_label": conf_label,
    }
    _format_enriched_numerics(out, national_avg_cpk)
    return out


def build_agent2_enrichment(
    *,
    top_codes: List[str],
    bottom_codes: List[str],
    census_by_state: Dict[str, Optional[Dict[str, Any]]],
    state_risk_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Agent 2 output: Census + derived context only (does not modify risk scores).
    """
    rank_map = _state_to_risk_rank(state_risk_df)
    national_avg_cpk, n_nat = compute_national_avg_cases_per_100k(state_risk_df, census_by_state)

    def row_for_full(code: str) -> Optional[pd.Series]:
        if state_risk_df is None or state_risk_df.empty:
            return None
        m = state_risk_df["state"].map(lambda x: str(x).strip().upper()) == code
        if not m.any():
            return None
        return state_risk_df.loc[m].iloc[0]

    staging: List[Tuple[str, str, pd.Series]] = []
    seen_top = set()
    for c in top_codes:
        r = row_for_full(c)
        if r is not None:
            staging.append((c, "top_risk", r))
            seen_top.add(c)
    for c in bottom_codes:
        if c in seen_top:
            continue
        r = row_for_full(c)
        if r is not None:
            staging.append((c, "bottom_risk", r))

    top_cpks_raw: List[float] = []
    for c in top_codes:
        rw = row_for_full(c)
        if rw is None:
            continue
        dem = census_by_state.get(c)
        pop = dem.get("population") if dem else None
        if pop is not None:
            try:
                pop = int(pop)
            except (TypeError, ValueError):
                pop = None
        cpk = _cases_per_100k(rw.get("cases_recent"), pop)
        if cpk is not None:
            top_cpks_raw.append(cpk)
    top_cohort_median_cpk = float(np.median(top_cpks_raw)) if top_cpks_raw else None

    top_enriched: List[Dict[str, Any]] = []
    bottom_enriched: List[Dict[str, Any]] = []

    for code, cohort, rw in staging:
        dem = census_by_state.get(code)
        rr = rank_map.get(code)
        enriched = _enrich_one_state(
            rw,
            dem,
            rr,
            cohort,
            national_avg_cpk=national_avg_cpk,
            top_cohort_median_cpk=top_cohort_median_cpk,
        )
        if cohort == "top_risk":
            top_enriched.append(enriched)
        else:
            bottom_enriched.append(enriched)

    def _avg_cpk(rows: List[Dict[str, Any]]) -> Optional[float]:
        vals = [float(r["cases_per_100k"]) for r in rows if r.get("cases_per_100k") is not None]
        return round(float(np.mean(vals)), 2) if vals else None

    def _pct_high_income(rows: List[Dict[str, Any]]) -> Optional[float]:
        inc = [r.get("income_bucket") for r in rows]
        known = [x for x in inc if x is not None]
        if not known:
            return None
        hi = sum(1 for x in known if x == "high")
        return round(100.0 * hi / len(known), 1)

    summary_stats = {
        "national_avg_cases_per_100k": national_avg_cpk,
        "n_states_in_national_avg": n_nat,
        "top_cohort_median_cases_per_100k": round(top_cohort_median_cpk, 2) if top_cohort_median_cpk is not None else None,
        "avg_cases_per_100k_top": _avg_cpk(top_enriched),
        "avg_cases_per_100k_bottom": _avg_cpk(bottom_enriched),
        "pct_high_income_top": _pct_high_income(top_enriched),
        "pct_high_income_bottom": _pct_high_income(bottom_enriched),
        "n_top": len(top_enriched),
        "n_bottom": len(bottom_enriched),
    }

    return {
        "top_states_enriched": top_enriched,
        "bottom_states_enriched": bottom_enriched,
        "summary_stats": summary_stats,
    }


def build_precomputed_payload(
    *,
    data_bundle: Dict[str, Any],
    agent1: Dict[str, Any],
    agent2_enrichment: Dict[str, Any],
) -> Dict[str, Any]:
    """Structured JSON for the report-context tool + markdown appendix."""
    d = data_bundle
    alarm = float(d.get("alarm_prob", 0.5) or 0.5)
    btier = str(d.get("baseline_tier", "low"))
    bval = float(d.get("baseline_val", 0.0) or 0.0)
    data_as_of = d.get("data_as_of")

    top_e = agent2_enrichment.get("top_states_enriched") or []
    bottom_e = agent2_enrichment.get("bottom_states_enriched") or []

    last_snap = agent1.get("last_snapshot") or {}
    summary = last_snap.get("summary") if isinstance(last_snap, dict) else {}

    md_lines = [
        "# PRECOMPUTED_REPORT_CONTEXT",
        "",
        f"- Data as of: {data_as_of}",
        f"- National alarm probability (stage-1): {alarm:.3f}",
        f"- Baseline tier/score: {btier} / {bval:.1f}",
        f"- {_forecast_note(d.get('forecast_df'))}",
        "",
        "## Snapshot summary (from tool)",
        json.dumps(summary, indent=2) if summary else "(no summary)",
        "",
        "## Agent 2 summary_stats (Census context only; risk scores unchanged)",
        json.dumps(agent2_enrichment.get("summary_stats") or {}, indent=2),
        "",
        "## Top cohort (enriched)",
    ]
    for r in top_e:
        md_lines.append(
            f"- **{r.get('state')}** rank={r.get('risk_rank')} tier={r.get('risk_tier')} "
            f"total_risk={r.get('total_risk')} narrative={r.get('risk_narrative')} "
            f"dominance={r.get('signal_dominance')} cp100k={r.get('cases_per_100k')} "
            f"delta_nat={r.get('delta_vs_national')} conf={r.get('confidence_label')}/{r.get('confidence_score')} "
            f"ww={r.get('data_completeness_flag')}"
        )
    md_lines.append("## Bottom cohort (enriched)")
    for r in bottom_e:
        md_lines.append(
            f"- **{r.get('state')}** rank={r.get('risk_rank')} total_risk={r.get('total_risk')} "
            f"cp100k={r.get('cases_per_100k')} narrative={r.get('risk_narrative')}"
        )
    md_lines.extend(
        [
            "",
            "## Demographic context",
            "Demographic data are included for context only and are not used in risk scoring.",
            "",
            "## Disclaimer",
            SNAPSHOT_DISCLAIMER,
        ]
    )

    return {
        "schema_version": "measles_report_context_1.2",
        "data_as_of": data_as_of,
        "agent1_assistant_text": agent1.get("assistant_text"),
        "last_snapshot_summary": summary,
        "national": {
            "alarm_probability": round(alarm, 3),
            "baseline_tier": btier,
            "baseline_val": round(bval, 1),
            "forecast_note": _forecast_note(d.get("forecast_df")),
        },
        "agent2": agent2_enrichment,
        "demographic_context_note": (
            "Demographic data are included for context only and are not used in risk scoring."
        ),
        "disclaimer": SNAPSHOT_DISCLAIMER,
        "precomputed_markdown": "\n".join(md_lines),
    }


def run_measles_report_agent(
    payload: Dict[str, Any],
    *,
    model: Optional[str] = None,
    max_tool_rounds: int = 6,
) -> Dict[str, Any]:
    """Agent 3: first completion must call get_precomputed_report_inputs; then final markdown."""
    _load_env()
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set; add it to repo or dashboard_v3 .env")

    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)
    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": REPORT_SYSTEM_PROMPT},
        {"role": "user", "content": REPORT_USER_PROMPT},
    ]

    first_turn = True
    rounds = 0
    assistant_text = ""

    while rounds < max_tool_rounds:
        rounds += 1
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": REPORT_CONTEXT_TOOLS,
        }
        kwargs["tool_choice"] = "required" if first_turn else "auto"
        first_turn = False

        completion = client.chat.completions.create(**kwargs)
        msg = completion.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if rounds == 1 and not tool_calls:
            kwargs["tool_choice"] = "required"
            completion = client.chat.completions.create(**kwargs)
            msg = completion.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

        if rounds == 1 and not tool_calls:
            raise RuntimeError("Expected get_precomputed_report_inputs on the first turn (tool_choice=required).")

        if not tool_calls:
            assistant_text = (msg.content or "").strip()
            return {"assistant_text": assistant_text, "model": model}

        messages.append(_assistant_message_dict(msg))

        for tc in tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            content = execute_report_context_tool(name, raw_args, payload=payload)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )

    raise RuntimeError(f"Exceeded max_tool_rounds={max_tool_rounds} without final report text.")


def run_measles_multi_agent_pipeline(
    data_bundle: Dict[str, Any],
    *,
    kg: pd.DataFrame,
    nndss: pd.DataFrame,
    ww: pd.DataFrame,
    agent1_user_message: str = AGENT1_USER_MESSAGE,
    model: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run Agent 1 (state risk forecaster), Agent 2 (Census via tool executor), Agent 3 (grounded report).

    Parameters
    ----------
    data_bundle
        Output of ``load_and_model`` (needs state_risk_df, alarm_prob, forecast_df, etc.).
    """
    _load_env()
    data_as_of = data_bundle.get("data_as_of")
    if isinstance(data_as_of, pd.Timestamp):
        data_as_of = data_as_of.isoformat()

    agent1 = run_state_risk_forecaster(
        kg,
        nndss,
        ww,
        user_message=agent1_user_message,
        model=model,
        data_as_of=data_as_of if isinstance(data_as_of, str) else None,
    )

    sr = data_bundle.get("state_risk_df")
    if isinstance(sr, pd.DataFrame) and sr.empty:
        sr = None

    top_codes: List[str] = []
    bottom_codes: List[str] = []
    cohort_df = pd.DataFrame()
    census_by_state: Dict[str, Optional[Dict[str, Any]]] = {}

    agent2_enrichment: Dict[str, Any] = {
        "top_states_enriched": [],
        "bottom_states_enriched": [],
        "summary_stats": {
            "national_avg_cases_per_100k": None,
            "n_states_in_national_avg": 0,
            "top_cohort_median_cases_per_100k": None,
            "avg_cases_per_100k_top": None,
            "avg_cases_per_100k_bottom": None,
            "pct_high_income_top": None,
            "pct_high_income_bottom": None,
            "n_top": 0,
            "n_bottom": 0,
        },
    }

    if sr is not None:
        top_codes, bottom_codes, cohort_df = select_top_bottom_states(sr, n=5)
        all_codes = sorted({str(s).strip().upper() for s in sr["state"].dropna()})
        if all_codes:
            raw_json = execute_census_tool(CENSUS_TOOL_NAME, json.dumps({"state_codes": all_codes}))
            try:
                census_by_state = json.loads(raw_json)
                if not isinstance(census_by_state, dict):
                    census_by_state = {}
            except json.JSONDecodeError:
                census_by_state = {}
        if top_codes or bottom_codes:
            agent2_enrichment = build_agent2_enrichment(
                top_codes=top_codes,
                bottom_codes=bottom_codes,
                census_by_state=census_by_state,
                state_risk_df=sr,
            )

    payload = build_precomputed_payload(
        data_bundle=data_bundle,
        agent1=agent1,
        agent2_enrichment=agent2_enrichment,
    )

    agent3 = run_measles_report_agent(payload, model=model)

    return {
        "agent1": agent1,
        "agent2": {
            **agent2_enrichment,
            "meta": {
                "top_state_codes": top_codes,
                "bottom_state_codes": bottom_codes,
                "cohort_row_count": int(len(cohort_df)),
            },
        },
        "report_context_payload": payload,
        "agent3": agent3,
    }
