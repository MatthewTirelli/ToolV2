"""
Measles outbreak risk dashboard — Streamlit UI.
Data loading and modeling: model_runner.load_and_model (unchanged).
"""
from __future__ import annotations

import csv
import html
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from plotly import graph_objects as go

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(APP_DIR.parent / ".env")
    load_dotenv(APP_DIR / ".env")
except ImportError:
    pass

from components.charts import (
    baseline_gauge_figure,
    nndss_only_figure,
    ww_vs_nndss_dual_axis,
)
from components.map import kindergarten_coverage_map_figure, state_risk_map_figure
from agents.measles_multi_agent import run_measles_multi_agent_pipeline
from kg_helpers import filter_kg_by_year, kg_state_pct_columns, prepare_kg_years
from model_runner import load_and_model
from risk import compute_ww_detection_frequency, validate_ww_nndss_audit
from utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("app")

DASH_CSS = """
<style>
[data-testid="stAppViewContainer"] { background-color: #f5f7fa !important; }
[data-testid="stHeader"] { background: #f5f7fa !important; }
section[data-testid="stSidebar"] > div { background: #f0f2f6 !important; }
.block-container { padding-top: 1.25rem !important; padding-bottom: 2rem !important; max-width: 1680px !important; }
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
h1.page-title { font-size: 1.75rem !important; font-weight: 700 !important; color: #0f172a !important; margin-bottom: 0.25rem !important; letter-spacing: -0.02em; }
p.page-sub { color: #64748b; font-size: 0.95rem; margin-bottom: 1.25rem; }
.disclaimer { font-size: 0.75rem; color: #94a3b8; margin-bottom: 1rem; }
.card {
  background: #ffffff;
  border-radius: 12px;
  padding: 20px;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08), 0 4px 12px rgba(15, 23, 42, 0.06);
  border: 1px solid #e8ecf1;
  margin-bottom: 0;
}
.card-tight { padding: 16px 20px; }
.kpi-title { font-size: 0.72rem; text-transform: uppercase; letter-spacing: 0.06em; color: #64748b; font-weight: 600; }
.kpi-value { font-size: 1.85rem; font-weight: 700; color: #0f172a; margin-top: 0.35rem; line-height: 1.15; }
.kpi-sub { font-size: 0.82rem; color: #94a3b8; margin-top: 0.35rem; }
.ai-panel-inner {
  min-height: 320px;
  max-height: 520px;
  overflow-y: auto;
  padding: 1rem 1.1rem;
  font-size: 0.92rem;
  line-height: 1.65;
  color: #475569;
  background: #fafbfc;
  border-radius: 8px;
  border: 1px solid #e2e8f0;
}
.section-title { font-size: 0.95rem; font-weight: 600; margin-bottom: 0.75rem; color: #0f172a; }
.ai-subhead { font-size: 0.88rem; font-weight: 600; color: #0f172a; margin: 0.5rem 0 0.35rem 0; }
.ai-section-divider { height: 1px; background: #e2e8f0; margin: 16px 0; }
.watch-bullets { font-size: 0.82rem; color: #334155; margin: 0.25rem 0 0.75rem 0; padding-left: 1.15rem; line-height: 1.5; }
.watch-bullets li { margin-bottom: 0.2rem; }
.risk-state-card {
  background: #ffffff;
  border-radius: 10px;
  padding: 14px 16px;
  margin-bottom: 10px;
  border: 1px solid #e2e8f0;
  border-left: 4px solid #94a3b8;
  transition: box-shadow 0.15s ease, border-color 0.15s ease;
  font-size: 0.875rem;
}
.risk-state-card:hover {
  box-shadow: 0 4px 14px rgba(15, 23, 42, 0.08);
}
.risk-state-card.tier-high { border-left-color: #dc2626; }
.risk-state-card.tier-medium { border-left-color: #d97706; }
.risk-state-card.tier-low { border-left-color: #64748b; }
.risk-state-card.tier-neutral { border-left-color: #94a3b8; }
.risk-card-head {
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 10px;
  margin-bottom: 8px;
}
.risk-card-title { font-weight: 600; color: #0f172a; font-size: 0.95rem; }
.risk-card-conf { font-size: 0.8rem; color: #475569; white-space: nowrap; }
.risk-card-metrics {
  font-size: 0.78rem;
  color: #475569;
  margin-bottom: 6px;
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 4px 0;
}
.risk-card-metrics .metric-sep { color: #cbd5e1; margin: 0 8px; user-select: none; }
.risk-card-metrics span[title] { cursor: help; border-bottom: 1px dotted #cbd5e1; }
.risk-card-narr {
  font-size: 0.8rem;
  font-style: italic;
  color: #475569;
  margin-bottom: 4px;
  line-height: 1.45;
}
.risk-card-why { font-size: 0.75rem; color: #94a3b8; line-height: 1.45; }
div[data-testid="column"] { min-width: 0; }
[data-testid="stPlotlyChart"] {
  background: #ffffff !important;
  border-radius: 12px !important;
  padding: 12px 16px 20px !important;
  box-shadow: 0 1px 3px rgba(15, 23, 42, 0.08), 0 4px 12px rgba(15, 23, 42, 0.06) !important;
  border: 1px solid #e8ecf1 !important;
  margin-bottom: 0.75rem;
}
</style>
"""


def _inject_css() -> None:
    st.markdown(DASH_CSS, unsafe_allow_html=True)


def _kpi_derived(d: dict | None) -> dict:
    """Same KPI derivation as dashboard/server.py `kpi_derived`."""
    out = {
        "alarm": 0.5,
        "hi_state": "—",
        "hi_score": None,
        "latest_cases": None,
        "cases_delta_pct": None,
    }
    if not d:
        return out
    out["alarm"] = float(d.get("alarm_prob", 0.5))
    sr = d.get("state_risk_df")
    if sr is not None and not sr.empty:
        score_col = "risk_score" if "risk_score" in sr.columns else "total_risk"
        if score_col in sr.columns:
            i = sr[score_col].idxmax()
            row = sr.loc[i]
            out["hi_state"] = str(row.get("state", "—"))
            out["hi_score"] = float(row[score_col]) if pd.notna(row[score_col]) else None
    nndss_agg = d.get("nndss_agg")
    if nndss_agg is not None and isinstance(nndss_agg, pd.DataFrame) and len(nndss_agg) >= 1:
        tail = nndss_agg.sort_values(["year", "week"]).reset_index(drop=True)
        last = tail.iloc[-1]
        out["latest_cases"] = float(last["cases"])
        if len(tail) >= 2:
            prev = tail.iloc[-2]
            pv = float(prev["cases"])
            if pv != 0:
                out["cases_delta_pct"] = (float(last["cases"]) - pv) / pv * 100.0
            elif float(last["cases"]) == 0:
                out["cases_delta_pct"] = 0.0
            else:
                out["cases_delta_pct"] = float("inf")
    return out


def _ww_year_choices(d: dict | None) -> list[str]:
    years_avail: list[int] = []
    if not d:
        return ["All"]
    nndss_agg = d.get("nndss_agg")
    ww = d.get("ww")
    if nndss_agg is not None and not nndss_agg.empty:
        na = nndss_agg.copy()
        na["year"] = pd.to_numeric(na["year"], errors="coerce")
        na = na.dropna(subset=["year"])
        years_avail = sorted(na["year"].astype(int).unique().tolist())
    if ww is not None and not ww.empty and "year" in ww.columns:
        ww_y = sorted(pd.to_numeric(ww["year"], errors="coerce").dropna().astype(int).unique().tolist())
        years_avail = sorted(set(years_avail) | set(ww_y)) if years_avail else ww_y
    return ["All"] + [str(y) for y in years_avail]


def _ww_nndss_context(d: dict | None, ym: str | None, yx: str | None) -> dict:
    """Port of dashboard/server.py `ww_nndss_context` reactive calc (same logic)."""
    empty = {
        "ww_weekly": pd.DataFrame(),
        "ww_val": {},
        "nndss_filtered": pd.DataFrame(),
        "audit": {},
        "merged_inner": pd.DataFrame(),
        "merged_nndss_only": pd.DataFrame(),
        "show_ww_only": False,
        "show_nd_only": False,
        "warn": [],
    }
    if not d:
        return empty
    ww = d.get("ww")
    nndss_agg = d.get("nndss_agg")
    if nndss_agg is None:
        nndss_agg = pd.DataFrame()
    if not nndss_agg.empty:
        nndss_agg = nndss_agg.copy()
        nndss_agg["year"] = pd.to_numeric(nndss_agg["year"], errors="coerce")
        nndss_agg = nndss_agg.dropna(subset=["year"])
        nndss_agg["year"] = nndss_agg["year"].astype(int)
    year_min = int(ym) if ym not in (None, "", "All") else None
    year_max = int(yx) if yx not in (None, "", "All") else None
    ww_weekly, ww_val = compute_ww_detection_frequency(
        ww if ww is not None else pd.DataFrame(), year_min=year_min, year_max=year_max
    )
    nndss_filtered = nndss_agg.copy()
    if not nndss_filtered.empty:
        if year_min is not None:
            nndss_filtered = nndss_filtered[nndss_filtered["year"].astype(int) >= int(year_min)]
        if year_max is not None:
            nndss_filtered = nndss_filtered[nndss_filtered["year"].astype(int) <= int(year_max)]
    audit = validate_ww_nndss_audit(ww_val, ww_weekly, nndss_filtered, year_min, year_max)
    warns = []
    nndss_sum = audit.get("nndss_cases_sum", 0)
    if nndss_sum == 0 and not nndss_filtered.empty:
        warns.append(
            "NNDSS cases sum for selected period: 0. No reported measles cases in this window; charts may be empty."
        )
    _, ww_val_range = compute_ww_detection_frequency(
        ww if ww is not None else pd.DataFrame(), year_min=None, year_max=None
    )
    weeks_min = ww_val_range.get("weeks_min")
    weeks_max = ww_val_range.get("weeks_max")
    ww_year_min = int(weeks_min[0]) if weeks_min else None
    ww_year_max = int(weeks_max[0]) if weeks_max else None
    if ww_year_min is not None and year_max is not None and int(year_max) < ww_year_min:
        warns.append(
            f"No wastewater data for selected years. Measles wastewater monitoring began in {ww_year_min}."
        )
    merged_inner = pd.DataFrame()
    if not ww_weekly.empty and not nndss_filtered.empty:
        merged_inner = ww_weekly.copy()
        merged_inner["year"] = merged_inner["year"].astype(int)
        nndss_f = nndss_filtered.copy()
        nndss_f["year"] = nndss_f["year"].astype(int)
        merged_inner = merged_inner.merge(nndss_f[["year", "week", "cases"]], on=["year", "week"], how="inner")
        if merged_inner.empty:
            warns.append(
                "No overlapping wastewater + NNDSS weeks for selected range. Check year filters."
            )
        else:
            merged_inner = merged_inner.sort_values(["year", "week"]).reset_index(drop=True)
            merged_inner["year_week"] = (
                merged_inner["year"].astype(str)
                + "-W"
                + merged_inner["week"].astype(int).astype(str)
            )
    merged_nndss_only = pd.DataFrame()
    if not nndss_filtered.empty and ww_weekly.empty:
        merged_nndss_only = nndss_filtered.sort_values(["year", "week"])
        merged_nndss_only["year_week"] = (
            merged_nndss_only["year"].astype(str) + "-W" + merged_nndss_only["week"].astype(int).astype(str)
        )
    if ww_weekly.empty and ww is not None and not ww.empty:
        warns.append(
            "0 rows after wastewater filters (pcr_target = measles, QC rules). Try widening the year range."
        )
    show_ww_only = not ww_weekly.empty and nndss_filtered.empty
    show_nd_only = ww_weekly.empty and not nndss_filtered.empty
    return {
        "ww_weekly": ww_weekly,
        "ww_val": ww_val,
        "nndss_filtered": nndss_filtered,
        "audit": audit,
        "merged_inner": merged_inner,
        "merged_nndss_only": merged_nndss_only,
        "show_ww_only": show_ww_only,
        "show_nd_only": show_nd_only,
        "warn": warns,
        "ww_year_min_meta": ww_year_min,
        "ww_year_max_meta": ww_year_max,
    }


def _kg_context(d: dict | None, kg_sel: str | None) -> dict:
    """Port of dashboard/server.py `kg_context`."""
    if not d or d.get("kg") is None or d.get("kg").empty:
        return {"kg_work": pd.DataFrame(), "state_col": "", "pct_col": None, "year_label": ""}
    kg = d["kg"]
    kg_full, year_options, _ = prepare_kg_years(kg)
    if year_options:
        if kg_sel in (None, "", "all", "All"):
            y = year_options[-1]
        else:
            try:
                y = int(kg_sel)
            except (TypeError, ValueError):
                y = year_options[-1]
        kg_work = filter_kg_by_year(kg_full, year_options, y)
        ylabel = str(y)
    else:
        kg_work = kg_full
        ylabel = "all years"
    st_c, pc = kg_state_pct_columns(kg_work if not kg_work.empty else kg)
    return {
        "kg_work": kg_work,
        "state_col": st_c,
        "pct_col": pc,
        "year_label": ylabel,
    }


def _forecast_cases_week_str(d: dict | None) -> str:
    if not d:
        return "—"
    fd = d.get("forecast_df")
    if fd is None or fd.empty or "forecast" not in fd.columns:
        return "—"
    v = pd.to_numeric(fd["forecast"], errors="coerce").dropna()
    if v.empty:
        return "—"
    return f"{int(round(float(v.mean())))} cases/week"


def _snapshot_tier_badge(ma: dict | None) -> str:
    if not ma:
        return "—"
    snap = (ma.get("agent1") or {}).get("last_snapshot") or {}
    summ = snap.get("summary") or {}
    tiers = summ.get("tiers") or {}
    if not tiers:
        return "—"
    h = int(tiers.get("high", 0))
    m = int(tiers.get("medium", 0))
    l = int(tiers.get("low", 0))
    return f"{h} High / {m} Medium / {l} Low"


def _national_avg_cpk_str(ma: dict | None) -> str:
    if not ma:
        return "—"
    a2 = ma.get("agent2") or {}
    ss = a2.get("summary_stats") or {}
    v = ss.get("national_avg_cases_per_100k")
    if v is None:
        return "—"
    return f"{float(v):.2f}"


def _sorted_top_states_for_ui(
    agent2: dict | None,
    *,
    limit: int = 3,
    exclude: frozenset[str] | None = None,
) -> list[dict]:
    """Top states by risk_rank, skipping excluded state codes (e.g. FL, UT for default cards)."""
    ex = {x.strip().upper() for x in (exclude or frozenset())}
    top = list((agent2 or {}).get("top_states_enriched") or [])
    top.sort(key=lambda x: int(x["risk_rank"]) if x.get("risk_rank") is not None else 999)
    out: list[dict] = []
    for s in top:
        code = str(s.get("state", "")).strip().upper()
        if code in ex:
            continue
        out.append(s)
        if len(out) >= limit:
            break
    return out


def _risk_tier_card_class(tier: object) -> str:
    t = str(tier or "").lower()
    if "high" in t:
        return "tier-high"
    if "medium" in t:
        return "tier-medium"
    if "low" in t:
        return "tier-low"
    return "tier-neutral"


def _risk_state_card_html(state: dict) -> str:
    code = html.escape(str(state.get("state", "—")))
    rs = state.get("risk_score")
    if rs is None:
        rs = state.get("total_risk")
    try:
        rs_s = f"{float(rs):.1f}" if rs is not None and pd.notna(rs) else "—"
    except (TypeError, ValueError):
        rs_s = "—"
    trt = html.escape(str(state.get("risk_tier", "—")).title())
    tier_cls = _risk_tier_card_class(state.get("risk_tier"))
    conf_raw = state.get("confidence_label")
    conf_disp = _confidence_badge(conf_raw)
    cpk = state.get("cases_per_100k")
    try:
        cpk_s = f"{float(cpk):.2f}" if cpk is not None and pd.notna(cpk) else "—"
    except (TypeError, ValueError):
        cpk_s = "—"
    dvn = state.get("delta_vs_national")
    try:
        dlt_s = f"{float(dvn):+.2f}" if dvn is not None and pd.notna(dvn) else "—"
    except (TypeError, ValueError):
        dlt_s = "—"
    sig = html.escape(str(state.get("signal_dominance", "—")))
    narr = html.escape(str(state.get("risk_narrative", "—")))
    why = html.escape(str(state.get("why_it_matters", "—")))
    title_line = f"{code} — {trt} ({html.escape(str(rs_s))})"
    return f"""<div class="risk-state-card {tier_cls}">
<div class="risk-card-head">
  <span class="risk-card-title">{title_line}</span>
  <span class="risk-card-conf">{conf_disp}</span>
</div>
<div class="risk-card-metrics">
  <span>Cases/100k: {html.escape(cpk_s)}</span>
  <span class="metric-sep">|</span>
  <span title="Difference in cases per 100k vs national average (payload).">Δ vs avg: {html.escape(dlt_s)}</span>
  <span class="metric-sep">|</span>
  <span title="Whether recent signal is driven more by wastewater detection vs reported cases (snapshot).">Signal: {sig}</span>
</div>
<div class="risk-card-narr">{narr}</div>
<div class="risk-card-why">{why}</div>
</div>"""


def _enrichment_for_map(ma: dict | None) -> dict:
    if not ma:
        return {}
    a2 = ma.get("agent2") or {}
    bym: dict[str, dict] = {}
    for s in (a2.get("top_states_enriched") or []) + (a2.get("bottom_states_enriched") or []):
        ab = str(s.get("state", "")).strip().upper()
        if ab:
            bym[ab] = {
                "risk_narrative": s.get("risk_narrative", "—"),
                "confidence_label": s.get("confidence_label", "—"),
            }
    return bym


def _callout_info_warning(top: list[dict]) -> tuple[str | None, str | None]:
    if not top:
        return None, None
    info = None
    mediums = [s for s in top if str(s.get("risk_tier", "")).lower() == "medium"]
    if len(mediums) == 1:
        s = mediums[0]
        info = (
            f"**{s['state']}** is the only **medium-risk** state in the top cohort — "
            f"{s.get('signal_dominance', '—')}; **{s.get('risk_narrative', '—')}**."
        )
    elif len(mediums) > 1:
        codes = ", ".join(str(x["state"]) for x in mediums)
        info = f"**Medium-risk** states in the top cohort: {codes}."

    warn = None
    pool = [
        s
        for s in top
        if s.get("delta_vs_national") is not None
        and (s.get("confidence_label") == "Low" or s.get("data_completeness_flag") == "partial")
    ]
    if pool:
        s = max(pool, key=lambda x: float(x["delta_vs_national"]))
        dlt = float(s["delta_vs_national"])
        cpk = s.get("cases_per_100k")
        cpk_s = f"{float(cpk):.2f}" if cpk is not None else "—"
        warn = (
            f"**{s['state']}** shows the strongest per-capita burden among lower-confidence states "
            f"({cpk_s} cases/100k; **{dlt:+.2f}** vs national avg) with **{s.get('confidence_label', '—')}** confidence."
        )
    return info, warn


def _confidence_badge(label: str | None) -> str:
    if label == "High":
        return "🟢 High"
    if label == "Medium":
        return "🟡 Medium"
    if label == "Low":
        return "🔴 Low"
    return label or "—"


def _do_load(use_cache: bool) -> None:
    st.session_state.load_error = None
    try:
        b = load_and_model(use_cache=use_cache, outbreak_percentile=95.0)
        st.session_state.data_bundle = b
        st.session_state.alarm_prob = float(b.get("alarm_prob", 0.5))
    except Exception:
        logger.exception("load_and_model failed")
        st.session_state.load_error = "Load failed. Check logs and .env (e.g. SOCRATA_APP_TOKEN)."
        st.session_state.data_bundle = None


def main() -> None:
    st.set_page_config(
        page_title="Measles outbreak risk — US",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_css()

    if "data_bundle" not in st.session_state:
        st.session_state.data_bundle = None
    if "load_error" not in st.session_state:
        st.session_state.load_error = None
    if "alarm_prob" not in st.session_state:
        st.session_state.alarm_prob = 0.5
    if "multi_agent_result" not in st.session_state:
        st.session_state.multi_agent_result = None
    if "multi_agent_error" not in st.session_state:
        st.session_state.multi_agent_error = None

    with st.sidebar:
        st.markdown("### Measles risk")
        st.caption("Situational awareness · CDC data")
        if st.button("Refresh data", type="primary", use_container_width=True):
            _do_load(use_cache=False)
            st.rerun()

        if st.session_state.data_bundle is None and st.session_state.load_error is None:
            with st.spinner("Loading data and fitting models…"):
                _do_load(use_cache=True)

        if st.session_state.load_error:
            st.error(st.session_state.load_error)
        elif st.session_state.data_bundle is None:
            st.info("Waiting for data…")
        else:
            d0 = st.session_state.data_bundle
            lines = []
            for source, status in (d0.get("load_status") or {}).items():
                label = "temporarily unavailable" if status == "fail" else status
                lines.append(f"{source}: {label}")
            st.caption("**Data sources**")
            st.caption(" · ".join(lines) if lines else "—")
            st.caption(f"Data as of: **{d0.get('data_as_of') or 'N/A'}**")

    st.markdown('<h1 class="page-title">Risk of Measles Outbreak in US</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="page-sub">Situational awareness dashboard — wastewater, NNDSS, and vaccination context.</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="disclaimer">For situational awareness only; not for clinical or policy decisions. Data: CDC.</p>',
        unsafe_allow_html=True,
    )

    d = st.session_state.data_bundle
    kd = _kpi_derived(d)
    ap = float(st.session_state.alarm_prob) if d else kd["alarm"]

    tab_overview, tab_analysis = st.tabs(["Overview", "Analysis"])

    with tab_overview:
        r1c1, r1c2, r1c3 = st.columns(3, gap="large")
        with r1c1:
            sub_a = "Probability of exceeding threshold in next 4 weeks"
            st.markdown(
                f"""<div class="card">
<div class="kpi-title">Outbreak Alarm</div>
<div class="kpi-value">{ap:.0%}</div>
<div class="kpi-sub">{sub_a}</div>
</div>""",
                unsafe_allow_html=True,
            )
        with r1c2:
            hs = kd["hi_state"]
            sc = kd["hi_score"]
            sub_h = f"Score: {sc:.0f}" if sc is not None else ""
            st.markdown(
                f"""<div class="card">
<div class="kpi-title">Highest Risk State</div>
<div class="kpi-value">{hs}</div>
<div class="kpi-sub">{sub_h}</div>
</div>""",
                unsafe_allow_html=True,
            )
        with r1c3:
            lc = kd["latest_cases"]
            if lc is None:
                sub_c = "NNDSS weekly (national)"
                val = "—"
            else:
                delta = kd["cases_delta_pct"]
                if delta is None or (isinstance(delta, float) and np.isinf(delta)):
                    sub_c = "vs prior week: n/a"
                else:
                    sub_c = f"vs prior week: {delta:+.1f}%"
                val = f"{lc:.0f}"
            st.markdown(
                f"""<div class="card">
<div class="kpi-title">Latest Case Count</div>
<div class="kpi-value">{val}</div>
<div class="kpi-sub">{sub_c}</div>
</div>""",
                unsafe_allow_html=True,
            )

        st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

        ma = st.session_state.multi_agent_result
        enrich = _enrichment_for_map(ma)

        col_map, col_base = st.columns([2, 1], gap="large")
        with col_map:
            st.markdown('<p class="section-title">State risk map</p>', unsafe_allow_html=True)
            if not d:
                fig_m = state_risk_map_figure(pd.DataFrame(), enrichment_by_abbr=None)
            else:
                sr = d.get("state_risk_df")
                fig_m = state_risk_map_figure(
                    sr if sr is not None else pd.DataFrame(),
                    enrichment_by_abbr=enrich or None,
                )
            fig_m.update_layout(
                height=560,
                margin=dict(t=40, b=24, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_m, use_container_width=True, key="overview_map")

        with col_base:
            with st.expander("Baseline risk & model detail", expanded=False):
                if d:
                    v = float(d.get("baseline_val", 0))
                    st.plotly_chart(baseline_gauge_figure(v), use_container_width=True)
                else:
                    st.plotly_chart(baseline_gauge_figure(0), use_container_width=True)

                def _overview_csv() -> str:
                    buf = io.StringIO()
                    w = csv.writer(buf)
                    w.writerow(["metric", "value"])
                    if d:
                        w.writerow(["alarm_probability", d.get("alarm_prob", 0.5)])
                        w.writerow(["baseline_tier", d.get("baseline_tier", "")])
                        w.writerow(["baseline_score", d.get("baseline_val", 0)])
                        w.writerow(["data_as_of", d.get("data_as_of", "")])
                    return buf.getvalue()

                st.download_button(
                    "Download summary CSV",
                    data=_overview_csv(),
                    file_name="overview_summary.csv",
                    mime="text/csv",
                )

                st.markdown("##### How is alarm probability calculated?")
                coef_df = d.get("coef_df") if d else None
                st.markdown(
                    "**Inputs used:** Recent national cases, wastewater trend (prior 8–12 weeks of detection data), "
                    "kindergarten MMR coverage (national), and week of year (seasonality)."
                )
                if coef_df is not None and not coef_df.empty:
                    pos = coef_df[coef_df["coefficient"] > 0].sort_values("coefficient", ascending=False)
                    neg = coef_df[coef_df["coefficient"] < 0].sort_values("coefficient", ascending=True)
                    if not pos.empty:
                        st.markdown(
                            "**Top positive drivers** (push alarm up): "
                            + ", ".join(
                                [f"{r['feature']} ({r['coefficient']:.2f})" for _, r in pos.head(3).iterrows()]
                            )
                            + "."
                        )
                    if not neg.empty:
                        st.markdown(
                            "**Top negative drivers** (push alarm down): "
                            + ", ".join(
                                [f"{r['feature']} ({r['coefficient']:.2f})" for _, r in neg.head(3).iterrows()]
                            )
                            + "."
                        )
                    st.dataframe(coef_df, use_container_width=True, hide_index=True)
                st.markdown(
                    "**Plain language:** Risk increases when recent wastewater levels are higher, when kindergarten coverage is lower, "
                    "or when the time of year is typically associated with more cases. The model combines these into a single probability."
                )

                st.markdown("##### How is baseline risk/score calculated?")
                from risk import get_baseline_risk_components

                nndss = d.get("nndss") if d else None
                nndss = nndss if nndss is not None else pd.DataFrame()
                comp = get_baseline_risk_components(nndss)
                st.markdown(
                    "**Inputs used:** National **weekly** measles cases from NNDSS (pipeline aggregation) and the same "
                    "**Stage 2 recency-weighted baseline projection** used elsewhere in the dashboard."
                )
                st.markdown(
                    "**Baseline risk** compares the **projection baseline** (mean weekly cases implied by the Stage 2 model) "
                    "to a **reference** median of weekly cases from **earlier** NNDSS weeks (the latest 12 weeks are excluded "
                    "when there is enough history). The **ratio** (projection baseline ÷ reference median) sets the tier and 0–100 score."
                )
                st.markdown(
                    f"Projection baseline (mean weekly cases): **{comp.get('projection_baseline_weekly', '—')}**. "
                    f"Reference median (earlier weeks): **{comp.get('reference_median_weekly', '—')}**. "
                    f"Ratio: **{comp.get('ratio', '—')}**."
                    + (
                        f" Recent-week volatility (Stage 2): **{comp.get('recent_std')}**."
                        if comp.get("recent_std") is not None
                        else ""
                    )
                )
                if comp.get("formula"):
                    st.markdown(comp["formula"])
                st.markdown(
                    "**Plain language:** When the short-horizon baseline sits well above a typical pre-recent week, "
                    "the baseline score rises toward medium or high tier."
                )

        st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">AI report</p>', unsafe_allow_html=True)
        gen = st.button(
            "Generate AI briefing",
            type="primary",
            use_container_width=True,
            key="btn_multi_agent",
            disabled=not d,
        )
        st.caption("Multi-agent pipeline (OpenAI + Census). Requires `OPENAI_API_KEY` in `.env`.")

        if gen and d:
            st.session_state.multi_agent_error = None
            try:
                with st.spinner("Running briefing…"):
                    st.session_state.multi_agent_result = run_measles_multi_agent_pipeline(
                        d,
                        kg=d["kg"],
                        nndss=d["nndss"],
                        ww=d["ww"],
                    )
            except Exception as exc:
                logger.exception("multi_agent pipeline failed")
                st.session_state.multi_agent_error = str(exc)
                st.session_state.multi_agent_result = None
            st.rerun()

        if st.session_state.multi_agent_error:
            st.error(st.session_state.multi_agent_error)

        ma = st.session_state.multi_agent_result
        a2 = (ma or {}).get("agent2") or {}
        top3 = _sorted_top_states_for_ui(a2, limit=3, exclude=frozenset({"FL", "UT"}))

        if ma:
            st.caption(
                f"{_snapshot_tier_badge(ma)} · Nat avg /100k: {_national_avg_cpk_str(ma)} · "
                f"Forecast: {_forecast_cases_week_str(d)}"
            )
            if top3:
                inf, war = _callout_info_warning(top3)
                if inf:
                    st.info(inf)
                if war:
                    st.warning(war)
                for state in top3:
                    st.markdown(_risk_state_card_html(state), unsafe_allow_html=True)
                st.markdown('<div class="ai-section-divider"></div>', unsafe_allow_html=True)

            st.markdown('<p class="ai-subhead">What to watch</p>', unsafe_allow_html=True)
            st.markdown(
                """<ul class="watch-bullets">
<li>Wastewater-driven signals may precede case increases</li>
<li>States above national avg per-capita burden need monitoring</li>
<li>Low-confidence states may understate risk</li>
</ul>""",
                unsafe_allow_html=True,
            )

            st.markdown('<p class="ai-subhead">Low-risk baseline</p>', unsafe_allow_html=True)
            st.markdown(
                """<ul class="watch-bullets">
<li>Near-zero recent cases</li>
<li>Minimal wastewater signal</li>
<li>Stable conditions</li>
</ul>""",
                unsafe_allow_html=True,
            )

            st.markdown('<div class="ai-section-divider"></div>', unsafe_allow_html=True)
            with st.expander("Full briefing (expand)", expanded=False):
                st.markdown((ma.get("agent3") or {}).get("assistant_text") or "—")

        elif d:
            st.markdown(
                '<div class="card"><div class="ai-panel-inner">'
                "AI-generated outbreak insights will appear here after "
                "<strong>Generate AI briefing</strong>."
                "</div></div>",
                unsafe_allow_html=True,
            )

    with tab_analysis:
        ychoices = _ww_year_choices(d)
        st.markdown('<p class="section-title">Wastewater vs NNDSS filters</p>', unsafe_allow_html=True)
        fa, fb = st.columns(2)
        with fa:
            ym = st.selectbox("From year", ychoices, index=0, key="ww_year_min")
        with fb:
            yx = st.selectbox("To year", ychoices, index=0, key="ww_year_max")
        ww = d.get("ww") if d else None
        _, ww_val_range = compute_ww_detection_frequency(
            ww if ww is not None else pd.DataFrame(), year_min=None, year_max=None
        )
        weeks_min = ww_val_range.get("weeks_min")
        weeks_max = ww_val_range.get("weeks_max")
        wymin = int(weeks_min[0]) if weeks_min else None
        wymax = int(weeks_max[0]) if weeks_max else None
        if wymin is not None and wymax is not None:
            st.caption(
                f"Wastewater measles surveillance is available from **{wymin}** through **{wymax}**. "
                "Years before this period will not display wastewater data."
            )

        ctx = _ww_nndss_context(d, ym, yx)

        st.markdown('<p class="section-title">Wastewater vs NNDSS</p>', unsafe_allow_html=True)
        for w in ctx["warn"]:
            st.warning(w)
        mi = ctx["merged_inner"]
        if mi.empty and not ctx["merged_nndss_only"].empty:
            st.caption("Showing NNDSS cases only (no wastewater series in selected period).")
        ww_w = ctx["ww_weekly"]
        nnd_f = ctx["nndss_filtered"]
        if ctx.get("show_ww_only") and not ww_w.empty:
            st.info("NNDSS has no data in selected window.")
        if not ww_w.empty:
            latest = ww_w.sort_values(["year", "week"], ascending=[False, False]).iloc[0]
            pct = 100 * latest["detection_frequency"] if latest["total_sites"] else 0
            st.caption(
                f"Latest week in data: {pct:.0f}% of reporting wastewater sites detected measles RNA."
            )
        if not ctx["warn"] and mi.empty and ctx["merged_nndss_only"].empty and ww_w.empty and nnd_f.empty:
            st.caption("No data for selected year range. Try All for From/To year.")

        if not mi.empty:
            fig_ww = ww_vs_nndss_dual_axis(mi)
        else:
            mo = ctx["merged_nndss_only"]
            if not mo.empty:
                fig_ww = nndss_only_figure(mo)
            else:
                fig_ww = go.Figure(
                    layout_title_text="No chart — adjust year filters or wait for overlapping WW + NNDSS weeks",
                    height=520,
                )
        fig_ww.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_ww, use_container_width=True, key="ww_analysis")

        with st.expander("Wastewater data audit"):
            wv = ctx.get("ww_val") or {}
            st.markdown(f"**Detection rule:** {wv.get('detection_rule_used') or '—'}")
            st.markdown(f"**Rows before filters:** {wv.get('n_rows_raw', 0)}")
            st.markdown(f"**% passing QC:** {wv.get('pct_passing_qc', 0):.1f} %")
            st.markdown(f"**Unique sites:** {wv.get('n_unique_sites', 0)}")
            wmin, wmax = wv.get("weeks_min"), wv.get("weeks_max")
            st.markdown(f"**Wastewater starts:** {wmin} | **ends:** {wmax}")
            if not ww_w.empty:
                st.markdown(
                    f"**Wastewater year range:** {int(ww_w['year'].min())} – {int(ww_w['year'].max())}"
                )
            st.markdown(f"**pcr_target used:** {wv.get('pcr_target_used')}")
            missing = wv.get("missing_columns")
            if missing:
                st.warning(f"Required column(s) missing: {missing}")

        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        st.markdown('<p class="section-title">Kindergarten MMR coverage</p>', unsafe_allow_html=True)
        kg = d.get("kg") if d else None
        kg_year_opts: list[str] = []
        if kg is not None and not kg.empty:
            kg_full, year_options, _ = prepare_kg_years(kg)
            if year_options:
                kg_year_opts = [str(y) for y in year_options]
            else:
                kg_year_opts = ["All available"]
        kgy = st.selectbox(
            "Coverage year",
            kg_year_opts if kg_year_opts else ["—"],
            index=len(kg_year_opts) - 1 if kg_year_opts else 0,
            key="kg_year",
        )
        kx = _kg_context(d, kgy)
        kw = kx["kg_work"]
        pc = kx["pct_col"]
        sc = kx["state_col"]
        if kw is None or kw.empty or not pc:
            st.plotly_chart(
                go.Figure(layout_title_text="No kindergarten coverage data", height=520),
                use_container_width=True,
                key="kg_map",
            )
        else:
            kw = kw.copy()
            kw["coverage"] = pd.to_numeric(kw[pc], errors="coerce")
            fig_kg = kindergarten_coverage_map_figure(kw, sc, "coverage")
            fig_kg.update_layout(height=520, paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_kg, use_container_width=True, key="kg_map")

        if kw is not None and not kw.empty and pc and sc:
            kw_tbl = kw.copy()
            cov_num = (
                kw_tbl["coverage"]
                if "coverage" in kw_tbl.columns
                else pd.to_numeric(kw_tbl[pc], errors="coerce")
            )
            kw_tbl = kw_tbl.assign(_coverage_pct=cov_num)
            cov_agg = kw_tbl.groupby(sc, as_index=False)["_coverage_pct"].mean()
            cov_agg = cov_agg.rename(columns={"_coverage_pct": "coverage"})
            table_df = cov_agg.dropna(subset=["coverage"])[[sc, "coverage"]].rename(
                columns={sc: "State", "coverage": "Percent coverage"}
            )
            st.dataframe(table_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
