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
    if "ww_nndss_ai" not in st.session_state:
        st.session_state.ww_nndss_ai = ""

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
        col_map, col_ai = st.columns([2, 1], gap="large")

        with col_map:
            st.markdown('<p class="section-title">State risk map</p>', unsafe_allow_html=True)
            if not d:
                fig_m = state_risk_map_figure(pd.DataFrame())
            else:
                sr = d.get("state_risk_df")
                fig_m = state_risk_map_figure(sr if sr is not None else pd.DataFrame())
            fig_m.update_layout(
                height=480,
                margin=dict(t=40, b=24, l=0, r=0),
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_m, use_container_width=True, key="overview_map")

        with col_ai:
            st.markdown('<p class="section-title">AI report</p>', unsafe_allow_html=True)
            st.markdown(
                """<div class="card"><div class="ai-panel-inner">AI-generated outbreak insights will appear here.</div></div>""",
                unsafe_allow_html=True,
            )

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
        c_left, c_right = st.columns(2, gap="large")

        with c_left:
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
                        height=400,
                    )
            fig_ww.update_layout(paper_bgcolor="rgba(0,0,0,0)")
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

            st.markdown('<p class="section-title">AI Reporter: Wastewater vs NNDSS</p>', unsafe_allow_html=True)
            if st.button("Generate AI report", key="btn_ww_nndss_report", type="primary"):
                ww_nndss_summary_parts = []
                if not ww_w.empty:
                    last8_ww = ww_w.sort_values(["year", "week"], ascending=[False, False]).head(8)
                    if not last8_ww.empty:
                        freq = last8_ww["detection_frequency"].values
                        mn, mx = float(freq.min()), float(freq.max())
                        last_val = float(freq[0])
                        direction_ww = (
                            "up"
                            if len(freq) >= 2 and freq[0] > freq[-1]
                            else ("down" if len(freq) >= 2 and freq[0] < freq[-1] else "flat")
                        )
                        ww_nndss_summary_parts.append(
                            f"Wastewater detection_frequency (last 8 weeks): min={mn:.2f}, max={mx:.2f}, latest={last_val:.2f}; trend {direction_ww}."
                        )
                if not nnd_f.empty:
                    last8_nd = nnd_f.sort_values(["year", "week"], ascending=[False, False]).head(8)
                    if not last8_nd.empty:
                        cases = last8_nd["cases"].values
                        mn, mx = int(cases.min()), int(cases.max())
                        last_c = int(cases[0])
                        direction_nd = (
                            "up"
                            if len(cases) >= 2 and cases[0] > cases[-1]
                            else ("down" if len(cases) >= 2 and cases[0] < cases[-1] else "flat")
                        )
                        ww_nndss_summary_parts.append(
                            f"NNDSS cases (last 8 weeks): min={mn}, max={mx}, latest={last_c}; trend {direction_nd}."
                        )
                wvv = ctx.get("ww_val") or {}
                wmin2, wmax2 = wvv.get("weeks_min"), wvv.get("weeks_max")
                ww_nndss_summary_parts.append(f"Wastewater data availability: weeks {wmin2 or '?'} to {wmax2 or '?'}.")

                text = " ".join(ww_nndss_summary_parts) if ww_nndss_summary_parts else "No summary data available."
                data_as_of = (d or {}).get("data_as_of", "")
                try:
                    from ollama_client import get_ollama_ww_nndss_report

                    report = get_ollama_ww_nndss_report(text, data_as_of)
                    st.session_state.ww_nndss_ai = report or "Could not generate (add OLLAMA_API_KEY to .env or check dashboard.log)."
                except Exception as e:
                    st.session_state.ww_nndss_ai = f"Error: {str(e)[:200]}"
                st.rerun()

            t_ai = st.session_state.ww_nndss_ai
            if not t_ai:
                st.caption("Click **Generate AI report** for a narrative summary.")
            else:
                _safe = (
                    html.escape(str(t_ai))
                    .replace("\n\n", "<br/><br/>")
                    .replace("\n", "<br/>")
                )
                st.markdown(f'<div class="ai-panel-inner">{_safe}</div>', unsafe_allow_html=True)
            with st.expander("AI Reporter: understanding wastewater detection"):
                st.markdown(
                    "- **Detection frequency** = share of reporting wastewater sites that had measurable measles RNA in that week. It is *not* a count of patients or cases.\n"
                    "- **Why it matters:** When more sites detect virus, community circulation may be higher; it can sometimes appear in wastewater before confirmed cases are reported.\n"
                    "- **How to interpret:** An *increase* in detection frequency suggests more sites seeing virus; a *decrease* may mean less circulation or fewer sites reporting.\n"
                    "- **Lag correlation:** A positive correlation at lag K means detection frequency K weeks ago lines up with cases this week. Correlation does not prove causation; reporting and lab delays affect timing."
                )
            st.caption("Requires **OLLAMA_API_KEY** in `.env` for AI text. Compares wastewater detection trend vs NNDSS cases.")

        with c_right:
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
                    go.Figure(layout_title_text="No kindergarten coverage data"),
                    use_container_width=True,
                    key="kg_map",
                )
            else:
                kw = kw.copy()
                kw["coverage"] = pd.to_numeric(kw[pc], errors="coerce")
                fig_kg = kindergarten_coverage_map_figure(kw, sc, "coverage")
                fig_kg.update_layout(height=460, paper_bgcolor="rgba(0,0,0,0)")
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

            with st.expander("Kindergarten data audit"):
                if not d or d.get("kg") is None:
                    st.write("No kindergarten data.")
                else:
                    kg2 = d["kg"]
                    kg_full, _yo, year_source = prepare_kg_years(kg2)
                    st.markdown(f"**Year source:** {year_source or 'none'}")
                    if "_year_derived" in kg_full.columns:
                        st.markdown(
                            f"**Derived year logic:** {kg_full['_year_source'].iloc[0] if kg_full['_year_source'].notna().any() else '—'}"
                        )
                        ys = (
                            sorted(pd.to_numeric(kg_full["_year_derived"], errors="coerce").dropna().astype(int).unique().tolist())
                            if kg_full["_year_derived"].notna().any()
                            else []
                        )
                        st.markdown(f"**Available years:** {ys}")


if __name__ == "__main__":
    main()
