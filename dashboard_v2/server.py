"""Shiny server: reactive data, KPIs, Plotly outputs, AI hooks."""
from __future__ import annotations

import csv
import io
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from shiny import reactive, render, ui
from shiny import Session
from shinywidgets import render_plotly

# App directory on path for loaders, risk, utils
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
    historical_annual_figure,
    nndss_only_figure,
    nndss_weekly_figure,
    ww_vs_nndss_dual_axis,
)
from components.kpi_card import kpi_card, kpi_row
from components.map import kindergarten_coverage_map_figure, state_risk_map_figure
from kg_helpers import filter_kg_by_year, kg_state_pct_columns, prepare_kg_years
from model_runner import load_and_model
from utils.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger("server")


def server(input, output, session: Session):
    data_bundle = reactive.Value(None)
    loading = reactive.Value(True)
    load_error = reactive.Value(None)
    ww_nndss_ai = reactive.Value("")
    state_ai_text = reactive.Value("")
    forecast_ai = reactive.Value(None)

    def _run_load(use_cache: bool) -> None:
        loading.set(True)
        load_error.set(None)
        try:
            data_bundle.set(load_and_model(use_cache=use_cache, outbreak_percentile=95.0))
        except Exception as e:
            logger.exception("load_and_model failed")
            load_error.set(str(e))
            data_bundle.set(None)
        finally:
            loading.set(False)

    @reactive.Effect
    def _load_initial():
        _run_load(use_cache=True)

    @reactive.Effect
    @reactive.event(input.refresh_btn)
    def _load_refresh():
        _run_load(use_cache=False)

    @reactive.calc
    def bundle():
        return data_bundle.get()

    @reactive.calc
    def kpi_derived():
        """Highest risk state, latest NNDSS cases, week-over-week change."""
        d = bundle()
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

    # --- Dynamic select choices (wastewater years, kg year, NNDSS view) ---
    @reactive.Effect
    def _update_ww_selects():
        d = bundle()
        if not d:
            return
        nndss_agg = d.get("nndss_agg")
        ww = d.get("ww")
        years_avail: list[int] = []
        if nndss_agg is not None and not nndss_agg.empty:
            na = nndss_agg.copy()
            na["year"] = pd.to_numeric(na["year"], errors="coerce")
            na = na.dropna(subset=["year"])
            years_avail = sorted(na["year"].astype(int).unique().tolist())
        if ww is not None and not ww.empty and "year" in ww.columns:
            ww_y = sorted(pd.to_numeric(ww["year"], errors="coerce").dropna().astype(int).unique().tolist())
            years_avail = sorted(set(years_avail) | set(ww_y)) if years_avail else ww_y
        ch = {"": "All"}
        for y in years_avail:
            ch[str(y)] = str(y)
        ui.update_select("ww_year_min", choices=ch, session=session)
        ui.update_select("ww_year_max", choices=ch, session=session)

    @reactive.Effect
    def _update_kg_year():
        d = bundle()
        if not d:
            return
        kg = d.get("kg")
        if kg is None or kg.empty:
            ui.update_select("kg_year", choices={}, selected=None, session=session)
            return
        kg_full, year_options, _ = prepare_kg_years(kg)
        if not year_options:
            ui.update_select("kg_year", choices={"all": "All available"}, session=session)
            return
        ch = {str(y): str(y) for y in year_options}
        ui.update_select("kg_year", choices=ch, session=session)

    @reactive.Effect
    def _update_nndss_view():
        d = bundle()
        if not d:
            return
        nndss_agg = d.get("nndss_agg")
        ch = {"104": "Last 104 weeks", "all": "All available weeks"}
        if nndss_agg is not None and not nndss_agg.empty:
            nac = nndss_agg.copy()
            nac["year"] = pd.to_numeric(nac["year"], errors="coerce")
            years_in = sorted(nac["year"].dropna().astype(int).unique().tolist())
            for y in years_in:
                ch[str(y)] = str(y)
        ui.update_select("nndss_view", choices=ch, session=session)

    @reactive.Effect
    def _update_state_report_choices():
        d = bundle()
        if not d:
            return
        sr = d.get("state_risk_df")
        from utils.state_maps import STATE_TO_ABBR, state_to_abbr

        ch = {"": "— Select a state —"}
        if sr is not None and not sr.empty:
            abbr_to_name = {v: k for k, v in STATE_TO_ABBR.items()}
            seen: set[str] = set()
            for _state in sorted(sr["state"].astype(str).unique()):
                ab = state_to_abbr(_state)
                if ab and ab not in seen:
                    seen.add(ab)
                    name = abbr_to_name.get(ab, str(_state))
                    ch[name] = name
        ui.update_select("state_report_select", choices=ch, session=session)

    # --- Loading banner ---
    @render.ui
    def load_status_ui():
        if loading.get():
            return ui.div(
                {"class": "dashboard-card", "style": "display:flex;align-items:center;gap:0.75rem;"},
                ui.tags.span(
                    {"class": "spinner-border spinner-border-sm text-primary", "role": "status"},
                ),
                ui.tags.span("Loading data and fitting models…", {"style": "color:#64748b;"}),
            )
        err = load_error.get()
        if err:
            return ui.div(
                {"class": "alert alert-danger", "role": "alert"},
                "Load failed. Check logs and .env (e.g. SOCRATA_APP_TOKEN).",
            )
        d = bundle()
        if not d:
            return ui.div(
                {"class": "alert alert-warning"},
                "No data loaded.",
            )
        lines = []
        for source, status in (d.get("load_status") or {}).items():
            label = "temporarily unavailable" if status == "fail" else status
            lines.append(f"{source}: {label}")
        return ui.div(
            {"style": "font-size:0.8rem;color:#64748b;margin-bottom:0.75rem;"},
            ui.tags.strong("Data sources: "),
            " · ".join(lines) if lines else "—",
            ui.tags.br(),
            ui.tags.span(f"Data as of: {d.get('data_as_of') or 'N/A'}"),
        )

    @render.ui
    def kpi_cards_ui():
        if loading.get():
            return kpi_row(
                kpi_card("Outbreak alarm", "…", "Loading"),
                kpi_card("Highest risk state", "…", ""),
                kpi_card("Latest case count", "…", ""),
            )
        kd = kpi_derived()
        ap = kd["alarm"]
        sub_a = "Probability of exceeding threshold in next 4 weeks"
        card1 = kpi_card("Outbreak alarm", f"{ap:.0%}", sub_a)
        hs = kd["hi_state"]
        sc = kd["hi_score"]
        sub_h = f"Score: {sc:.0f}" if sc is not None else ""
        card2 = kpi_card("Highest risk state", hs, sub_h)
        lc = kd["latest_cases"]
        if lc is None:
            card3 = kpi_card("Latest case count", "—", "NNDSS weekly (national)")
        else:
            delta = kd["cases_delta_pct"]
            if delta is None or (isinstance(delta, float) and np.isinf(delta)):
                sub_c = "vs prior week: n/a"
            else:
                sub_c = f"vs prior week: {delta:+.1f}%"
            card3 = kpi_card("Latest case count", f"{lc:.0f}", sub_c)
        return kpi_row(card1, card2, card3)

    @render_plotly
    def overview_map_plot():
        d = bundle()
        if not d:
            return state_risk_map_figure(pd.DataFrame())
        sr = d.get("state_risk_df")
        if sr is None:
            return state_risk_map_figure(pd.DataFrame())
        return state_risk_map_figure(sr)

    @render_plotly
    def baseline_gauge_plot():
        d = bundle()
        v = float(d.get("baseline_val", 0)) if d else 0.0
        return baseline_gauge_figure(v)

    @render.ui
    def alarm_help_ui():
        d = bundle()
        coef_df = d.get("coef_df") if d else None
        parts = [
            ui.p(
                "**Inputs used:** Recent national cases, wastewater trend (prior 8–12 weeks of detection data), "
                "kindergarten MMR coverage (national), and week of year (seasonality)."
            ),
        ]
        if coef_df is not None and not coef_df.empty:
            pos = coef_df[coef_df["coefficient"] > 0].sort_values("coefficient", ascending=False)
            neg = coef_df[coef_df["coefficient"] < 0].sort_values("coefficient", ascending=True)
            if not pos.empty:
                parts.append(
                    ui.p(
                        "**Top positive drivers** (push alarm up): "
                        + ", ".join(
                            [f"{r['feature']} ({r['coefficient']:.2f})" for _, r in pos.head(3).iterrows()]
                        )
                        + "."
                    )
                )
            if not neg.empty:
                parts.append(
                    ui.p(
                        "**Top negative drivers** (push alarm down): "
                        + ", ".join(
                            [f"{r['feature']} ({r['coefficient']:.2f})" for _, r in neg.head(3).iterrows()]
                        )
                        + "."
                    )
                )
            parts.append(ui.p("Coefficients (positive = increases probability, negative = decreases):"))
            parts.append(ui.HTML(coef_df.to_html(classes="table table-sm", index=False)))
        parts.append(
            ui.p(
                "**Plain language:** Risk increases when recent wastewater levels are higher, when kindergarten coverage is lower, "
                "or when the time of year is typically associated with more cases. The model combines these into a single probability."
            )
        )
        return ui.div(*parts)

    @render.ui
    def baseline_help_ui():
        from risk import get_baseline_risk_components

        d = bundle()
        hist = d.get("hist") if d else None
        nndss = d.get("nndss") if d else None
        hist = hist if hist is not None else pd.DataFrame()
        nndss = nndss if nndss is not None else pd.DataFrame()
        comp = get_baseline_risk_components(hist, nndss)
        parts = [
            ui.p("**Inputs used:** Historical national annual cases (CSV) and recent NNDSS weekly national cases."),
            ui.p(
                "**Baseline risk** compares **recent** measles case levels to **historical** (annual national cases from the CSV). "
                "We take the **average of the last 5 years** and the **overall average**; the **ratio** (recent ÷ overall) determines the tier and score."
            ),
            ui.p(
                f"Recent 5-year average: **{comp.get('recent_5yr_avg', '—')}**. "
                f"Overall average: **{comp.get('overall_avg', '—')}**. Ratio: **{comp.get('ratio', '—')}**."
            ),
        ]
        if comp.get("formula"):
            parts.append(ui.p(comp["formula"]))
        parts.append(
            ui.p(
                "**Plain language:** Higher recent case levels relative to history mean a higher baseline score and medium or high tier."
            )
        )
        return ui.div(*parts)

    @render.download(filename="overview_summary.csv")
    def dl_overview():
        d = bundle()
        buf = io.StringIO()
        w = csv.writer(buf)
        w.writerow(["metric", "value"])
        if d:
            w.writerow(["alarm_probability", d.get("alarm_prob", 0.5)])
            w.writerow(["baseline_tier", d.get("baseline_tier", "")])
            w.writerow(["baseline_score", d.get("baseline_val", 0)])
            w.writerow(["data_as_of", d.get("data_as_of", "")])
        yield buf.getvalue()

    # --- Analysis: WW vs NNDSS ---
    @reactive.calc
    def ww_nndss_context():
        from risk import compute_ww_detection_frequency, validate_ww_nndss_audit

        d = bundle()
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
        ym = input.ww_year_min()
        yx = input.ww_year_max()
        year_min = int(ym) if ym not in (None, "") else None
        year_max = int(yx) if yx not in (None, "") else None
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

    @render.ui
    def ww_info_ui():
        from risk import compute_ww_detection_frequency

        ww = bundle().get("ww") if bundle() else None
        _, ww_val_range = compute_ww_detection_frequency(
            ww if ww is not None else pd.DataFrame(), year_min=None, year_max=None
        )
        weeks_min = ww_val_range.get("weeks_min")
        weeks_max = ww_val_range.get("weeks_max")
        wymin = int(weeks_min[0]) if weeks_min else None
        wymax = int(weeks_max[0]) if weeks_max else None
        if wymin is not None and wymax is not None:
            return ui.p(
                f"Wastewater measles surveillance is available from **{wymin}** through **{wymax}**. "
                "Years before this period will not display wastewater data."
            )
        return ui.p("")

    @render.ui
    def ww_chart_ui():
        ctx = ww_nndss_context()
        parts = []
        for w in ctx["warn"]:
            parts.append(ui.div({"class": "alert alert-warning py-2"}, w))
        mi = ctx["merged_inner"]
        if not mi.empty:
            pass
        elif not ctx["merged_nndss_only"].empty:
            parts.append(ui.p({"style": "font-size:0.85rem;color:#64748b;"}, "Showing NNDSS cases only (no wastewater series in selected period)."))
        ww_w = ctx["ww_weekly"]
        nnd_f = ctx["nndss_filtered"]
        if ctx.get("show_ww_only") and not ww_w.empty:
            parts.append(ui.div({"class": "alert alert-info"}, "NNDSS has no data in selected window."))
        if not ww_w.empty:
            latest = ww_w.sort_values(["year", "week"], ascending=[False, False]).iloc[0]
            pct = 100 * latest["detection_frequency"] if latest["total_sites"] else 0
            parts.append(
                ui.p(
                    {"style": "font-size:0.88rem;"},
                    f"Latest week in data: {pct:.0f}% of reporting wastewater sites detected measles RNA.",
                )
            )
        if not ctx["warn"] and mi.empty and ctx["merged_nndss_only"].empty and ww_w.empty and nnd_f.empty:
            parts.append(ui.p("No data for selected year range. Try All for From/To year."))
        return ui.div(*parts) if parts else ui.p("")

    @render_plotly
    def ww_analysis_plot():
        from plotly import graph_objects as go

        ctx = ww_nndss_context()
        mi = ctx["merged_inner"]
        if not mi.empty:
            return ww_vs_nndss_dual_axis(mi)
        mo = ctx["merged_nndss_only"]
        if not mo.empty:
            return nndss_only_figure(mo)
        return go.Figure(
            layout_title_text="No chart — adjust year filters or wait for overlapping WW + NNDSS weeks",
            height=400,
        )

    @render.ui
    def ww_audit_ui():
        ctx = ww_nndss_context()
        wv = ctx.get("ww_val") or {}
        parts = [
            ui.p(f"**Detection rule:** {wv.get('detection_rule_used') or '—'}"),
            ui.p(f"**Rows before filters:** {wv.get('n_rows_raw', 0)}"),
            ui.p(f"**% passing QC:** {wv.get('pct_passing_qc', 0):.1f} %"),
            ui.p(f"**Unique sites:** {wv.get('n_unique_sites', 0)}"),
        ]
        wmin, wmax = wv.get("weeks_min"), wv.get("weeks_max")
        parts.append(ui.p(f"**Wastewater starts:** {wmin} | **ends:** {wmax}"))
        ww_w = ctx["ww_weekly"]
        if not ww_w.empty:
            parts.append(
                ui.p(
                    f"**Wastewater year range:** {int(ww_w['year'].min())} – {int(ww_w['year'].max())}"
                )
            )
        parts.append(ui.p(f"**pcr_target used:** {wv.get('pcr_target_used')}"))
        missing = wv.get("missing_columns")
        if missing:
            parts.insert(0, ui.div({"class": "alert alert-warning"}, f"Required column(s) missing: {missing}"))
        return ui.div(*parts)

    # --- Kindergarten map (Analysis tab) ---
    @reactive.calc
    def kg_context():
        d = bundle()
        if not d or d.get("kg") is None or d.get("kg").empty:
            return {"kg_work": pd.DataFrame(), "state_col": "", "pct_col": None, "year_label": ""}
        kg = d["kg"]
        kg_full, year_options, _ = prepare_kg_years(kg)
        sel = input.kg_year()
        if year_options:
            if sel in (None, "", "all"):
                y = year_options[-1]
            else:
                try:
                    y = int(sel)
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

    @render_plotly
    def kg_map_plot():
        kx = kg_context()
        kw = kx["kg_work"]
        pc = kx["pct_col"]
        sc = kx["state_col"]
        if kw is None or kw.empty or not pc:
            from plotly import graph_objects as go

            return go.Figure(layout_title_text="No kindergarten coverage data")
        kw = kw.copy()
        kw["coverage"] = pd.to_numeric(kw[pc], errors="coerce")
        return kindergarten_coverage_map_figure(kw, sc, "coverage")

    @render.ui
    def kg_table_ui():
        kx = kg_context()
        kw = kx["kg_work"]
        pc = kx["pct_col"]
        sc = kx["state_col"]
        if kw is None or kw.empty or not pc:
            return ui.p("No coverage rows for current filters.")
        cov_agg = kw.groupby(sc, as_index=False)[pc].mean()
        cov_agg = cov_agg.rename(columns={pc: "coverage"})
        table_df = cov_agg.dropna(subset=["coverage"])[[sc, "coverage"]].rename(
            columns={sc: "State", "coverage": "Percent coverage"}
        )
        return ui.HTML(
            table_df.to_html(classes="table table-sm table-striped", index=False, float_format=lambda x: f"{x:.1f}")
        )

    @render.ui
    def kg_audit_ui():
        d = bundle()
        if not d or d.get("kg") is None:
            return ui.p("No kindergarten data.")
        kg = d["kg"]
        kg_full, _yo, year_source = prepare_kg_years(kg)
        parts = [ui.p(f"**Year source:** {year_source or 'none'}")]
        if "_year_derived" in kg_full.columns:
            parts.append(
                ui.p(
                    f"**Derived year logic:** {kg_full['_year_source'].iloc[0] if kg_full['_year_source'].notna().any() else '—'}"
                )
            )
            ys = (
                sorted(pd.to_numeric(kg_full["_year_derived"], errors="coerce").dropna().astype(int).unique().tolist())
                if kg_full["_year_derived"].notna().any()
                else []
            )
            parts.append(ui.p(f"**Available years:** {ys}"))
        return ui.div(*parts)

    # --- WW vs NNDSS AI ---
    @reactive.Effect
    @reactive.event(input.btn_ww_nndss_report)
    def _ww_nndss_ai():
        from risk import compute_ww_detection_frequency

        ctx = ww_nndss_context()
        d = bundle()
        ww_nndss_summary_parts = []
        ww_w = ctx["ww_weekly"]
        nnd_f = ctx["nndss_filtered"]
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
        wv = ctx.get("ww_val") or {}
        wmin, wmax = wv.get("weeks_min"), wv.get("weeks_max")
        ww_nndss_summary_parts.append(f"Wastewater data availability: weeks {wmin or '?'} to {wmax or '?'}.")

        text = " ".join(ww_nndss_summary_parts) if ww_nndss_summary_parts else "No summary data available."
        data_as_of = (d or {}).get("data_as_of", "")
        try:
            from ollama_client import get_ollama_ww_nndss_report

            report = get_ollama_ww_nndss_report(text, data_as_of)
            ww_nndss_ai.set(report or "Could not generate (add OLLAMA_API_KEY to .env or check dashboard.log).")
        except Exception as e:
            ww_nndss_ai.set(f"Error: {str(e)[:200]}")

    @render.ui
    def ww_nndss_ai_ui():
        t = ww_nndss_ai.get()
        if not t:
            return ui.p({"style": "color:#94a3b8;font-size:0.88rem;"}, "Click **Generate AI report** for a narrative summary.")
        s = (
            str(t)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n\n", "<br/><br/>")
            .replace("\n", "<br/>")
        )
        return ui.div(ui.HTML(f'<div class="ai-panel-inner">{s}</div>'))

    # --- Historical tab ---
    @render_plotly
    def hist_annual_plot():
        d = bundle()
        h = d.get("hist") if d else None
        return historical_annual_figure(h if h is not None else pd.DataFrame())

    @reactive.calc
    def nndss_agg_view():
        d = bundle()
        if not d:
            return pd.DataFrame()
        nndss_agg = d.get("nndss_agg")
        if nndss_agg is None or nndss_agg.empty:
            return pd.DataFrame()
        nndss_agg_copy = nndss_agg.copy()
        nndss_agg_copy["year"] = pd.to_numeric(nndss_agg_copy["year"], errors="coerce")
        choice = input.nndss_view()
        if choice == "104":
            return nndss_agg_copy.tail(104).copy()
        if choice == "all":
            return nndss_agg_copy.copy()
        try:
            y = int(choice)
            sub = nndss_agg_copy[nndss_agg_copy["year"].astype(int) == y].copy()
            return sub.sort_values(["year", "week"]).reset_index(drop=True)
        except (ValueError, TypeError):
            return nndss_agg_copy.tail(104).copy()

    @render_plotly
    def nndss_weekly_plot():
        agg = nndss_agg_view()
        return nndss_weekly_figure(agg)

    @render.ui
    def nndss_recent_table_ui():
        agg = nndss_agg_view()
        if agg is None or agg.empty:
            return ui.p("No NNDSS weekly data.")
        recent5 = agg[["year", "week", "cases"]].tail(5).copy()
        recent5["Year-Week"] = (
            recent5["year"].astype(int).astype(str)
            + "-W"
            + recent5["week"].astype(int).apply(lambda x: str(x).zfill(2))
        )
        show = recent5[["Year-Week", "cases"]].rename(columns={"cases": "Cases"})
        return ui.HTML(show.to_html(classes="table table-sm", index=False))

    @render.ui
    def nndss_audit_ui():
        d = bundle()
        if not d:
            return ui.p("No data.")
        au = d.get("nndss_audit") or {}
        na = d.get("nndss_agg")
        parts = [
            ui.p(
                f"**Case column used:** {au.get('case_column_used')} | **Year range:** {au.get('year_min')} – {au.get('year_max')} | **Most recent:** {au.get('year_week_max')}"
            ),
            ui.p(
                f"**National rows before agg:** {au.get('n_national_rows')} | **year_max before agg:** {au.get('year_max_before_agg')} | **after:** {au.get('year_max_after_agg')}"
            ),
        ]
        if au.get("candidate_case_columns"):
            parts.append(ui.p(f"**Candidate case columns:** {au.get('candidate_case_columns')}"))
        if na is None or na.empty:
            parts.append(ui.p("No NNDSS national weekly data."))
        return ui.div(*parts)

    # --- State & risk tab ---
    @render_plotly
    def state_tab_map_plot():
        d = bundle()
        sr = d.get("state_risk_df") if d else None
        if sr is None:
            return state_risk_map_figure(pd.DataFrame())
        return state_risk_map_figure(sr)

    @render.ui
    def state_table_ui():
        d = bundle()
        sr = d.get("state_risk_df") if d else None
        if sr is None or sr.empty:
            return ui.div({"class": "alert alert-info"}, "No state risk table. Ensure kindergarten data loaded.")
        display_cols = ["state", "cases_recent", "ww_recent", "wastewater_coverage", "total_risk"]
        if not all(c in sr.columns for c in display_cols):
            display_cols = ["state", "cases_recent", "ww_recent", "total_risk"]
        if all(c in sr.columns for c in display_cols):

            def _ww_display(val):
                if val is None or (isinstance(val, float) and pd.isna(val)):
                    return "No coverage"
                if isinstance(val, (int, float)):
                    return str(round(val / 100_000, 2))
                return str(val)

            tbl = sr[display_cols].copy()
            tbl["Wastewater signal (per 100k)"] = tbl["ww_recent"].apply(_ww_display)
            if "wastewater_coverage" in tbl.columns:
                tbl["Wastewater coverage"] = tbl["wastewater_coverage"].map({True: "Yes", False: "No"})
            show_cols = ["state", "cases_recent", "Wastewater signal (per 100k)"] + (
                ["Wastewater coverage"] if "wastewater_coverage" in tbl.columns else []
            ) + ["total_risk"]
            tbl = tbl[show_cols].rename(
                columns={
                    "state": "State",
                    "cases_recent": "Recent cases",
                    "total_risk": "Final score",
                }
            )
        else:
            tbl = sr[["state", "risk_tier", "risk_score"]].copy().rename(
                columns={"state": "State", "risk_tier": "Risk tier", "risk_score": "Risk score"}
            )
        return ui.HTML(tbl.to_html(classes="table table-sm table-striped", index=False))

    @render.ui
    def state_risk_help_ui():
        d = bundle()
        sr = d.get("state_risk_df") if d else None
        parts = [
            ui.p(
                "**State risk** combines: **Coverage risk** (0–50 pts) = max(0, 95 − coverage%) × 2; "
                "**Case activity** (0–30 pts) = percentile of last 4-week cases vs other states; "
                "**Wastewater activity** (0–20 pts) = percentile of last 4-week wastewater signal by state."
            ),
            ui.p(
                "**Risk tier:** **High** = total ≥ 70; **Medium** = 40–69; **Low** < 40. "
                "Thresholds are for situational awareness only."
            ),
        ]
        if sr is not None and not sr.empty:
            has_ww_pts = "wastewater_points" in sr.columns and (sr["wastewater_points"].fillna(0) != 0).any()
            if not has_ww_pts:
                parts.append(
                    ui.p(
                        "States without wastewater coverage get wastewater_points=0 and total score capped at 80."
                    )
                )
            br_cols = ["coverage_points", "case_points", "wastewater_points", "total_risk"]
            if all(c in sr.columns for c in br_cols):
                ex = sr.iloc[0]
                breakdown = pd.DataFrame(
                    [
                        ["Coverage", ex.get("coverage_points", 0)],
                        ["Cases", ex.get("case_points", 0)],
                        ["Wastewater", ex.get("wastewater_points", 0)],
                        ["Total", ex.get("total_risk", 0)],
                    ],
                    columns=["Component", "Points"],
                )
                parts.append(ui.HTML(breakdown.to_html(classes="table table-sm", index=False)))
        return ui.div(*parts)

    @reactive.Effect
    @reactive.event(input.btn_state_report)
    def _state_ai():
        d = bundle()
        sr = d.get("state_risk_df") if d else None
        sel = input.state_report_select()
        if not d or sr is None or sr.empty or not sel or sel.startswith("—"):
            state_ai_text.set("")
            return
        from utils.state_maps import state_to_abbr

        selected_abbr = state_to_abbr(sel) or sel
        row = sr[sr["state"].astype(str).apply(lambda x: state_to_abbr(str(x)) == selected_abbr or str(x) == sel)].iloc[0]
        cov_pct = None
        kg = d.get("kg")
        if kg is not None and not kg.empty:
            sc = next((c for c in ["state", "State", "jurisdiction", "geography"] if c in kg.columns), None)
            pc = next((c for c in kg.columns if "pct" in c.lower() or "coverage" in c.lower()), None)
            if sc and pc:

                def _kg_state_match(val):
                    v = str(val).strip()
                    return v == sel or state_to_abbr(v) == selected_abbr

                sub = kg[kg[sc].astype(str).apply(_kg_state_match)]
                if not sub.empty:
                    cov_pct = pd.to_numeric(sub[pc], errors="coerce").mean()
        try:
            from ollama_client import get_ollama_state_report

            report = get_ollama_state_report(
                sel,
                str(row["risk_tier"]),
                float(row["risk_score"]),
                coverage_pct=cov_pct,
                data_as_of=d.get("data_as_of", ""),
            )
            state_ai_text.set(report or "Could not generate. Add OLLAMA_API_KEY or check dashboard.log.")
        except Exception as e:
            state_ai_text.set(str(e)[:300])

    @render.ui
    def state_report_ai_ui():
        t = state_ai_text.get()
        if not t:
            return ui.p({"style": "color:#94a3b8;font-size:0.85rem;"}, "Select a state and click Generate.")
        s = (
            str(t)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n\n", "<br/><br/>")
            .replace("\n", "<br/>")
        )
        return ui.div(ui.HTML(f'<div class="ai-panel-inner">{s}</div>'))

    # --- Forecast tab ---
    @render.ui
    def forecast_table_ui():
        d = bundle()
        sr = d.get("state_risk_df") if d else None
        if sr is None or sr.empty:
            return ui.div({"class": "alert alert-info"}, "No state risk data yet.")

        def _outlook(tier):
            if tier == "high":
                return "High"
            if tier == "medium":
                return "Watch"
            return "Low"

        def _drivers(row):
            parts = []
            if "coverage_points" in row and row.get("coverage_points", 0) > 10:
                parts.append("low coverage")
            if "case_points" in row and row.get("case_points", 0) > 10:
                parts.append("recent cases")
            if "wastewater_points" in row and row.get("wastewater_points", 0) > 5:
                parts.append("wastewater signal")
            if not parts:
                return "Coverage and case levels in normal range."
            return "Mainly: " + ", ".join(parts) + "."

        forecast_table = sr.copy()
        forecast_table["Outlook"] = forecast_table["risk_tier"].apply(_outlook)
        forecast_table["What's driving it"] = forecast_table.apply(_drivers, axis=1)
        out = forecast_table[["state", "Outlook", "What's driving it"]].rename(columns={"state": "State"})
        return ui.HTML(out.to_html(classes="table table-sm", index=False))

    @render.ui
    def forecast_national_ui():
        d = bundle()
        fc = d.get("forecast_df") if d else None
        if fc is None or fc.empty:
            return ui.p("")
        mean_fc = float(fc["forecast"].mean())
        line = (
            f"Expected national cases in next 4 weeks: about {mean_fc * 4:.0f} "
            f"(range {fc['forecast'].min():.0f}–{fc['forecast'].max():.0f} per week)."
        )
        return ui.p({"style": "font-size:0.88rem;"}, ui.markdown(f"**{line}**"))

    @reactive.Effect
    @reactive.event(input.forecast_ai_btn)
    def _forecast_ai_gen():
        d = bundle()
        sr = d.get("state_risk_df") if d else None
        if not d or sr is None or sr.empty:
            forecast_ai.set("No data for AI summary.")
            return

        def _outlook(tier):
            if tier == "high":
                return "High"
            if tier == "medium":
                return "Watch"
            return "Low"

        def _drivers(row):
            parts = []
            if "coverage_points" in row and row.get("coverage_points", 0) > 10:
                parts.append("low coverage")
            if "case_points" in row and row.get("case_points", 0) > 10:
                parts.append("recent cases")
            if "wastewater_points" in row and row.get("wastewater_points", 0) > 5:
                parts.append("wastewater signal")
            if not parts:
                return "Coverage and case levels in normal range."
            return "Mainly: " + ", ".join(parts) + "."

        ft = sr.copy()
        ft["Outlook"] = ft["risk_tier"].apply(_outlook)
        ft["What's driving it"] = ft.apply(_drivers, axis=1)
        outlook_counts = ft["Outlook"].value_counts()
        summary = "State outlooks: " + "; ".join([f"{k}: {v} states" for k, v in outlook_counts.items()])
        summary += ". Sample drivers: " + "; ".join(ft["What's driving it"].head(5).tolist())
        rank_col = "risk_score" if "risk_score" in ft.columns else "total_risk"
        if rank_col in ft.columns:
            top_states = ft.nlargest(10, rank_col)
            hotspots_list = [f"{row['state']}: {float(row[rank_col]):.0f}" for _, row in top_states.iterrows()]
            summary += " Hotspots: " + "; ".join(hotspots_list) + "."
        national_line = ""
        fc = d.get("forecast_df")
        if fc is not None and not fc.empty:
            mean_fc = float(fc["forecast"].mean())
            national_line = (
                f"Expected national cases in next 4 weeks: about {mean_fc * 4:.0f} "
                f"(range {fc['forecast'].min():.0f}–{fc['forecast'].max():.0f} per week)."
            )
        try:
            from ollama_client import get_ollama_forecast_interpretation

            report = get_ollama_forecast_interpretation(summary, national_line, d.get("data_as_of", ""))
            forecast_ai.set(report or "Could not generate.")
        except Exception as e:
            forecast_ai.set(str(e)[:400])

    @reactive.Effect
    @reactive.event(input.forecast_ai_regen)
    def _forecast_ai_clear():
        forecast_ai.set(None)

    @render.ui
    def forecast_ai_ui():
        t = forecast_ai.get()
        if not t:
            return ui.p({"style": "color:#94a3b8;font-size:0.85rem;"}, "Click **Generate AI interpretation** (requires OLLAMA_API_KEY).")
        s = (
            str(t)
            .replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n\n", "<br/><br/>")
            .replace("\n", "<br/>")
        )
        return ui.div(ui.HTML(f'<div class="ai-panel-inner">{s}</div>'))
