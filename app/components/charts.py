"""Plotly figures: wastewater vs NNDSS, historical series, overview gauge."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def baseline_gauge_figure(baseline_val: float) -> go.Figure:
    """Baseline risk meter (0–100), same semantics as Streamlit Overview."""
    v = float(baseline_val) if baseline_val is not None else 0.0
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=v,
            number={"suffix": ""},
            title={"text": "Baseline risk meter"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "darkgray"},
                "steps": [
                    {"range": [0, 33], "color": "lightgreen"},
                    {"range": [33, 67], "color": "lightyellow"},
                    {"range": [67, 100], "color": "lightcoral"},
                ],
                "threshold": {"line": {"color": "black", "width": 2}, "value": v},
            },
        )
    )
    fig.update_layout(height=200, margin=dict(l=40, r=40, t=50, b=30))
    return fig


def historical_annual_figure(hist: pd.DataFrame) -> go.Figure:
    if hist is None or hist.empty:
        return go.Figure(layout_title_text="No historical data")
    case_col = "Measles Cases" if "Measles Cases" in hist.columns else hist.columns[1]
    xcol = hist.columns[0]
    fig = px.line(hist, x=xcol, y=case_col, title="National annual measles cases (historical CSV)")
    fig.update_layout(height=380, font=dict(size=12, color="#334155"))
    return fig


def nndss_weekly_figure(agg: pd.DataFrame) -> go.Figure:
    if agg is None or agg.empty:
        return go.Figure(layout_title_text="No NNDSS weekly data")
    d = agg.copy()
    d["year"] = pd.to_numeric(d["year"], errors="coerce")
    d["year_week"] = (
        d["year"].astype(int).astype(str)
        + "-W"
        + d["week"].astype(int).apply(lambda x: str(x).zfill(2))
    )
    fig = px.line(d, x="year_week", y="cases", title="National weekly cases (NNDSS)")
    fig.update_layout(
        xaxis_title="Week ending",
        yaxis_title="Reported measles cases (weekly)",
        height=400,
        font=dict(size=12, color="#334155"),
    )
    return fig


def ww_vs_nndss_dual_axis(merged_inner: pd.DataFrame) -> go.Figure:
    """Dual-axis: detection frequency vs NNDSS cases (same as Streamlit)."""
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged_inner["year_week"],
            y=merged_inner["detection_frequency"].values,
            name="Wastewater detection rate",
            line=dict(color="steelblue", width=3),
            mode="lines+markers",
            marker=dict(size=4),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=merged_inner["year_week"],
            y=merged_inner["cases"].values,
            name="Reported measles cases",
            yaxis="y2",
            line=dict(color="darkorange", width=2),
            mode="lines+markers",
            marker=dict(size=4),
            opacity=0.8,
        )
    )
    fig.update_layout(
        height=480,
        font=dict(size=12, color="#334155"),
        yaxis=dict(title="Detection frequency (share of sites)", tickformat=".0%"),
        yaxis2=dict(overlaying="y", side="right", title="Reported measles cases (weekly)"),
        title="Wastewater detection frequency vs NNDSS cases",
        xaxis=dict(title="Week", type="category", tickangle=-45),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=60, b=80),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def nndss_only_figure(merged: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=merged["year_week"],
            y=merged["cases"],
            name="Reported cases (NNDSS)",
            mode="lines+markers",
        )
    )
    fig.update_layout(title="Reported cases by week (no wastewater detection data in selected period)")
    return fig
