"""Plotly choropleth maps (state risk, kindergarten coverage)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def state_risk_map_figure(
    sr: pd.DataFrame,
    enrichment_by_abbr: dict | None = None,
) -> go.Figure:
    """
    US choropleth: color by risk_score, blue gradient.
    Hover: state + score; optional enrichment adds narrative + confidence (Agent 2 JSON).
    ``enrichment_by_abbr``: maps USPS code (e.g. ``\"CO\"``) to dict with
    ``risk_narrative``, ``confidence_label``, optional ``risk_score`` display.
    """
    from utils.state_maps import state_to_abbr

    if sr is None or sr.empty:
        fig = go.Figure()
        fig.update_layout(
            title="State risk (no data)",
            height=420,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        return fig

    d = sr.copy()
    d["abbr"] = d["state"].apply(state_to_abbr)
    d = d.dropna(subset=["abbr"])
    if d.empty:
        fig = go.Figure()
        fig.update_layout(title="State risk (no mappable states)", height=420)
        return fig

    score_col = "risk_score" if "risk_score" in d.columns else "total_risk"
    d["_hover_score"] = (
        pd.to_numeric(d[score_col], errors="coerce").round().fillna(0).astype(int)
    )

    fig = px.choropleth(
        d,
        locations="abbr",
        locationmode="USA-states",
        color=score_col,
        scope="usa",
        color_continuous_scale=[
            [0, "#dbeafe"],
            [0.35, "#93c5fd"],
            [0.65, "#3b82f6"],
            [1, "#1e3a8a"],
        ],
    )

    hover_texts: list[str] = []
    for _, row in d.iterrows():
        abbr = str(row["abbr"])
        st_name = str(row["state"])
        sc = str(int(row["_hover_score"]))
        base = f"<b>{st_name}</b><br>Risk score: {sc}"
        if enrichment_by_abbr:
            e = enrichment_by_abbr.get(abbr) or enrichment_by_abbr.get(abbr.upper())
            if isinstance(e, dict):
                nar = e.get("risk_narrative") or "—"
                conf = e.get("confidence_label") or "—"
                base += f"<br>Narrative: {nar}<br>Confidence: {conf}"
        hover_texts.append(base)

    fig.update_traces(
        hovertemplate="%{hovertext}<extra></extra>",
        hovertext=hover_texts,
    )
    fig.update_geos(scope="usa", showlakes=True, lakecolor="rgb(255,255,255)")
    fig.update_layout(
        title="State risk score by state",
        height=420,
        margin=dict(l=0, r=0, t=50, b=0),
        coloraxis_colorbar=dict(title="Risk score", tickformat="d"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#334155"),
    )
    return fig


def kindergarten_coverage_map_figure(
    kg_work: pd.DataFrame,
    state_col: str,
    pct_col: str,
) -> go.Figure:
    """MMR kindergarten coverage choropleth (Blues), with gray states for missing."""
    from utils.state_maps import state_to_abbr, STATE_TO_ABBR

    if kg_work is None or kg_work.empty or not pct_col:
        fig = go.Figure()
        fig.update_layout(title="MMR coverage (no data)", height=400)
        return fig

    cov_agg = kg_work.groupby(state_col, as_index=False)[pct_col].mean()
    cov_agg = cov_agg.rename(columns={pct_col: "coverage"})
    cov_agg["coverage"] = pd.to_numeric(cov_agg["coverage"], errors="coerce")
    cov_agg["abbr"] = cov_agg[state_col].astype(str).apply(state_to_abbr)

    all_states = pd.DataFrame([{"abbr": abbr} for abbr in STATE_TO_ABBR.values()])
    map_df = all_states.merge(cov_agg[["abbr", "coverage"]], on="abbr", how="left")
    map_df["hover_text"] = map_df["coverage"].apply(
        lambda x: f"{x:.1f}%" if pd.notna(x) else "Data Unavailable"
    )

    map_with_data = map_df[map_df["coverage"].notna()]
    map_no_data = map_df[map_df["coverage"].isna()]
    r = map_df["coverage"].dropna()
    range_color = (float(r.min()), float(r.max())) if len(r) else (0, 100)

    fig = px.choropleth(
        map_with_data,
        locations="abbr",
        locationmode="USA-states",
        color="coverage",
        scope="usa",
        color_continuous_scale="Blues",
        title="Kindergarten MMR coverage % by state",
        custom_data=["hover_text"],
        range_color=range_color,
    )
    fig.update_traces(hovertemplate="%{location}<br>%{customdata[0]}<extra></extra>")
    if not map_no_data.empty:
        fig.add_trace(
            go.Choropleth(
                locations=map_no_data["abbr"],
                locationmode="USA-states",
                z=[0] * len(map_no_data),
                colorscale=[[0, "lightgray"], [1, "lightgray"]],
                showscale=False,
                hoverinfo="text",
                text=[f"{abbr}<br>Data Unavailable" for abbr in map_no_data["abbr"]],
                hoverlabel=dict(bgcolor="white", font=dict(color="black")),
            )
        )
    fig.update_layout(
        height=400,
        margin=dict(l=0, r=0, t=40, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(size=12, color="#334155"),
        coloraxis_colorbar=dict(title="Coverage %"),
    )
    return fig
