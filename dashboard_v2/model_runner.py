"""
Shared data load + model orchestration (no UI framework).
Used by the Shiny app; mirrors the former Streamlit load_and_model pipeline.
"""
from __future__ import annotations

import pandas as pd

from utils.logging_config import get_logger

logger = get_logger("model_runner")


def load_and_model(use_cache: bool = True, outbreak_percentile: float = 95.0) -> dict:
    """
    Load all sources, fit model, compute aggregates. Returns a dict of all session fields
    the dashboard needs (same keys previously stored on st.session_state).
    """
    from loaders import load_all, clear_cache
    from risk import (
        fit_stage1,
        predict_alarm_probability,
        get_forecast,
        get_baseline_risk,
        get_state_risk_df,
        get_national_weekly_cases,
    )

    if not use_cache:
        clear_cache()
    hist, kg, ww, nndss, load_status, data_as_of = load_all(use_cache=use_cache)

    nndss_agg = pd.DataFrame()
    nndss_audit: dict = {}
    if nndss is not None and not nndss.empty:
        nndss_agg, nndss_audit = get_national_weekly_cases(nndss)
    if not nndss_agg.empty:
        recent = nndss_agg.tail(5)
        logger.info(
            "Five most recent NNDSS (national weekly cases) available to app: %s",
            recent[["year", "week", "cases"]].to_dict("records"),
        )

    alarm_prob = 0.5
    model_stage1 = None
    coef_df = None
    auc = None
    try:
        model, coef_df, auc, _ = fit_stage1(nndss, ww, kg, outbreak_percentile=outbreak_percentile)
        model_stage1 = model
        alarm_prob = predict_alarm_probability(
            model, nndss, ww, kg, outbreak_percentile=outbreak_percentile
        )
    except Exception:
        logger.exception("Stage 1 fit/predict failed")
        alarm_prob = 0.5
        model_stage1 = None
        coef_df = None
        auc = None

    forecast_df = None
    try:
        forecast_df, ok = get_forecast(nndss)
        if not ok:
            forecast_df = None
    except Exception:
        forecast_df = None

    baseline_tier = "low"
    baseline_val = 0.0
    try:
        baseline_tier, baseline_val = get_baseline_risk(hist, nndss)
    except Exception:
        baseline_tier = "low"
        baseline_val = 0.0

    state_risk_df = None
    try:
        state_risk_df = get_state_risk_df(kg, nndss, ww)
    except Exception:
        state_risk_df = None

    return {
        "hist": hist,
        "kg": kg,
        "ww": ww,
        "nndss": nndss,
        "load_status": load_status,
        "data_as_of": data_as_of,
        "nndss_agg": nndss_agg,
        "nndss_audit": nndss_audit,
        "model_stage1": model_stage1,
        "coef_df": coef_df,
        "auc": auc,
        "alarm_prob": alarm_prob,
        "forecast_df": forecast_df,
        "baseline_tier": baseline_tier,
        "baseline_val": baseline_val,
        "state_risk_df": state_risk_df,
    }
