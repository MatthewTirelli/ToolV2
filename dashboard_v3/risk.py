"""
Two-stage Alarm-then-Size risk model for the Predictive Measles Risk Dashboard.
Stage 1: Logistic regression for outbreak_next_4w (early warning).
Stage 2: 4-8 week case forecast.
Plus baseline and state-level risk.
"""
from __future__ import annotations

import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd

from utils.logging_config import get_logger

logger = get_logger("risk")

# Default outbreak threshold: 1 if total cases in next 4 weeks > this (used when not using percentile)
DEFAULT_OUTBREAK_THRESHOLD = 0
# When using data-driven threshold: "outbreak" = next-4-weeks cases above this percentile of historical 4-week counts
DEFAULT_OUTBREAK_PERCENTILE = 95.0
# How many weeks to forecast
FORECAST_WEEKS = 8
# Time-based split: last N weeks as test (for metrics); train on rest
TEST_WEEKS = 12


def _safe_numeric(s: pd.Series) -> pd.Series:
    """Coerce to numeric, non-numeric -> NaN."""
    return pd.to_numeric(s, errors="coerce")


def _national_weekly_cases(nndss: pd.DataFrame) -> pd.DataFrame:
    """Aggregate NNDSS to national weekly case counts. Returns df with year, week, cases."""
    agg, _ = get_national_weekly_cases(nndss)
    return agg


# Prefer m2 (100% non-null per audit); then m1, current_week, etc.
NNDSS_CASE_COLUMN_PRIORITY = ["m2", "m1", "current_week", "Current week", "weekly_cases", "cases"]
# Columns that must never be used as case counts (time, geography, ids)
NNDSS_EXCLUDE_CASE_COLUMNS = {"mmwr_year", "mmwr_week", "year", "week", "sort_order", "Reporting Area", "states", "label", "geography", "location1", "reporting_area"}


def _pick_case_col_and_agg(df: pd.DataFrame, year_col: str, week_col: str, audit: dict, n_rows_label: str) -> Tuple[pd.DataFrame, Optional[str]]:
    """Pick case column from priority, attach cases, return (df with _y,_w,cases, groupby agg; case_col). Used for national or jurisdiction slice."""
    df = df.copy()
    df["_y"] = pd.to_numeric(df[year_col], errors="coerce")
    df["_w"] = pd.to_numeric(df[week_col], errors="coerce")
    df = df.dropna(subset=["_y", "_w"])
    if df.empty:
        return pd.DataFrame(columns=["year", "week", "cases"]), None
    num_cols = [c for c in df.select_dtypes(include=[np.number]).columns.tolist()
                if c not in NNDSS_EXCLUDE_CASE_COLUMNS and "flag" not in c.lower()]
    case_col = None
    for c in NNDSS_CASE_COLUMN_PRIORITY:
        if c in df.columns and c not in NNDSS_EXCLUDE_CASE_COLUMNS:
            case_col = c
            break
    if not case_col:
        for c in num_cols:
            case_col = c
            break
    if not case_col:
        return pd.DataFrame(columns=["year", "week", "cases"]), None
    df["cases"] = _safe_numeric(df[case_col]).fillna(0)
    agg = df.groupby(["_y", "_w"], as_index=False)["cases"].sum()
    agg = agg.rename(columns={"_y": "year", "_w": "week"}).sort_values(["year", "week"]).reset_index(drop=True)
    return agg, case_col


def get_national_weekly_cases(nndss: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Canonical NNDSS national weekly aggregation. Returns (df with year, week, cases; audit dict).
    If US RESIDENTS is missing or incomplete (max year < overall max), builds national by summing
    valid jurisdiction rows only (state_to_abbr). Uses m2-first case column priority.
    """
    from utils.state_maps import state_to_abbr
    empty_df = pd.DataFrame(columns=["year", "week", "cases"])
    audit = {"case_column_used": None, "year_min": None, "year_max": None, "year_week_max": None, "columns_available_top10": None,
             "n_national_rows": None, "year_max_before_agg": None, "year_max_after_agg": None, "candidate_case_columns": None,
             "total_cases_latest_year": None, "source": "US RESIDENTS"}
    if nndss is None or nndss.empty:
        return empty_df, audit
    reporting_col = "Reporting Area" if "Reporting Area" in nndss.columns else ("states" if "states" in nndss.columns else None)
    if not reporting_col:
        return empty_df, audit
    year_col = "year" if "year" in nndss.columns else None
    week_col = "week" if "week" in nndss.columns else None
    if not year_col or not week_col:
        return empty_df, audit
    overall_max_year = int(pd.to_numeric(nndss[year_col], errors="coerce").max()) if nndss[year_col].notna().any() else None
    national = nndss[nndss[reporting_col].astype(str).str.strip() == "US RESIDENTS"].copy()
    national_max_year = None
    if not national.empty:
        national["_y"] = pd.to_numeric(national[year_col], errors="coerce")
        national["_w"] = pd.to_numeric(national[week_col], errors="coerce")
        national = national.dropna(subset=["_y", "_w"])
        if not national.empty:
            national_max_year = int(national["_y"].max())
    use_jurisdictions = national.empty or (overall_max_year is not None and (national_max_year is None or national_max_year < overall_max_year))
    if use_jurisdictions:
        # Sum from jurisdiction rows; only include valid US states (state_to_abbr)
        jur = nndss[nndss[reporting_col].astype(str).str.strip() != "US RESIDENTS"].copy()
        jur["_area"] = jur[reporting_col].astype(str).str.strip()
        jur = jur[jur["_area"].apply(lambda x: state_to_abbr(x) is not None)]
        jur = jur.drop(columns=["_area"], errors="ignore")
        if jur.empty:
            return empty_df, audit
        audit["n_national_rows"] = len(jur)
        audit["year_max_before_agg"] = int(pd.to_numeric(jur[year_col], errors="coerce").max())
        audit["source"] = "jurisdictions (summed)"
        agg, case_col = _pick_case_col_and_agg(jur, year_col, week_col, audit, "jurisdiction")
    else:
        audit["n_national_rows"] = len(national)
        audit["year_max_before_agg"] = int(national["_y"].max())
        agg, case_col = _pick_case_col_and_agg(national, year_col, week_col, audit, "national")
    if agg.empty or not case_col:
        audit["columns_available_top10"] = list(nndss.columns)[:10]
        return empty_df, audit
    audit["case_column_used"] = case_col
    audit["year_min"] = int(agg["year"].min())
    audit["year_max"] = int(agg["year"].max())
    audit["year_max_after_agg"] = int(agg["year"].max())
    last = agg.iloc[-1]
    audit["year_week_max"] = (int(last["year"]), int(last["week"]))
    latest_year = int(last["year"])
    audit["total_cases_latest_year"] = int(agg[agg["year"] == latest_year]["cases"].sum())
    logger.info("NNDSS national weekly: case_column=%s year_min=%s year_max=%s year_week_max=%s total_cases_latest_year=%s source=%s",
                audit["case_column_used"], audit["year_min"], audit["year_max"], audit["year_week_max"], audit["total_cases_latest_year"], audit["source"])
    return agg, audit


# Priority list for CDC wastewater signal column (dataset column names vary)
WW_SIGNAL_COLUMN_PRIORITY = [
    "normalized_viral_load",
    "target_concentration",
    "value",
    "signal",
    "viral_load",
    "pcr_target_flowpop_lin",
    "pcr_target_avg_conc_lin",
    "pcr_target_avg_conc",
    "pcr_target_mic_lin",
]


def _detect_ww_signal_column(ww: pd.DataFrame) -> Optional[str]:
    """Return first column from WW_SIGNAL_COLUMN_PRIORITY that exists in ww, else None."""
    for c in WW_SIGNAL_COLUMN_PRIORITY:
        if c in ww.columns:
            return c
    return None


def _wastewater_national_weekly(ww: pd.DataFrame) -> pd.DataFrame:
    """Aggregate wastewater to national weekly (mean of signal). Returns df with year, week, ww_signal. No silent zeros."""
    if ww.empty:
        return pd.DataFrame(columns=["year", "week", "ww_signal"])
    logger.info("wastewater dataframe columns: %s", list(ww.columns))
    year_col = next((c for c in ("year", "mmwr_year", "reporting_year") if c in ww.columns), None)
    week_col = next((c for c in ("week", "mmwr_week", "reporting_week") if c in ww.columns), None)
    if not year_col or not week_col:
        logger.warning("wastewater: missing year/week columns (have %s)", list(ww.columns))
        return pd.DataFrame(columns=["year", "week", "ww_signal"])
    signal_col = _detect_ww_signal_column(ww)
    if not signal_col:
        logger.warning("wastewater: no valid signal column found (checked %s). Returning empty.", WW_SIGNAL_COLUMN_PRIORITY)
        return pd.DataFrame(columns=["year", "week", "ww_signal"])
    logger.info("wastewater aggregate: signal_col=%s", signal_col)
    ww = ww.copy()
    ww["year"] = ww[year_col].astype(str)
    ww["week"] = _safe_numeric(ww[week_col]).fillna(0).astype(int)
    ww["ww_signal"] = pd.to_numeric(ww[signal_col], errors="coerce")
    agg = ww.groupby(["year", "week"], as_index=False)["ww_signal"].mean()
    if not agg.empty and "ww_signal" in agg.columns:
        valid = agg["ww_signal"].dropna()
        n_valid = len(valid)
        n_rows = len(agg)
        pct_non_null = round(100.0 * n_valid / n_rows, 1) if n_rows > 0 else 0
        logger.info("wastewater weekly aggregate: rows=%s pct_non_null=%.1f min=%s median=%s max=%s",
                    n_rows, pct_non_null,
                    float(valid.min()) if n_valid else None,
                    float(valid.median()) if n_valid else None,
                    float(valid.max()) if n_valid else None)
        if n_valid == 0 or (valid == 0).all():
            logger.warning("wastewater: all weekly ww_signal are zero or null; dataset may have no measurable signal")
    return agg


def get_wastewater_diagnostics(ww: pd.DataFrame) -> dict:
    """
    Return diagnostics for wastewater data: signal column used, min/median/max, non-null rate, all_zero.
    Use for UI data-sanity display; do not fill missing with zeros.
    """
    out = {"signal_col": None, "min": None, "median": None, "max": None, "non_null_pct": 0.0, "all_zero": True, "rows": 0}
    if ww is None or ww.empty:
        return out
    out["rows"] = len(ww)
    signal_col = _detect_ww_signal_column(ww)
    out["signal_col"] = signal_col
    if not signal_col:
        return out
    ser = _safe_numeric(ww[signal_col])
    n = ser.notna().sum()
    out["non_null_pct"] = round(100.0 * n / len(ww), 1) if len(ww) > 0 else 0
    valid = ser.dropna()
    if len(valid) > 0:
        out["min"] = float(valid.min())
        out["median"] = float(valid.median())
        out["max"] = float(valid.max())
        out["all_zero"] = (valid == 0).all()
    return out


def _state_weekly_cases(nndss: pd.DataFrame) -> pd.DataFrame:
    """NNDSS cases by state, year, week. Uses Reporting Area / states as state id. Returns df with state, year, week, cases."""
    if nndss.empty:
        return pd.DataFrame(columns=["state", "year", "week", "cases"])
    reporting_col = "Reporting Area" if "Reporting Area" in nndss.columns else ("states" if "states" in nndss.columns else None)
    if not reporting_col:
        return pd.DataFrame(columns=["state", "year", "week", "cases"])
    state_col = "_state" if "_state" in nndss.columns else reporting_col
    if state_col not in nndss.columns:
        nndss = nndss.copy()
        nndss["_state"] = nndss[reporting_col].astype(str).str.strip()
        state_col = "_state"
    year_col = "year" if "year" in nndss.columns else None
    week_col = "week" if "week" in nndss.columns else None
    if not year_col or not week_col:
        return pd.DataFrame(columns=["state", "year", "week", "cases"])
    case_col = None
    for c in ("m1", "m2", "current_week"):
        if c in nndss.columns:
            case_col = c
            break
    if not case_col:
        num_cols = [c for c in nndss.select_dtypes(include=[np.number]).columns if "flag" not in c.lower()]
        case_col = num_cols[0] if num_cols else None
    if not case_col:
        return pd.DataFrame(columns=["state", "year", "week", "cases"])
    df = nndss.copy()
    df["year"] = df[year_col].astype(str)
    df["week"] = _safe_numeric(df[week_col]).fillna(0).astype(int)
    df["cases"] = _safe_numeric(df[case_col]).fillna(0)
    agg = df.groupby([state_col, "year", "week"], as_index=False)["cases"].sum()
    return agg.rename(columns={state_col: "state"})


def _wastewater_state_weekly(ww: pd.DataFrame) -> pd.DataFrame:
    """Wastewater signal by state (normalized to state abbr via state_to_abbr). Returns df with state=abbr, year, week, ww_signal."""
    from utils.state_maps import state_to_abbr
    if ww.empty or "wwtp_jurisdiction" not in ww.columns:
        return pd.DataFrame(columns=["state", "year", "week", "ww_signal"])
    year_col = next((c for c in ("year", "mmwr_year") if c in ww.columns), None)
    week_col = next((c for c in ("week", "mmwr_week") if c in ww.columns), None)
    if not year_col or not week_col:
        return pd.DataFrame(columns=["state", "year", "week", "ww_signal"])
    signal_col = _detect_ww_signal_column(ww)
    if not signal_col:
        return pd.DataFrame(columns=["state", "year", "week", "ww_signal"])
    ww = ww.copy()
    ww["state"] = ww["wwtp_jurisdiction"].astype(str).str.strip().apply(state_to_abbr)
    ww = ww.dropna(subset=["state"])
    if ww.empty:
        return pd.DataFrame(columns=["state", "year", "week", "ww_signal"])
    ww["year"] = ww[year_col].astype(str)
    ww["week"] = _safe_numeric(ww[week_col]).fillna(0).astype(int)
    ww["ww_signal"] = pd.to_numeric(ww[signal_col], errors="coerce")
    return ww.groupby(["state", "year", "week"], as_index=False)["ww_signal"].mean()


# --- Wastewater detection frequency (STEP 1-4) and lag correlation (STEP 5-6) ---

def compute_ww_detection_frequency(
    ww_df: pd.DataFrame,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
) -> Tuple[pd.DataFrame, dict]:
    """
    STEP 1-4: Filter valid measles measurements, define is_detected, collapse to site-week, compute weekly detection_frequency.
    Returns (weekly_df with year, week, total_sites, positive_sites, detection_frequency), validation_info dict.
    """
    required_ww = ["pcr_target", "ntc_amplify", "inhibition_detect", "inhibition_adjust", "pcr_target_avg_conc"]
    validation_info = {"n_rows_raw": 0, "n_after_qc": 0, "n_unique_sites": 0, "all_zero_freq": True, "pct_passing_qc": 0.0, "pcr_target_used": None, "weeks_min": None, "weeks_max": None, "missing_columns": None, "columns_present": None, "detection_rule_used": None}
    empty_weekly = pd.DataFrame(columns=["year", "week", "total_sites", "positive_sites", "detection_frequency"])
    if ww_df is None or ww_df.empty:
        return empty_weekly, validation_info
    validation_info["n_rows_raw"] = len(ww_df)
    validation_info["columns_present"] = list(ww_df.columns)

    # STEP 1 — Filter: pcr_target == "Measles virus" or MEV (actual value from data); ntc_amplify == "no"; (inhibition_detect == "no") OR (inhibition_adjust == "yes")
    df = ww_df.copy()
    if "pcr_target" not in df.columns:
        validation_info["missing_columns"] = [c for c in required_ww if c not in df.columns]
        logger.warning("wastewater detection_frequency: missing pcr_target column; present: %s", list(df.columns))
        return empty_weekly, validation_info
    pt = df["pcr_target"].astype(str).str.strip()
    pt_upper = pt.str.upper()
    measles_ok = pt_upper.str.contains("MEV", na=False) | (pt_upper == "MEASLES VIRUS")
    df = df.loc[measles_ok].copy()
    if not df.empty:
        validation_info["pcr_target_used"] = df["pcr_target"].astype(str).iloc[0].strip() if (df["pcr_target"].notna().any()) else None
    if "ntc_amplify" in df.columns:
        ntc_ok = df["ntc_amplify"].astype(str).str.strip().str.lower() == "no"
        df = df.loc[ntc_ok]
    if "inhibition_detect" in df.columns and "inhibition_adjust" in df.columns:
        inh_ok = (df["inhibition_detect"].astype(str).str.strip().str.lower() == "no") | (df["inhibition_adjust"].astype(str).str.strip().str.lower() == "yes")
        df = df.loc[inh_ok]
    if df.empty:
        logger.warning("wastewater detection_frequency: no rows after QC filters (pcr_target, ntc_amplify, inhibition)")
        validation_info["n_after_qc"] = 0
        return empty_weekly, validation_info
    validation_info["n_after_qc"] = len(df)
    validation_info["pct_passing_qc"] = round(100.0 * len(df) / validation_info["n_rows_raw"], 1) if validation_info["n_rows_raw"] > 0 else 0

    # year/week
    year_col = next((c for c in ("year", "mmwr_year") if c in df.columns), None)
    week_col = next((c for c in ("week", "mmwr_week") if c in df.columns), None)
    if not year_col or not week_col:
        validation_info["missing_columns"] = (["year"] if not year_col else []) + (["week"] if not week_col else [])
        logger.warning("wastewater detection_frequency: missing year/week columns; present: %s", list(df.columns))
        return empty_weekly, validation_info
    df["year"] = pd.to_numeric(df[year_col], errors="coerce").fillna(0).astype(int)
    df["week"] = pd.to_numeric(df[week_col], errors="coerce").fillna(0).astype(int)
    if year_min is not None:
        df = df[df["year"] >= year_min]
    if year_max is not None:
        df = df[df["year"] <= year_max]
    if df.empty:
        return empty_weekly, validation_info

    # STEP 2 — Detection variable (primary: pcr_target_avg_conc > 0; fallback: proxy signal column if conc missing)
    conc_col = None
    if "pcr_target_avg_conc" in df.columns:
        conc_col = "pcr_target_avg_conc"
        validation_info["detection_rule_used"] = "primary: pcr_target_avg_conc > 0, QC (ntc_amplify=no, inhibition)"
    else:
        for proxy in WW_SIGNAL_COLUMN_PRIORITY:
            if proxy in df.columns:
                conc_col = proxy
                validation_info["detection_rule_used"] = "fallback: %s > 0 (pcr_target_avg_conc not in dataset)" % proxy
                logger.info("wastewater detection_frequency: using fallback signal column %s", proxy)
                break
    if not conc_col:
        validation_info["missing_columns"] = [c for c in required_ww if c not in ww_df.columns]
        logger.warning("wastewater detection_frequency: missing pcr_target_avg_conc and no proxy; present: %s", list(ww_df.columns))
        return empty_weekly, validation_info
    conc = pd.to_numeric(df[conc_col], errors="coerce")
    df["is_detected"] = (conc > 0).fillna(False)

    # STEP 3 — Site-week level (one row per site per week; detected if any measurement detected)
    site_col = "sewershed_id" if "sewershed_id" in df.columns else ("sample_id" if "sample_id" in df.columns else None)
    if not site_col:
        validation_info["missing_columns"] = ["sewershed_id or sample_id"]
        logger.warning("wastewater detection_frequency: no site identifier (sewershed_id or sample_id); present: %s", list(df.columns))
        return empty_weekly, validation_info
    df["site_id"] = df[site_col].astype(str).str.strip()
    site_week = df.groupby(["site_id", "year", "week"], as_index=False)["is_detected"].any()
    validation_info["n_unique_sites"] = int(site_week["site_id"].nunique())

    # STEP 4 — Weekly detection frequency
    total_sites = site_week.groupby(["year", "week"], as_index=False)["site_id"].nunique().rename(columns={"site_id": "total_sites"})
    positive = site_week[site_week["is_detected"]].groupby(["year", "week"], as_index=False)["site_id"].nunique().rename(columns={"site_id": "positive_sites"})
    weekly = total_sites.merge(positive, on=["year", "week"], how="left")
    weekly["positive_sites"] = weekly["positive_sites"].fillna(0).astype(int)
    weekly["detection_frequency"] = (weekly["positive_sites"] / weekly["total_sites"].replace(0, np.nan)).fillna(0)
    validation_info["all_zero_freq"] = (weekly["detection_frequency"].fillna(0) == 0).all()
    if not weekly.empty and "year" in weekly.columns and "week" in weekly.columns:
        validation_info["weeks_min"] = (int(weekly["year"].min()), int(weekly["week"].min()))
        validation_info["weeks_max"] = (int(weekly["year"].max()), int(weekly["week"].max()))
    logger.info("ww detection_frequency: rows_after_qc=%s unique_sites=%s weekly_rows=%s all_zero_freq=%s pcr_target_used=%s",
                validation_info["n_after_qc"], validation_info["n_unique_sites"], len(weekly), validation_info["all_zero_freq"], validation_info["pcr_target_used"])
    return weekly, validation_info


def compute_ww_lag_correlation(
    ww_weekly: pd.DataFrame,
    nndss_weekly: pd.DataFrame,
    max_lag: int = 12,
    use_log_cases: bool = True,
) -> Tuple[pd.DataFrame, dict]:
    """
    STEP 5-6: Merge wastewater weekly with NNDSS weekly; compute detection_frequency lagged 1..max_lag vs cases.
    Returns (df with lag_weeks, correlation, p_value), summary dict (best_lag, best_r, best_p).
    """
    from scipy import stats as scipy_stats
    empty_lag = pd.DataFrame(columns=["lag_weeks", "correlation", "p_value"])
    summary = {"best_lag": None, "best_r": None, "best_p": None}
    if ww_weekly.empty or "detection_frequency" not in ww_weekly.columns:
        return empty_lag, summary
    if nndss_weekly.empty or "cases" not in nndss_weekly.columns:
        return empty_lag, summary
    ww_weekly = ww_weekly.copy()
    nndss_weekly = nndss_weekly.copy()
    nndss_weekly["year"] = nndss_weekly["year"].astype(str)
    nndss_weekly["week"] = pd.to_numeric(nndss_weekly["week"], errors="coerce").fillna(0).astype(int)
    ww_weekly["year"] = ww_weekly["year"].astype(str)
    ww_weekly["week"] = pd.to_numeric(ww_weekly["week"], errors="coerce").fillna(0).astype(int)
    merged = ww_weekly.merge(nndss_weekly[["year", "week", "cases"]], on=["year", "week"], how="inner")
    if merged.empty or len(merged) < 10:
        return empty_lag, summary
    cases = merged["cases"].values.astype(float)
    if use_log_cases:
        cases = np.log10(cases + 1)
    freq = merged["detection_frequency"].values.astype(float)
    rows = []
    for k in range(1, min(max_lag + 1, len(merged))):
        freq_lag = np.roll(freq, k)
        freq_lag[:k] = np.nan
        valid = ~(np.isnan(freq_lag) | np.isnan(cases))
        if valid.sum() < 5:
            continue
        r, p = scipy_stats.pearsonr(freq_lag[valid], cases[valid])
        rows.append({"lag_weeks": k, "correlation": round(float(r), 4), "p_value": round(float(p), 4)})
    if not rows:
        return empty_lag, summary
    lag_df = pd.DataFrame(rows)
    best_idx = lag_df["correlation"].idxmax()
    summary["best_lag"] = int(lag_df.loc[best_idx, "lag_weeks"])
    summary["best_r"] = float(lag_df.loc[best_idx, "correlation"])
    summary["best_p"] = float(lag_df.loc[best_idx, "p_value"])
    return lag_df, summary


def validate_ww_nndss_audit(
    ww_validation: dict,
    ww_weekly: pd.DataFrame,
    nndss_weekly: pd.DataFrame,
    year_min: Optional[int],
    year_max: Optional[int],
) -> dict:
    """Lightweight audit: log and return summary for UI (wastewater QC stats, NNDSS sum in window)."""
    audit = {
        "ww_pct_passing_qc": ww_validation.get("pct_passing_qc", 0),
        "ww_unique_sites": ww_validation.get("n_unique_sites", 0),
        "ww_all_zero_freq": ww_validation.get("all_zero_freq", True),
        "nndss_cases_sum": 0,
        "nndss_window": f"year_min={year_min} year_max={year_max}",
    }
    if nndss_weekly is not None and not nndss_weekly.empty and "cases" in nndss_weekly.columns:
        if year_min is not None or year_max is not None:
            nw = nndss_weekly.copy()
            nw["year"] = pd.to_numeric(nw["year"], errors="coerce")
            if year_min is not None:
                nw = nw[nw["year"] >= year_min]
            if year_max is not None:
                nw = nw[nw["year"] <= year_max]
            audit["nndss_cases_sum"] = int(nw["cases"].sum())
        else:
            audit["nndss_cases_sum"] = int(nndss_weekly["cases"].sum())
    logger.info("audit: ww pct_passing_qc=%.1f unique_sites=%s all_zero_freq=%s nndss_cases_sum=%s",
                audit["ww_pct_passing_qc"], audit["ww_unique_sites"], audit["ww_all_zero_freq"], audit["nndss_cases_sum"])
    return audit


def _build_modeling_dataset(
    nndss: pd.DataFrame,
    wastewater: pd.DataFrame,
    kindergarten: pd.DataFrame,
    outbreak_threshold: float = DEFAULT_OUTBREAK_THRESHOLD,
) -> pd.DataFrame:
    """
    Build one row per (year, week) with target outbreak_next_4w and features.
    Target = 1 if sum(cases in next 4 weeks) > outbreak_threshold.
    Features: wastewater lags 1..12 (prior 8–12 weeks), seasonality (week_of_year), kindergarten (national avg if available).
    """
    national = _national_weekly_cases(nndss)
    if national.empty or len(national) < 5:
        return pd.DataFrame()
    national = national.sort_values(["year", "week"]).reset_index(drop=True)
    national["cases_next4"] = (
        national["cases"].shift(-1) + national["cases"].shift(-2) + national["cases"].shift(-3) + national["cases"].shift(-4)
    )
    national["outbreak_next_4w"] = (national["cases_next4"] > outbreak_threshold).astype(int)
    ww = _wastewater_national_weekly(wastewater)
    # Coerce year/week to int in both so merge does not mix int64 and object
    national = national.copy()
    national["year"] = pd.to_numeric(national["year"], errors="coerce").fillna(0).astype(int)
    national["week"] = pd.to_numeric(national["week"], errors="coerce").fillna(0).astype(int)
    ww = ww.copy()
    ww["year"] = pd.to_numeric(ww["year"], errors="coerce").fillna(0).astype(int)
    ww["week"] = pd.to_numeric(ww["week"], errors="coerce").fillna(0).astype(int)
    national = national.merge(ww, on=["year", "week"], how="left")
    national["ww_signal"] = national.get("ww_signal", pd.Series(dtype=float)).fillna(0).astype(float)
    # Use prior 8–12 weeks of wastewater (lags 1–12) for alarm model
    for lag in range(1, 13):
        national[f"ww_lag{lag}"] = national["ww_signal"].shift(lag).fillna(0).astype(float)
    national["week_of_year"] = national["week"].clip(1, 53)
    # Kindergarten: use latest national average coverage if column exists
    if not kindergarten.empty and "mmr_covered_pct" in kindergarten.columns:
        pct = _safe_numeric(kindergarten["mmr_covered_pct"]).mean()
        national["kg_coverage"] = pct
    elif not kindergarten.empty and "coverage" in kindergarten.columns:
        national["kg_coverage"] = _safe_numeric(kindergarten["coverage"]).mean()
    else:
        national["kg_coverage"] = 0
    feature_cols = [c for c in national.columns if c.startswith("ww_lag") or c in ("week_of_year", "kg_coverage")]
    national[feature_cols] = national[feature_cols].fillna(0)
    return national.dropna(subset=["outbreak_next_4w"]).reset_index(drop=True)


def get_outbreak_threshold_from_data(
    nndss: pd.DataFrame,
    percentile: float = DEFAULT_OUTBREAK_PERCENTILE,
) -> Optional[float]:
    """
    Compute outbreak threshold from the distribution of 4-week case counts in the data.
    Returns the given percentile of (sum of cases in next 4 weeks) across all year-weeks.
    "Outbreak" = next-4-weeks cases above this value (unusually high).
    Returns None if insufficient data.
    """
    national = _national_weekly_cases(nndss)
    if national.empty or len(national) < 5:
        return None
    national = national.sort_values(["year", "week"]).reset_index(drop=True)
    cases_next4 = (
        national["cases"].shift(-1)
        + national["cases"].shift(-2)
        + national["cases"].shift(-3)
        + national["cases"].shift(-4)
    )
    valid = cases_next4.dropna()
    if valid.empty or len(valid) < 10:
        return None
    threshold = float(np.percentile(valid, percentile))
    logger.info(
        "Outbreak threshold from data: %.1f (%.0fth percentile of next-4-weeks cases, n=%s)",
        threshold,
        percentile,
        len(valid),
    )
    return threshold


def fit_stage1(
    nndss: pd.DataFrame,
    wastewater: pd.DataFrame,
    kindergarten: pd.DataFrame,
    outbreak_threshold: float = DEFAULT_OUTBREAK_THRESHOLD,
    outbreak_percentile: Optional[float] = DEFAULT_OUTBREAK_PERCENTILE,
) -> Tuple[Optional[object], Optional[pd.DataFrame], float, Optional[dict]]:
    """
    Fit Stage 1 logistic regression. Returns (model, coefficients_df, auc, metrics_dict).
    On failure returns (None, None, 0.5, None) and logs.
    If outbreak_percentile is set (e.g. 75), threshold is computed from data (that percentile of
    next-4-weeks case counts); otherwise outbreak_threshold is used.
    """
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import train_test_split
    except ImportError:
        logger.error("sklearn not installed")
        return None, None, 0.5, None
    threshold = outbreak_threshold
    if outbreak_percentile is not None:
        data_threshold = get_outbreak_threshold_from_data(nndss, outbreak_percentile)
        if data_threshold is not None:
            threshold = data_threshold
    df = _build_modeling_dataset(nndss, wastewater, kindergarten, outbreak_threshold=threshold)
    if df.empty or len(df) < 20:
        logger.warning("Stage 1 fit skipped: insufficient rows after build (%s)", len(df))
        return None, None, 0.5, None
    feature_cols = [c for c in df.columns if c.startswith("ww_lag") or c in ("week_of_year", "kg_coverage")]
    if not feature_cols:
        return None, None, 0.5, None
    X = df[feature_cols]
    y = df["outbreak_next_4w"]
    if y.nunique() < 2:
        logger.warning("Stage 1 fit skipped: no variance in target")
        return None, None, 0.5, None
    # Time-based split: last TEST_WEEKS as test
    n = len(df)
    if n > TEST_WEEKS:
        train_idx = np.arange(0, n - TEST_WEEKS)
        test_idx = np.arange(n - TEST_WEEKS, n)
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    if y_train.nunique() < 2:
        logger.warning("Stage 1 fit skipped: training set has only one class (try higher outbreak_percentile or different threshold)")
        return None, None, 0.5, None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = LogisticRegression(max_iter=1000, random_state=42)
        model.fit(X_train, y_train)
    pred_proba = model.predict_proba(X_test)[:, 1]
    try:
        auc = roc_auc_score(y_test, pred_proba)
    except Exception:
        auc = 0.5
    coef_df = pd.DataFrame({"feature": feature_cols, "coefficient": model.coef_.ravel()})
    metrics = {"auc": auc, "n_train": len(X_train), "n_test": len(X_test)}
    logger.info("Stage 1 fit n_train=%s n_test=%s auc=%.3f", len(X_train), len(X_test), auc)
    return model, coef_df, auc, metrics


def predict_alarm_probability(
    model: Optional[object],
    nndss: pd.DataFrame,
    wastewater: pd.DataFrame,
    kindergarten: pd.DataFrame,
    outbreak_threshold: float = DEFAULT_OUTBREAK_THRESHOLD,
    outbreak_percentile: Optional[float] = DEFAULT_OUTBREAK_PERCENTILE,
) -> float:
    """Predict P(outbreak in next 4 weeks) for the most recent week. Returns 0.5 if model is None."""
    if model is None:
        return 0.5
    threshold = outbreak_threshold
    if outbreak_percentile is not None:
        data_threshold = get_outbreak_threshold_from_data(nndss, outbreak_percentile)
        if data_threshold is not None:
            threshold = data_threshold
    df = _build_modeling_dataset(nndss, wastewater, kindergarten, outbreak_threshold=threshold)
    if df.empty:
        return 0.5
    feature_cols = [c for c in df.columns if c.startswith("ww_lag") or c in ("week_of_year", "kg_coverage")]
    if not feature_cols:
        return 0.5
    X = df[feature_cols].iloc[-1:]
    try:
        return float(model.predict_proba(X)[0, 1])
    except Exception:
        return 0.5


def get_forecast(nndss: pd.DataFrame, weeks: int = FORECAST_WEEKS) -> Tuple[pd.DataFrame, bool]:
    """
    Stage 2: forecast next 4-8 weeks of national cases. Returns (df with week_ahead, forecast), success.
    Uses simple average of recent weeks; empty if insufficient data.
    """
    national = _national_weekly_cases(nndss)
    if national.empty or len(national) < 4:
        return pd.DataFrame(columns=["week_ahead", "forecast"]), False
    national = national.sort_values(["year", "week"]).reset_index(drop=True)
    recent = national["cases"].tail(12)
    if recent.empty:
        return pd.DataFrame(columns=["week_ahead", "forecast"]), False
    mean_cases = float(recent.mean())
    last_year = national.iloc[-1]["year"]
    last_week = int(national.iloc[-1]["week"])
    rows = []
    for i in range(1, min(weeks + 1, 9)):
        w = last_week + i
        y = last_year
        if w > 53:
            w -= 53
            y = str(int(y) + 1)
        rows.append({"week_ahead": i, "forecast": round(mean_cases, 1)})
    logger.info("Stage 2 forecast: %s weeks, mean=%.1f", len(rows), mean_cases)
    return pd.DataFrame(rows), True


def get_baseline_risk(historical: pd.DataFrame, nndss: pd.DataFrame) -> Tuple[str, float]:
    """
    Baseline risk from historical trend. Returns (tier "low"|"medium"|"high", value 0-100).
    Uses deviation from 5-year average if available.
    """
    if historical.empty:
        return "low", 0.0
    case_col = "Measles Cases" if "Measles Cases" in historical.columns else historical.columns[1]
    hist = _safe_numeric(historical[case_col]).dropna()
    if len(hist) < 5:
        return "low", 0.0
    recent_avg = hist.tail(5).mean()
    overall_avg = hist.mean()
    if overall_avg <= 0:
        return "low", 0.0
    ratio = recent_avg / overall_avg
    if ratio > 1.5:
        return "high", min(100, 50 + (ratio - 1.5) * 50)
    if ratio > 1.0:
        return "medium", min(100, 30 + (ratio - 1.0) * 40)
    return "low", max(0, 20 * ratio)


def get_baseline_risk_components(historical: pd.DataFrame, nndss: pd.DataFrame) -> dict:
    """
    Same as get_baseline_risk but returns a dict with tier, score, and component breakdown for UI explanation.
    """
    out = {"tier": "low", "score": 0.0, "recent_5yr_avg": None, "overall_avg": None, "ratio": None, "formula": ""}
    if historical.empty:
        return out
    case_col = "Measles Cases" if "Measles Cases" in historical.columns else historical.columns[1]
    hist = _safe_numeric(historical[case_col]).dropna()
    if len(hist) < 5:
        return out
    recent_avg = float(hist.tail(5).mean())
    overall_avg = float(hist.mean())
    out["recent_5yr_avg"] = round(recent_avg, 1)
    out["overall_avg"] = round(overall_avg, 2)
    if overall_avg <= 0:
        return out
    ratio = recent_avg / overall_avg
    out["ratio"] = round(ratio, 2)
    if ratio > 1.5:
        out["tier"] = "high"
        out["score"] = min(100, 50 + (ratio - 1.5) * 50)
        out["formula"] = "Recent 5-year average is >1.5× overall average → high tier; score = 50 + (ratio − 1.5)×50."
    elif ratio > 1.0:
        out["tier"] = "medium"
        out["score"] = min(100, 30 + (ratio - 1.0) * 40)
        out["formula"] = "Recent 5-year average is 1–1.5× overall → medium tier; score = 30 + (ratio − 1)×40."
    else:
        out["tier"] = "low"
        out["score"] = max(0, 20 * ratio)
        out["formula"] = "Recent 5-year average ≤ overall → low tier; score = 20×ratio."
    out["score"] = round(out["score"], 1)
    return out


def get_state_risk_df(
    kindergarten: pd.DataFrame,
    nndss: pd.DataFrame,
    wastewater: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Per-state risk: Coverage Risk (0–50), Case Activity (0–30), Wastewater Activity (0–20).
    If state-level wastewater unavailable, only coverage + cases are used and rescaled to 100.
    Returns df with state, coverage, cases_recent, ww_recent, wastewater_coverage, coverage_points,
    case_points, wastewater_points, total_risk, risk_tier, risk_score.
    States with no wastewater data get wastewater_points=0 and total_risk capped at 80 (no rescale).
    """
    out_cols = [
        "state", "coverage", "cases_recent", "ww_recent", "wastewater_coverage",
        "coverage_points", "case_points", "wastewater_points", "total_risk", "risk_tier", "risk_score",
    ]
    if kindergarten.empty:
        return pd.DataFrame(columns=out_cols)
    state_col = None
    for c in ["state", "State", "jurisdiction", "geography", "location1"]:
        if c in kindergarten.columns:
            state_col = c
            break
    if not state_col:
        return pd.DataFrame(columns=out_cols)
    pct_col = None
    for c in kindergarten.columns:
        if "pct" in c.lower() or "coverage" in c.lower() or "rate" in c.lower() or c == "coverage_estimate":
            pct_col = c
            break
    if not pct_col:
        pct_col = kindergarten.columns[1] if len(kindergarten.columns) > 1 else None

    # Last 4 weeks: state-level cases (keyed by state abbreviation for consistent lookup)
    from utils.state_maps import state_to_abbr
    state_cases = _state_weekly_cases(nndss) if nndss is not None and not nndss.empty else pd.DataFrame()
    state_ww = _wastewater_state_weekly(wastewater) if wastewater is not None and not wastewater.empty else pd.DataFrame()
    cases_recent_by_state = {}
    if not state_cases.empty and "state" in state_cases.columns:
        state_cases = state_cases.copy()
        state_cases["_abbr"] = state_cases["state"].astype(str).str.strip().apply(state_to_abbr)
        state_cases = state_cases.dropna(subset=["_abbr"])
        if not state_cases.empty:
            state_cases = state_cases.sort_values(["year", "week"], ascending=[False, False])
            recent_keys = state_cases[["year", "week"]].drop_duplicates().head(4)
            merged = state_cases.merge(recent_keys, on=["year", "week"])
            agg = merged.groupby("_abbr", as_index=False)["cases"].sum()
            agg["_abbr"] = agg["_abbr"].astype(str).str.upper()
            cases_recent_by_state = dict(zip(agg["_abbr"], agg["cases"]))
    ww_recent_by_state = {}
    if not state_ww.empty and "state" in state_ww.columns:
        state_ww["_abbr"] = state_ww["state"].astype(str).str.strip().str.upper()
        state_ww = state_ww.sort_values(["year", "week"], ascending=[False, False])
        recent_keys_ww = state_ww[["year", "week"]].drop_duplicates().head(4)
        merged_ww = state_ww.merge(recent_keys_ww, on=["year", "week"])
        agg_ww = merged_ww.groupby("_abbr", as_index=False)["ww_signal"].mean()
        ww_recent_by_state = dict(zip(agg_ww["_abbr"], agg_ww["ww_signal"]))

    has_ww = len(ww_recent_by_state) > 0

    # Case percentile: 0–30 (higher recent cases -> higher points)
    states_with_cases = list(cases_recent_by_state.keys())
    case_points_by_state = {}
    if states_with_cases:
        case_vals = np.array([cases_recent_by_state.get(s, 0) for s in states_with_cases])
        if len(case_vals) > 0 and (np.nanmax(case_vals) - np.nanmin(case_vals)) > 0:
            ranks = pd.Series(case_vals).rank(pct=True, method="average")
            case_points_by_state = dict(zip(states_with_cases, (ranks.values * 30).tolist()))
        else:
            case_points_by_state = {s: 0.0 for s in states_with_cases}

    # Wastewater percentile: 0–20 (higher signal -> higher points)
    ww_states = list(ww_recent_by_state.keys())
    ww_points_by_state = {}
    if has_ww and ww_states:
        ww_vals = np.array([ww_recent_by_state[s] for s in ww_states], dtype=float)
        if len(ww_vals) > 0 and (np.nanmax(ww_vals) - np.nanmin(ww_vals)) > 0:
            ranks_ww = pd.Series(ww_vals).rank(pct=True, method="average")
            ww_points_by_state = dict(zip(ww_states, (ranks_ww.values * 20).tolist()))
        else:
            ww_points_by_state = {s: 10.0 for s in ww_states}

    rows = []
    for state in kindergarten[state_col].dropna().unique():
        st = str(state).strip()
        if not st or st.upper() == "US":
            continue
        abbr = state_to_abbr(st)
        if not abbr:
            continue
        key = abbr.upper() if isinstance(abbr, str) else abbr
        sub = kindergarten[kindergarten[state_col] == state]
        cov = float(_safe_numeric(sub[pct_col]).mean()) if pct_col else 50.0
        coverage_pts = min(50.0, max(0.0, (95.0 - cov) * 2.0))
        cases_recent = float(cases_recent_by_state.get(key, 0))
        ww_recent = ww_recent_by_state.get(key)
        if ww_recent is not None and (np.isnan(ww_recent) or pd.isna(ww_recent)):
            ww_recent = None
        has_ww_state = ww_recent is not None
        case_pts = float(case_points_by_state.get(key, 0))
        ww_pts = float(ww_points_by_state.get(key, 0)) if has_ww_state else 0.0
        total_raw = coverage_pts + case_pts + ww_pts
        cap = 100.0 if has_ww_state else 80.0
        total_risk = min(cap, total_raw)
        if cases_recent == 0 and not has_ww_state:
            total_risk = min(total_risk, 60.0)
        if total_risk >= 70:
            tier = "high"
        elif total_risk >= 40:
            tier = "medium"
        else:
            tier = "low"
        rows.append({
            "state": st,
            "coverage": round(cov, 1),
            "cases_recent": round(cases_recent, 1),
            "ww_recent": round(ww_recent, 4) if ww_recent is not None else None,
            "wastewater_coverage": has_ww_state,
            "coverage_points": round(coverage_pts, 1),
            "case_points": round(case_pts, 1),
            "wastewater_points": round(ww_pts, 1) if has_ww_state else 0.0,
            "total_risk": round(total_risk, 1),
            "risk_tier": tier,
            "risk_score": round(total_risk, 1),
        })
    return pd.DataFrame(rows).sort_values("total_risk", ascending=False).reset_index(drop=True)
