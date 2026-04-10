from __future__ import annotations

"""
Fast, dashboard-first measles risk pipeline for Shiny.

Design goals
------------
- Simple and stable
- Fast enough for interactive dashboard refreshes
- Honest labeling: surveillance/risk support, not a definitive outbreak oracle
- Minimal dependencies beyond pandas/numpy/sklearn

Outputs
-------
build_dashboard_payload(...) returns a dict with:
- national_risk: dict for top-line card
- forecast: short-horizon baseline projection
- state_risk: per-state composite surveillance index
- diagnostics: data quality and model quality info
- modeling_frame: optional debugging table

Modeling choices
----------------
Stage 1 alarm model:
- Regularized logistic regression
- Predicts elevated measles activity in the next 4 weeks
- Uses recent national cases + wastewater detection/signal + seasonality

Stage 2 forecast:
- Simple recency-weighted baseline projection
- Explicitly labeled as a baseline projection, not a mechanistic forecast

State risk:
- Composite surveillance index
- Coverage + recent cases + wastewater activity
"""

import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# Configuration
# -------------------------------------------------------------------

OUTBREAK_LOOKAHEAD_WEEKS = 4
FORECAST_WEEKS = 8
TEST_WEEKS = 12
MIN_MODEL_ROWS = 30
DEFAULT_ALERT_THRESHOLD = 3  # elevated activity in next 4 weeks
EPS = 1e-9

NNDSS_CASE_COLUMN_PRIORITY = ["m2", "m1", "current_week", "Current week", "weekly_cases", "cases"]
NNDSS_EXCLUDE_CASE_COLUMNS = {
    "mmwr_year",
    "mmwr_week",
    "year",
    "week",
    "sort_order",
    "Reporting Area",
    "states",
    "label",
    "geography",
    "location1",
    "reporting_area",
}

WW_SIGNAL_COLUMN_PRIORITY = [
    "detection_frequency",
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

# CDC NWSS measles (data.cdc.gov akvg-8vrb) uses state_territory, not wwtp_jurisdiction.
WW_JURISDICTION_COLUMN_PRIORITY = [
    "wwtp_jurisdiction",
    "state_territory",
    "state",
    "jurisdiction",
    "geography",
]


# -------------------------------------------------------------------
# Small utilities
# -------------------------------------------------------------------

def _safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clip_week(x: pd.Series) -> pd.Series:
    return _safe_numeric(x).clip(lower=1, upper=53)


def _pct_rank_points(values: pd.Series, max_points: float) -> pd.Series:
    if values.empty:
        return pd.Series(dtype=float)
    vals = _safe_numeric(values)
    if vals.nunique(dropna=True) <= 1:
        return pd.Series(np.zeros(len(vals)), index=values.index, dtype=float)
    return vals.rank(pct=True, method="average") * max_points


def _sigmoid_confidence(prob: float) -> str:
    distance = abs(prob - 0.5)
    if distance >= 0.30:
        return "high"
    if distance >= 0.15:
        return "moderate"
    return "low"


def _latest_year_value(df: pd.DataFrame, value_col: str, year_candidates: List[str]) -> Optional[float]:
    if df is None or df.empty or value_col not in df.columns:
        return None
    year_col = next((c for c in year_candidates if c in df.columns), None)
    tmp = df.copy()
    tmp[value_col] = _safe_numeric(tmp[value_col])
    if year_col:
        tmp[year_col] = _safe_numeric(tmp[year_col])
        tmp = tmp.dropna(subset=[year_col, value_col]).sort_values(year_col)
        if not tmp.empty:
            latest_year = tmp[year_col].max()
            sub = tmp[tmp[year_col] == latest_year]
            if not sub.empty:
                return float(sub[value_col].mean())
    tmp = tmp.dropna(subset=[value_col])
    if tmp.empty:
        return None
    return float(tmp[value_col].mean())


# -------------------------------------------------------------------
# State normalization
# -------------------------------------------------------------------

STATE_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR",
    "CALIFORNIA": "CA", "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE",
    "FLORIDA": "FL", "GEORGIA": "GA", "HAWAII": "HI", "IDAHO": "ID",
    "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA", "KANSAS": "KS",
    "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS",
    "MISSOURI": "MO", "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV",
    "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ", "NEW MEXICO": "NM", "NEW YORK": "NY",
    "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH", "OKLAHOMA": "OK",
    "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT",
    "VERMONT": "VT", "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV",
    "WISCONSIN": "WI", "WYOMING": "WY", "DISTRICT OF COLUMBIA": "DC",
    "PUERTO RICO": "PR",
}


def state_to_abbr(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    s = str(value).strip().upper()
    if not s:
        return None
    if s in STATE_ABBR.values():
        return s
    return STATE_ABBR.get(s)


_WW_VALID_ABBR = frozenset(STATE_ABBR.values())

# Longest names first; word-boundary patterns reduce accidental in-word matches.
_WW_STATE_NAME_PATTERNS: List[Tuple[str, re.Pattern[str]]] = []
for _name in sorted(STATE_ABBR.keys(), key=len, reverse=True):
    _parts = _name.split()
    _pat = r"\b" + r"\s+".join(re.escape(p) for p in _parts) + r"\b"
    _WW_STATE_NAME_PATTERNS.append((_name, re.compile(_pat)))


def normalize_wastewater_jurisdiction_to_abbr(value: Any) -> Optional[str]:
    """
    Map raw wastewater jurisdiction strings to a 2-letter USPS-style abbreviation.

    Conservative: returns None when no state/territory in STATE_ABBR can be identified.
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    raw = str(value).strip()
    if not raw:
        return None

    su = raw.upper().replace(".", " ")
    su = " ".join(su.split())
    if len(su) == 2 and su in _WW_VALID_ABBR:
        return su

    s_norm = re.sub(r"[^A-Z0-9\s]", " ", su)
    s_upper = " ".join(s_norm.split())
    if not s_upper:
        return None

    if s_upper in STATE_ABBR:
        return STATE_ABBR[s_upper]

    tokens = s_upper.split()
    if len(tokens) == 1 and len(tokens[0]) == 2 and tokens[0] in _WW_VALID_ABBR:
        return tokens[0]

    for _name, rx in _WW_STATE_NAME_PATTERNS:
        if rx.search(s_upper):
            return STATE_ABBR[_name]

    for t in reversed(tokens):
        if len(t) == 2 and t in _WW_VALID_ABBR:
            return t

    return None


# -------------------------------------------------------------------
# NNDSS aggregation
# -------------------------------------------------------------------

def _pick_case_col(df: pd.DataFrame) -> Optional[str]:
    for c in NNDSS_CASE_COLUMN_PRIORITY:
        if c in df.columns and c not in NNDSS_EXCLUDE_CASE_COLUMNS:
            return c
    numeric_cols = [
        c for c in df.select_dtypes(include=[np.number]).columns.tolist()
        if c not in NNDSS_EXCLUDE_CASE_COLUMNS and "flag" not in c.lower()
    ]
    return numeric_cols[0] if numeric_cols else None


def get_national_weekly_cases(nndss: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    empty = pd.DataFrame(columns=["year", "week", "cases"])
    audit = {
        "source": None,
        "case_column_used": None,
        "year_min": None,
        "year_max": None,
        "latest_year_week": None,
        "n_rows_used": 0,
    }
    if nndss is None or nndss.empty:
        return empty, audit

    reporting_col = "Reporting Area" if "Reporting Area" in nndss.columns else ("states" if "states" in nndss.columns else None)
    year_col = "year" if "year" in nndss.columns else None
    week_col = "week" if "week" in nndss.columns else None
    if not reporting_col or not year_col or not week_col:
        return empty, audit

    df = nndss.copy()
    df[year_col] = _safe_numeric(df[year_col])
    df[week_col] = _clip_week(df[week_col])
    df = df.dropna(subset=[year_col, week_col])
    if df.empty:
        return empty, audit

    overall_max_year = int(df[year_col].max())
    national = df[df[reporting_col].astype(str).str.strip().str.upper() == "US RESIDENTS"].copy()

    use_national = False
    if not national.empty:
        national_max_year = int(_safe_numeric(national[year_col]).max())
        use_national = national_max_year >= overall_max_year

    if use_national:
        source_df = national
        audit["source"] = "US RESIDENTS"
    else:
        tmp = df.copy()
        tmp["state_abbr"] = tmp[reporting_col].apply(state_to_abbr)
        source_df = tmp.dropna(subset=["state_abbr"])
        audit["source"] = "summed_jurisdictions"

    if source_df.empty:
        return empty, audit

    case_col = _pick_case_col(source_df)
    if not case_col:
        return empty, audit

    source_df["cases"] = _safe_numeric(source_df[case_col]).fillna(0)
    agg = (
        source_df.groupby([year_col, week_col], as_index=False)["cases"]
        .sum()
        .rename(columns={year_col: "year", week_col: "week"})
        .sort_values(["year", "week"])
        .reset_index(drop=True)
    )
    if agg.empty:
        return empty, audit

    audit["case_column_used"] = case_col
    audit["year_min"] = int(agg["year"].min())
    audit["year_max"] = int(agg["year"].max())
    audit["latest_year_week"] = (int(agg.iloc[-1]["year"]), int(agg.iloc[-1]["week"]))
    audit["n_rows_used"] = len(source_df)
    return agg, audit


def get_state_weekly_cases(nndss: pd.DataFrame) -> pd.DataFrame:
    empty = pd.DataFrame(columns=["state", "year", "week", "cases"])
    if nndss is None or nndss.empty:
        return empty

    reporting_col = "Reporting Area" if "Reporting Area" in nndss.columns else ("states" if "states" in nndss.columns else None)
    year_col = "year" if "year" in nndss.columns else None
    week_col = "week" if "week" in nndss.columns else None
    if not reporting_col or not year_col or not week_col:
        return empty

    df = nndss.copy()
    df["state"] = df[reporting_col].apply(state_to_abbr)
    df[year_col] = _safe_numeric(df[year_col])
    df[week_col] = _clip_week(df[week_col])
    case_col = _pick_case_col(df)
    if not case_col:
        return empty

    df["cases"] = _safe_numeric(df[case_col]).fillna(0)
    df = df.dropna(subset=["state", year_col, week_col])
    if df.empty:
        return empty

    return (
        df.groupby(["state", year_col, week_col], as_index=False)["cases"]
        .sum()
        .rename(columns={year_col: "year", week_col: "week"})
    )


# -------------------------------------------------------------------
# Wastewater aggregation
# -------------------------------------------------------------------

def detect_ww_signal_column(ww: pd.DataFrame) -> Optional[str]:
    if ww is None or ww.empty:
        return None
    for c in WW_SIGNAL_COLUMN_PRIORITY:
        if c in ww.columns:
            return c
    return None


def compute_ww_detection_frequency(ww: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    empty = pd.DataFrame(columns=["year", "week", "total_sites", "positive_sites", "detection_frequency"])
    audit = {
        "signal_mode": None,
        "n_rows_raw": 0,
        "n_rows_after_filters": 0,
        "n_unique_sites": 0,
        "all_zero": True,
        "weeks_min": None,
        "weeks_max": None,
    }
    if ww is None or ww.empty:
        return empty, audit

    audit["n_rows_raw"] = len(ww)
    df = ww.copy()

    year_col = next((c for c in ["year", "mmwr_year", "reporting_year"] if c in df.columns), None)
    week_col = next((c for c in ["week", "mmwr_week", "reporting_week"] if c in df.columns), None)
    site_col = next((c for c in ["sewershed_id", "sample_id", "wwtp_id", "site_id"] if c in df.columns), None)
    if not year_col or not week_col or not site_col:
        return empty, audit

    if "pcr_target" in df.columns:
        pt = df["pcr_target"].astype(str).str.upper().str.strip()
        df = df[pt.str.contains("MEASLES|MEV", na=False)].copy()

    signal_col = None
    if "pcr_target_avg_conc" in df.columns:
        signal_col = "pcr_target_avg_conc"
        audit["signal_mode"] = "concentration"
    else:
        signal_col = detect_ww_signal_column(df)
        audit["signal_mode"] = f"proxy:{signal_col}" if signal_col else None

    if not signal_col:
        return empty, audit

    df["year"] = _safe_numeric(df[year_col])
    df["week"] = _clip_week(df[week_col])
    df["site_id"] = df[site_col].astype(str).str.strip()
    df["signal"] = _safe_numeric(df[signal_col])

    if "ntc_amplify" in df.columns:
        df = df[df["ntc_amplify"].astype(str).str.lower().str.strip() == "no"]
    if "inhibition_detect" in df.columns and "inhibition_adjust" in df.columns:
        ok = (
            df["inhibition_detect"].astype(str).str.lower().str.strip().eq("no") |
            df["inhibition_adjust"].astype(str).str.lower().str.strip().eq("yes")
        )
        df = df[ok]

    df = df.dropna(subset=["year", "week", "site_id"])
    if df.empty:
        return empty, audit

    audit["n_rows_after_filters"] = len(df)
    df["is_detected"] = (df["signal"].fillna(0) > 0).astype(int)

    site_week = (
        df.groupby(["site_id", "year", "week"], as_index=False)["is_detected"]
        .max()
    )
    weekly = site_week.groupby(["year", "week"], as_index=False).agg(
        total_sites=("site_id", "nunique"),
        positive_sites=("is_detected", "sum"),
    )
    weekly["detection_frequency"] = weekly["positive_sites"] / weekly["total_sites"].clip(lower=1)
    weekly = weekly.sort_values(["year", "week"]).reset_index(drop=True)

    audit["n_unique_sites"] = int(site_week["site_id"].nunique())
    audit["all_zero"] = bool((weekly["detection_frequency"] <= 0).all()) if not weekly.empty else True
    if not weekly.empty:
        audit["weeks_min"] = (int(weekly.iloc[0]["year"]), int(weekly.iloc[0]["week"]))
        audit["weeks_max"] = (int(weekly.iloc[-1]["year"]), int(weekly.iloc[-1]["week"]))
    return weekly, audit


def get_national_ww_signal(ww: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    empty = pd.DataFrame(columns=["year", "week", "ww_signal"])
    detection_weekly, det_audit = compute_ww_detection_frequency(ww)
    if not detection_weekly.empty:
        out = detection_weekly[["year", "week", "detection_frequency"]].rename(
            columns={"detection_frequency": "ww_signal"}
        )
        return out, {"mode": "detection_frequency", **det_audit}

    if ww is None or ww.empty:
        return empty, {"mode": None}

    df = ww.copy()
    year_col = next((c for c in ["year", "mmwr_year", "reporting_year"] if c in df.columns), None)
    week_col = next((c for c in ["week", "mmwr_week", "reporting_week"] if c in df.columns), None)
    signal_col = detect_ww_signal_column(df)
    if not year_col or not week_col or not signal_col:
        return empty, {"mode": None}

    df["year"] = _safe_numeric(df[year_col])
    df["week"] = _clip_week(df[week_col])
    df["ww_signal"] = _safe_numeric(df[signal_col])
    df = df.dropna(subset=["year", "week"])
    if df.empty:
        return empty, {"mode": None}

    agg = df.groupby(["year", "week"], as_index=False)["ww_signal"].mean().sort_values(["year", "week"])
    return agg, {"mode": f"mean:{signal_col}"}


def get_state_ww_signal(ww: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    empty = pd.DataFrame(columns=["state", "year", "week", "ww_signal"])
    audit: Dict[str, Any] = {
        "status": "empty_input",
        "jurisdiction_column": None,
        "non_null_jurisdiction_rows": 0,
        "unique_raw_values": 0,
        "first_100_unique_raw": [],
        "rows_mapped_to_state": 0,
        "rows_jurisdiction_but_unmapped": 0,
        "example_unmapped": [],
        "rows_after_state_year_week_dropna": 0,
        "state_week_row_groups": 0,
    }
    if ww is None or ww.empty:
        return empty, audit

    df = ww.copy()
    state_col = next((c for c in WW_JURISDICTION_COLUMN_PRIORITY if c in df.columns), None)
    year_col = next((c for c in ["year", "mmwr_year", "reporting_year"] if c in df.columns), None)
    week_col = next((c for c in ["week", "mmwr_week", "reporting_week"] if c in df.columns), None)
    signal_col = detect_ww_signal_column(df)
    if not state_col or not year_col or not week_col or not signal_col:
        audit["status"] = "missing_columns"
        audit["jurisdiction_column"] = state_col
        audit["year_column"] = year_col
        audit["week_column"] = week_col
        audit["signal_column"] = signal_col
        return empty, audit

    raw_series = df[state_col]
    n_jur_nonnull = int(raw_series.notna().sum())
    uniq = raw_series.dropna().astype(str).unique()
    uniq_sample = list(uniq[:100])
    audit["jurisdiction_column"] = state_col
    audit["non_null_jurisdiction_rows"] = n_jur_nonnull
    audit["unique_raw_values"] = int(len(uniq))
    audit["first_100_unique_raw"] = uniq_sample
    logger.info(
        "wastewater state mapping: column=%s non_null_jurisdiction=%s unique_values=%s first_100=%s",
        state_col,
        n_jur_nonnull,
        len(uniq),
        uniq_sample,
    )

    df["state"] = raw_series.map(normalize_wastewater_jurisdiction_to_abbr)
    n_mapped = int(df["state"].notna().sum())
    n_dropped_mapping = n_jur_nonnull - n_mapped
    bad_mask = df["state"].isna() & raw_series.notna()
    unmapped_examples = (
        raw_series.loc[bad_mask].astype(str).drop_duplicates().head(25).tolist()
    )
    audit["rows_mapped_to_state"] = n_mapped
    audit["rows_jurisdiction_but_unmapped"] = n_dropped_mapping
    audit["example_unmapped"] = unmapped_examples
    logger.info(
        "wastewater state mapping: rows_mapped_to_state=%s rows_jurisdiction_but_unmapped=%s example_unmapped=%s",
        n_mapped,
        n_dropped_mapping,
        unmapped_examples,
    )

    df["year"] = _safe_numeric(df[year_col])
    df["week"] = _clip_week(df[week_col])
    df["ww_signal"] = _safe_numeric(df[signal_col])
    df = df.dropna(subset=["state", "year", "week"])
    n_after_drop = len(df)
    audit["rows_after_state_year_week_dropna"] = n_after_drop
    logger.info(
        "wastewater state aggregation: rows_after_state_year_week_dropna=%s (signal/year/week nulls removed from mapped rows)",
        n_after_drop,
    )
    if df.empty:
        audit["status"] = "no_rows_after_state_year_week_coerce"
        return empty, audit

    agg = (
        df.groupby(["state", "year", "week"], as_index=False)["ww_signal"]
        .mean()
        .sort_values(["state", "year", "week"])
        .reset_index(drop=True)
    )
    audit["status"] = "ok"
    audit["state_week_row_groups"] = int(len(agg))
    return agg, audit


# -------------------------------------------------------------------
# Feature engineering
# -------------------------------------------------------------------

def build_modeling_frame(
    nndss: pd.DataFrame,
    wastewater: pd.DataFrame,
    kindergarten: pd.DataFrame,
    outbreak_threshold: int = DEFAULT_ALERT_THRESHOLD,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cases_df, cases_audit = get_national_weekly_cases(nndss)
    ww_df, ww_audit = get_national_ww_signal(wastewater)

    audit = {
        "nndss": cases_audit,
        "wastewater": ww_audit,
        "kindergarten_latest_coverage": None,
        "rows_modeling": 0,
    }

    if cases_df.empty:
        return pd.DataFrame(), audit

    df = cases_df.copy().sort_values(["year", "week"]).reset_index(drop=True)
    df["row_id"] = np.arange(len(df))
    df["week_of_year"] = df["week"].clip(1, 53)
    df["sin_week"] = np.sin(2 * np.pi * df["week_of_year"] / 52.0)
    df["cos_week"] = np.cos(2 * np.pi * df["week_of_year"] / 52.0)

    df["cases_next4"] = sum(df["cases"].shift(-k) for k in range(1, OUTBREAK_LOOKAHEAD_WEEKS + 1))
    df["elevated_next4w"] = (df["cases_next4"] >= outbreak_threshold).astype(float)

    for lag in [1, 2, 3, 4, 8, 12]:
        df[f"cases_lag{lag}"] = df["cases"].shift(lag)

    df["cases_roll4"] = df["cases"].shift(1).rolling(4, min_periods=1).mean()
    df["cases_roll8"] = df["cases"].shift(1).rolling(8, min_periods=1).mean()

    if not ww_df.empty:
        tmp_ww = ww_df.copy()
        tmp_ww["year"] = _safe_numeric(tmp_ww["year"])
        tmp_ww["week"] = _clip_week(tmp_ww["week"])
        df = df.merge(tmp_ww, on=["year", "week"], how="left")
    else:
        df["ww_signal"] = np.nan

    df["ww_missing"] = df["ww_signal"].isna().astype(int)
    df["ww_signal_ffill"] = df["ww_signal"].ffill()
    df["ww_signal_filled"] = df["ww_signal_ffill"].fillna(df["ww_signal"].median() if df["ww_signal"].notna().any() else 0.0)

    for lag in [1, 2, 3, 4, 8]:
        df[f"ww_lag{lag}"] = df["ww_signal_filled"].shift(lag)
    df["ww_roll4"] = df["ww_signal_filled"].shift(1).rolling(4, min_periods=1).mean()

    kg_col = None
    for c in ["mmr_covered_pct", "coverage", "coverage_estimate", "coverage_pct"]:
        if kindergarten is not None and not kindergarten.empty and c in kindergarten.columns:
            kg_col = c
            break

    kg_latest = _latest_year_value(
        kindergarten,
        kg_col,
        ["year", "school_year", "survey_year", "_year_derived"],
    ) if kg_col else None
    audit["kindergarten_latest_coverage"] = kg_latest
    df["kg_coverage"] = kg_latest if kg_latest is not None else np.nan
    df["kg_coverage"] = df["kg_coverage"].fillna(df["kg_coverage"].median() if df["kg_coverage"].notna().any() else 95.0)
    df["coverage_gap"] = (95.0 - df["kg_coverage"]).clip(lower=0)

    feature_cols = [
        "cases_lag1", "cases_lag2", "cases_lag3", "cases_lag4", "cases_lag8", "cases_lag12",
        "cases_roll4", "cases_roll8",
        "ww_lag1", "ww_lag2", "ww_lag3", "ww_lag4", "ww_lag8", "ww_roll4",
        "ww_missing",
        "sin_week", "cos_week",
        "coverage_gap",
    ]

    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ["elevated_next4w"]).reset_index(drop=True)
    audit["rows_modeling"] = len(df)
    return df, audit


# -------------------------------------------------------------------
# Stage 1 alarm model
# -------------------------------------------------------------------

@dataclass
class AlarmModelResult:
    model: Any
    scaler: Any
    features: List[str]
    auc: float
    brier: float
    n_train: int
    n_test: int
    status: str


def fit_alarm_model(modeling_df: pd.DataFrame) -> AlarmModelResult:
    try:
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import brier_score_loss, roc_auc_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except Exception:
        return AlarmModelResult(None, None, [], 0.5, 0.25, 0, 0, "sklearn_unavailable")

    if modeling_df is None or modeling_df.empty or len(modeling_df) < MIN_MODEL_ROWS:
        return AlarmModelResult(None, None, [], 0.5, 0.25, 0, 0, "insufficient_rows")

    feature_cols = [
        c for c in modeling_df.columns
        if c.startswith("cases_") or c.startswith("ww_") or c in {"sin_week", "cos_week", "coverage_gap"}
    ]
    X = modeling_df[feature_cols].copy()
    y = modeling_df["elevated_next4w"].astype(int)

    if y.nunique() < 2:
        return AlarmModelResult(None, None, feature_cols, 0.5, 0.25, 0, 0, "single_class_target")

    n = len(modeling_df)
    if n <= TEST_WEEKS + 8:
        split = int(max(0.7 * n, 1))
    else:
        split = n - TEST_WEEKS

    X_train = X.iloc[:split]
    X_test = X.iloc[split:]
    y_train = y.iloc[:split]
    y_test = y.iloc[split:]

    if len(X_test) < 4 or y_train.nunique() < 2:
        return AlarmModelResult(None, None, feature_cols, 0.5, 0.25, len(X_train), len(X_test), "weak_split")

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(C=0.5, max_iter=2000, class_weight="balanced", random_state=42)),
    ])
    pipe.fit(X_train, y_train)
    proba = pipe.predict_proba(X_test)[:, 1]

    try:
        auc = float(roc_auc_score(y_test, proba))
    except Exception:
        auc = 0.5

    try:
        brier = float(brier_score_loss(y_test, proba))
    except Exception:
        brier = 0.25

    return AlarmModelResult(pipe, None, feature_cols, auc, brier, len(X_train), len(X_test), "ok")


def predict_alarm_probability(model_result: AlarmModelResult, modeling_df: pd.DataFrame) -> float:
    if model_result is None or model_result.model is None or modeling_df is None or modeling_df.empty:
        return 0.5
    x_latest = modeling_df[model_result.features].iloc[[-1]].copy()
    try:
        return float(model_result.model.predict_proba(x_latest)[0, 1])
    except Exception:
        return 0.5


# -------------------------------------------------------------------
# Stage 2 baseline projection
# -------------------------------------------------------------------

def build_baseline_projection(cases_df: pd.DataFrame, weeks: int = FORECAST_WEEKS) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    empty = pd.DataFrame(columns=["week_ahead", "forecast", "lower", "upper"])
    audit = {"status": "unavailable", "method": "recency_weighted_baseline"}
    if cases_df is None or cases_df.empty or len(cases_df) < 8:
        return empty, audit

    recent = _safe_numeric(cases_df["cases"]).tail(12).fillna(0)
    if recent.empty:
        return empty, audit

    weights = np.linspace(1.0, 2.0, len(recent))
    baseline = float(np.average(recent, weights=weights))
    vol = float(recent.std(ddof=0)) if len(recent) > 1 else 0.0
    lower = max(0.0, baseline - 1.28 * vol)
    upper = max(lower, baseline + 1.28 * vol)

    rows = []
    for i in range(1, weeks + 1):
        rows.append({
            "week_ahead": i,
            "forecast": round(baseline, 1),
            "lower": round(lower, 1),
            "upper": round(upper, 1),
        })

    audit["status"] = "ok"
    audit["baseline_mean"] = round(baseline, 2)
    audit["recent_std"] = round(vol, 2)
    return pd.DataFrame(rows), audit


# -------------------------------------------------------------------
# State composite risk index
# -------------------------------------------------------------------

def build_state_risk_index(
    kindergarten: pd.DataFrame,
    nndss: pd.DataFrame,
    wastewater: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    cols = [
        "state", "coverage", "coverage_points",
        "cases_recent", "case_points",
        "ww_recent", "wastewater_points",
        "total_risk", "risk_tier", "wastewater_available",
    ]
    ww_diag: Dict[str, Any] = {"status": "skipped_no_kindergarten"}

    if kindergarten is None or kindergarten.empty:
        return pd.DataFrame(columns=cols), {"wastewater_state": ww_diag}

    state_col = next((c for c in ["state", "State", "jurisdiction", "geography", "location1"] if c in kindergarten.columns), None)
    cov_col = next((c for c in ["mmr_covered_pct", "coverage", "coverage_estimate", "coverage_pct"] if c in kindergarten.columns), None)
    if not state_col or not cov_col:
        return pd.DataFrame(columns=cols), {"wastewater_state": {"status": "skipped_missing_kindergarten_columns"}}

    kg = kindergarten.copy()
    kg["state"] = kg[state_col].apply(state_to_abbr)
    kg["coverage"] = _safe_numeric(kg[cov_col])
    kg = kg.dropna(subset=["state", "coverage"])
    if kg.empty:
        return pd.DataFrame(columns=cols), {"wastewater_state": {"status": "skipped_empty_kindergarten"}}

    state_cases = get_state_weekly_cases(nndss)
    state_ww, ww_diag = get_state_ww_signal(wastewater)

    cases_recent = {}
    if not state_cases.empty:
        tmp = state_cases.sort_values(["year", "week"])
        last_rows = tmp[["year", "week"]].drop_duplicates().tail(4)
        sub = tmp.merge(last_rows, on=["year", "week"], how="inner")
        cases_recent = sub.groupby("state")["cases"].sum().to_dict()

    ww_recent = {}
    if not state_ww.empty:
        tmp = state_ww.sort_values(["year", "week"])
        last_rows = tmp[["year", "week"]].drop_duplicates().tail(4)
        sub = tmp.merge(last_rows, on=["year", "week"], how="inner")
        ww_recent = sub.groupby("state")["ww_signal"].mean().to_dict()

    coverage_df = kg.groupby("state", as_index=False)["coverage"].mean()
    coverage_df["coverage_points"] = ((95.0 - coverage_df["coverage"]).clip(lower=0) * 2.0).clip(upper=50)
    coverage_df["cases_recent"] = coverage_df["state"].map(cases_recent).fillna(0.0)
    coverage_df["ww_recent"] = coverage_df["state"].map(ww_recent)
    coverage_df["case_points"] = _pct_rank_points(coverage_df["cases_recent"], 30).fillna(0.0)
    coverage_df["wastewater_points"] = _pct_rank_points(coverage_df["ww_recent"], 20).fillna(0.0)
    coverage_df["wastewater_available"] = coverage_df["ww_recent"].notna()

    coverage_df["total_risk"] = (
        coverage_df["coverage_points"] +
        coverage_df["case_points"] +
        coverage_df["wastewater_points"]
    )

    def tier(score: float) -> str:
        if score >= 70:
            return "high"
        if score >= 40:
            return "medium"
        return "low"

    coverage_df["risk_tier"] = coverage_df["total_risk"].apply(tier)
    out_df = coverage_df[cols].sort_values("total_risk", ascending=False).reset_index(drop=True)
    return out_df, {"wastewater_state": ww_diag}


# -------------------------------------------------------------------
# Dashboard payload
# -------------------------------------------------------------------

def build_dashboard_payload(
    historical: Optional[pd.DataFrame],
    nndss: pd.DataFrame,
    wastewater: pd.DataFrame,
    kindergarten: pd.DataFrame,
    outbreak_threshold: int = DEFAULT_ALERT_THRESHOLD,
    include_debug_tables: bool = False,
) -> Dict[str, Any]:
    # Defensive normalization for loader compatibility
    if kindergarten is not None and not kindergarten.empty:
        kindergarten = kindergarten.copy()
        if "coverage" not in kindergarten.columns and "coverage_pct" in kindergarten.columns:
            kindergarten["coverage"] = pd.to_numeric(kindergarten["coverage_pct"], errors="coerce")
        if "year" not in kindergarten.columns and "_year_derived" in kindergarten.columns:
            kindergarten["year"] = pd.to_numeric(kindergarten["_year_derived"], errors="coerce")

    cases_df, nndss_audit = get_national_weekly_cases(nndss)
    modeling_df, modeling_audit = build_modeling_frame(
        nndss=nndss,
        wastewater=wastewater,
        kindergarten=kindergarten,
        outbreak_threshold=outbreak_threshold,
    )
    model_result = fit_alarm_model(modeling_df)
    alarm_probability = predict_alarm_probability(model_result, modeling_df)
    projection_df, projection_audit = build_baseline_projection(cases_df)
    state_risk_df, state_aux = build_state_risk_index(kindergarten, nndss, wastewater)

    latest_cases = float(cases_df.iloc[-1]["cases"]) if not cases_df.empty else 0.0
    latest_week = (
        {"year": int(cases_df.iloc[-1]["year"]), "week": int(cases_df.iloc[-1]["week"])}
        if not cases_df.empty else None
    )

    national_risk = {
        "alarm_probability": round(alarm_probability, 3),
        "alert_threshold_next4w_cases": outbreak_threshold,
        "signal_level": (
            "high" if alarm_probability >= 0.70 else
            "medium" if alarm_probability >= 0.40 else
            "low"
        ),
        "confidence": _sigmoid_confidence(alarm_probability),
        "latest_cases": latest_cases,
        "latest_week": latest_week,
        "model_status": model_result.status,
        "auc": round(model_result.auc, 3),
        "brier": round(model_result.brier, 3),
        "n_train": model_result.n_train,
        "n_test": model_result.n_test,
        "label": "Probability of elevated national measles activity in the next 4 weeks",
    }

    diagnostics = {
        "nndss": nndss_audit,
        "modeling": modeling_audit,
        "projection": projection_audit,
        "state_rows": int(len(state_risk_df)),
        "wastewater_state": state_aux.get("wastewater_state", {}),
        "notes": [
            "Stage 1 is a regularized logistic alarm model.",
            "Stage 2 is a recency-weighted baseline projection for dashboard speed.",
            "State map is a composite surveillance index, not a causal model.",
            "Missing wastewater is handled with an explicit missingness flag and forward-fill for recent continuity.",
        ],
    }

    payload: Dict[str, Any] = {
        "national_risk": national_risk,
        "forecast": projection_df,
        "state_risk": state_risk_df,
        "diagnostics": diagnostics,
    }

    if include_debug_tables:
        payload["modeling_frame"] = modeling_df
        payload["national_cases"] = cases_df

    return payload


# -------------------------------------------------------------------
# Shiny helpers
# -------------------------------------------------------------------

def payload_to_value_boxes(payload: Dict[str, Any]) -> Dict[str, Any]:
    risk = payload["national_risk"]
    fc = payload["forecast"]
    next4_baseline = float(fc["forecast"].head(4).sum()) if isinstance(fc, pd.DataFrame) and not fc.empty else None
    return {
        "alarm_probability_pct": round(100 * risk["alarm_probability"], 1),
        "signal_level": risk["signal_level"],
        "confidence": risk["confidence"],
        "latest_cases": risk["latest_cases"],
        "next4w_baseline_cases": round(next4_baseline, 1) if next4_baseline is not None else None,
        "model_auc": risk["auc"],
    }


def payload_to_plot_df(payload: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
    return {
        "forecast": payload.get("forecast", pd.DataFrame()).copy(),
        "state_risk": payload.get("state_risk", pd.DataFrame()).copy(),
        "modeling_frame": payload.get("modeling_frame", pd.DataFrame()).copy(),
    }


# -------------------------------------------------------------------
# Example usage
# -------------------------------------------------------------------

# payload = build_dashboard_payload(
#     historical=historical_df,
#     nndss=nndss_df,
#     wastewater=wastewater_df,
#     kindergarten=kindergarten_df,
#     outbreak_threshold=3,
#     include_debug_tables=True,
# )
#
# boxes = payload_to_value_boxes(payload)
# forecast_df = payload["forecast"]
# state_risk_df = payload["state_risk"]
# diagnostics = payload["diagnostics"]
