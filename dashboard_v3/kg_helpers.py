"""Kindergarten coverage: year options and filtered frame (logic preserved from Streamlit)."""
from __future__ import annotations

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd


def prepare_kg_years(kg: pd.DataFrame) -> Tuple[pd.DataFrame, list, Optional[str]]:
    """
    Returns (kg_full with helper columns, year_options, year_source).
    year_options empty means all years in one frame (no filter).
    """
    if kg is None or kg.empty:
        return pd.DataFrame(), [], None

    kg_full = kg.copy()
    year_options: list = []
    year_source = None

    if "_year_derived" in kg_full.columns and "_year_source" in kg_full.columns:
        year_source = kg_full["_year_source"].iloc[0] if kg_full["_year_source"].notna().any() else None
        valid = (
            kg_full["_year_derived"].notna()
            & (pd.to_numeric(kg_full["_year_derived"], errors="coerce") >= 2000)
            & (pd.to_numeric(kg_full["_year_derived"], errors="coerce") < 2100)
        )
        year_options = sorted(
            pd.to_numeric(kg_full.loc[valid, "_year_derived"], errors="coerce")
            .dropna()
            .astype(int)
            .unique()
            .tolist()
        )

    if not year_options:
        year_col = next(
            (c for c in ["school_year", "reporting_year", "year", "school year", "coverage_school_year"] if c in kg.columns),
            None,
        )
        if not year_col:
            year_col = next((c for c in kg.columns if "year" in c.lower() or "school" in c.lower()), None)
        if year_col:
            raw = kg_full[year_col].astype(str).str.strip()
            kg_full["_year"] = pd.to_numeric(raw, errors="coerce")
            valid = kg_full["_year"].notna() & (kg_full["_year"] >= 2000) & (kg_full["_year"] < 2100)
            kg_full["_year_key"] = np.nan
            kg_full.loc[valid, "_year_key"] = kg_full.loc[valid, "_year"].astype(int)
            year_options = sorted(kg_full["_year_key"].dropna().unique().astype(int).tolist())
            year_source = year_col

    return kg_full, year_options, year_source


def filter_kg_by_year(kg_full: pd.DataFrame, year_options: list, selected_year: Optional[int]) -> pd.DataFrame:
    """Subset kg to selected school year; if no year column, return full frame."""
    if kg_full is None or kg_full.empty:
        return pd.DataFrame()
    if not year_options:
        return kg_full
    y = selected_year if selected_year is not None else year_options[-1]
    if "_year_derived" in kg_full.columns:
        return kg_full[pd.to_numeric(kg_full["_year_derived"], errors="coerce") == y].copy()
    if "_year_key" in kg_full.columns:
        return kg_full[kg_full["_year_key"] == y].copy()
    return kg_full


def kg_state_pct_columns(kg: pd.DataFrame) -> Tuple[str, Optional[str]]:
    state_col = next(
        (c for c in ["state", "State", "jurisdiction", "geography", "location1"] if c in kg.columns),
        kg.columns[0],
    )
    pct_col = next((c for c in kg.columns if "pct" in c.lower() or "coverage" in c.lower()), None)
    if pct_col is None and len(kg.columns) > 1:
        pct_col = kg.columns[1]
    return state_col, pct_col
