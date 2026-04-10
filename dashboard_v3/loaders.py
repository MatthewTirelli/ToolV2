"""
Data loaders for the Predictive Measles Risk Dashboard.
Loads historical CSV and fetches kindergarten, wastewater, and NNDSS from CDC Socrata APIs.
Per-source error handling; returns load status and optional cache.
"""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any, Optional, Tuple

import pandas as pd
import requests

from utils.logging_config import get_logger

logger = get_logger("loaders")

# Paths: assume run from project root (e.g. streamlit run dashboard/app.py)
DASHBOARD_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = DASHBOARD_DIR.parent
HISTORICAL_CSV_PATHS = [
    PROJECT_ROOT / "Shiny App V1" / "measles_annual_1985.csv",
    PROJECT_ROOT / "measles_annual_1985.csv",
    DASHBOARD_DIR / "measles_annual_1985.csv",
]

# CDC Socrata view IDs and defaults
KG_VIEW_ID = "ijqb-a7ye"
KG_WHERE = "vaccine in ('MMR', 'MMR (PAC)')"
WW_VIEW_ID = "akvg-8vrb"
# Wastewater: do not hard-filter pcr_target until we audit; see load_wastewater()
NNDSS_VIEW_ID = "x9gk-5huc"
NNDSS_WHERE = "label in ('Measles, Indigenous', 'Measles, Imported')"
NNDSS_ORDER = "year DESC, week DESC"  # get latest data first
API_LIMIT = 100000  # Request more rows so we get all years (e.g. through 2025–2026)

# Simple in-memory cache: (data, timestamp); TTL seconds
_cache: dict[str, tuple[Any, float]] = {}
CACHE_TTL = 3600  # 1 hour


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv(DASHBOARD_DIR / ".env")
    except ImportError:
        pass


def _get_token() -> Optional[str]:
    _load_env()
    return (os.environ.get("SOCRATA_APP_TOKEN") or "").strip() or None


def _soda3_post(
    token: str, view_id: str, where_clause: Optional[str], limit: int = API_LIMIT, order_clause: Optional[str] = None
) -> list:
    """Single-page fetch. Use _soda3_post_all for all available rows."""
    url = f"https://data.cdc.gov/api/v3/views/{view_id}/query.json"
    soql = "SELECT *" + (f" WHERE {where_clause}" if where_clause else "")
    if order_clause:
        soql += f" ORDER BY {order_clause}"
    payload = {"query": soql, "page": {"pageNumber": 1, "pageSize": limit}}
    headers = {"X-App-Token": token, "Content-Type": "application/json"}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    if isinstance(data, dict) and "data" in data:
        return data["data"]
    if isinstance(data, list):
        return data
    return []


def _soda3_post_all(
    token: str, view_id: str, where_clause: Optional[str], order_clause: Optional[str] = None, page_size: int = 50000
) -> list:
    """Fetch all pages so we get all available data (no time-frame limit)."""
    all_rows = []
    page = 1
    while True:
        url = f"https://data.cdc.gov/api/v3/views/{view_id}/query.json"
        soql = "SELECT *" + (f" WHERE {where_clause}" if where_clause else "")
        if order_clause:
            soql += f" ORDER BY {order_clause}"
        payload = {"query": soql, "page": {"pageNumber": page, "pageSize": page_size}}
        headers = {"X-App-Token": token, "Content-Type": "application/json"}
        r = requests.post(url, headers=headers, json=payload, timeout=90)
        r.raise_for_status()
        data = r.json()
        rows = data.get("data", []) if isinstance(data, dict) else (data if isinstance(data, list) else [])
        if not rows:
            break
        all_rows.extend(rows)
        # Log first page's year range to verify ordering (NNDSS: we want year DESC so page 1 has 2026)
        if page == 1 and rows and view_id == NNDSS_VIEW_ID:
            try:
                first_yr = next((r.get("year") for r in rows if r.get("year") is not None), None)
                last_yr = next((r.get("year") for r in reversed(rows) if r.get("year") is not None), None)
                if first_yr is not None or last_yr is not None:
                    logger.info("NNDSS page 1 year range: first= %s last= %s (expect newest first if ORDER BY applied)", first_yr, last_yr)
            except Exception:
                pass
        if len(rows) < page_size:
            break
        page += 1
    logger.info("_soda3_post_all view=%s pages=%s total_rows=%s", view_id, page, len(all_rows))
    return all_rows


def load_historical(cache_key: str = "historical", use_cache: bool = True) -> Tuple[pd.DataFrame, str]:
    """Load historical national measles annual data from CSV. Returns (df, status)."""
    if use_cache and cache_key in _cache:
        data, ts = _cache[cache_key]
        if time.time() - ts < CACHE_TTL:
            return data, "ok"
    status = "ok"
    df = pd.DataFrame()
    for path in HISTORICAL_CSV_PATHS:
        try:
            if path.exists():
                df = pd.read_csv(path)
                logger.info("Loaded historical CSV rows=%s path=%s", len(df), path)
                if use_cache:
                    _cache[cache_key] = (df.copy(), time.time())
                return df, status
        except Exception as e:
            logger.warning("historical load failed path=%s reason=%s", path, e)
            status = "fail"
    if df.empty:
        logger.warning("historical CSV not found at any path")
        status = "fail"
    return df, status


def load_kindergarten(use_cache: bool = True) -> Tuple[pd.DataFrame, str]:
    """Fetch kindergarten MMR coverage; clean to MMR/MMR (PAC) only. Returns (df, status)."""
    cache_key = "kindergarten"
    if use_cache and cache_key in _cache:
        data, ts = _cache[cache_key]
        if time.time() - ts < CACHE_TTL:
            return data, "ok"
    token = _get_token()
    if not token:
        logger.error("kindergarten fetch skipped: SOCRATA_APP_TOKEN not set")
        return pd.DataFrame(), "fail"
    try:
        raw = _soda3_post(token, KG_VIEW_ID, KG_WHERE)
        df = pd.DataFrame(raw)
        if df.empty:
            return df, "ok"
        if "vaccine" in df.columns:
            mask = df["vaccine"].astype(str).str.strip().isin({"MMR", "MMR (PAC)"})
            df = df.loc[mask].copy()
        if not df.empty:
            if "geography" in df.columns and "jurisdiction" not in df.columns:
                df = df.rename(columns={"geography": "jurisdiction"})
            if "coverage_estimate" in df.columns:
                df = df.rename(columns={"coverage_estimate": "coverage_pct"})
            # Log which year column exists (for by-year filter in app)
            year_candidates = ["school_year", "year", "season", "report_year", "academic_year", "end_year", "coverage_school_year", "school year"]
            found = [c for c in year_candidates if c in df.columns]
            logger.info("kindergarten full column list: %s", list(df.columns))
            logger.info("kindergarten sample rows (head 3): %s", df.head(3).to_dict("records"))
            logger.info("kindergarten year column(s) present: %s (use for by-year filter)", found if found else "none")
            if not found:
                # Derive _year_derived: regex 20xx from any column, or single-year constant
                def _first_20xx(val):
                    m = re.search(r"20\d{2}", str(val))
                    return int(m.group()) if m else None
                derived = df.apply(lambda row: next((_first_20xx(row[c]) for c in df.columns if _first_20xx(row[c]) is not None), None), axis=1)
                years_found = set(derived.dropna().astype(int).unique().tolist())
                years_found = {y for y in years_found if 2000 <= y < 2100}
                if len(years_found) == 1:
                    (single_year,) = years_found
                    df["_year_derived"] = single_year
                    df["_year_source"] = "single_year"
                    logger.info("kindergarten: dataset contains a single year (derived): %s", single_year)
                elif years_found:
                    df["_year_derived"] = derived
                    df["_year_source"] = "regex_20xx"
                    logger.info("kindergarten: year derived from regex 20xx; unique years: %s", sorted(years_found))
                else:
                    df["_year_derived"] = pd.NA
                    df["_year_source"] = "none"
                    logger.warning("kindergarten: no year-like values (20xx) found; year dropdown will be empty")
            else:
                df["_year_derived"] = pd.to_numeric(df[found[0]].astype(str).str.strip(), errors="coerce")
                df["_year_source"] = found[0]
        logger.info("Loaded kindergarten rows=%s", len(df))
        if use_cache:
            _cache[cache_key] = (df.copy(), time.time())
        return df, "ok"
    except requests.RequestException as e:
        logger.warning("kindergarten fetch failed reason=%s", e)
        return pd.DataFrame(), "fail"
    except Exception as e:
        logger.exception("kindergarten load error")
        return pd.DataFrame(), "fail"


def _wastewater_audit_pcr_target(token: str) -> Tuple[Optional[str], list]:
    """Fetch a small sample of wastewater WITHOUT pcr_target filter; return (chosen_filter_value, list of unique pcr_target)."""
    sample = _soda3_post(token, WW_VIEW_ID, where_clause=None, limit=3000)
    if not sample:
        return None, []
    df = pd.DataFrame(sample)
    logger.info("wastewater audit columns: %s", list(df.columns))
    if "pcr_target" not in df.columns:
        logger.warning("wastewater audit: no pcr_target column; columns present: %s", list(df.columns))
        return None, []
    unique_pt = df["pcr_target"].dropna().astype(str).str.strip().unique().tolist()
    logger.info("wastewater audit unique pcr_target values: %s", unique_pt)
    # Prefer "Measles virus", then MEV_WT (any case)
    for candidate in ["Measles virus", "MEV_WT", "mev_wt", "MeV_WT"]:
        for u in unique_pt:
            if u and (u == candidate or u.upper() == candidate.upper()):
                logger.info("wastewater audit: using pcr_target filter value '%s'", u)
                return u, unique_pt
    if unique_pt:
        logger.warning("wastewater audit: no measles pcr_target in ['Measles virus','MEV_WT']; found %s. Passing all rows; risk layer will filter by MEV/Measles.", unique_pt)
    return None, unique_pt


def load_wastewater(use_cache: bool = True) -> Tuple[pd.DataFrame, str]:
    """Fetch wastewater measles. Audits pcr_target values then applies correct filter. Returns (df, status)."""
    cache_key = "wastewater"
    if use_cache and cache_key in _cache:
        data, ts = _cache[cache_key]
        if time.time() - ts < CACHE_TTL:
            return data, "ok"
    token = _get_token()
    if not token:
        logger.error("wastewater fetch skipped: SOCRATA_APP_TOKEN not set")
        return pd.DataFrame(), "fail"
    try:
        pcr_filter_val, _ = _wastewater_audit_pcr_target(token)
        ww_where = f"pcr_target = '{pcr_filter_val}'" if pcr_filter_val else None
        raw = _soda3_post_all(token, WW_VIEW_ID, ww_where)
        df = pd.DataFrame(raw)
        if not df.empty and pcr_filter_val and "pcr_target" in df.columns:
            df = df[df["pcr_target"].astype(str).str.strip() == pcr_filter_val].copy()
        if not df.empty:
            logger.info("wastewater raw columns: %s", list(df.columns))
            # Audit: confirm required fields for detection pipeline (akvg-8vrb)
            has_site = "site_id" in df.columns or "sewershed_id" in df.columns or "sample_id" in df.columns
            has_conc = "pcr_target_avg_conc" in df.columns
            has_ntc = "ntc_amplify" in df.columns
            has_inh_d = "inhibition_detect" in df.columns
            has_inh_a = "inhibition_adjust" in df.columns
            has_date = "sample_collect_date" in df.columns or ("year" in df.columns and "week" in df.columns)
            logger.info("wastewater required-for-detection: site_id/sewershed_id/sample_id=%s, pcr_target_avg_conc=%s, ntc_amplify=%s, inhibition_detect=%s, inhibition_adjust=%s, date_field=%s",
                        has_site, has_conc, has_ntc, has_inh_d, has_inh_a, has_date)
            # CDC wastewater has sample_collect_date, not year/week; derive year and week for alignment with NNDSS
            if "sample_collect_date" in df.columns and ("year" not in df.columns or "week" not in df.columns):
                try:
                    dates = pd.to_datetime(df["sample_collect_date"], errors="coerce")
                    cal = dates.dt.isocalendar()
                    df["year"] = cal["year"].fillna(0).astype(int)
                    df["week"] = cal["week"].fillna(0).astype(int)
                except Exception as e:
                    logger.warning("wastewater sample_collect_date parse failed: %s", e)
            for c in ("year", "week"):
                if c in df.columns:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
            # Diagnostic: which signal column exists and its summary (for risk layer to use)
            for sig in ("pcr_target_flowpop_lin", "pcr_target_avg_conc_lin", "pcr_target_avg_conc", "pcr_target_mic_lin"):
                if sig in df.columns:
                    ser = pd.to_numeric(df[sig], errors="coerce")
                    n = ser.notna().sum()
                    if n > 0:
                        logger.info("wastewater signal col=%s non_null=%s min=%s median=%s max=%s",
                                    sig, int(n), ser.min(), ser.median(), ser.max())
                    break
        logger.info("Loaded wastewater rows=%s", len(df))
        if use_cache:
            _cache[cache_key] = (df.copy(), time.time())
        return df, "ok"
    except requests.RequestException as e:
        logger.warning("wastewater fetch failed reason=%s", e)
        return pd.DataFrame(), "fail"
    except Exception as e:
        logger.exception("wastewater load error")
        return pd.DataFrame(), "fail"


def load_nndss(use_cache: bool = True) -> Tuple[pd.DataFrame, str]:
    """Fetch NNDSS measles (Indigenous/Imported); clean to national + state-level. Returns (df, status)."""
    cache_key = "nndss"
    if use_cache and cache_key in _cache:
        data, ts = _cache[cache_key]
        if time.time() - ts < CACHE_TTL:
            return data, "ok"
    token = _get_token()
    if not token:
        logger.error("nndss fetch skipped: SOCRATA_APP_TOKEN not set")
        return pd.DataFrame(), "fail"
    allowed = {"Measles, Indigenous", "Measles, Imported"}
    try:
        raw = _soda3_post_all(token, NNDSS_VIEW_ID, NNDSS_WHERE, order_clause=NNDSS_ORDER)
        df = pd.DataFrame(raw)
        if df.empty:
            if use_cache:
                _cache[cache_key] = (df, time.time())
            return df, "ok"
        reporting_col = "Reporting Area" if "Reporting Area" in df.columns else ("states" if "states" in df.columns else None)
        if not reporting_col or "label" not in df.columns:
            logger.warning("nndss missing required columns (need reporting area, label)")
            if use_cache:
                _cache[cache_key] = (df.copy(), time.time())
            return df, "ok"
        label_ok = df["label"].astype(str).str.strip().isin(allowed)
        df = df.loc[label_ok].copy()
        # Prefer state from Reporting Area: national = "US RESIDENTS", state = state name (e.g. CONNECTICUT)
        if "states" in df.columns and "Reporting Area" not in df.columns:
            df = df.rename(columns={"states": "Reporting Area"})
            reporting_col = "Reporting Area"
        # Ensure we have a single state column for grouping: use Reporting Area value as state identifier
        df["_state"] = df[reporting_col].astype(str).str.strip()
        national_count = (df["_state"] == "US RESIDENTS").sum()
        state_count = (df["_state"] != "US RESIDENTS").sum()
        logger.info("nndss after filter: national rows=%s state rows=%s", int(national_count), int(state_count))
        for col in ("year", "week"):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        # Log actual year range so we can see if 2025/2026 are present (ordering may be server-dependent)
        if "year" in df.columns:
            y = df["year"].dropna()
            if len(y) > 0:
                y_min, y_max = float(y.min()), float(y.max())
                logger.info("NNDSS loaded: rows=%s year range min=%.0f max=%.0f", len(df), y_min, y_max)
                if y_max < 2025:
                    logger.warning("NNDSS most recent year is %.0f; 2025/2026 may exist but not in this fetch. Click Refresh data to bypass cache; API may return oldest-first so all pages are needed.", y_max)
        logger.info("Loaded nndss rows=%s", len(df))
        if use_cache:
            _cache[cache_key] = (df.copy(), time.time())
        return df, "ok"
    except requests.RequestException as e:
        logger.warning("nndss fetch failed reason=%s", e)
        return pd.DataFrame(), "fail"
    except Exception as e:
        logger.exception("nndss load error")
        return pd.DataFrame(), "fail"


def load_all(use_cache: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict, str]:
    """
    Load all four sources. Returns:
    (historical_df, kindergarten_df, wastewater_df, nndss_df, load_status, data_as_of).
    data_as_of is an ISO date string for display.
    """
    start = time.time()
    hist, s_hist = load_historical(use_cache=use_cache)
    kg, s_kg = load_kindergarten(use_cache=use_cache)
    ww, s_ww = load_wastewater(use_cache=use_cache)
    nndss, s_nndss = load_nndss(use_cache=use_cache)
    load_status = {"historical": s_hist, "kindergarten": s_kg, "wastewater": s_ww, "nndss": s_nndss}
    data_as_of = time.strftime("%Y-%m-%d %H:%M", time.gmtime())
    logger.info("load_all completed in %.1fs status=%s", time.time() - start, load_status)
    return hist, kg, ww, nndss, load_status, data_as_of


def clear_cache() -> None:
    """Clear in-memory cache (e.g. after 'Refresh data')."""
    _cache.clear()
    logger.info("Cache cleared")
