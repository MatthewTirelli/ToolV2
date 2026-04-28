"""
State-level ACS5 demographics via Census API (same variables as HW2 reference).
"""
from __future__ import annotations

import sys
from typing import Any, Dict, Optional

import requests

STATE_FIPS: Dict[str, str] = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}

ACS5_URL = "https://api.census.gov/data/2023/acs/acs5"

_state_census_cache: Dict[str, Optional[Dict[str, Any]]] = {}


def get_state_demographics(state_abbr: str) -> Optional[Dict[str, Any]]:
    """
    Fetch state-level ACS5: median household income, population, % White (one-race).
    Returns None on failure.
    """
    if not state_abbr:
        return None
    state_code = state_abbr.strip().upper()
    fips = STATE_FIPS.get(state_code)
    if not fips:
        return None

    if state_code in _state_census_cache:
        return _state_census_cache[state_code]

    params = {
        "get": "B19013_001E,B01003_001E,B02001_002E",
        "for": f"state:{fips}",
    }

    try:
        r = requests.get(ACS5_URL, params=params, timeout=60)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            _state_census_cache[state_code] = None
            return None

        row = data[1]
        income_raw = row[0]
        pop_raw = row[1]
        white_raw = row[2]

        income_val = float(income_raw) if income_raw not in (None, "", "null") else None
        pop_val = float(pop_raw) if pop_raw not in (None, "", "null") else None
        white_val = float(white_raw) if white_raw not in (None, "", "null") else None
        if income_val is None or pop_val in (None, 0) or white_val is None:
            _state_census_cache[state_code] = None
            return None

        out = {
            "state": state_code,
            "median_household_income": income_val,
            "population": int(pop_val),
            "pct_white": (white_val / pop_val) * 100.0,
            "acs_dataset": "acs5",
            "acs_year": 2023,
        }
        _state_census_cache[state_code] = out
        return out
    except Exception as e:
        print(f"[CENSUS_ERROR] state={state_code!r} err={e}", file=sys.stderr, flush=True)
        _state_census_cache[state_code] = None
        return None


def fetch_demographics_for_states(state_codes: list[str]) -> Dict[str, Optional[Dict[str, Any]]]:
    """Fetch demographics for each two-letter state code; missing/invalid codes map to None."""
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for s in state_codes:
        code = (s or "").strip().upper()
        if not code:
            continue
        out[code] = get_state_demographics(code)
    return out
