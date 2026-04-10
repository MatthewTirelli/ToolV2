#!/usr/bin/env python3
"""One-off debug: prove why Wisconsin (or any state) tops state_risk_df."""
from __future__ import annotations

import os
import sys
from pathlib import Path

# Run from repo: python scripts/debug_state_risk_wi.py from dashboard_v3/
APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

import pandas as pd

from loaders import clear_cache, load_all
from risk import build_state_risk_index, get_state_risk_df, get_state_weekly_cases
from risk import get_state_ww_signal


def main() -> None:
    os.chdir(APP_DIR)
    clear_cache()
    hist, kg, ww, nndss, status, _ = load_all(use_cache=False)
    print("load_status:", status)
    print("kg rows:", len(kg), "ww rows:", len(ww), "nndss rows:", len(nndss))

    sr, aux = build_state_risk_index(kg, nndss, ww)
    sr_public = get_state_risk_df(kg, nndss, ww)

    print("\n=== Wisconsin row (build_state_risk_index) ===")
    wi = sr[sr["state"] == "WI"]
    print(wi.to_string() if not wi.empty else "NO WI ROW")

    print("\n=== Top 10 by total_risk ===")
    print(sr.head(10).to_string())

    for c in ["coverage_points", "case_points", "wastewater_points", "total_risk"]:
        print(f"\n=== describe {c} ===")
        print(sr[c].describe().to_string())

    nz_cases = (sr["cases_recent"] > 0).sum()
    ww_avail = sr["wastewater_available"].sum()
    print(f"\nstates with cases_recent > 0: {nz_cases} / {len(sr)}")
    print(f"states with wastewater_available: {ww_avail} / {len(sr)}")

    compare = ["WI", "NY", "CA", "TX", "FL"]
    top2 = sr.iloc[1]["state"] if len(sr) > 1 else None
    if top2 and top2 not in compare:
        compare.append(str(top2))
    print("\n=== Compare WI, NY, CA, TX, FL, + 2nd-ranked if not listed ===")
    mask = sr["state"].isin(compare)
    print(sr.loc[mask].sort_values("total_risk", ascending=False).to_string())

    # Raw kindergarten WI vs others
    print("\n=== Kindergarten: WI raw rows (jurisdiction + coverage cols) ===")
    if not kg.empty:
        jcol = next((c for c in ["jurisdiction", "geography", "state", "State"] if c in kg.columns), kg.columns[0])
        pcol = next((c for c in kg.columns if "pct" in c.lower() or "coverage" in c.lower()), None)
        wi_kg = kg[kg[jcol].astype(str).str.upper().str.contains("WISCONSIN", na=False)]
        print("cols:", list(kg.columns))
        print(wi_kg.head(10).to_string())

    sc = get_state_weekly_cases(nndss)
    if not sc.empty and "state" in sc.columns:
        wi_c = sc[sc["state"] == "WI"].tail(8)
        print("\n=== NNDSS state weekly last 8 rows for WI ===")
        print(wi_c.to_string())

    st_ww, ww_diag = get_state_ww_signal(ww)
    print("\n=== get_state_ww_signal audit (excerpt) ===")
    print({k: ww_diag.get(k) for k in ["status", "jurisdiction_column", "rows_mapped_to_state", "state_week_row_groups"]})
    if not st_ww.empty:
        wi_w = st_ww[st_ww["state"] == "WI"].tail(6)
        print("\n=== State WW last 6 rows for WI ===")
        print(wi_w.to_string())


if __name__ == "__main__":
    main()
