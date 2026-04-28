"""
JSON-serializable state risk snapshot built on `build_state_risk_index` from `risk.py`.
For downstream agents and OpenAI tool output.

Stable payload shape (dict, JSON-serializable):

- ``schema_version`` (str), ``role`` (``"measles_state_risk_snapshot"``)
- ``generated_at`` (ISO8601 UTC), optional ``data_as_of`` (str)
- ``summary``: ``n_states``, ``tiers`` (``high`` / ``medium`` / ``low`` counts)
- ``states_top``: ordered list of state records (top ``top_n`` by ``total_risk``)
- ``states_full``: only if ``include_full_table=True``
- ``wastewater_diagnostics``: small sanitized dict from the index helper
- ``disclaimer``: fixed surveillance-limitations string
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from app.risk import build_state_risk_index

SCHEMA_VERSION = "1.0"
ROLE = "measles_state_risk_snapshot"

DISCLAIMER = (
    "This snapshot is derived from public surveillance-style inputs (NNDSS-style case data, "
    "kindergarten MMR coverage, and wastewater monitoring). It supports risk awareness only; "
    "it is not a clinical diagnosis or definitive outbreak prediction."

)

DEFAULT_TOP_N = 15


def _json_safe_value(v: Any) -> Any:
    if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
        return None
    if isinstance(v, (np.integer, np.floating)):
        return float(v) if isinstance(v, np.floating) else int(v)
    if isinstance(v, (float, int, str, bool)):
        return v
    if pd.isna(v):
        return None
    return str(v)


def _dataframe_to_records(df: pd.DataFrame) -> List[Dict[str, Any]]:
    if df is None or df.empty:
        return []
    out: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        rec = {k: _json_safe_value(row[k]) for k in df.columns}
        out.append(rec)
    return out


def _sanitize_ww_diag(diag: Dict[str, Any], max_keys: int = 20) -> Dict[str, Any]:
    """Keep a small, JSON-friendly subset of wastewater diagnostics."""
    if not isinstance(diag, dict):
        return {"note": "invalid_diagnostic"}
    out: Dict[str, Any] = {}
    for i, (k, v) in enumerate(diag.items()):
        if i >= max_keys:
            break
        if isinstance(v, (dict, list)) and len(str(v)) > 2000:
            out[k] = "<truncated>"
        else:
            try:
                json.dumps(v)
                out[k] = v
            except TypeError:
                out[k] = str(v)[:500]
    return out


def compute_state_risk_snapshot(
    kindergarten: pd.DataFrame,
    nndss: pd.DataFrame,
    wastewater: pd.DataFrame,
    *,
    top_n: int = DEFAULT_TOP_N,
    include_full_table: bool = False,
    data_as_of: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run `build_state_risk_index` and return a versioned dict safe for `json.dumps`
    and for feeding other agents.

    Parameters
    ----------
    kindergarten, nndss, wastewater
        Same DataFrames as the Streamlit pipeline (`load_all` / `load_and_model`).
    top_n
        Number of highest `total_risk` states to list in `states_top`.
    include_full_table
        If True, include `states_full` with all rows (can be large).
    data_as_of
        Optional ISO or display string for when data were pulled.
    """
    df, ww_diag = build_state_risk_index(kindergarten, nndss, wastewater)
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat()

    summary: Dict[str, Any] = {
        "n_states": int(len(df)),
        "tiers": {"high": 0, "medium": 0, "low": 0},
    }
    if not df.empty and "risk_tier" in df.columns:
        vc = df["risk_tier"].astype(str).str.lower().value_counts()
        for t in ("high", "medium", "low"):
            summary["tiers"][t] = int(vc.get(t, 0))

    states_top = _dataframe_to_records(df.head(max(0, top_n)).copy()) if not df.empty else []
    payload: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "role": ROLE,
        "generated_at": generated_at,
        "data_as_of": data_as_of,
        "summary": summary,
        "states_top": states_top,
        "disclaimer": DISCLAIMER,
        "wastewater_diagnostics": _sanitize_ww_diag(ww_diag if isinstance(ww_diag, dict) else {}),
    }
    if include_full_table and not df.empty:
        payload["states_full"] = _dataframe_to_records(df)
    return payload


def snapshot_to_json(snapshot: Dict[str, Any]) -> str:
    """Serialize snapshot to a compact JSON string."""
    return json.dumps(snapshot, indent=2, ensure_ascii=False)
