"""
OpenAI Chat Completions function tool schema and executor for state risk snapshot.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any, Dict

if TYPE_CHECKING:
    import pandas as pd

STATE_RISK_TOOL_NAME = "get_state_risk_snapshot"

STATE_RISK_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": STATE_RISK_TOOL_NAME,
            "description": (
                "Compute the current measles-related state risk snapshot from kindergarten MMR coverage, "
                "recent case activity, and wastewater signals. Returns JSON with ranked states and summary tiers. "
                "Call this before answering user questions about state-level risk."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "top_n": {
                        "type": "integer",
                        "description": "Number of highest-risk states to include in states_top (default 15).",
                    },
                    "include_full_table": {
                        "type": "boolean",
                        "description": "If true, include states_full with all jurisdictions (larger payload).",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]


def execute_state_risk_tool(
    name: str,
    arguments: str | dict[str, Any],
    *,
    kg: pd.DataFrame,
    nndss: pd.DataFrame,
    ww: pd.DataFrame,
    data_as_of: str | None = None,
) -> str:
    """
    Execute the registered tool by name; returns JSON string for the tool message content.
    """
    from .state_risk_snapshot import DEFAULT_TOP_N, compute_state_risk_snapshot

    if name != STATE_RISK_TOOL_NAME:
        return json.dumps({"error": f"unknown_tool:{name}"})

    if isinstance(arguments, str):
        try:
            args = json.loads(arguments) if arguments.strip() else {}
        except json.JSONDecodeError:
            args = {}
    else:
        args = dict(arguments)

    top_n = args.get("top_n")
    if top_n is not None:
        try:
            top_n = int(top_n)
        except (TypeError, ValueError):
            top_n = None
    if top_n is None:
        top_n = DEFAULT_TOP_N

    include_full = bool(args.get("include_full_table", False))

    snap = compute_state_risk_snapshot(
        kg,
        nndss,
        ww,
        top_n=top_n,
        include_full_table=include_full,
        data_as_of=data_as_of,
    )
    return json.dumps(snap, ensure_ascii=False)
