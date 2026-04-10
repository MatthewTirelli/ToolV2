"""
OpenAI function tool schema and executor for Census ACS5 state demographics (batch).
"""
from __future__ import annotations

import json
from typing import Any, Dict

from .census_acs import fetch_demographics_for_states

CENSUS_TOOL_NAME = "get_demographics_for_states"

CENSUS_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": CENSUS_TOOL_NAME,
            "description": (
                "Fetch U.S. Census ACS5 state-level demographics for a list of two-letter state codes: "
                "median household income, population, and approximate percent White (one-race). "
                "Returns JSON mapping each state code to a record or null if unavailable."
            ),
            "parameters": {
                "type": "object",
                "required": ["state_codes"],
                "properties": {
                    "state_codes": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "List of two-letter U.S. state abbreviations (e.g. [\"CA\", \"TX\"]).",
                    },
                },
                "additionalProperties": False,
            },
        },
    },
]


def execute_census_tool(
    name: str,
    arguments: str | dict[str, Any],
) -> str:
    """Execute Census tool; returns JSON string for OpenAI tool message content."""
    if name != CENSUS_TOOL_NAME:
        return json.dumps({"error": f"unknown_tool:{name}"})

    if isinstance(arguments, str):
        try:
            args = json.loads(arguments.strip() or "{}")
        except json.JSONDecodeError:
            args = {}
    else:
        args = dict(arguments)

    raw = args.get("state_codes")
    if not isinstance(raw, list):
        return json.dumps({"error": "state_codes must be a list of strings"})

    codes = [str(x).strip().upper() for x in raw if str(x).strip()]
    result = fetch_demographics_for_states(codes)
    return json.dumps(result, ensure_ascii=False)
