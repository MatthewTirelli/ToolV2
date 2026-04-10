"""
OpenAI tool: returns precomputed report inputs (grounding payload) as JSON.
The pipeline builds `payload` in Python; the executor ignores model arguments.
"""
from __future__ import annotations

import json
from typing import Any, Dict

REPORT_CONTEXT_TOOL_NAME = "get_precomputed_report_inputs"

REPORT_CONTEXT_TOOLS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": REPORT_CONTEXT_TOOL_NAME,
            "description": (
                "Load the authoritative precomputed measles risk report context: national alarm/baseline/forecast "
                "summaries, Census-enriched top/bottom state cohort, and observational correlation notes. "
                "You must call this before writing the final report so all numbers come from this payload."
            ),
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
]


def execute_report_context_tool(
    name: str,
    arguments: str | dict[str, Any],
    *,
    payload: Dict[str, Any],
) -> str:
    """Return JSON string of the precomputed payload (model args ignored)."""
    if name != REPORT_CONTEXT_TOOL_NAME:
        return json.dumps({"error": f"unknown_tool:{name}"})
    try:
        return json.dumps(payload, ensure_ascii=False)
    except (TypeError, ValueError) as e:
        return json.dumps({"error": f"payload_not_serializable:{e}"})
