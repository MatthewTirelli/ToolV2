"""Standalone agents for state-level measles risk (Streamlit wiring optional)."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "DEFAULT_OPENAI_MODEL",
    "DEFAULT_TOP_N",
    "SCHEMA_VERSION",
    "SYSTEM_PROMPT",
    "STATE_RISK_TOOL_NAME",
    "STATE_RISK_TOOLS",
    "CENSUS_TOOL_NAME",
    "CENSUS_TOOLS",
    "REPORT_CONTEXT_TOOL_NAME",
    "REPORT_CONTEXT_TOOLS",
    "compute_state_risk_snapshot",
    "execute_state_risk_tool",
    "execute_census_tool",
    "execute_report_context_tool",
    "run_state_risk_forecaster",
    "run_measles_multi_agent_pipeline",
    "build_agent2_enrichment",
    "snapshot_to_json",
]

_LAZY = {
    "DEFAULT_OPENAI_MODEL": ("openai_state_risk_agent", "DEFAULT_OPENAI_MODEL"),
    "SYSTEM_PROMPT": ("openai_state_risk_agent", "SYSTEM_PROMPT"),
    "run_state_risk_forecaster": ("openai_state_risk_agent", "run_state_risk_forecaster"),
    "DEFAULT_TOP_N": ("state_risk_snapshot", "DEFAULT_TOP_N"),
    "SCHEMA_VERSION": ("state_risk_snapshot", "SCHEMA_VERSION"),
    "compute_state_risk_snapshot": ("state_risk_snapshot", "compute_state_risk_snapshot"),
    "snapshot_to_json": ("state_risk_snapshot", "snapshot_to_json"),
    "STATE_RISK_TOOL_NAME": ("state_risk_tools", "STATE_RISK_TOOL_NAME"),
    "STATE_RISK_TOOLS": ("state_risk_tools", "STATE_RISK_TOOLS"),
    "execute_state_risk_tool": ("state_risk_tools", "execute_state_risk_tool"),
    "CENSUS_TOOL_NAME": ("census_tools", "CENSUS_TOOL_NAME"),
    "CENSUS_TOOLS": ("census_tools", "CENSUS_TOOLS"),
    "execute_census_tool": ("census_tools", "execute_census_tool"),
    "REPORT_CONTEXT_TOOL_NAME": ("report_context_tools", "REPORT_CONTEXT_TOOL_NAME"),
    "REPORT_CONTEXT_TOOLS": ("report_context_tools", "REPORT_CONTEXT_TOOLS"),
    "execute_report_context_tool": ("report_context_tools", "execute_report_context_tool"),
    "run_measles_multi_agent_pipeline": ("measles_multi_agent", "run_measles_multi_agent_pipeline"),
    "build_agent2_enrichment": ("measles_multi_agent", "build_agent2_enrichment"),
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        mod = import_module(f"{__name__}.{mod_name}")
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
