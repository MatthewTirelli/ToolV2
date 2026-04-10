"""Standalone agents for state-level measles risk (not wired into Streamlit app yet)."""

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
    "compute_state_risk_snapshot",
    "execute_state_risk_tool",
    "run_state_risk_forecaster",
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
}


def __getattr__(name: str) -> Any:
    if name in _LAZY:
        mod_name, attr = _LAZY[name]
        mod = import_module(f"{__name__}.{mod_name}")
        return getattr(mod, attr)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)
