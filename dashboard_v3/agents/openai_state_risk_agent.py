"""
OpenAI Chat Completions agent: measles state-risk forecaster with mandatory tool use.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from utils.logging_config import get_logger

from .state_risk_tools import STATE_RISK_TOOL_NAME, STATE_RISK_TOOLS, execute_state_risk_tool

logger = get_logger("openai_state_risk_agent")

# dashboard_v3/agents/this_file -> repo root (ToolV2) and dashboard_v3
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DASHBOARD_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_OPENAI_MODEL = "gpt-5.4-mini"

SYSTEM_PROMPT = """You are a measles state-level risk forecaster assistant for U.S. public health surveillance.
Your answers combine: (1) recent case activity, (2) kindergarten MMR coverage, and (3) wastewater-related signals where available.
You must call the tool `get_state_risk_snapshot` before giving substantive guidance so your reasoning is grounded in the computed index.
Scores and tiers are supporting evidence for situational awareness—not a clinical diagnosis, not a guarantee of outbreak timing or size.
Be concise, name specific states when relevant, and note data gaps (e.g., missing wastewater) clearly."""


def _load_env() -> None:
    try:
        from dotenv import dotenv_values, load_dotenv
    except ImportError:
        return

    # Repo first; then dashboard_v3 with override=True so local .env wins over empty/partial repo or shell env.
    load_dotenv(REPO_ROOT / ".env")
    load_dotenv(DASHBOARD_ROOT / ".env", override=True, encoding="utf-8-sig")

    # Bulletproof: read dashboard_v3/.env directly so OPENAI_API_KEY is set even if load_dotenv skipped
    # (e.g. odd permissions, path quirks, or dotenv version differences).
    dash = DASHBOARD_ROOT / ".env"
    if dash.is_file():
        try:
            vals = dotenv_values(dash, encoding="utf-8-sig")
        except TypeError:
            vals = dotenv_values(dash)
        key = (vals.get("OPENAI_API_KEY") or "").strip()
        if key:
            os.environ["OPENAI_API_KEY"] = key


def _assistant_message_dict(msg: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"role": "assistant", "content": msg.content}
    if getattr(msg, "tool_calls", None):
        out["tool_calls"] = [
            {
                "id": tc.id,
                "type": "function",
                "function": {
                    "name": tc.function.name,
                    "arguments": tc.function.arguments or "{}",
                },
            }
            for tc in msg.tool_calls
        ]
    return out


def run_state_risk_forecaster(
    kg: pd.DataFrame,
    nndss: pd.DataFrame,
    ww: pd.DataFrame,
    *,
    user_message: str = "Summarize current state-level measles risk and highlight the highest-risk jurisdictions.",
    model: Optional[str] = None,
    data_as_of: Optional[str] = None,
    max_tool_rounds: int = 5,
) -> Dict[str, Any]:
    """
    Run the forecaster: first model turn MUST invoke `get_state_risk_snapshot` (tool_choice required).

    Returns
    -------
    dict with keys:
      - ``assistant_text``: final natural-language reply (empty string if missing).
      - ``last_snapshot``: dict from the last successful tool run, or None.
      - ``tool_invoked``: True if the snapshot tool was executed at least once.
      - ``model``: model id used.
    """
    _load_env()
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set; add it to .env (see agents/README.md).")

    model = model or os.environ.get("OPENAI_MODEL", DEFAULT_OPENAI_MODEL)

    from openai import OpenAI

    client = OpenAI(api_key=api_key)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    last_snapshot: Optional[Dict[str, Any]] = None
    tool_invoked = False
    first_turn = True
    rounds = 0

    while rounds < max_tool_rounds:
        rounds += 1
        kwargs: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "tools": STATE_RISK_TOOLS,
        }
        if first_turn:
            kwargs["tool_choice"] = "required"
            first_turn = False
        else:
            kwargs["tool_choice"] = "auto"

        try:
            completion = client.chat.completions.create(**kwargs)
        except Exception as e:
            logger.exception("OpenAI chat.completions failed: %s", e)
            raise

        msg = completion.choices[0].message
        tool_calls = getattr(msg, "tool_calls", None)

        if rounds == 1 and not tool_calls:
            logger.warning("First completion had no tool_calls; retrying once with tool_choice=required.")
            kwargs["tool_choice"] = "required"
            completion = client.chat.completions.create(**kwargs)
            msg = completion.choices[0].message
            tool_calls = getattr(msg, "tool_calls", None)

        if rounds == 1 and not tool_calls:
            raise RuntimeError(
                "Expected a tool call on the first turn (tool_choice=required) but the model returned no tool_calls."
            )

        if not tool_calls:
            text = (msg.content or "").strip()
            return {
                "assistant_text": text,
                "last_snapshot": last_snapshot,
                "tool_invoked": tool_invoked,
                "model": model,
            }

        messages.append(_assistant_message_dict(msg))

        for tc in tool_calls:
            name = tc.function.name
            raw_args = tc.function.arguments or "{}"
            try:
                content = execute_state_risk_tool(
                    name,
                    raw_args,
                    kg=kg,
                    nndss=nndss,
                    ww=ww,
                    data_as_of=data_as_of,
                )
            except Exception as e:
                logger.exception("Tool execution failed: %s", e)
                content = json.dumps({"error": str(e)})
            if name == STATE_RISK_TOOL_NAME:
                tool_invoked = True
                try:
                    last_snapshot = json.loads(content)
                except json.JSONDecodeError:
                    last_snapshot = None
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )

    raise RuntimeError(f"Exceeded max_tool_rounds={max_tool_rounds} without a final assistant message.")
