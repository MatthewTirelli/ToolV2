#!/usr/bin/env python3
"""
CLI: run full measles multi-agent pipeline (state risk + Census + grounded report).

Loads repo root `.env` (ToolV2) then `dashboard_v3/.env` so OPENAI_API_KEY from repo root works.

```bash
cd dashboard_v3
source .venv/bin/activate   # if used
pip install -r requirements.txt   # once

# Tool schemas only (no network)
python scripts/run_measles_multi_agent.py --dry-run

# Full pipeline (OpenAI + Census; needs OPENAI_API_KEY)
python scripts/run_measles_multi_agent.py
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent  # dashboard_v3/
REPO_ROOT = APP_DIR.parent  # ToolV2 repo root (parent of dashboard_v3)
sys.path.insert(0, str(APP_DIR))


def _load_env() -> None:
    try:
        from dotenv import dotenv_values, load_dotenv

        load_dotenv(REPO_ROOT / ".env")
        load_dotenv(APP_DIR / ".env", override=True, encoding="utf-8-sig")
        dash = APP_DIR / ".env"
        if dash.is_file():
            try:
                vals = dotenv_values(dash, encoding="utf-8-sig")
            except TypeError:
                vals = dotenv_values(dash)
            key = (vals.get("OPENAI_API_KEY") or "").strip()
            if key:
                os.environ["OPENAI_API_KEY"] = key
    except ImportError:
        pass


def main() -> None:
    _load_env()

    parser = argparse.ArgumentParser(description="Measles multi-agent pipeline (state risk + Census + report).")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tool schemas and exit (no data load, no API).",
    )
    args = parser.parse_args()

    if args.dry_run:
        from agents.state_risk_tools import STATE_RISK_TOOLS
        from agents.census_tools import CENSUS_TOOLS
        from agents.report_context_tools import REPORT_CONTEXT_TOOLS

        print("=== STATE_RISK_TOOLS ===")
        print(json.dumps(STATE_RISK_TOOLS, indent=2))
        print("=== CENSUS_TOOLS ===")
        print(json.dumps(CENSUS_TOOLS, indent=2))
        print("=== REPORT_CONTEXT_TOOLS ===")
        print(json.dumps(REPORT_CONTEXT_TOOLS, indent=2))
        return

    from model_runner import load_and_model
    from agents.measles_multi_agent import run_measles_multi_agent_pipeline

    print("Loading data and models…", flush=True)
    d = load_and_model(use_cache=True)
    kg, nndss, ww = d["kg"], d["nndss"], d["ww"]

    print("Running multi-agent pipeline (OpenAI + Census)…", flush=True)
    out = run_measles_multi_agent_pipeline(d, kg=kg, nndss=nndss, ww=ww)

    print("\n=== AGENT 1 (state risk forecaster) ===\n")
    print(out["agent1"].get("assistant_text", ""))
    print("\n--- tool_invoked:", out["agent1"].get("tool_invoked"))
    print("--- model:", out["agent1"].get("model"))

    print("\n=== AGENT 2 (enriched cohort; risk scores unchanged) ===\n")
    a2 = out["agent2"]
    meta = a2.get("meta", {})
    body = {k: v for k, v in a2.items() if k != "meta"}
    print(json.dumps(body, indent=2, default=str))
    print("\n--- meta ---\n")
    print(json.dumps(meta, indent=2, default=str))

    print("\n=== AGENT 3 (report) ===\n")
    print(out["agent3"].get("assistant_text", ""))
    print("\n--- model:", out["agent3"].get("model"))


if __name__ == "__main__":
    main()
