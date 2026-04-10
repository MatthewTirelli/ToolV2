#!/usr/bin/env python3
"""Manual test for state risk snapshot + OpenAI forecaster (run from dashboard_v3 with .venv).

```bash
cd dashboard_v3
source .venv/bin/activate
pip install -r requirements.txt   # once

# No API calls — snapshot JSON only
python scripts/run_state_risk_agent_demo.py --snapshot-only

# Full agent + OpenAI (requires OPENAI_API_KEY)
python scripts/run_state_risk_agent_demo.py
```
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))


def main() -> None:
    # Load .env before any code that reads OPENAI_API_KEY (same paths as openai_state_risk_agent._load_env).
    try:
        from dotenv import dotenv_values, load_dotenv

        load_dotenv(APP_DIR.parent.parent / ".env")
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

    parser = argparse.ArgumentParser(description="State risk snapshot / OpenAI forecaster demo.")
    parser.add_argument(
        "--snapshot-only",
        action="store_true",
        help="Compute JSON snapshot only (no OpenAI; no API key needed).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print tool schema and exit (no data load, no API).",
    )
    args = parser.parse_args()

    if args.dry_run:
        from agents.state_risk_tools import STATE_RISK_TOOLS

        print(json.dumps(STATE_RISK_TOOLS, indent=2))
        return

    from loaders import load_all

    _, kg, ww, nndss, _, data_as_of = load_all(use_cache=False)

    from agents.state_risk_snapshot import compute_state_risk_snapshot, snapshot_to_json

    if args.snapshot_only:
        snap = compute_state_risk_snapshot(kg, nndss, ww, data_as_of=data_as_of)
        print(snapshot_to_json(snap))
        return

    from agents.openai_state_risk_agent import run_state_risk_forecaster

    out = run_state_risk_forecaster(kg, nndss, ww, data_as_of=data_as_of)
    print("--- assistant ---")
    print(out.get("assistant_text", ""))
    print("--- meta ---")
    print("model:", out.get("model"))
    print("tool_invoked:", out.get("tool_invoked"))
    if out.get("last_snapshot") is not None:
        s = out["last_snapshot"]
        print("snapshot schema_version:", s.get("schema_version"))
        print("summary:", json.dumps(s.get("summary"), indent=2))


if __name__ == "__main__":
    main()
