# State risk agents (standalone)

OpenAI-powered **measles state-risk forecaster** plus a JSON snapshot tool built on `risk.build_state_risk_index`. This package is **not** wired into `app.py` yet; use it from scripts or a future Streamlit integration (see [INTEGRATION.md](INTEGRATION.md)).

## Requirements

- Install from `dashboard_v3`: `pip install -r requirements.txt` (includes `openai`).
- Set `OPENAI_API_KEY` in `dashboard_v3/.env` (or repo root `.env`). Both paths are loaded; **`dashboard_v3/.env` overrides** the repo root so a key defined only there is used even if the shell or root `.env` left an empty value.
- If you still see “OPENAI_API_KEY is not set”, check the line is exactly `OPENAI_API_KEY=sk-...` with no quotes issues, and run only the `python ...` line (do not paste comment lines like `# needs OPENAI_API_KEY` as extra arguments).
- Optional: `OPENAI_MODEL` to override the default model **`gpt-5.4-mini`** if your account rejects that id.
- For live data via `load_all`, you may need `SOCRATA_APP_TOKEN` per the main app docs.

## Mandatory tool behavior

`run_state_risk_forecaster` issues the **first** `chat.completions.create` with **`tool_choice="required"`**, so the model must emit a `function` call to `get_state_risk_snapshot` before a final answer. If the first response has no tool calls, the run **raises** `RuntimeError` instead of returning a free-form answer that skipped the tool. After the tool result is appended, later turns use `tool_choice="auto"` for the natural-language reply.

## Modules

| Module | Role |
|--------|------|
| `state_risk_snapshot.py` | `compute_state_risk_snapshot(kg, nndss, ww, **opts)` → JSON-safe dict |
| `state_risk_tools.py` | OpenAI function schema + `execute_state_risk_tool(...)` |
| `openai_state_risk_agent.py` | `SYSTEM_PROMPT`, `run_state_risk_forecaster(...)` |

## Quick demo (no Streamlit)

From `dashboard_v3`:

```bash
source .venv/bin/activate
pip install -r requirements.txt   # once

# No API calls — snapshot JSON only
python scripts/run_state_risk_agent_demo.py --snapshot-only

# Full agent + OpenAI (requires OPENAI_API_KEY)
python scripts/run_state_risk_agent_demo.py

# Tool schema only (no network)
python scripts/run_state_risk_agent_demo.py --dry-run
```

## See also

- **[INTEGRATION.md](INTEGRATION.md)** — Streamlit session keys, data contract, multi-agent handoff, failure modes.
