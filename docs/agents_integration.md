# Integrating the state-risk agent (Streamlit & orchestration)

## Audience & scope

This doc is for engineers wiring **dashboard_v3** or a **multi-agent** orchestrator to the standalone package under `dashboard_v3/agents/`.

- **`risk.py` / `build_state_risk_index`** still owns the scoring math. The agent layer **does not** duplicate it; it wraps it in `compute_state_risk_snapshot` for a stable JSON payload.
- **`model_runner.load_and_model`** (and `loaders.load_all`) remain the data sources; the agent does not fetch raw files by itself.

## Dependencies

- `pip install -r dashboard_v3/requirements.txt` (includes `openai`).
- **`OPENAI_API_KEY`** in `dashboard_v3/.env` (or repo root `.env`) for `run_state_risk_forecaster`.
- Optional **`SOCRATA_APP_TOKEN`** if you use live `load_all()` against Socrata.

## Data contract

Inputs are **pandas** `DataFrame`s:

| Name | Role |
|------|------|
| `kg` | Kindergarten / coverage (MMR columns as expected by `risk.build_state_risk_index`) |
| `nndss` | Case line list (NNDSS-style) |
| `ww` | Wastewater |

**Where to get them in this app**

- `d = load_and_model(...)` → use `d["kg"]`, `d["nndss"]`, `d["ww"]` (same keys as elsewhere in the pipeline).
- Or `load_all()` and unpack the frames in the same order as existing scripts.

Pass an optional **`data_as_of`** string (ISO or display) into `compute_state_risk_snapshot(..., data_as_of=...)` for auditability in the snapshot.

## Streamlit integration (recipe)

1. Run with `dashboard_v3` on `sys.path` (or run the app from that directory).
2. After loading data, **snapshot-only** (no key, fast):

   ```python
   from agents import compute_state_risk_snapshot

   snap = compute_state_risk_snapshot(
       d["kg"], d["nndss"], d["ww"],
       data_as_of=d.get("data_as_of"),
   )
   ```

3. **Full forecaster** (blocking network I/O to OpenAI — use `st.spinner`, consider async later):

   ```python
   from agents import run_state_risk_forecaster

   out = run_state_risk_forecaster(d["kg"], d["nndss"], d["ww"], data_as_of=d.get("data_as_of"))
   st.markdown(out["assistant_text"])
   ```

4. Cache **snapshots** with `st.cache_data` if inputs are stable; do not cache OpenAI calls blindly.

## Orchestration / other agents

- `compute_state_risk_snapshot` returns a **dict** (`schema_version`, `role`, `generated_at`, `summary`, `states_top`, `disclaimer`, `wastewater_diagnostics`, optional `states_full`).
- Pass **`json.dumps(snapshot)`** or the dict into a parent orchestrator / second LLM; compose payloads:

  ```python
  combined = {"state_risk": snapshot, "other_tool": ...}
  ```

- Downstream agents should **preserve** tool results when wrapping `run_state_risk_forecaster`; do not strip `tool` messages from the OpenAI message history if you replay or log the run.

## Mandatory tool behavior

`run_state_risk_forecaster` **forces** the first model turn to call `get_state_risk_snapshot` via OpenAI **`tool_choice="required"`**. If that call does not include tool calls, the Python layer **raises** so you never silently accept a hallucinated summary without the tool.

## Failure modes

- **Missing API key** → `ValueError` at start of `run_state_risk_forecaster`.
- **Rate limits / API errors** → exception from OpenAI SDK; check logs under logger `openai_state_risk_agent` via `utils.logging_config.get_logger`.
- **Empty kindergarten frame** → `build_state_risk_index` returns an empty table; snapshot has `summary.n_states == 0` and empty `states_top` (no crash).
- **Model id rejected** → set **`OPENAI_MODEL`** or pass `model=` to `run_state_risk_forecaster`.

## Testing

Use the standalone CLI (no Streamlit):

```bash
cd dashboard_v3
python scripts/run_state_risk_agent_demo.py --snapshot-only
python scripts/run_state_risk_agent_demo.py   # full run with key
```

See [README.md](README.md) for `--dry-run`.
