# Repository Inventory

## Scope and tracing method
Dependency tracing started from active entry points:
- `app/main.py` (Streamlit app entry)
- `scripts/run_measles_multi_agent.py`
- `scripts/run_state_risk_agent_demo.py`
- `agents/report_writer_agent.py`
- `agents/state_risk_agent.py`

Imports were traced through `app.*`, `agents.*`, and `agents.tools.*` modules.

## Active files used by app/agents
- App entry and orchestration: `app/main.py`, `app/model_runner.py`, `app/loaders.py`, `app/risk.py`, `app/kg_helpers.py`
- UI components: `app/components/charts.py`, `app/components/map.py`
- Logging and support: `app/utils/logging_config.py`, `app/utils/state_maps.py`
- Agent pipeline: `agents/state_risk_agent.py`, `agents/report_writer_agent.py`, `agents/enrichment_agent.py`
- Agent tools: `agents/tools/state_risk_tools.py`, `agents/tools/state_risk_snapshot.py`, `agents/tools/census_tools.py`, `agents/tools/census_acs.py`, `agents/tools/report_context_tools.py`
- Runner scripts: `scripts/run_measles_multi_agent.py`, `scripts/run_state_risk_agent_demo.py`
- Required reference data: `data/reference/measles_annual_1985.csv`
- Runtime/deps config: `requirements.txt`, `app/.streamlit/config.toml`, `README.md`, `.gitignore`

## Redundant / non-active files moved to archive
Moved (not deleted) to preserve reproducibility and grading context:
- Legacy app folder and prior structure: `archive/dashboard_v3_legacy/`
- Legacy run scripts: `archive/run_app.sh`, `archive/run_dashboard.sh`
- Legacy debug scripts: `archive/debug_alarm_pipeline.py`, `archive/debug_state_risk_wi.py`
- Generated logs and snapshots: `archive/dashboard.log`, `archive/dashboard_v3.zip`
- Unclear grading artifact retained: `archive/VaxFax App V1.1 - TOOL1 Report.docx`

## Generated/cache files
- Python cache dirs: `__pycache__/` (ignored)
- Virtual environments: `.venv/`, `venv/` (ignored)
- Runtime outputs: `qc/outputs/` contents (ignored; folder retained with `.gitkeep`)
- Working data outputs: `data/raw/` and `data/processed/` contents (ignored; folders retained with `.gitkeep`)
- Logs: `*.log` (ignored)

## Environment/secrets files (should not be committed)
- `.env`
- `.env.local`
- `.env.*.local`
- `app/.env` (local runtime secret file; if used, should remain untracked)

## Files whose purpose is unclear (retained safely)
- `archive/VaxFax App V1.1 - TOOL1 Report.docx` — possibly grading or report artifact
- `archive/dashboard_v3_legacy/` — retains historical context and old structure for rollback/comparison

## Notes
- No core model logic was rewritten in this cleanup pass.
- Imports/paths were updated to new package locations and entry points.
