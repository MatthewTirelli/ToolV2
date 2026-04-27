# ToolV2 Measles Risk Dashboard

Cleaned repository layout for the Streamlit app and multi-agent pipeline.

## Run the app

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/main.py
```

## Run the agent pipeline

```bash
python scripts/run_measles_multi_agent.py --dry-run
python scripts/run_measles_multi_agent.py
```

## Repository layout

- `app/`: Streamlit UI, modeling, loaders, shared app utils
- `agents/`: state risk agent, enrichment/report pipeline, tool adapters
- `data/reference/`: checked-in reference data
- `scripts/`: runnable CLI entry points
- `qc/outputs/`: reserved output folder (git-ignored contents)
- `docs/`: inventory and migration notes
- `archive/`: legacy and non-active artifacts retained for reproducibility
