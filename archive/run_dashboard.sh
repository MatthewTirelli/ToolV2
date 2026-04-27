#!/usr/bin/env bash
# Run the Measles Risk Dashboard (Streamlit).
# From project root: ./run_dashboard.sh   OR   bash run_dashboard.sh

cd "$(dirname "$0")"
export PYTHONPATH="$PWD"
streamlit run dashboard/app.py --server.port 8501
