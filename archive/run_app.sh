#!/usr/bin/env bash
# Create .venv if missing, install requirements, start Streamlit (run from anywhere).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VENV_DIR="${VENV_DIR:-.venv}"
PYTHON="${PYTHON:-python3}"

if [[ ! -d "$VENV_DIR" ]]; then
  echo "Creating virtual environment in $VENV_DIR ..."
  "$PYTHON" -m venv "$VENV_DIR"
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip
pip install -r requirements.txt

echo "Starting Streamlit..."
exec streamlit run app.py "$@"
