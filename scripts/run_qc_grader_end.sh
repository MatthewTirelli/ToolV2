#!/usr/bin/env bash
# Run QC with the LLM grader in a second pass (after all reports are generated).
# Requires OPENAI_API_KEY in the environment or in .env at repo root.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv ]]; then
  # shellcheck source=/dev/null
  source .venv/bin/activate
fi

exec python3 qc/run_qc_experiment.py --with-grader "$@"
