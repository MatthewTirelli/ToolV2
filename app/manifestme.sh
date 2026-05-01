#!/usr/bin/env bash
# Regenerate Posit Connect bundle metadata (manifest.json) for this Streamlit app.
#
# Prerequisites:
#   pip install rsconnect-python
#
# Usage (from repo root or this directory):
#   ./app/manifestme.sh
#   cd app && ./manifestme.sh
#
# After this runs, deploy with Posit Connect / Connect Cloud, for example:
#   rsconnect deploy manifest . \
#     --server https://YOUR-SERVER.posit.co \
#     --api-key YOUR_API_KEY \
#     --name measles-risk-dashboard
#
# Set application environment variables in the Connect UI (do not commit secrets):
#   SOCRATA_APP_TOKEN   - required for CDC data.cdc.gov Socrata API
#   OPENAI_API_KEY      - required for AI briefing
#   OPENAI_MODEL        - optional override of the default OpenAI model id

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

pip install -q rsconnect-python 2>/dev/null || pip install rsconnect-python

rsconnect write-manifest streamlit . \
  --entrypoint main \
  --overwrite \
  -x ".venv" \
  -x "__pycache__" \
  -x "*.pyc" \
  -x ".DS_Store" \
  -x "dashboard.log" \
  -x "manifestme.sh"

echo "Wrote manifest.json in $SCRIPT_DIR"
echo "Next: rsconnect deploy manifest . --server ... --api-key ... --name ..."
