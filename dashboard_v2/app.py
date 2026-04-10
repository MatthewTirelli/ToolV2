"""
Measles outbreak risk dashboard — Shiny for Python entrypoint.
Loads `.env` from project root or this directory. Run: `shiny run app.py` from `dashboard/`.
"""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent
if str(APP_DIR) not in sys.path:
    sys.path.insert(0, str(APP_DIR))

try:
    from dotenv import load_dotenv

    load_dotenv(APP_DIR.parent / ".env")
    load_dotenv(APP_DIR / ".env")
except ImportError:
    pass

from shiny import App

from server import server
from ui import app_ui

app = App(app_ui(), server)
