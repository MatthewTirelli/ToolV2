"""
Structured logging for the dashboard. Log to file and console; no secrets or PII.
"""
import logging
import sys
from pathlib import Path

DASHBOARD_DIR = Path(__file__).resolve().parent.parent
LOG_FILE = DASHBOARD_DIR / "dashboard.log"


def setup_logging(level: int = logging.INFO, log_file: "Path | None" = None) -> None:
    """Configure root logger for dashboard: file + console, no duplicate propagation."""
    root = logging.getLogger("dashboard")
    root.setLevel(level)
    if root.handlers:
        return
    if log_file is None:
        log_file = LOG_FILE
    fmt = logging.Formatter("%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
    if log_file:
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(fmt)
        root.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(fmt)
    root.addHandler(ch)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for module `name` (e.g. 'dashboard.loaders')."""
    return logging.getLogger(f"dashboard.{name}")
