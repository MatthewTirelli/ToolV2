"""
Ollama Cloud client for the Predictive Measles Risk Dashboard.
Sends a summary of data and risk metrics; returns plain-language interpretation.
Reads OLLAMA_API_KEY from .env (project root). No secrets in logs.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import requests

from utils.logging_config import get_logger

logger = get_logger("ollama_client")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OLLAMA_URL = "https://ollama.com/api/chat"
# Prefer cloud model tag; some setups use short name
OLLAMA_MODELS = ("gpt-oss:20b-cloud", "gpt-oss:20b", "llama3.2:3b")
MAX_PROMPT_CHARS = 8000


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(PROJECT_ROOT / ".env")
        load_dotenv(Path(__file__).resolve().parent / ".env")
    except ImportError:
        pass


def get_ollama_summary(
    alarm_probability: float,
    baseline_tier: str,
    forecast_summary: str,
    top_drivers: Optional[list] = None,
    data_as_of: str = "",
    extra_context: str = "",
) -> Optional[str]:
    """
    Send current risk summary to Ollama Cloud; return AI-generated interpretation.
    On failure (missing key, timeout, 4xx/5xx) returns None and logs (no key).
    """
    _load_env()
    key = (os.environ.get("OLLAMA_API_KEY") or "").strip()
    if not key:
        logger.warning("Ollama skipped: OLLAMA_API_KEY not set")
        return None
    drivers_str = ""
    if top_drivers:
        drivers_str = "Top drivers of outbreak alarm: " + "; ".join(str(d) for d in top_drivers[:5])
    payload_text = (
        f"Measles risk dashboard summary (data as of {data_as_of or 'N/A'}):\n"
        f"- Probability of outbreak in next 4 weeks (alarm): {alarm_probability:.1%}\n"
        f"- Baseline risk tier: {baseline_tier}\n"
        f"- {forecast_summary}\n"
        f"{drivers_str}\n"
        f"{extra_context}"
    )
    if len(payload_text) > MAX_PROMPT_CHARS:
        payload_text = payload_text[:MAX_PROMPT_CHARS] + "\n... (truncated)"
    prompt = (
        "Below is a short summary of measles surveillance data and a predictive risk model (outbreak alarm and case forecast). "
        "Write 2-3 short paragraphs in plain language: (1) what the current alarm and baseline risk suggest, "
        "(2) what the forecast implies for the next few weeks, and (3) one cautious recommendation. "
        "Do not make up numbers; only use the provided data. Use clear, non-technical language.\n\n"
        f"{payload_text}"
    )
    headers = {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }
    for model in OLLAMA_MODELS:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
        }
        try:
            response = requests.post(OLLAMA_URL, headers=headers, json=body, timeout=90)
        except requests.RequestException as e:
            logger.error("Ollama request failed reason=%s", e)
            return None
        if response.status_code == 200:
            break
        if response.status_code in (404, 422):
            logger.warning("Ollama model %s not available status=%s", model, response.status_code)
            continue
        logger.error("Ollama request failed status=%s body=%s", response.status_code, response.text[:300])
        return None
    else:
        logger.error("Ollama: no model succeeded")
        return None
    try:
        out = response.json()
        report = out.get("message", {}).get("content", "")
        return (report or "").strip() or None
    except (KeyError, json.JSONDecodeError) as e:
        logger.error("Ollama parse failed reason=%s", e)
        return None


def get_ollama_follow_up(question: str, context_summary: str) -> Optional[str]:
    """
    Send a user follow-up question with current context to Ollama. Returns answer or None.
    """
    _load_env()
    key = (os.environ.get("OLLAMA_API_KEY") or "").strip()
    if not key:
        return None
    prompt = f"Context from the measles risk dashboard:\n{context_summary[:4000]}\n\nUser question: {question}\n\nAnswer in 1-2 short paragraphs using only the context and data provided."
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": OLLAMA_MODELS[0], "messages": [{"role": "user", "content": prompt}], "stream": False}
    try:
        response = requests.post(OLLAMA_URL, headers=headers, json=body, timeout=90)
    except requests.RequestException as e:
        logger.error("Ollama follow-up request failed reason=%s", e)
        return None
    if response.status_code != 200:
        logger.error("Ollama follow-up status=%s", response.status_code)
        return None
    try:
        return response.json().get("message", {}).get("content", "") or None
    except (KeyError, json.JSONDecodeError):
        return None


def get_ollama_forecast_interpretation(
    forecast_table_summary: str,
    national_forecast_line: str = "",
    data_as_of: str = "",
) -> Optional[str]:
    """
    Generate a short AI interpretation of the forecast-by-state table and national outlook.
    Output must include: Hotspots (top states), Universal precautions, CDC reference link, and key CDC facts.
    """
    _load_env()
    key = (os.environ.get("OLLAMA_API_KEY") or "").strip()
    if not key:
        return None
    cdc_facts = (
        "Key CDC facts (paraphrase in your response where relevant): measles is highly contagious (airborne; virus can remain in a room up to about 2 hours). "
        "Symptoms typically appear 7–14 days after exposure: fever, cough, runny nose, conjunctivitis, rash. "
        "MMR vaccine: 2 doses about 97%% effective; 1 dose about 93%%."
    )
    prompt = (
        f"Measles risk dashboard — forecast (data as of {data_as_of or 'N/A'}):\n"
        f"{forecast_table_summary}\n"
        f"{national_forecast_line}\n\n"
        f"{cdc_facts}\n\n"
        "Your response MUST include these sections:\n"
        "(1) **Hotspots** — List the current hotspot states (from the summary above) with their risk scores.\n"
        "(2) **Outlook** — What the state-level outlooks (High / Watch / Low) and drivers suggest overall; one practical takeaway.\n"
        "(3) **Universal precautions** — Vaccination/MMR, hand hygiene, staying home when sick, masking in high-risk settings, seeking care for concerning symptoms. Keep practical and non-alarmist.\n"
        "(4) **CDC reference** — Include this line: CDC measles information: https://www.cdc.gov/measles/about/index.html\n"
        "Use clear, non-technical language. Do not make up numbers."
    )
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS] + "\n... (truncated)"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": OLLAMA_MODELS[0], "messages": [{"role": "user", "content": prompt}], "stream": False}
    try:
        response = requests.post(OLLAMA_URL, headers=headers, json=body, timeout=90)
    except requests.RequestException as e:
        logger.error("Ollama forecast interpretation failed reason=%s", e)
        return None
    if response.status_code != 200:
        logger.error("Ollama forecast interpretation status=%s", response.status_code)
        return None
    try:
        return (response.json().get("message", {}).get("content", "") or "").strip() or None
    except (KeyError, json.JSONDecodeError):
        return None


def get_ollama_ww_nndss_report(summary_text: str, data_as_of: str = "") -> Optional[str]:
    """
    Generate an AI report comparing wastewater detection to NNDSS cases: trends, lead time, divergence, and cautions.
    """
    _load_env()
    key = (os.environ.get("OLLAMA_API_KEY") or "").strip()
    if not key:
        logger.warning("Ollama skipped: OLLAMA_API_KEY not set")
        return None
    prompt = (
        "Measles risk dashboard — Wastewater vs NNDSS (data as of %s):\n\n%s\n\n"
        "Write a short comparative report in plain language. Include: (1) Current NNDSS weekly cases trend over the last 4–8 weeks (up/down/flat). "
        "(2) Current wastewater detection_frequency trend (this is the share of reporting sites that detected measles RNA; explain what that means in one sentence). "
        "(3) Whether wastewater appears to lead reported cases and by how many weeks, based on the best lag. "
        "(4) Any divergence (e.g. wastewater rising while cases flat). "
        "(5) A brief caution: correlation does not prove causation; reporting delays and sparse case counts can make correlation noisy. "
        "Use clear, non-technical language. Do not make up numbers."
        % (data_as_of or "N/A", summary_text[:5000])
    )
    if len(prompt) > MAX_PROMPT_CHARS:
        prompt = prompt[:MAX_PROMPT_CHARS] + "\n... (truncated)"
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": OLLAMA_MODELS[0], "messages": [{"role": "user", "content": prompt}], "stream": False}
    try:
        response = requests.post(OLLAMA_URL, headers=headers, json=body, timeout=90)
    except requests.RequestException as e:
        logger.error("Ollama WW vs NNDSS report failed reason=%s", e)
        return None
    if response.status_code != 200:
        logger.error("Ollama WW vs NNDSS report status=%s", response.status_code)
        return None
    try:
        return (response.json().get("message", {}).get("content", "") or "").strip() or None
    except (KeyError, json.JSONDecodeError):
        return None


def get_ollama_state_report(
    state_name: str,
    risk_tier: str,
    risk_score: float,
    coverage_pct: Optional[float] = None,
    data_as_of: str = "",
) -> Optional[str]:
    """
    Generate a short AI narrative report for a single state. Returns plain-language summary or None.
    """
    _load_env()
    key = (os.environ.get("OLLAMA_API_KEY") or "").strip()
    if not key:
        return None
    cov_str = f"Kindergarten MMR coverage: {coverage_pct:.1f}%." if coverage_pct is not None else "Coverage data not shown."
    prompt = (
        f"Measles risk dashboard — state report for **{state_name}** (data as of {data_as_of or 'N/A'}):\n"
        f"- Risk tier: {risk_tier}\n"
        f"- Risk score (0–100): {risk_score}\n"
        f"- {cov_str}\n\n"
        "Write 2 short paragraphs in plain language: (1) what this state's risk tier and score mean for measles outbreak risk, "
        "and (2) one practical takeaway (e.g. for public health awareness or vaccination). Use clear, non-technical language. Do not make up numbers."
    )
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    body = {"model": OLLAMA_MODELS[0], "messages": [{"role": "user", "content": prompt}], "stream": False}
    try:
        response = requests.post(OLLAMA_URL, headers=headers, json=body, timeout=90)
    except requests.RequestException as e:
        logger.error("Ollama state report failed reason=%s", e)
        return None
    if response.status_code != 200:
        logger.error("Ollama state report status=%s", response.status_code)
        return None
    try:
        return (response.json().get("message", {}).get("content", "") or "").strip() or None
    except (KeyError, json.JSONDecodeError):
        return None
