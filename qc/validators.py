from __future__ import annotations

import re
from typing import Any, Dict, List

STATE_SECTION_HEADERS = [
    "## Executive summary",
    "## National outlook",
    "## Top 3 states",
    "## Confidence and uncertainty",
    "## Limitations",
]

GROUNDED_SECTION_HEADERS = [
    "## National Overview",
    "## Highest Risk States",
    "## Key Grounded Insights",
    "## Limitations",
    "## Conclusion",
]


def _collapse_ws_lower(s: str) -> str:
    """Lowercase and collapse internal whitespace for tolerant header matching."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def _header_present_in_report(report_text: str, header: str) -> bool:
    """Case-insensitive; allows extra spaces/newlines inside or around the header line."""
    return _collapse_ws_lower(header) in _collapse_ws_lower(report_text)


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _norm(s: Any) -> str:
    return str(s or "").strip().lower()


def extract_ground_truth(payload: Dict[str, Any], top_n: int = 3) -> Dict[str, Any]:
    top_states: List[Dict[str, Any]] = []
    agent2 = payload.get("agent2") or {}
    candidates = list(agent2.get("top_states_enriched") or [])
    if not candidates:
        candidates = list((payload.get("report_context_payload") or {}).get("agent2", {}).get("top_states_enriched") or [])

    def rank_key(x: Dict[str, Any]) -> int:
        try:
            return int(x.get("risk_rank"))
        except (TypeError, ValueError):
            return 10**6

    candidates.sort(key=rank_key)
    for row in candidates[:top_n]:
        top_states.append(
            {
                "state": row.get("state"),
                "risk_level": row.get("risk_tier") or row.get("risk_narrative"),
                "case_count": row.get("cases_recent"),
                "mmr_coverage": row.get("coverage"),
                "wastewater_signal": row.get("signal_dominance") or row.get("ww_recent"),
                "confidence_label": row.get("confidence_label"),
                "confidence_score": row.get("confidence_score"),
            }
        )

    national = (payload.get("report_context_payload") or {}).get("national") or {}
    national_risk_level = national.get("baseline_tier") or payload.get("baseline_tier") or "unknown"

    return {"top_states": top_states, "national_risk_level": national_risk_level}


def _states_in_report(report_text: str) -> List[str]:
    return re.findall(r"\b[A-Z]{2}\b", report_text)


def _numbers_in_text(text: str) -> List[float]:
    return [float(x) for x in re.findall(r"[-+]?\d*\.?\d+", text)]


def validate_report(
    report_text: str,
    ground_truth: Dict[str, Any],
    *,
    section_headers: List[str] | None = None,
) -> Dict[str, Any]:
    text = report_text or ""
    lower = text.lower()

    headers = section_headers if section_headers is not None else STATE_SECTION_HEADERS
    required_sections_present = sum(1 for h in headers if _header_present_in_report(text, h))
    required_sections_rate = required_sections_present / max(1, len(headers))

    gt_states = [str(s.get("state", "")).upper() for s in ground_truth.get("top_states", []) if s.get("state")]
    mentioned_state_codes = {s.upper() for s in _states_in_report(text)}

    covered = [s for s in gt_states if s in mentioned_state_codes or s.lower() in lower]
    top_state_coverage_rate = len(covered) / max(1, len(gt_states))

    hallucinated_states = [s for s in mentioned_state_codes if s not in set(gt_states)]

    risk_match_hits = 0
    wastewater_match_hits = 0
    enrichment_hits = 0

    expected_numbers: List[float] = []
    matched_numbers = 0
    for row in ground_truth.get("top_states", []):
        state = str(row.get("state", "")).upper()
        if state:
            risk_label = _norm(row.get("risk_level"))
            if risk_label and risk_label in lower:
                risk_match_hits += 1
            ws = _norm(row.get("wastewater_signal"))
            if ws and ws in lower:
                wastewater_match_hits += 1
            if _norm(row.get("confidence_label")) in lower:
                enrichment_hits += 1

        for k in ("case_count", "mmr_coverage", "confidence_score"):
            val = _to_float(row.get(k))
            if val is None:
                continue
            expected_numbers.append(val)
            if re.search(rf"\b{re.escape(str(int(round(val))))}\b", text) or re.search(rf"{val:.1f}", text) or re.search(rf"{val:.2f}", text):
                matched_numbers += 1

    numeric_accuracy_rate = matched_numbers / max(1, len(expected_numbers))
    risk_match_rate = risk_match_hits / max(1, len(ground_truth.get("top_states", [])))
    wastewater_match_rate = wastewater_match_hits / max(1, len(ground_truth.get("top_states", [])))
    enrichment_usage_rate = enrichment_hits / max(1, len(ground_truth.get("top_states", [])))

    all_report_numbers = _numbers_in_text(text)
    hallucinated_number_count = max(0, len(all_report_numbers) - matched_numbers - 10)

    confidence_metrics = validate_confidence_usage(report_text=text, ground_truth=ground_truth)

    return {
        "hallucinated_state_count": len(hallucinated_states),
        "hallucinated_number_count": hallucinated_number_count,
        "top_state_coverage_rate": top_state_coverage_rate,
        "numeric_accuracy_rate": numeric_accuracy_rate,
        "risk_match_rate": risk_match_rate,
        "wastewater_match_rate": wastewater_match_rate,
        "enrichment_usage_rate": enrichment_usage_rate,
        "required_sections_rate": required_sections_rate,
        **confidence_metrics,
    }


def validate_confidence_usage(report_text: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    text = report_text or ""
    lower = text.lower()

    label_total = 0
    label_matches = 0
    low_mod_total = 0
    low_mod_disclosed = 0
    score_total = 0
    score_matches = 0
    unsupported_confidence_claim_count = 0

    misuse: List[str] = []

    for row in ground_truth.get("top_states", []):
        state = str(row.get("state", "")).upper()
        label = _norm(row.get("confidence_label"))
        score = _to_float(row.get("confidence_score"))

        if label:
            label_total += 1
            if label in lower:
                label_matches += 1
            else:
                misuse.append(f"Missing confidence label mention for {state}: expected {label}")

            if label in {"low", "moderate", "medium"}:
                low_mod_total += 1
                has_uncertainty = any(x in lower for x in ["uncertain", "uncertainty", "limited", "low confidence", "moderate confidence"]) and "limitations" in lower
                if has_uncertainty:
                    low_mod_disclosed += 1
                else:
                    misuse.append(f"No uncertainty disclosure for {state} with {label} confidence")

        if score is not None:
            score_total += 1
            if re.search(rf"{score:.2f}", text) or re.search(rf"{score:.1f}", text):
                score_matches += 1

    if "high confidence" in lower and all(_norm(s.get("confidence_label")) != "high" for s in ground_truth.get("top_states", [])):
        unsupported_confidence_claim_count += 1
        misuse.append("Report claims high confidence unsupported by input labels")

    return {
        "confidence_label_match_rate": label_matches / max(1, label_total),
        "low_confidence_disclosure_rate": low_mod_disclosed / max(1, low_mod_total),
        "confidence_score_match_rate": score_matches / max(1, score_total),
        "unsupported_confidence_claim_count": unsupported_confidence_claim_count,
        "confidence_misuse": misuse,
    }
