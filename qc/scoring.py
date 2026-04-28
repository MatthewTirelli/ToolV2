from __future__ import annotations

from typing import Dict

WEIGHTS = {
    "numeric_accuracy_rate": 25,
    "top_state_coverage_rate": 15,
    "risk_match_rate": 10,
    "wastewater_match_rate": 10,
    "enrichment_usage_rate": 10,
    "confidence_fidelity": 15,
    "required_sections_rate": 10,
    "concision_score": 5,
}


def _clamp(v: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, v))


def compute_validity_score(metrics: Dict[str, float], concision_score_1_5: float | None = None) -> Dict[str, float | bool]:
    confidence_fidelity = (
        0.5 * _clamp(float(metrics.get("confidence_label_match_rate", 0.0)))
        + 0.3 * _clamp(float(metrics.get("low_confidence_disclosure_rate", 0.0)))
        + 0.2 * _clamp(float(metrics.get("confidence_score_match_rate", 0.0)))
    )

    concision_norm = 0.6
    if concision_score_1_5 is not None:
        concision_norm = _clamp((float(concision_score_1_5) - 1.0) / 4.0)

    weighted = 0.0
    weighted += WEIGHTS["numeric_accuracy_rate"] * _clamp(float(metrics.get("numeric_accuracy_rate", 0.0)))
    weighted += WEIGHTS["top_state_coverage_rate"] * _clamp(float(metrics.get("top_state_coverage_rate", 0.0)))
    weighted += WEIGHTS["risk_match_rate"] * _clamp(float(metrics.get("risk_match_rate", 0.0)))
    weighted += WEIGHTS["wastewater_match_rate"] * _clamp(float(metrics.get("wastewater_match_rate", 0.0)))
    weighted += WEIGHTS["enrichment_usage_rate"] * _clamp(float(metrics.get("enrichment_usage_rate", 0.0)))
    weighted += WEIGHTS["confidence_fidelity"] * confidence_fidelity
    weighted += WEIGHTS["required_sections_rate"] * _clamp(float(metrics.get("required_sections_rate", 0.0)))
    weighted += WEIGHTS["concision_score"] * concision_norm

    hallucination_penalty = min(
        25.0,
        8.0 * float(metrics.get("hallucinated_state_count", 0))
        + 2.0 * float(metrics.get("hallucinated_number_count", 0))
        + 6.0 * float(metrics.get("unsupported_confidence_claim_count", 0)),
    )

    validity = max(0.0, weighted - hallucination_penalty)
    passed = bool(
        validity >= 80.0
        and float(metrics.get("hallucinated_state_count", 0)) == 0.0
        and float(metrics.get("hallucinated_number_count", 0)) <= 1.0
        and float(metrics.get("numeric_accuracy_rate", 0.0)) >= 0.8
        and float(metrics.get("top_state_coverage_rate", 0.0)) >= 1.0
    )

    return {
        "validity_score_0_100": round(validity, 2),
        "passed_absolute_validity": passed,
    }
