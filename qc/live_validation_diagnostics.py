"""
Read-only diagnostics for live report validation (no scoring/validator changes).

Compares live wiring vs batch QC (`run_qc_experiment`) expectations.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List

from qc.scoring import compute_validity_score
from qc.validators import (
    GROUNDED_SECTION_HEADERS,
    STATE_SECTION_HEADERS,
    extract_ground_truth,
    validate_report,
)
import qc.validators as _v

_REPO_ROOT = Path(__file__).resolve().parents[1]


def grounded_prompt_file_info() -> Dict[str, Any]:
    """Resolved path to qc/prompts/grounded_prompt.txt (same file Agent 3 loads via report_writer)."""
    path = _REPO_ROOT / "qc" / "prompts" / "grounded_prompt.txt"
    p = path.resolve()
    info: Dict[str, Any] = {"path": str(p), "exists": path.is_file()}
    if path.is_file():
        info["size_bytes"] = path.stat().st_size
    return info


def tool_payload_from_multi_agent(ma: Dict[str, Any]) -> Dict[str, Any]:
    """JSON object returned by get_precomputed_report_inputs (same as Agent 3 tool body)."""
    p = ma.get("report_context_payload")
    return p if isinstance(p, dict) else {}


def payloads_align_for_validation(ma: Dict[str, Any]) -> Dict[str, Any]:
    """
    Report generator receives `payload` = build_precomputed_payload dict (tool JSON).
    extract_ground_truth(ma) prefers ma['agent2'], else report_context_payload['agent2'].
    """
    tool_body = tool_payload_from_multi_agent(ma)

    def _state_codes(rows: List[Any]) -> List[str]:
        return [str(x.get("state")) for x in rows if isinstance(x, dict)]

    top_tool = (tool_body.get("agent2") or {}).get("top_states_enriched") or []
    top_ma = (ma.get("agent2") or {}).get("top_states_enriched") or []
    codes_tool = _state_codes(list(top_tool))
    codes_ma = _state_codes(list(top_ma))
    ma_agent2_core = {k: v for k, v in (ma.get("agent2") or {}).items() if k != "meta"}
    tool_agent2_only = {"agent2": tool_body.get("agent2")}
    same_core = json.dumps(tool_agent2_only, sort_keys=True, default=str) == json.dumps(
        {"agent2": ma_agent2_core}, sort_keys=True, default=str
    )
    return {
        "generator_tool_returns": "multi_agent_result['report_context_payload'] (serialized as tool output)",
        "validator_uses": "full multi_agent_result; extract_ground_truth uses top-level agent2 first",
        "top_states_enriched_count_tool": len(codes_tool),
        "top_states_enriched_count_ma_agent2": len(codes_ma),
        "top_state_codes_tool": codes_tool,
        "top_state_codes_ma_agent2": codes_ma,
        "agent2_enriched_block_matches_tool": same_core,
    }


def section_header_diagnosis(report_text: str) -> Dict[str, Any]:
    def scan(headers: List[str]) -> Dict[str, Any]:
        present = [h for h in headers if _v._header_present_in_report(report_text, h)]
        missing = [h for h in headers if h not in present]
        return {"required": list(headers), "detected": present, "missing": missing}

    return {
        "grounded_prompt_headers": scan(GROUNDED_SECTION_HEADERS),
        "batch_baseline_headers": scan(STATE_SECTION_HEADERS),
    }


def _to_float(v: Any) -> float | None:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def numeric_diagnosis(report_text: str, ground_truth: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Per expected numeric field: expected value, regex matched, closest number in report."""
    text = report_text or ""
    all_nums = _v._numbers_in_text(text)
    rows_out: List[Dict[str, Any]] = []
    for row in ground_truth.get("top_states", []):
        state = str(row.get("state", "")).upper()
        for field in ("case_count", "mmr_coverage", "confidence_score"):
            val = _to_float(row.get(field))
            if val is None:
                continue
            pat_int = rf"\b{re.escape(str(int(round(val))))}\b"
            pat_1 = rf"{val:.1f}"
            pat_2 = rf"{val:.2f}"
            found = bool(
                re.search(pat_int, text) or re.search(pat_1, text) or re.search(pat_2, text)
            )
            closest = None
            if all_nums:
                closest = min(all_nums, key=lambda x: abs(x - val))
            rows_out.append(
                {
                    "state": state,
                    "field": field,
                    "expected": val,
                    "matched_by_validator_rules": found,
                    "closest_number_in_report": closest,
                    "delta_to_closest": (abs(closest - val) if closest is not None else None),
                }
            )
    return rows_out


def hallucination_diagnosis(report_text: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Mirror validator state lists: extra US codes vs expected top-3 set."""
    text = report_text or ""
    gt_states = {str(s.get("state", "")).upper() for s in ground_truth.get("top_states", []) if s.get("state")}
    mentioned = {s.upper() for s in _v._states_in_report(text)}
    hallucinated = sorted(mentioned - gt_states)
    return {"expected_top_state_codes": sorted(gt_states), "all_state_codes_in_report": sorted(mentioned), "codes_not_in_ground_truth_top3": hallucinated}


def missing_top_states(report_text: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    text = report_text or ""
    lower = text.lower()
    gt_states = [str(s.get("state", "")).upper() for s in ground_truth.get("top_states", []) if s.get("state")]
    mentioned = {s.upper() for s in _v._states_in_report(text)}
    covered = [s for s in gt_states if s in mentioned or s.lower() in lower]
    missing = [s for s in gt_states if s not in covered]
    extra = sorted(mentioned - set(gt_states))
    return {"expected_top_states": gt_states, "covered": covered, "missing": missing, "extra_state_codes_vs_top3": extra}


def summarize_failed_checks(metrics: Dict[str, Any]) -> List[str]:
    """Human-readable failures vs common pass thresholds (mirrors intent of scoring, not new logic)."""
    issues: List[str] = []
    m = metrics
    if float(m.get("hallucinated_state_count", 0) or 0) > 0:
        issues.append(f"Hallucinated/extra state signals: count={m.get('hallucinated_state_count')}")
    if float(m.get("hallucinated_number_count", 0) or 0) > 1:
        issues.append(f"Hallucinated number penalty signal: count={m.get('hallucinated_number_count')}")
    if float(m.get("top_state_coverage_rate", 1) or 0) < 1.0:
        issues.append(f"Top-state coverage below 100%: {m.get('top_state_coverage_rate')}")
    if float(m.get("numeric_accuracy_rate", 1) or 0) < 0.8:
        issues.append(f"Numeric accuracy below 0.8: {m.get('numeric_accuracy_rate')}")
    if float(m.get("required_sections_rate", 1) or 0) < 1.0:
        issues.append(f"Required sections incomplete: rate={m.get('required_sections_rate')}")
    misuse = m.get("confidence_misuse") or []
    if misuse:
        issues.append(f"Confidence wording issues ({len(misuse)}): see list")
    if float(m.get("unsupported_confidence_claim_count", 0) or 0) > 0:
        issues.append(f"Unsupported confidence claims: {m.get('unsupported_confidence_claim_count')}")
    return issues


def run_header_variant_metrics(report_text: str, ground_truth: Dict[str, Any]) -> Dict[str, Any]:
    """Same validator, different section header lists (batch QC uses mode-specific headers)."""
    m_default = validate_report(report_text=report_text, ground_truth=ground_truth)
    m_grounded = validate_report(
        report_text=report_text,
        ground_truth=ground_truth,
        section_headers=GROUNDED_SECTION_HEADERS,
    )
    m_baseline = validate_report(
        report_text=report_text,
        ground_truth=ground_truth,
        section_headers=STATE_SECTION_HEADERS,
    )
    return {
        "production_grounded_headers": {
            "section_headers": "GROUNDED_SECTION_HEADERS (batch QC grounded + live app)",
            "metrics": m_grounded,
            "score": compute_validity_score(m_grounded, concision_score_1_5=None),
        },
        "validator_state_headers_only": {
            "section_headers": "STATE_SECTION_HEADERS (validate_report default)",
            "metrics": m_default,
            "score": compute_validity_score(m_default, concision_score_1_5=None),
        },
        "batch_qc_baseline_wiring": {
            "section_headers": "STATE_SECTION_HEADERS",
            "metrics": m_baseline,
            "score": compute_validity_score(m_baseline, concision_score_1_5=None),
        },
    }


def write_live_validation_debug_bundle(
    out_dir: Path,
    *,
    ma: Dict[str, Any],
    report_text: str,
    ground_truth: Dict[str, Any],
    prompt_info: Dict[str, Any],
    alignment: Dict[str, Any],
) -> Dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = tool_payload_from_multi_agent(ma)
    (out_dir / "current_input_payload.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )
    (out_dir / "current_report.md").write_text(report_text, encoding="utf-8")
    (out_dir / "validator_truth_object.json").write_text(
        json.dumps(ground_truth, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    variants = run_header_variant_metrics(report_text, ground_truth)
    sections = section_header_diagnosis(report_text)
    nums = numeric_diagnosis(report_text, ground_truth)
    states = missing_top_states(report_text, ground_truth)
    hall = hallucination_diagnosis(report_text, ground_truth)
    prod_metrics = variants["production_grounded_headers"]["metrics"]

    bundle: Dict[str, Any] = {
        "prompt_source": prompt_info,
        "payload_alignment": alignment,
        "section_headers": sections,
        "missing_top_states": states,
        "state_code_hallucinations_vs_top3": hall,
        "numeric_expected_vs_found": nums,
        "failed_checks": summarize_failed_checks(prod_metrics),
        "header_and_wiring_comparison": variants,
        "confidence_misuse_detail": prod_metrics.get("confidence_misuse", []),
    }
    (out_dir / "live_validation_result.json").write_text(
        json.dumps(bundle, indent=2, ensure_ascii=False, default=str),
        encoding="utf-8",
    )

    report_md = _build_diagnostic_markdown(bundle)
    (out_dir / "DIAGNOSTIC_REPORT.md").write_text(report_md, encoding="utf-8")
    return bundle


def _build_diagnostic_markdown(bundle: Dict[str, Any]) -> str:
    vprod = bundle["header_and_wiring_comparison"]["production_grounded_headers"]
    vstate = bundle["header_and_wiring_comparison"]["validator_state_headers_only"]
    lines = [
        "# Live validation diagnostic report",
        "",
        "## 1. Prompt source",
        f"- Resolved path: `{bundle['prompt_source'].get('path')}`",
        f"- Exists: {bundle['prompt_source'].get('exists')}",
        f"- Bytes: {bundle['prompt_source'].get('size_bytes')}",
        "- Agent 3 user content is loaded from this file (with `{structured_input_json}` replaced by tool instructions).",
        "- No full grounded template is hardcoded in `AGENT3_SYSTEM_BRIEF` (only a short tool contract).",
        "",
        "## 2. Payload alignment",
        json.dumps(bundle.get("payload_alignment"), indent=2),
        "",
        "## 3. Section headers (Grounded Prompt vs Baseline-style list)",
        "### Grounded Prompt (Production) / offline QC (`GROUNDED_SECTION_HEADERS`)",
        f"- Missing: {bundle['section_headers']['grounded_prompt_headers']['missing']}",
        f"- Detected: {bundle['section_headers']['grounded_prompt_headers']['detected']}",
        "### Baseline Prompt-style (`STATE_SECTION_HEADERS`, contrast only)",
        f"- Missing: {bundle['section_headers']['batch_baseline_headers']['missing']}",
        f"- Detected: {bundle['section_headers']['batch_baseline_headers']['detected']}",
        "",
        "## 4. Scores: Grounded Prompt (Production) headers vs Baseline-style headers",
        f"- **Grounded Prompt (Production) / offline parity**: report quality score={vprod['score'].get('validity_score_0_100')}, "
        f"required_sections_rate={vprod['metrics'].get('required_sections_rate')}",
        f"- **Baseline-style headers only** (contrast): report quality score={vstate['score'].get('validity_score_0_100')}, "
        f"required_sections_rate={vstate['metrics'].get('required_sections_rate')}",
        "",
        "## 5. Failed checks (Grounded Prompt production wiring)",
        "\n".join(f"- {x}" for x in bundle.get("failed_checks", [])) or "- (none flagged by heuristics)",
        "",
        "## 6. Top states",
        json.dumps(bundle.get("missing_top_states"), indent=2),
        "",
        "## 7. Numeric expectations (validator rules)",
        json.dumps(bundle.get("numeric_expected_vs_found"), indent=2),
        "",
        "## 8. Concision",
        "- Live `compute_validity_score(..., concision_score_1_5=None)` uses fixed concision_norm=0.6; "
        "batch QC may pass a grader concision score. Small score deltas can come from that alone.",
        "",
        "## 9. Notes (automated)",
    ]
    rs_prod = float(vprod["metrics"].get("required_sections_rate") or 0)
    rs_state = float(vstate["metrics"].get("required_sections_rate") or 0)
    causes: List[str] = []
    if rs_prod > rs_state + 0.2:
        causes.append(
            "- **Contrast:** Grounded Prompt (Production) markdown scores higher on `required_sections_rate` with "
            "`GROUNDED_SECTION_HEADERS` than with the Baseline-style header list (expected)."
        )
    if bundle.get("missing_top_states", {}).get("missing"):
        causes.append("- **Also:** Missing mentions of expected top state(s).")
    if bundle.get("numeric_expected_vs_found"):
        bad = [x for x in bundle["numeric_expected_vs_found"] if not x.get("matched_by_validator_rules")]
        if bad:
            causes.append(f"- **Also:** {len(bad)} expected numeric field(s) not matched by strict regex rules.")
    if not causes:
        causes.append("- No extra automated notes.")
    lines.extend(causes)
    lines.append("")
    lines.append(f"Strict pass (Grounded Prompt production headers): **{vprod['score'].get('passed_absolute_validity')}**")
    return "\n".join(lines)
