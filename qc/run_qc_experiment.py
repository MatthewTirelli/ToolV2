from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.model_runner import load_and_model
from agents.report_writer_agent import build_agent2_enrichment, build_precomputed_payload, select_top_bottom_states
from agents.tools.census_tools import CENSUS_TOOL_NAME, execute_census_tool
from qc.scoring import compute_validity_score
from qc.statistical_analysis import analyze_results
from qc.validators import GROUNDED_SECTION_HEADERS, STATE_SECTION_HEADERS, extract_ground_truth, validate_report
from qc.report_generation import write_charts, write_summary_markdown

PROMPTS = ROOT / "qc" / "prompts"
OUT_DIR = ROOT / "qc" / "outputs"
RESULTS_CSV = OUT_DIR / "qc_results.csv"
SUMMARY_MD = OUT_DIR / "qc_summary.md"
CHART_DIR = OUT_DIR / "qc_charts"


def _section_headers_for_mode(mode: str) -> List[str]:
    return GROUNDED_SECTION_HEADERS if mode == "grounded" else STATE_SECTION_HEADERS


def _load_qc_env() -> None:
    """Load repo root (and optional app) .env for standalone `python qc/run_qc_experiment.py`."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv(ROOT / ".env")
    app_env = ROOT / "app" / ".env"
    if app_env.is_file():
        load_dotenv(app_env, override=True, encoding="utf-8-sig")


def _load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text()


def _openai_client() -> OpenAI:
    key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    if not key:
        raise ValueError("OPENAI_API_KEY is required for QC experiment runs.")
    return OpenAI(api_key=key)


def _dominant_confidence(gt: Dict[str, Any]) -> tuple[str, float | None]:
    labels = [str(s.get("confidence_label", "")).lower() for s in gt.get("top_states", []) if s.get("confidence_label")]
    scores = [float(s.get("confidence_score")) for s in gt.get("top_states", []) if s.get("confidence_score") is not None]
    if not labels:
        return ("unknown", None)
    priority = {"low": 0, "moderate": 1, "medium": 1, "high": 2}
    labels.sort(key=lambda x: priority.get(x, -1))
    return labels[0], (sum(scores) / len(scores) if scores else None)


def build_trial_payload(seed: int) -> Dict[str, Any]:
    random.seed(seed)
    d = load_and_model(use_cache=True)
    sr = d.get("state_risk_df")
    if sr is None or sr.empty:
        raise RuntimeError("state_risk_df unavailable; cannot run QC payload extraction")

    top_codes, bottom_codes, _ = select_top_bottom_states(sr, n=5)
    all_codes = sorted({str(s).strip().upper() for s in sr["state"].dropna()})
    census_by_state: Dict[str, Any] = {}
    if all_codes:
        raw_json = execute_census_tool(CENSUS_TOOL_NAME, json.dumps({"state_codes": all_codes}))
        census_by_state = json.loads(raw_json)

    agent2 = build_agent2_enrichment(
        top_codes=top_codes,
        bottom_codes=bottom_codes,
        census_by_state=census_by_state,
        state_risk_df=sr,
    )

    agent1_stub = {"assistant_text": "", "last_snapshot": {"summary": {}}}
    payload = build_precomputed_payload(data_bundle=d, agent1=agent1_stub, agent2_enrichment=agent2)
    return {"report_context_payload": payload, "agent2": agent2, **d}


def _generate_report(client: OpenAI, structured_input: Dict[str, Any], mode: str, model: str) -> str:
    if mode == "baseline":
        system = "You are a concise measles situational-awareness briefer. Use only provided structured JSON."
        user = _load_prompt("baseline_prompt.txt") + "\n\nStructured input JSON:\n" + json.dumps(structured_input, ensure_ascii=False)
    else:
        system = "You are a strict grounded report writer."
        tpl = _load_prompt("grounded_prompt.txt")
        user = tpl.replace("{structured_input_json}", json.dumps(structured_input, ensure_ascii=False))

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return (resp.choices[0].message.content or "").strip()


def _grade_optional(client: OpenAI, structured_input: Dict[str, Any], report_text: str, model: str) -> Dict[str, Any]:
    prompt = _load_prompt("grader_prompt.txt")
    msg = (
        prompt
        + "\n\nSTRUCTURED_GROUND_TRUTH_JSON:\n"
        + json.dumps(structured_input, ensure_ascii=False)
        + "\n\nREPORT_TEXT:\n"
        + report_text
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": msg}],
            temperature=0.0,
        )
        text = (resp.choices[0].message.content or "{}").strip()
        return json.loads(text)
    except Exception:
        return {}


def _apply_graders_at_end(client: OpenAI, df: pd.DataFrame, model: str) -> pd.DataFrame:
    """Run LLM grader on each row after all reports exist (separate pass from report generation)."""
    if "ground_truth_json" not in df.columns:
        return df

    grader_runs = []
    for _, row in df.iterrows():
        gt = json.loads(row["ground_truth_json"])
        report_text = str(row.get("report_text", ""))
        grader = _grade_optional(client, gt, report_text, model=model)
        grader_runs.append(grader)

    out_rows: List[Dict[str, Any]] = []
    for i, (_, row) in enumerate(df.iterrows()):
        gt = json.loads(row["ground_truth_json"])
        report_text = str(row.get("report_text", ""))
        grader = grader_runs[i]
        mode = str(row.get("mode", ""))
        metrics = validate_report(
            report_text=report_text,
            ground_truth=gt,
            section_headers=_section_headers_for_mode(mode),
        )
        score = compute_validity_score(metrics, concision_score_1_5=grader.get("concision_score"))
        d = row.to_dict()
        d.update(metrics)
        d.update(score)
        d["grader_overall_score"] = grader.get("overall_score")
        d["grader_payload"] = json.dumps(grader, ensure_ascii=False) if grader else ""
        d["unsupported_claims"] = json.dumps(grader.get("unsupported_claims", []), ensure_ascii=False) if grader else "[]"
        d["confidence_misuse"] = json.dumps(
            grader.get("confidence_misuse", []), ensure_ascii=False
        ) if grader else json.dumps(metrics.get("confidence_misuse", []), ensure_ascii=False)
        d["missing_required_elements"] = json.dumps(
            grader.get("missing_required_elements", []), ensure_ascii=False
        ) if grader else "[]"
        out_rows.append(d)

    return pd.DataFrame(out_rows).drop(columns=["ground_truth_json"], errors="ignore")


def run_experiment(
    n_trials: int,
    mode: str,
    model: str,
    *,
    with_grader: bool,
    grader_interleaved: bool,
) -> pd.DataFrame:
    client = _openai_client()
    rows: List[Dict[str, Any]] = []
    run_modes = [mode] if mode in {"baseline", "grounded"} else ["baseline", "grounded"]
    use_interleaved = with_grader and grader_interleaved
    use_end = with_grader and not grader_interleaved

    for trial_id in range(1, n_trials + 1):
        trial_payload = build_trial_payload(seed=trial_id)
        gt = extract_ground_truth(trial_payload, top_n=3)
        conf_label, conf_score = _dominant_confidence(gt)

        for m in run_modes:
            t0 = time.time()
            report_text = _generate_report(client, gt, mode=m, model=model)
            runtime_s = time.time() - t0

            metrics = validate_report(
                report_text=report_text,
                ground_truth=gt,
                section_headers=_section_headers_for_mode(m),
            )
            if use_interleaved:
                grader = _grade_optional(client, gt, report_text, model=model)
            else:
                grader = {}
            score = compute_validity_score(
                metrics,
                concision_score_1_5=grader.get("concision_score") if use_interleaved else None,
            )

            row: Dict[str, Any] = {
                "trial_id": trial_id,
                "mode": m,
                "runtime_s": round(runtime_s, 3),
                "dominant_confidence_label": conf_label,
                "dominant_confidence_score": conf_score,
                "report_text": report_text,
                **metrics,
                **score,
                "grader_overall_score": grader.get("overall_score") if use_interleaved else None,
                "grader_payload": json.dumps(grader, ensure_ascii=False) if (use_interleaved and grader) else "",
                "unsupported_claims": json.dumps(grader.get("unsupported_claims", []), ensure_ascii=False)
                if use_interleaved
                else "[]",
                "confidence_misuse": json.dumps(grader.get("confidence_misuse", []), ensure_ascii=False)
                if use_interleaved
                else json.dumps(metrics.get("confidence_misuse", []), ensure_ascii=False),
                "missing_required_elements": json.dumps(grader.get("missing_required_elements", []), ensure_ascii=False)
                if use_interleaved
                else "[]",
            }
            if use_end:
                row["ground_truth_json"] = json.dumps(gt, ensure_ascii=False)
            rows.append(row)

    df = pd.DataFrame(rows)
    if use_end and not df.empty:
        df = _apply_graders_at_end(client, df, model)
    return df


def main() -> None:
    _load_qc_env()
    parser = argparse.ArgumentParser(description="Run QC experiment for final AI reports")
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--mode", choices=["baseline", "grounded", "compare"], default="compare")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", "gpt-5.4-mini"))
    parser.add_argument("--with-grader", action="store_true", help="Call LLM grader (default: one batch pass after all reports are done)")
    parser.add_argument(
        "--grader-interleaved",
        action="store_true",
        help="Run grader immediately after each report (old behavior; 2x API interleaving with generation)",
    )
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    CHART_DIR.mkdir(parents=True, exist_ok=True)

    if args.with_grader and not args.grader_interleaved:
        print("Grader will run in a second pass after all report generations complete.", flush=True)

    df = run_experiment(
        n_trials=args.n_trials,
        mode=args.mode,
        model=args.model,
        with_grader=args.with_grader,
        grader_interleaved=args.grader_interleaved,
    )

    df.to_csv(RESULTS_CSV, index=False)
    summary = analyze_results(df)
    write_summary_markdown(summary, df, SUMMARY_MD)
    write_charts(df, CHART_DIR)

    print(f"Wrote {RESULTS_CSV}")
    print(f"Wrote {SUMMARY_MD}")
    print(f"Wrote charts in {CHART_DIR}")


if __name__ == "__main__":
    main()
