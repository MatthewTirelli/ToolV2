from __future__ import annotations

import json
import math
from statistics import mean, pstdev
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd


def _as_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    sl = s.astype(str).str.lower().str.strip()
    return sl.isin(["true", "1", "yes"])


def _json_list_nonempty(cell: Any) -> bool:
    if cell is None or pd.isna(cell):
        return False
    raw = str(cell).strip()
    if not raw:
        return False
    try:
        v = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return False
    return isinstance(v, list) and len(v) > 0


def _failure_mode_table(df: pd.DataFrame) -> Tuple[List[Dict[str, Any]], str]:
    """Row-level failure flags for interpretability (not mutually exclusive)."""
    n = len(df)
    if n == 0:
        return [], "No rows in results; failure modes were not computed."

    def row(name: str, mask: pd.Series) -> Dict[str, Any]:
        c = int(mask.sum())
        return {"name": name, "count": c, "percent": 100.0 * c / n}

    rows: List[Dict[str, Any]] = []
    if "passed_absolute_validity" in df.columns:
        passed = _as_bool_series(df["passed_absolute_validity"])
        rows.append(row("Did not pass absolute validity", ~passed))

    if "hallucinated_state_count" in df.columns:
        rows.append(row("Hallucinated state mention(s)", pd.to_numeric(df["hallucinated_state_count"], errors="coerce").fillna(0) > 0))
    if "hallucinated_number_count" in df.columns:
        rows.append(row("Hallucinated number signal(s)", pd.to_numeric(df["hallucinated_number_count"], errors="coerce").fillna(0) > 0))
    if "top_state_coverage_rate" in df.columns:
        rows.append(row("Incomplete top-state coverage", pd.to_numeric(df["top_state_coverage_rate"], errors="coerce") < 1.0))
    if "numeric_accuracy_rate" in df.columns:
        rows.append(row("Numeric accuracy below 0.95", pd.to_numeric(df["numeric_accuracy_rate"], errors="coerce") < 0.95))
    if "required_sections_rate" in df.columns:
        rows.append(row("Missing required sections", pd.to_numeric(df["required_sections_rate"], errors="coerce") < 1.0))
    if "confidence_label_match_rate" in df.columns:
        rows.append(row("Confidence label match below 1.0", pd.to_numeric(df["confidence_label_match_rate"], errors="coerce") < 1.0))
    if "low_confidence_disclosure_rate" in df.columns:
        lcd = pd.to_numeric(df["low_confidence_disclosure_rate"], errors="coerce")
        dom = df["dominant_confidence_label"].astype(str).str.lower().str.strip() if "dominant_confidence_label" in df.columns else pd.Series([""] * n)
        # When all top states are high, the validator uses a vacuous denominator and yields 0.0 — not a disclosure failure.
        disclosure_applicable = ~((lcd == 0.0) & dom.eq("high"))
        rows.append(
            row(
                "Low/moderate confidence disclosure below 1.0 (where applicable)",
                (lcd < 1.0) & disclosure_applicable,
            )
        )
    if "unsupported_confidence_claim_count" in df.columns:
        rows.append(
            row(
                "Unsupported structured confidence claim(s)",
                pd.to_numeric(df["unsupported_confidence_claim_count"], errors="coerce").fillna(0) > 0,
            )
        )
    if "unsupported_claims" in df.columns:
        rows.append(row("Grader: unsupported_claims non-empty", df["unsupported_claims"].map(_json_list_nonempty)))
    if "confidence_misuse" in df.columns:
        rows.append(row("Grader: confidence_misuse non-empty", df["confidence_misuse"].map(_json_list_nonempty)))

    rows.sort(key=lambda x: x["count"], reverse=True)
    top = [r for r in rows if r["count"] > 0][:3]
    if not top:
        interp = (
            "No rows triggered the listed structural failure flags. This does not imply absence of subtle errors; "
            "see validator definitions and row-level columns in `qc_results.csv`."
        )
    else:
        parts = [f"{r['name']} ({r['percent']:.1f}% of rows)" for r in top]
        interp = (
            "The most common failure signals were: " + "; ".join(parts) + ". "
            "Counts can overlap (one row may trigger multiple flags). "
            "Use the table above and per-row metrics in `qc_results.csv` to prioritize fixes."
        )
    return rows, interp


def _pearson_confidence_validity(df: pd.DataFrame) -> Tuple[float | None, str | None]:
    """Pearson r between upstream confidence score column and validity; returns (r, column_used)."""
    score_col: str | None = None
    if "confidence_score" in df.columns and pd.to_numeric(df["confidence_score"], errors="coerce").notna().any():
        score_col = "confidence_score"
    elif "dominant_confidence_score" in df.columns:
        score_col = "dominant_confidence_score"
    if score_col is None or "validity_score_0_100" not in df.columns:
        return None, None
    numeric = pd.to_numeric(df[score_col], errors="coerce")
    valid = pd.to_numeric(df["validity_score_0_100"], errors="coerce")
    mask = numeric.notna() & valid.notna()
    if mask.sum() <= 2:
        return None, score_col
    a = numeric[mask].astype(float)
    b = valid[mask].astype(float)
    if float(a.std()) == 0 or float(b.std()) == 0:
        return None, score_col
    c = float(a.corr(b))
    if not math.isfinite(c):
        return None, score_col
    return c, score_col


def _ci95(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    m = mean(values)
    if len(values) == 1:
        return (m, m)
    sd = pstdev(values)
    margin = 1.96 * (sd / math.sqrt(len(values)))
    return (m - margin, m + margin)


def _paired_ttest(a: list[float], b: list[float]) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        return None
    diffs = [x - y for x, y in zip(a, b)]
    md = mean(diffs)
    sd = pstdev(diffs)
    if sd == 0:
        return 0.0
    return md / (sd / math.sqrt(len(diffs)))


def _cohens_d(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    ma, mb = mean(a), mean(b)
    sda = pstdev(a) if len(a) > 1 else 0.0
    sdb = pstdev(b) if len(b) > 1 else 0.0
    pooled = math.sqrt((sda * sda + sdb * sdb) / 2.0)
    if pooled == 0:
        return 0.0
    return (ma - mb) / pooled


def _wilson_ci(pass_count: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (95% default). Returns (low, high) in [0, 1]."""
    if n <= 0:
        return (0.0, 0.0)
    ph = pass_count / n
    z2 = z * z
    denom = 1.0 + z2 / n
    center = (ph + z2 / (2.0 * n)) / denom
    rad = z * math.sqrt((ph * (1.0 - ph) + z2 / (4.0 * n)) / n) / denom
    return (max(0.0, center - rad), min(1.0, center + rad))


def _binom_cdf_upto(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    acc = 0.0
    for x in range(0, k + 1):
        acc += math.comb(n, x) * (p**x) * ((1.0 - p) ** (n - x))
    return min(1.0, max(0.0, acc))


def _mcnemar_p_value_exact_two_sided(b: int, c: int) -> float:
    """
    Exact two-sided McNemar p-value for discordant counts b, c (Binomial(b+c, 0.5) null).
    p = 2 * min(P(X <= min(b,c)), 1 - P(X <= min(b,c)-1)), capped at 1.0.
    """
    n_disc = b + c
    if n_disc <= 0:
        return 1.0
    k = min(b, c)
    t1 = _binom_cdf_upto(k, n_disc, 0.5)
    t2 = 1.0 - _binom_cdf_upto(k - 1, n_disc, 0.5)
    return min(1.0, 2.0 * min(t1, t2))


def _mcnemar_via_statsmodels(a: int, b: int, c: int, d: int) -> float | None:
    """2x2 table rows (baseline fail, baseline pass) × cols (grounded fail, grounded pass): [[d,c],[b,a]]."""
    try:
        from statsmodels.stats.contingency_tables import mcnemar

        r = mcnemar([[d, c], [b, a]], exact=True)
        pv = float(r.pvalue)
        return pv if math.isfinite(pv) else None
    except Exception:
        return None


def _bootstrap_pass_rate_diff_ci(
    baseline_pass: np.ndarray,
    grounded_pass: np.ndarray,
    *,
    n_boot: int = 8000,
    seed: int = 42,
) -> tuple[float, float]:
    """95% bootstrap CI for (mean(grounded) - mean(baseline)) over paired trial_ids."""
    n = int(len(baseline_pass))
    if n < 2:
        diff = float(np.mean(grounded_pass) - np.mean(baseline_pass))
        return (diff, diff)
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        diffs[i] = float(grounded_pass[idx].mean() - baseline_pass[idx].mean())
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return (float(lo), float(hi))


def _absolute_validity_by_mode(df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    """Per prompt mode: n, pass_count, pass_rate + Wilson CI, mean validity, hallucination rate, numeric accuracy mean."""
    out: Dict[str, Dict[str, float]] = {}
    if "mode" not in df.columns:
        return out
    mode_col = df["mode"].astype(str).str.lower().str.strip()
    for m in ("baseline", "grounded"):
        sub = df.loc[mode_col == m]
        if sub.empty:
            continue
        n = int(len(sub))
        passed = _as_bool_series(sub["passed_absolute_validity"]) if "passed_absolute_validity" in sub.columns else pd.Series([False] * n)
        pass_count = int(passed.sum())
        pass_r = float(pass_count / n) if n else 0.0
        w_lo, w_hi = _wilson_ci(pass_count, n)
        mean_v = float(pd.to_numeric(sub["validity_score_0_100"], errors="coerce").mean())
        if "hallucinated_state_count" in sub.columns and "hallucinated_number_count" in sub.columns:
            hs = pd.to_numeric(sub["hallucinated_state_count"], errors="coerce").fillna(0) > 0
            hn = pd.to_numeric(sub["hallucinated_number_count"], errors="coerce").fillna(0) > 0
            hall = float((hs | hn).mean())
        else:
            hall = 0.0
        num_mean = (
            float(pd.to_numeric(sub["numeric_accuracy_rate"], errors="coerce").mean())
            if "numeric_accuracy_rate" in sub.columns
            else 0.0
        )
        out[m] = {
            "n": float(n),
            "pass_count": float(pass_count),
            "pass_rate": pass_r,
            "pass_rate_wilson_low": w_lo,
            "pass_rate_wilson_high": w_hi,
            "mean_validity": mean_v,
            "hallucination_rate": hall,
            "numeric_accuracy_mean": num_mean,
        }
    return out


def _paired_pass_rate_comparison(df: pd.DataFrame) -> Dict[str, Any] | None:
    """Paired pass/fail by trial_id: Wilson CIs per mode, bootstrap CI for difference, McNemar."""
    need = {"trial_id", "mode", "passed_absolute_validity"}
    if not need <= set(df.columns):
        return None
    modes = set(df["mode"].astype(str).str.lower().unique())
    if not {"baseline", "grounded"} <= modes:
        return None

    d2 = df.copy()
    d2["_pass"] = _as_bool_series(d2["passed_absolute_validity"])
    piv = d2.pivot_table(index="trial_id", columns="mode", values="_pass", aggfunc="first")
    if "baseline" not in piv.columns or "grounded" not in piv.columns:
        return None
    pairs = piv[["baseline", "grounded"]].dropna()
    if pairs.empty:
        return None

    b_pass = pairs["baseline"].astype(bool).to_numpy()
    g_pass = pairs["grounded"].astype(bool).to_numpy()
    n_paired = int(len(pairs))

    a = int(np.sum(b_pass & g_pass))
    b_disc = int(np.sum(b_pass & ~g_pass))
    c_disc = int(np.sum(~b_pass & g_pass))
    d_cell = int(np.sum(~b_pass & ~g_pass))

    base_rate = float(np.mean(b_pass))
    ground_rate = float(np.mean(g_pass))
    diff = ground_rate - base_rate

    d_lo, d_hi = _bootstrap_pass_rate_diff_ci(b_pass, g_pass)

    mcnemar_note: str | None = None
    p_sm = _mcnemar_via_statsmodels(a, b_disc, c_disc, d_cell)
    if b_disc + c_disc == 0:
        mcnemar_note = "McNemar test not applicable because there are no discordant pairs."
        p_final: float | None = None
        method = "n/a"
    else:
        p_exact = _mcnemar_p_value_exact_two_sided(b_disc, c_disc)
        if p_sm is not None:
            p_final = float(p_sm)
            method = "statsmodels exact"
        else:
            p_final = float(p_exact)
            method = "exact binomial"

    return {
        "n_paired_trials": n_paired,
        "mcnemar_a": a,
        "mcnemar_b": b_disc,
        "mcnemar_c": c_disc,
        "mcnemar_d": d_cell,
        "baseline_pass_rate": base_rate,
        "grounded_pass_rate": ground_rate,
        "pass_rate_difference": diff,
        "difference_ci95_low": d_lo,
        "difference_ci95_high": d_hi,
        "mcnemar_p_value": p_final,
        "mcnemar_method": method,
        "mcnemar_note": mcnemar_note,
    }


def analyze_results(df: pd.DataFrame) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    if df.empty:
        return out

    all_vals = df["validity_score_0_100"].astype(float).tolist()
    ci_lo, ci_hi = _ci95(all_vals)
    overall_pass = (
        float(_as_bool_series(df["passed_absolute_validity"]).mean())
        if "passed_absolute_validity" in df.columns
        else 0.0
    )
    overall_hall = (
        float(((df["hallucinated_state_count"] > 0) | (df["hallucinated_number_count"] > 0)).mean())
        if "hallucinated_state_count" in df.columns and "hallucinated_number_count" in df.columns
        else 0.0
    )
    out["absolute_validity"] = {
        "mean_score": float(mean(all_vals)),
        "ci95_low": float(ci_lo),
        "ci95_high": float(ci_hi),
        "pass_rate": overall_pass,
        "hallucination_rate": overall_hall,
        "by_mode": _absolute_validity_by_mode(df),
    }

    if set(df["mode"].unique()) >= {"baseline", "grounded"}:
        piv = df.pivot_table(index="trial_id", columns="mode", values="validity_score_0_100", aggfunc="first").dropna()
        if not piv.empty:
            b = piv["baseline"].astype(float).tolist()
            g = piv["grounded"].astype(float).tolist()
            out["comparative"] = {
                "baseline_mean": float(mean(b)),
                "grounded_mean": float(mean(g)),
                "paired_t_stat": _paired_ttest(g, b),
                "effect_size_cohens_d": _cohens_d(g, b),
            }

    conf_rows = []
    for label in ["high", "moderate", "medium", "low"]:
        sub = df[df["dominant_confidence_label"].astype(str).str.lower().str.strip() == label]
        if sub.empty:
            continue
        conf_rows.append(
            {
                "confidence_label": label,
                "mean_validity_score": float(sub["validity_score_0_100"].mean()),
                "hallucination_rate": float(((sub["hallucinated_state_count"] > 0) | (sub["hallucinated_number_count"] > 0)).mean()),
                "numeric_accuracy": float(sub["numeric_accuracy_rate"].mean()),
                "pass_rate": float(sub["passed_absolute_validity"].mean()),
            }
        )

    by_label_summary: List[Dict[str, Any]] = []
    diverse_labels_for_calibration = False
    unique_label_count = 0
    if "dominant_confidence_label" in df.columns:
        dl = df["dominant_confidence_label"]
        ok = dl.notna() & (dl.astype(str).str.strip() != "")
        df_lab = df.loc[ok].copy()
        df_lab["_dl_norm"] = df_lab["dominant_confidence_label"].astype(str).str.strip().str.lower()
        unique_label_count = int(df_lab["_dl_norm"].nunique()) if not df_lab.empty else 0
        diverse_labels_for_calibration = unique_label_count >= 2
        for lbl, sub in df_lab.groupby("_dl_norm", sort=True):
            by_label_summary.append(
                {
                    "confidence_label": str(lbl),
                    "mean_validity_score": float(sub["validity_score_0_100"].mean()),
                    "numeric_accuracy": float(sub["numeric_accuracy_rate"].mean()),
                    "pass_rate": float(_as_bool_series(sub["passed_absolute_validity"]).mean())
                    if "passed_absolute_validity" in sub.columns
                    else 0.0,
                }
            )

    corr, corr_col = _pearson_confidence_validity(df)

    fm_table, fm_interp = _failure_mode_table(df)
    out["failure_mode_analysis"] = {"table": fm_table, "interpretation": fm_interp}

    out["pass_rate_comparison"] = _paired_pass_rate_comparison(df)

    out["confidence_validation"] = {
        "groups": conf_rows,
        "by_label_summary": by_label_summary,
        "unique_label_count": unique_label_count,
        "diverse_labels_for_calibration": diverse_labels_for_calibration,
        "confidence_score_validity_correlation": corr,
        "confidence_score_correlation_column": corr_col,
        "calibration_flag": _calibration_flag(conf_rows),
    }
    return out


def _calibration_flag(groups: list[dict]) -> str:
    if not groups:
        return "insufficient-data"
    by = {g["confidence_label"]: g["mean_validity_score"] for g in groups}
    high = by.get("high")
    low = by.get("low")
    if high is None or low is None:
        return "partial-data"
    diff = high - low
    if diff >= 10:
        return "strong"
    if diff >= 5:
        return "moderate"
    return "weak"
