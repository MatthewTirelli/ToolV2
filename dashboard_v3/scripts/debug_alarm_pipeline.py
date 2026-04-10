#!/usr/bin/env python3
"""Diagnostics for Stage 1 national alarm (run from dashboard_v3 with .venv)."""
from __future__ import annotations

import sys
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(APP_DIR))

import numpy as np
import pandas as pd

from loaders import clear_cache, load_all
from risk import (
    ALARM_STAGE1_FEATURE_COLUMNS,
    DEFAULT_OUTBREAK_PERCENTILE,
    _outbreak_threshold_int,
    _pipeline_predict_alarm_prob,
    _squash_extreme_alarm_proba,
    build_modeling_frame,
    fit_alarm_model,
)


def main() -> None:
    clear_cache()
    _, kg, ww, nndss, _, _ = load_all(use_cache=False)
    thresh = _outbreak_threshold_int(nndss, 0, DEFAULT_OUTBREAK_PERCENTILE)
    print("outbreak_threshold (int):", thresh)
    df, audit = build_modeling_frame(nndss, ww, kg, outbreak_threshold=thresh)
    print("\n=== modeling frame ===")
    print("shape:", df.shape)
    print("columns (sample):", list(df.columns[:12]), "...")
    print("year/week range:", (df["year"].iloc[0], df["week"].iloc[0]), "->", (df["year"].iloc[-1], df["week"].iloc[-1]))
    dup = df.duplicated(subset=["year", "week"]).sum()
    print("duplicate (year,week) rows:", int(dup))
    print("head:\n", df.head(3))
    print("tail:\n", df.tail(3))

    y = df["elevated_next4w"].astype(int)
    print("\n=== target ===")
    print(y.value_counts())
    print("positive rate overall:", float(y.mean()))
    split = len(df) - 12
    print("train pos rate:", float(y.iloc[:split].mean()), "test pos rate:", float(y.iloc[split:].mean()))
    print("train n:", split, "test n:", len(df) - split)

    mr = fit_alarm_model(df)
    print("\n=== model ===")
    print("status:", mr.status, "auc:", mr.auc)
    print("features n:", len(mr.features))
    print("cases_next4 in features (must be False):", "cases_next4" in mr.features)
    print("feature list == allowlist:", mr.features == list(ALARM_STAGE1_FEATURE_COLUMNS))

    feat = mr.features
    X = df[feat]
    train = X.iloc[:split]
    last = X.iloc[[-1]]
    print("\n=== latest row (all features) ===")
    print(last.T.to_string())

    print("\n=== latest vs train distribution ===")
    for c in feat:
        tr = train[c]
        v = float(last[c].iloc[0])
        lo, hi = float(tr.min()), float(tr.max())
        flag = "OUTSIDE_TRAIN_RANGE" if (v < lo or v > hi) else ""
        print(
            f"{c:14} latest={v:12.4f} train_mean={float(tr.mean()):8.4f} std={float(tr.std()):8.4f} "
            f"min={lo:8.4f} max={hi:8.4f} {flag}"
        )

    pipe = mr.model
    im = pipe.named_steps["imputer"]
    sc = pipe.named_steps["scaler"]
    lr = pipe.named_steps["model"]
    xt = im.transform(last)
    xs = sc.transform(xt)
    logit = float(lr.decision_function(xs)[0])
    coef = pd.Series(lr.coef_.ravel(), index=feat)
    contrib = pd.Series(xs.ravel() * coef.values, index=feat).sort_values(ascending=False)
    print("\n=== linear model ===")
    print("intercept:", float(lr.intercept_[0]))
    print("decision_function (latest):", logit)
    print("top positive contributions (scaled x * coef):\n", contrib.head(8).to_string())
    print("top negative contributions:\n", contrib.tail(8).sort_values().to_string())

    raw = float(pipe.predict_proba(last)[0, 1])
    disp = _pipeline_predict_alarm_prob(mr, df)
    print("\n=== probability ===")
    print("raw sklearn proba:", raw)
    print("squashed (display helper test):", _squash_extreme_alarm_proba(raw))
    print("_pipeline_predict_alarm_prob returns:", disp)

    miss = last.isna().sum().sum()
    print("\nmissing in latest feature row:", int(miss))


if __name__ == "__main__":
    main()
