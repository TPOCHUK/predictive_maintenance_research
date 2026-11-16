"""Exploratory drift detector; does not diagnose physical failure causes."""

import numpy as np
import pandas as pd

from predictive_maintenance.config import load_pipeline_config

PIPELINE_CONFIG = load_pipeline_config()
BASELINE_CYCLES = PIPELINE_CONFIG.baseline_cycles


def sensor_overview(rows: list[dict]) -> dict:
    names = [f"sensor_{i}" for i in range(1, 22)]
    empty = {
        "ready": False,
        "baseline_cycles": BASELINE_CYCLES,
        "threshold": 3,
        "cycles": [],
        "scores": [],
        "events": [],
        "sensors": [],
    }
    if not rows:
        return empty
    df = pd.DataFrame(rows)
    empty["cycles"] = df.cycle.astype(int).tolist()
    if len(df) < BASELINE_CYCLES:
        empty["scores"] = [None] * len(df)
        return empty
    baseline = df[names].iloc[:BASELINE_CYCLES]
    std = baseline.std(ddof=0)
    # Ignore sensors with a constant baseline.
    active = std > np.maximum(baseline.mean().abs() * 1e-8, 1e-8)
    scores = (
        (df[names].rolling(5, min_periods=5).mean() - baseline.mean()) / std.where(active)
    ).abs()
    scores.iloc[:BASELINE_CYCLES] = np.nan
    events, summaries = [], []
    for name in names:
        series = scores[name]
        sustained = series.gt(3).rolling(3, min_periods=3).sum().eq(3)
        positions = np.flatnonzero(sustained.to_numpy())
        if len(positions):
            # Save the cycle when the change is confirmed.
            pos = int(positions[0])
            events.append(
                {
                    "cycle": int(df.cycle.iloc[pos]),
                    "sensor": name,
                    "score": round(float(series.iloc[pos]), 2),
                }
            )
        summaries.append(
            {
                "sensor": name,
                "value": float(df[name].iloc[-1]),
                "score": round(float(series.iloc[-1]), 2) if pd.notna(series.iloc[-1]) else None,
                "active": bool(active[name]),
            }
        )
    maximum = scores.max(axis=1)
    return {
        "ready": True,
        "baseline_cycles": BASELINE_CYCLES,
        "threshold": 3,
        "cycles": df.cycle.astype(int).tolist(),
        "scores": [float(value) if pd.notna(value) else None for value in maximum],
        "events": sorted(events, key=lambda event: (event["cycle"], -event["score"])),
        "sensors": summaries,
    }
