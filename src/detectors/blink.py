from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from signal_utils import contiguous_true_segments, segment_duration_ms
from models.blink_config import BlinkConfig


def detect_blinks(df: pd.DataFrame, cfg: BlinkConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or BlinkConfig()
    out = df.copy()

    required = {"Time_ms", "LeftBlinkWeight", "RightBlinkWeight"}
    if not required.issubset(out.columns):
        out["blink_avg"] = np.nan
        out["is_blink_sample"] = False
        return out, pd.DataFrame(columns=["start_idx", "end_idx", "start_ms", "end_ms", "duration_ms", "peak_weight"])

    out["blink_avg"] = (out["LeftBlinkWeight"].astype(float) + out["RightBlinkWeight"].astype(float)) / 2.0

    blink_mask = np.zeros(len(out), dtype=bool)
    active = False
    start_idx = 0

    for idx, value in enumerate(out["blink_avg"].to_numpy(dtype=float)):
        if not active and value >= cfg.onset_threshold:
            active = True
            start_idx = idx
        elif active and value <= cfg.offset_threshold:
            blink_mask[start_idx: idx + 1] = True
            active = False

    if active:
        blink_mask[start_idx:] = True

    events = []
    for start, end in contiguous_true_segments(blink_mask):
        duration_ms = segment_duration_ms(out["Time_ms"], start, end)
        if cfg.min_duration_ms <= duration_ms <= cfg.max_duration_ms:
            events.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "start_ms": float(out["Time_ms"].iloc[start]),
                    "end_ms": float(out["Time_ms"].iloc[end]),
                    "duration_ms": duration_ms,
                    "peak_weight": float(out["blink_avg"].iloc[start: end + 1].max()),
                }
            )
        else:
            blink_mask[start: end + 1] = False

    out["is_blink_sample"] = blink_mask
    return out, pd.DataFrame(events)


def summarize_blinks(df: pd.DataFrame, events: pd.DataFrame) -> dict[str, float]:
    blink_fraction = float(df.get("is_blink_sample", pd.Series(dtype=bool)).mean()) if len(df) else 0.0
    blink_avg = df.get("blink_avg", pd.Series(dtype=float))
    return {
        "blink_event_count": float(len(events)),
        "blink_sample_fraction": blink_fraction,
        "blink_peak_max": float(blink_avg.max()) if len(blink_avg) else 0.0,
        "blink_peak_q95": float(blink_avg.quantile(0.95)) if len(blink_avg) else 0.0,
        "blink_duration_mean_ms": float(events["duration_ms"].mean()) if len(events) else 0.0,
    }
