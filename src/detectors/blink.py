from __future__ import annotations

import numpy as np
import pandas as pd

from signal_utils import contiguous_true_segments, segment_duration_ms
from models.blink_config import BlinkConfig


BLINK_EVENT_COLUMNS = [
    "start_idx",
    "end_idx",
    "start_ms",
    "end_ms",
    "duration_ms",
    "peak_weight",
]


def detect_blinks(
    df: pd.DataFrame,
    cfg: BlinkConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cfg = cfg or BlinkConfig()
    out = df.copy()

    if not _has_blink_columns(out):
        return _empty_blink_result(out)

    out["blink_avg"] = _calculate_blink_average(out)
    blink_mask = _create_hysteresis_blink_mask(out["blink_avg"], cfg)
    blink_mask, events = _extract_valid_blink_events(out, blink_mask, cfg)

    out["is_blink_sample"] = blink_mask
    return out, events


def summarize_blinks(df: pd.DataFrame, events: pd.DataFrame) -> dict[str, float]:
    blink_fraction = (
        float(df.get("is_blink_sample", pd.Series(dtype=bool)).mean())
        if len(df)
        else 0.0
    )

    blink_avg = df.get("blink_avg", pd.Series(dtype=float))

    return {
        "blink_event_count": float(len(events)),
        "blink_sample_fraction": blink_fraction,
        "blink_peak_max": float(blink_avg.max()) if len(blink_avg) else 0.0,
        "blink_peak_q95": float(blink_avg.quantile(0.95)) if len(blink_avg) else 0.0,
        "blink_duration_mean_ms": (
            float(events["duration_ms"].mean())
            if len(events) and "duration_ms" in events.columns
            else 0.0
        ),
    }


def _has_blink_columns(df: pd.DataFrame) -> bool:
    required = {"Time_ms", "LeftBlinkWeight", "RightBlinkWeight"}
    return required.issubset(df.columns)


def _empty_blink_result(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["blink_avg"] = np.nan
    out["is_blink_sample"] = False
    return out, pd.DataFrame(columns=BLINK_EVENT_COLUMNS)


def _calculate_blink_average(df: pd.DataFrame) -> pd.Series:
    left = df["LeftBlinkWeight"].astype(float)
    right = df["RightBlinkWeight"].astype(float)
    return (left + right) / 2.0

def _create_hysteresis_blink_mask(
    blink_avg: pd.Series,
    cfg: BlinkConfig,
) -> np.ndarray:
    blink_mask = np.zeros(len(blink_avg), dtype=bool)

    active = False
    start_idx = 0

    for idx, value in enumerate(blink_avg.to_numpy(dtype=float)):
        if not active and value >= cfg.onset_threshold:
            active = True
            start_idx = idx

        elif active and value <= cfg.offset_threshold:
            blink_mask[start_idx : idx + 1] = True
            active = False

    if active:
        blink_mask[start_idx:] = True

    return blink_mask

def _extract_valid_blink_events(
    df: pd.DataFrame,
    blink_mask: np.ndarray,
    cfg: BlinkConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    events = []

    for start, end in contiguous_true_segments(blink_mask):
        duration_ms = segment_duration_ms(df["Time_ms"], start, end)

        if not cfg.min_duration_ms <= duration_ms <= cfg.max_duration_ms:
            blink_mask[start : end + 1] = False
            continue

        events.append(
            {
                "start_idx": start,
                "end_idx": end,
                "start_ms": float(df["Time_ms"].iloc[start]),
                "end_ms": float(df["Time_ms"].iloc[end]),
                "duration_ms": duration_ms,
                "peak_weight": float(df["blink_avg"].iloc[start : end + 1].max()),
            }
        )

    return blink_mask, pd.DataFrame(events, columns=BLINK_EVENT_COLUMNS)
