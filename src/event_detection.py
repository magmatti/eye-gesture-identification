from __future__ import annotations

import numpy as np
import pandas as pd

from .io_utils import split_by_phase


EVENT_COLUMNS = [
    "source_file",
    "phase",
    "gesture",
    "start_time_s",
    "end_time_s",
    "duration_ms",
    "peak_value",
    "sample_count",
]


# locate continuous true runs in a detection mask
def contiguous_true_segments(mask) -> list[tuple[int, int]]:
    values = np.asarray(mask, dtype=bool)
    if len(values) == 0:
        return []

    segments: list[tuple[int, int]] = []
    start_idx: int | None = None
    for idx, value in enumerate(values):
        if value and start_idx is None:
            start_idx = idx
        elif not value and start_idx is not None:
            segments.append((start_idx, idx - 1))
            start_idx = None

    if start_idx is not None:
        segments.append((start_idx, len(values) - 1))

    return segments


# convert one boolean detection column into timestamped gesture event rows
def find_events(
    df: pd.DataFrame,
    mask_column: str,
    gesture_name: str,
    min_duration_ms: float,
    max_duration_ms: float | None = None,
) -> pd.DataFrame:
    if mask_column not in df.columns or "Time_s" not in df.columns:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    rows = []
    phase = _phase_value(df)
    source_file = _first_value(df, "source_file", "unknown")

    for start, end in contiguous_true_segments(df[mask_column].fillna(False).to_numpy()):
        start_time_s = float(df["Time_s"].iloc[start])
        end_time_s = float(df["Time_s"].iloc[end])
        duration_ms = (end_time_s - start_time_s) * 1000.0

        if duration_ms < min_duration_ms:
            continue
        if max_duration_ms is not None and duration_ms > max_duration_ms:
            continue

        rows.append(
            {
                "source_file": source_file,
                "phase": phase,
                "gesture": gesture_name,
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "duration_ms": duration_ms,
                "peak_value": _peak_value(df.iloc[start : end + 1], gesture_name),
                "sample_count": int(end - start + 1),
            }
        )

    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


# detect every supported gesture type, phase by phase, using configured thresholds
def detect_all_events(df: pd.DataFrame, cfg: DetectionConfig) -> pd.DataFrame:
    all_events = []
    for _, phase_df in split_by_phase(df).items():
        all_events.append(
            _find_events_with_trim(
                phase_df,
                "is_blink",
                "blink",
                cfg.min_blink_duration_ms,
                cfg.max_blink_duration_ms,
                cfg.trim_start_ms,
            )
        )
        all_events.append(
            _find_events_with_trim(
                phase_df,
                "is_saccade",
                "saccade",
                cfg.min_saccade_duration_ms,
                cfg.max_saccade_duration_ms,
                cfg.trim_start_ms,
            )
        )
        all_events.append(
            _find_events_with_trim(
                phase_df,
                "is_fixation",
                "fixation",
                cfg.min_fixation_duration_ms,
                None,
                cfg.trim_start_ms,
            )
        )

    if not all_events:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    events = pd.concat(all_events, ignore_index=True)
    if events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    return events.sort_values(["source_file", "start_time_s", "gesture"]).reset_index(drop=True)


def summarize_events_by_file(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "source_file",
                "event_count",
                "total_duration_ms",
                "mean_duration_ms",
            ]
        )

    return (
        events.groupby("source_file", dropna=False)
        .agg(
            event_count=("gesture", "count"),
            total_duration_ms=("duration_ms", "sum"),
            mean_duration_ms=("duration_ms", "mean"),
        )
        .reset_index()
        .sort_values("source_file")
    )


def summarize_event_counts_by_file(events: pd.DataFrame) -> pd.DataFrame:
    columns = ["filename", "blink_count", "fixation_count", "saccade_count"]
    if events.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        events.groupby(["source_file", "gesture"], dropna=False)
        .size()
        .unstack(fill_value=0)
        .reset_index()
        .rename(columns={"source_file": "filename"})
    )

    for gesture in ["blink", "fixation", "saccade"]:
        if gesture not in summary.columns:
            summary[gesture] = 0

    summary = summary.rename(
        columns={
            "blink": "blink_count",
            "fixation": "fixation_count",
            "saccade": "saccade_count",
        }
    )
    return summary[columns].sort_values("filename").reset_index(drop=True)


def summarize_events_by_gesture(events: pd.DataFrame) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame(
            columns=[
                "gesture",
                "event_count",
                "total_duration_ms",
                "mean_duration_ms",
                "mean_peak_value",
            ]
        )

    return (
        events.groupby("gesture", dropna=False)
        .agg(
            event_count=("gesture", "count"),
            total_duration_ms=("duration_ms", "sum"),
            mean_duration_ms=("duration_ms", "mean"),
            mean_peak_value=("peak_value", "mean"),
        )
        .reset_index()
        .sort_values("gesture")
    )


# apply generic event detection and drop events that occur inside the startup trim window
def _find_events_with_trim(
    df: pd.DataFrame,
    mask_column: str,
    gesture_name: str,
    min_duration_ms: float,
    max_duration_ms: float | None,
    trim_start_ms: float,
) -> pd.DataFrame:
    events = find_events(df, mask_column, gesture_name, min_duration_ms, max_duration_ms)
    if events.empty:
        return events
    return events[events["start_time_s"] * 1000.0 >= trim_start_ms].reset_index(drop=True)


# pick the most useful signal value for reporting a detected event
def _peak_value(df: pd.DataFrame, gesture_name: str) -> float:
    if gesture_name == "blink":
        return float(df["blink_avg"].max()) if "blink_avg" in df.columns else np.nan
    if gesture_name == "fixation":
        return (
            float(df["gaze_speed_smooth_deg_s"].mean())
            if "gaze_speed_smooth_deg_s" in df.columns
            else np.nan
        )
    return (
        float(df["gaze_speed_smooth_deg_s"].max())
        if "gaze_speed_smooth_deg_s" in df.columns
        else np.nan
    )


def _phase_value(df: pd.DataFrame) -> str:
    if "Phase" in df.columns and len(df):
        return str(df["Phase"].iloc[0])
    return "recording"


def _first_value(df: pd.DataFrame, column: str, default: str) -> str:
    if column not in df.columns or df.empty:
        return default
    return str(df[column].iloc[0])
