from __future__ import annotations

import numpy as np
import pandas as pd

from .detection_config import DetectionConfig
from .gaze_signal import GAZE_VECTOR_COLUMNS
from .gesture_specs import DETECTION_GESTURE_SPECS
from .io_utils import split_by_phase
from .saccade_direction import SACCADE_DIRECTION_COLUMNS

EVENT_COLUMNS = [
    "source_file",
    "participant",
    "scenario",
    "phase",
    "gesture",
    "start_time_s",
    "end_time_s",
    "duration_ms",
    "sample_count",
    "peak_speed_deg_s",
    "mean_speed_deg_s",
    "peak_blink_weight",
    "fixation_centroid_horizontal_deg",
    "fixation_centroid_vertical_deg",
    *SACCADE_DIRECTION_COLUMNS,
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
    phase: str | None = None,
) -> pd.DataFrame:
    rows = []
    phase_value = _phase_value(df) if phase is None else phase
    for start, end in contiguous_true_segments(df[mask_column].to_numpy()):
        start_time_s = float(df["Time_s"].iloc[start])
        end_time_s = float(df["Time_s"].iloc[end])
        duration_ms = (end_time_s - start_time_s) * 1000.0
        if duration_ms < min_duration_ms:
            continue
        if max_duration_ms is not None and duration_ms > max_duration_ms:
            continue
        segment = df.iloc[start : end + 1]
        centroid_horizontal, centroid_vertical = _fixation_centroid(
            segment, gesture_name
        )
        rows.append(
            {
                "source_file": str(df["source_file"].iloc[0]),
                "participant": str(df["participant"].iloc[0]),
                "scenario": str(df["scenario"].iloc[0]),
                "phase": phase_value,
                "gesture": gesture_name,
                "start_time_s": start_time_s,
                "end_time_s": end_time_s,
                "duration_ms": duration_ms,
                "sample_count": int(end - start + 1),
                "peak_speed_deg_s": (
                    float(segment["gaze_speed_smooth_deg_s"].max())
                    if gesture_name == "saccade"
                    else np.nan
                ),
                "mean_speed_deg_s": (
                    float(segment["gaze_speed_smooth_deg_s"].mean())
                    if gesture_name == "fixation"
                    else np.nan
                ),
                "peak_blink_weight": (
                    float(segment["blink_avg"].max())
                    if gesture_name == "blink"
                    else np.nan
                ),
                "fixation_centroid_horizontal_deg": centroid_horizontal,
                "fixation_centroid_vertical_deg": centroid_vertical,
            }
        )
    return pd.DataFrame(rows, columns=EVENT_COLUMNS)


# detect every supported gesture type, phase by phase, using configured thresholds
def detect_all_events(df: pd.DataFrame, cfg: DetectionConfig) -> pd.DataFrame:
    all_events = []
    for phase, phase_df in split_by_phase(df).items():
        for spec in DETECTION_GESTURE_SPECS:
            max_duration_ms = (
                None
                if spec.max_duration_attr is None
                else getattr(cfg, spec.max_duration_attr)
            )
            all_events.append(
                find_events(
                    phase_df,
                    spec.mask_column,
                    spec.name,
                    getattr(cfg, spec.min_duration_attr),
                    max_duration_ms,
                    phase,
                )
            )
    events = pd.concat(all_events, ignore_index=True)
    if events.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    events = _remove_saccades_near_blinks(events, cfg.blink_guard_ms)
    return events.sort_values(["source_file", "start_time_s", "gesture"]).reset_index(
        drop=True
    )

# reject saccades whose start or end lies within a blink guard interval
def _remove_saccades_near_blinks(
    events: pd.DataFrame,
    blink_guard_ms: float,
) -> pd.DataFrame:
    guard_s = blink_guard_ms / 1000.0
    keep = pd.Series(True, index=events.index)
    blinks = events[events["gesture"] == "blink"]
    for index, saccade in events[events["gesture"] == "saccade"].iterrows():
        phase_blinks = blinks[
            (blinks["source_file"] == saccade["source_file"])
            & (blinks["phase"] == saccade["phase"])
        ]
        start_near = (
            saccade["start_time_s"] >= phase_blinks["start_time_s"] - guard_s
        ) & (saccade["start_time_s"] <= phase_blinks["end_time_s"] + guard_s)
        end_near = (saccade["end_time_s"] >= phase_blinks["start_time_s"] - guard_s) & (
            saccade["end_time_s"] <= phase_blinks["end_time_s"] + guard_s
        )
        if (start_near | end_near).any():
            keep.at[index] = False
    return events[keep].reset_index(drop=True)


# compute the angular centroid of gaze vectors within a fixation event
def _fixation_centroid(
    samples: pd.DataFrame,
    gesture_name: str,
) -> tuple[float, float]:
    if gesture_name != "fixation":
        return np.nan, np.nan
    x, y, z = samples[GAZE_VECTOR_COLUMNS].mean().to_numpy(dtype=float)
    horizontal = np.degrees(np.arctan2(x, z))
    vertical = np.degrees(np.arctan2(y, np.sqrt(x**2 + z**2)))
    return float(horizontal), float(vertical)


# preserve phase inference for direct find_events callers
def _phase_value(df: pd.DataFrame) -> str:
    if "Phase" in df.columns:
        return str(df["Phase"].iloc[0])
    return "recording"
