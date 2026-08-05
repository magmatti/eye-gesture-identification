from __future__ import annotations

import numpy as np
import pandas as pd

from .detection_config import DetectionConfig
from .gaze_signal import GAZE_VECTOR_COLUMNS, normalize_rows
from .io_utils import split_by_phase

SACCADE_DIRECTIONS = ["left", "right", "up", "down", "unknown"]

SACCADE_DIRECTION_COLUMNS = [
    "saccade_direction",
    "saccade_amplitude_deg",
    "saccade_delta_horizontal_deg",
    "saccade_delta_vertical_deg",
]


# classify each saccade event as left/right/up/down/unknown with direction metrics
def add_saccade_directions(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    cfg: DetectionConfig,
) -> pd.DataFrame:
    out = _initialize_direction_columns(events)

    saccade_indices = out.index[out["gesture"] == "saccade"]
    if len(saccade_indices) == 0:
        return out

    out.loc[saccade_indices, "saccade_direction"] = "unknown"
    if not all(column in samples.columns for column in GAZE_VECTOR_COLUMNS):
        return out

    samples_by_phase = split_by_phase(samples)
    for index in saccade_indices:
        metrics = _compute_saccade_direction_metrics(
            samples_by_phase,
            out.loc[index],
            cfg,
        )
        for column, value in metrics.items():
            out.at[index, column] = value

    return out


# pick the dominant displacement axis and its sign as the cardinal direction
def classify_direction(delta_horizontal: float, delta_vertical: float) -> str:
    if abs(delta_horizontal) >= abs(delta_vertical):
        return "right" if delta_horizontal > 0 else "left"

    return "up" if delta_vertical > 0 else "down"


# initialize empty direction columns so every event row shares the same schema
def _initialize_direction_columns(events: pd.DataFrame) -> pd.DataFrame:
    out = events.copy()
    out["saccade_direction"] = pd.NA
    for column in SACCADE_DIRECTION_COLUMNS:
        if column != "saccade_direction":
            out[column] = np.nan

    return out


# compute every direction column value for a single saccade event
def _compute_saccade_direction_metrics(
    samples_by_phase: dict[str, pd.DataFrame],
    event: pd.Series,
    cfg: DetectionConfig,
) -> dict[str, float | str]:
    phase_samples = samples_by_phase.get(str(event["phase"]))
    if phase_samples is None:
        return {}

    before, after = _extract_context_windows(
        phase_samples,
        event,
        cfg.saccade_direction_context_ms / 1000.0,
    )
    gaze_before = _compute_representative_gaze(before)
    gaze_after = _compute_representative_gaze(after)
    if gaze_before is None or gaze_after is None:
        return {}

    metrics = _compute_direction_metrics(gaze_before, gaze_after)
    if metrics["saccade_amplitude_deg"] >= cfg.min_saccade_amplitude_deg:
        metrics["saccade_direction"] = classify_direction(
            metrics["saccade_delta_horizontal_deg"],
            metrics["saccade_delta_vertical_deg"],
        )

    return metrics


# slice the samples into the context windows directly before and after the event
def _extract_context_windows(
    samples: pd.DataFrame,
    event: pd.Series,
    context_s: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    before = samples[
        (samples["Time_s"] >= event["start_time_s"] - context_s)
        & (samples["Time_s"] < event["start_time_s"])
    ]
    after = samples[
        (samples["Time_s"] > event["end_time_s"])
        & (samples["Time_s"] <= event["end_time_s"] + context_s)
    ]

    return before, after


# reduce a context window to one normalized gaze vector via component-wise median
def _compute_representative_gaze(window: pd.DataFrame) -> np.ndarray | None:
    if window.empty:
        return None

    gaze = window[GAZE_VECTOR_COLUMNS].median().to_numpy(dtype=float)
    norm = np.linalg.norm(gaze)
    if not np.isfinite(gaze).all() or norm == 0.0:
        return None

    return normalize_rows(gaze[np.newaxis, :])[0]


# convert before/after gaze vectors into displacement and amplitude values
def _compute_direction_metrics(
    gaze_before: np.ndarray,
    gaze_after: np.ndarray,
) -> dict[str, float]:
    horizontal_before, vertical_before = _compute_horizontal_vertical_angles(
        gaze_before
    )
    horizontal_after, vertical_after = _compute_horizontal_vertical_angles(gaze_after)
    amplitude = np.degrees(
        np.arccos(np.clip(np.dot(gaze_before, gaze_after), -1.0, 1.0))
    )

    return {
        "saccade_amplitude_deg": float(amplitude),
        "saccade_delta_horizontal_deg": float(horizontal_after - horizontal_before),
        "saccade_delta_vertical_deg": float(vertical_after - vertical_before),
    }


# convert a gaze vector into horizontal and vertical angles in degrees
def _compute_horizontal_vertical_angles(gaze: np.ndarray) -> tuple[float, float]:
    x, y, z = gaze
    horizontal = np.degrees(np.arctan2(x, z))
    vertical = np.degrees(np.arctan2(y, np.sqrt(x**2 + z**2)))

    return float(horizontal), float(vertical)
