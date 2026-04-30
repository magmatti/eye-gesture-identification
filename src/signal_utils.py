from __future__ import annotations

import numpy as np
import pandas as pd

from quaternion_utils import apply_unity_quaternions, vectors_to_yaw_pitch_deg, angular_distance_deg


def add_dt_seconds(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    dt = out["Time_ms"].diff().to_numpy(dtype=float) / 1000.0
    dt[0] = np.nan
    out["dt_s"] = dt
    return out


def add_gaze_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add gaze vectors, yaw/pitch, and angular velocity from local eye rotations."""
    required = [
        "LeftLocalRotX", "LeftLocalRotY", "LeftLocalRotZ", "LeftLocalRotW",
        "RightLocalRotX", "RightLocalRotY", "RightLocalRotZ", "RightLocalRotW",
    ]
    if not all(col in df.columns for col in required):
        return df.copy()

    out = add_dt_seconds(df)

    left_quat = out[["LeftLocalRotX", "LeftLocalRotY", "LeftLocalRotZ", "LeftLocalRotW"]].to_numpy(dtype=float)
    right_quat = out[["RightLocalRotX", "RightLocalRotY", "RightLocalRotZ", "RightLocalRotW"]].to_numpy(dtype=float)

    left_vec = apply_unity_quaternions(left_quat)
    right_vec = apply_unity_quaternions(right_quat)
    gaze_vec = left_vec + right_vec
    gaze_vec /= np.linalg.norm(gaze_vec, axis=1, keepdims=True)

    left_yaw, left_pitch = vectors_to_yaw_pitch_deg(left_vec)
    right_yaw, right_pitch = vectors_to_yaw_pitch_deg(right_vec)
    gaze_yaw, gaze_pitch = vectors_to_yaw_pitch_deg(gaze_vec)

    out["left_yaw_deg"] = left_yaw
    out["left_pitch_deg"] = left_pitch
    out["right_yaw_deg"] = right_yaw
    out["right_pitch_deg"] = right_pitch
    out["gaze_yaw_deg"] = gaze_yaw
    out["gaze_pitch_deg"] = gaze_pitch
    out["binocular_disparity_deg"] = np.sqrt((left_yaw - right_yaw) ** 2 + (left_pitch - right_pitch) ** 2)

    # Angular speed is the angle between consecutive gaze vectors divided by dt.
    angle_step = np.full(len(out), np.nan, dtype=float)
    angle_step[1:] = angular_distance_deg(gaze_vec[1:], gaze_vec[:-1])
    out["gaze_angle_step_deg"] = angle_step
    out["gaze_speed_deg_s"] = out["gaze_angle_step_deg"] / out["dt_s"]
    out["gaze_speed_deg_s"] = out["gaze_speed_deg_s"].replace([np.inf, -np.inf], np.nan)

    # Angular acceleration is the change in gaze speed over time.
    out["gaze_accel_deg_s2"] = out["gaze_speed_deg_s"].diff() / out["dt_s"]

    # Rolling dispersion: how spread-out the gaze is inside a short temporal window.
    # The default 8 samples correspond to ~110 ms at ~72 Hz.
    out["dispersion_deg"] = rolling_dispersion_2d(out["gaze_yaw_deg"], out["gaze_pitch_deg"], window=8)
    out["speed_smooth_deg_s"] = out["gaze_speed_deg_s"].rolling(window=5, center=True, min_periods=1).median()
    return out


def rolling_dispersion_2d(x: pd.Series | np.ndarray, y: pd.Series | np.ndarray, window: int = 8) -> np.ndarray:
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    out = np.full(len(x_arr), np.nan, dtype=float)
    for idx in range(len(x_arr)):
        start = max(0, idx - window + 1)
        x_slice = x_arr[start: idx + 1]
        y_slice = y_arr[start: idx + 1]
        out[idx] = (np.nanmax(x_slice) - np.nanmin(x_slice)) + (np.nanmax(y_slice) - np.nanmin(y_slice))
    return out


def contiguous_true_segments(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive index segments for runs of True values."""
    mask = np.asarray(mask, dtype=bool)
    if len(mask) == 0:
        return []

    segments: list[tuple[int, int]] = []
    start = None
    for idx, value in enumerate(mask):
        if value and start is None:
            start = idx
        elif not value and start is not None:
            segments.append((start, idx - 1))
            start = None
    if start is not None:
        segments.append((start, len(mask) - 1))
    return segments


def segment_duration_ms(time_ms: pd.Series, start_idx: int, end_idx: int) -> float:
    return float(time_ms.iloc[end_idx] - time_ms.iloc[start_idx])
