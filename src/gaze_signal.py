from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

LEFT_GAZE_QUATERNION_COLUMNS = [
    "LeftLocalRotX",
    "LeftLocalRotY",
    "LeftLocalRotZ",
    "LeftLocalRotW",
]
RIGHT_GAZE_QUATERNION_COLUMNS = [
    "RightLocalRotX",
    "RightLocalRotY",
    "RightLocalRotZ",
    "RightLocalRotW",
]

GAZE_QUATERNION_COLUMNS = LEFT_GAZE_QUATERNION_COLUMNS + RIGHT_GAZE_QUATERNION_COLUMNS

FORWARD_VECTOR = np.array([0.0, 0.0, 1.0], dtype=float)

GAZE_VECTOR_COLUMNS = ["gaze_vector_x", "gaze_vector_y", "gaze_vector_z"]


# convert a gaze vector into horizontal and vertical angles in degrees
def gaze_vector_to_angles_deg(gaze: np.ndarray) -> tuple[float, float]:
    x, y, z = gaze
    horizontal = np.degrees(np.arctan2(x, z))
    vertical = np.degrees(np.arctan2(y, np.sqrt(x**2 + z**2)))
    return float(horizontal), float(vertical)


# check if df has gaze columns
def has_gaze_columns(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in GAZE_QUATERNION_COLUMNS)


# normalizing gaze vectors e.g [2, 0, 0] -> [1, 0, 0]
def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)

    return values / norms


# converting left and right eye rotation quaternions into 3D gaze direction vectors
def quaternions_to_gaze_vectors(df: pd.DataFrame) -> np.ndarray:
    left_quat = df[LEFT_GAZE_QUATERNION_COLUMNS].to_numpy(dtype=float)
    right_quat = df[RIGHT_GAZE_QUATERNION_COLUMNS].to_numpy(dtype=float)

    left_vectors = Rotation.from_quat(left_quat).apply(FORWARD_VECTOR)
    right_vectors = Rotation.from_quat(right_quat).apply(FORWARD_VECTOR)

    gaze_vectors = left_vectors + right_vectors

    return normalize_rows(gaze_vectors)


# calculating gaze speed out of gaze direction vectors
def add_gaze_speed(df: pd.DataFrame, smoothing_window: int) -> pd.DataFrame:
    out = df.copy()

    if not has_gaze_columns(out):
        out["gaze_speed_deg_s"] = np.nan
        out["gaze_speed_smooth_deg_s"] = np.nan
        return out

    gaze_vectors = quaternions_to_gaze_vectors(out)
    out[GAZE_VECTOR_COLUMNS] = gaze_vectors

    angle_step = _calculate_angle_steps_deg(gaze_vectors)
    speed = _calculate_gaze_speed_deg_s(angle_step, out["Time_s"])
    out["gaze_speed_deg_s"] = speed
    out["gaze_speed_smooth_deg_s"] = _smooth_gaze_speed(
        speed, out.index, smoothing_window
    )

    return out


# calculating gaze direction changes between each frame
def _calculate_angle_steps_deg(gaze_vectors: np.ndarray) -> np.ndarray:
    angle_step = np.zeros(len(gaze_vectors), dtype=float)
    if len(gaze_vectors) > 1:
        dots = np.sum(gaze_vectors[1:] * gaze_vectors[:-1], axis=1)
        dots = np.clip(dots, -1.0, 1.0)
        angle_step[1:] = np.degrees(np.arccos(dots))

    return angle_step


# divide angle steps by sample time differences to get degrees per second
def _calculate_gaze_speed_deg_s(
    angle_step: np.ndarray, time_s: pd.Series
) -> np.ndarray:
    dt_s = time_s.diff().to_numpy(dtype=float)
    return np.divide(
        angle_step,
        dt_s,
        out=np.zeros(len(angle_step), dtype=float),
        where=dt_s > 0,
    )


# smooth short speed spikes with a centered average
def _smooth_gaze_speed(
    speed: np.ndarray,
    index: pd.Index,
    smoothing_window: int,
) -> pd.Series:
    return (
        pd.Series(speed, index=index)
        .rolling(window=smoothing_window, center=True, min_periods=1)
        .mean()
    )
