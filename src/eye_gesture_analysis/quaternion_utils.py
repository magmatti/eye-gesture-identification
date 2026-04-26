from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation


FORWARD_VECTOR = np.array([0.0, 0.0, 1.0], dtype=float)


def apply_unity_quaternions(quats_xyzw: np.ndarray) -> np.ndarray:
    """Rotate Unity's forward vector [0, 0, 1] using quaternions in x,y,z,w order."""
    rot = Rotation.from_quat(quats_xyzw)
    vecs = rot.apply(np.tile(FORWARD_VECTOR, (len(quats_xyzw), 1)))
    return normalize_rows(vecs)


def normalize_rows(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return arr / norms


def vectors_to_yaw_pitch_deg(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert 3D direction vectors to yaw / pitch in degrees.

    Yaw measures left-right angle.
    Pitch measures up-down angle.
    """
    x = vectors[:, 0]
    y = vectors[:, 1]
    z = vectors[:, 2]

    yaw = np.degrees(np.arctan2(x, z))
    pitch = np.degrees(np.arctan2(-y, np.sqrt(x * x + z * z)))
    return yaw, pitch


def angular_distance_deg(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    """Angular distance in degrees between matching rows of two vector arrays."""
    dots = np.sum(v1 * v2, axis=1)
    dots = np.clip(dots, -1.0, 1.0)
    return np.degrees(np.arccos(dots))


def wrap_signed_deg(values: np.ndarray) -> np.ndarray:
    """Convert 0..360 style angles into -180..180."""
    return ((values + 180.0) % 360.0) - 180.0
