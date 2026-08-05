from __future__ import annotations

import numpy as np
import pandas as pd

from .detection_config import DetectionConfig
from .event_detection import GROUP_COLUMNS, contiguous_true_segments
from .io_utils import split_by_phase
from .saccade_direction import classify_direction

TARGET_COLUMNS = ["TargetRotX", "TargetRotY"]

TARGET_JUMP_COLUMNS = [
    "source_file",
    "participant",
    "scenario",
    "phase",
    "target_time_s",
    "expected_direction",
    "target_delta_horizontal_deg",
    "target_delta_vertical_deg",
    "detected",
    "detected_direction",
    "direction_match",
]

BLINK_WINDOW_COLUMNS = [
    "source_file",
    "participant",
    "scenario",
    "phase",
    "duration_s",
    "expected_blink_count",
    "detected_blink_count",
]

_MATCH_COLUMNS = ["detected", "detected_direction", "direction_match"]

# a beep scheduled exactly at the end of the recording loses the race against
# the test-stop coroutine, so treat the recording end as an exclusive bound
_BEEP_STOP_TOLERANCE_S = 0.1


# per target jump: expected direction plus whether a detected saccade matched it
def evaluate_saccade_targets(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    cfg: DetectionConfig,
) -> pd.DataFrame:
    jumps = _extract_target_jumps(samples)
    if jumps.empty:
        return pd.DataFrame(columns=TARGET_JUMP_COLUMNS)

    saccades = events[events["gesture"] == "saccade"]

    return _match_jumps_to_saccades(jumps, saccades, cfg)


# per blink recording window: expected beep-driven blink count vs detected count
def evaluate_blink_targets(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    cfg: DetectionConfig,
) -> pd.DataFrame:
    if cfg.blink_beep_interval_s is None:
        return pd.DataFrame(columns=BLINK_WINDOW_COLUMNS)

    blinks = events[events["gesture"] == "blink"]
    rows = []
    for phase, phase_df in split_by_phase(samples).items():
        if not _is_blink_phase(phase_df, phase):
            continue
        duration_s = float(phase_df["Time_s"].iloc[-1] - phase_df["Time_s"].iloc[0])
        rows.append(
            {
                "source_file": str(phase_df["source_file"].iloc[0]),
                "participant": str(phase_df["participant"].iloc[0]),
                "scenario": str(phase_df["scenario"].iloc[0]),
                "phase": phase,
                "duration_s": duration_s,
                "expected_blink_count": _expected_beep_count(duration_s, cfg),
                "detected_blink_count": int((blinks["phase"] == phase).sum()),
            }
        )

    return pd.DataFrame(rows, columns=BLINK_WINDOW_COLUMNS)


# aggregate target/saccade matches into detection and direction accuracy rates
def summarize_saccade_ground_truth(matches: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *GROUP_COLUMNS,
        "target_count",
        "detected_count",
        "detection_rate",
        "direction_correct_count",
        "direction_accuracy",
    ]
    if matches.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        matches.groupby(GROUP_COLUMNS, dropna=False)
        .agg(
            target_count=("detected", "size"),
            detected_count=("detected", "sum"),
            direction_correct_count=(
                "direction_match",
                lambda s: int((s).sum()),
            ),
        )
        .reset_index()
    )
    summary["detection_rate"] = summary["detected_count"] / summary["target_count"]
    summary["direction_accuracy"] = (
        summary["direction_correct_count"] / summary["detected_count"]
    ).where(summary["detected_count"] > 0)

    return summary[columns].sort_values(GROUP_COLUMNS).reset_index(drop=True)


# aggregate blink windows into expected vs detected blink counts
def summarize_blink_ground_truth(windows: pd.DataFrame) -> pd.DataFrame:
    columns = [
        *GROUP_COLUMNS,
        "expected_blink_count",
        "detected_blink_count",
        "detection_rate",
    ]
    if windows.empty:
        return pd.DataFrame(columns=columns)

    summary = (
        windows.groupby(GROUP_COLUMNS, dropna=False)
        .agg(
            expected_blink_count=("expected_blink_count", "sum"),
            detected_blink_count=("detected_blink_count", "sum"),
        )
        .reset_index()
    )
    summary["detection_rate"] = (
        summary["detected_blink_count"] / summary["expected_blink_count"]
    ).where(summary["expected_blink_count"] > 0)

    return summary[columns].sort_values(GROUP_COLUMNS).reset_index(drop=True)


# locate animated target movements and describe each as one expected saccade
def _extract_target_jumps(samples: pd.DataFrame) -> pd.DataFrame:
    if not all(column in samples.columns for column in TARGET_COLUMNS):
        return pd.DataFrame(columns=TARGET_JUMP_COLUMNS)

    rows = []
    for phase, phase_df in split_by_phase(samples).items():
        # Unity yaw (Y) grows to the right, pitch (X) grows downward
        horizontal = _wrap_degrees(phase_df["TargetRotY"].to_numpy(dtype=float))
        vertical = -_wrap_degrees(phase_df["TargetRotX"].to_numpy(dtype=float))
        horizontal_change = np.diff(horizontal, prepend=horizontal[:1])
        vertical_change = np.diff(vertical, prepend=vertical[:1])
        moving = (horizontal_change != 0.0) | (vertical_change != 0.0)

        for start, end in contiguous_true_segments(moving):
            if start == 0:
                continue
            delta_horizontal = float(horizontal[end] - horizontal[start - 1])
            delta_vertical = float(vertical[end] - vertical[start - 1])
            rows.append(
                {
                    "source_file": str(phase_df["source_file"].iloc[0]),
                    "participant": str(phase_df["participant"].iloc[0]),
                    "scenario": str(phase_df["scenario"].iloc[0]),
                    "phase": phase,
                    "target_time_s": float(phase_df["Time_s"].iloc[start]),
                    "expected_direction": classify_direction(
                        delta_horizontal, delta_vertical
                    ),
                    "target_delta_horizontal_deg": delta_horizontal,
                    "target_delta_vertical_deg": delta_vertical,
                }
            )

    return pd.DataFrame(
        rows, columns=[c for c in TARGET_JUMP_COLUMNS if c not in _MATCH_COLUMNS]
    )


# pair each target jump with the first detected saccade inside the latency window
def _match_jumps_to_saccades(
    jumps: pd.DataFrame,
    saccades: pd.DataFrame,
    cfg: DetectionConfig,
) -> pd.DataFrame:
    max_latency_s = cfg.saccade_match_max_latency_ms / 1000.0
    out = jumps.sort_values("target_time_s").reset_index(drop=True)
    out["detected"] = False
    out["detected_direction"] = pd.NA
    out["direction_match"] = pd.NA

    used_indices: set[int] = set()
    for index, jump in out.iterrows():
        candidates = saccades[
            (saccades["phase"] == jump["phase"])
            & (saccades["start_time_s"] >= jump["target_time_s"])
            & (saccades["start_time_s"] <= jump["target_time_s"] + max_latency_s)
            & ~saccades.index.isin(used_indices)
        ]
        if candidates.empty:
            continue

        match = candidates.iloc[0]
        used_indices.add(int(candidates.index[0]))
        out.at[index, "detected"] = True
        out.at[index, "detected_direction"] = match["saccade_direction"]
        out.at[index, "direction_match"] = bool(
            match["saccade_direction"] == jump["expected_direction"]
        )

    return out[TARGET_JUMP_COLUMNS]


# count metronome beeps played during a recording window; the Unity
# MetronomeSequence beeps at t = initial_delay + k * interval
def _expected_beep_count(duration_s: float, cfg: DetectionConfig) -> int:
    window_s = duration_s - cfg.blink_beep_initial_delay_s - _BEEP_STOP_TOLERANCE_S
    if window_s < 0.0:
        return 0

    return int(window_s // cfg.blink_beep_interval_s) + 1


# wrap Unity euler angles from [0, 360) into [-180, 180)
def _wrap_degrees(angles: np.ndarray) -> np.ndarray:
    return (angles + 180.0) % 360.0 - 180.0


# blink target applies to dedicated blink recordings and Blink phases of combined runs
def _is_blink_phase(phase_df: pd.DataFrame, phase: str) -> bool:
    if phase.lower() == "blink":
        return True

    return str(phase_df["scenario"].iloc[0]) == "blink"
