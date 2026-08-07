from __future__ import annotations

import numpy as np
import pandas as pd

from .detection_config import ScenarioConfig
from .event_detection import contiguous_true_segments
from .io_utils import split_by_phase
from .saccade_direction import classify_direction

TARGET_COLUMNS = ["TargetRotX", "TargetRotY"]
EXPECTED_COLUMNS = [
    "source_file",
    "participant",
    "scenario",
    "phase",
    "gesture",
    "expected_time_s",
    "expected_direction",
    "expected_direction_deg",
    "expected_amplitude_deg",
]
MATCH_COLUMNS = [
    *EXPECTED_COLUMNS,
    "detected",
    "detected_event_index",
    "latency_ms",
    "detected_direction",
    "detected_direction_deg",
    "detected_amplitude_deg",
    "direction_match",
    "direction_error_deg",
    "amplitude_error_deg",
]
EVALUATION_SCOPE = {
    ("fixation", "recording"): ("saccade",),
    ("saccade", "recording"): ("saccade",),
    ("blink", "recording"): ("blink",),
    ("combined", "Fixation"): ("saccade", "blink"),
    ("combined", "Saccade"): ("saccade", "blink"),
    ("combined", "Blink"): ("saccade", "blink"),
}
# a beep scheduled exactly at the end of the recording loses the race against
# the test-stop coroutine, so treat the recording end as an exclusive bound
_BEEP_STOP_TOLERANCE_S = 0.1


# build expected saccades from animated target movements in each recording phase
def build_expected_saccades(samples: pd.DataFrame) -> pd.DataFrame:
    if not all(column in samples.columns for column in TARGET_COLUMNS):
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
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
                    "gesture": "saccade",
                    "expected_time_s": float(phase_df["Time_s"].iloc[start]),
                    "expected_direction": classify_direction(
                        delta_horizontal, delta_vertical
                    ),
                    "expected_direction_deg": float(
                        np.degrees(np.arctan2(delta_vertical, delta_horizontal))
                    ),
                    "expected_amplitude_deg": float(
                        np.hypot(delta_horizontal, delta_vertical)
                    ),
                }
            )
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


# build expected blink times from the metronome schedule in blink phases
def build_expected_blinks(
    samples: pd.DataFrame,
    scenario_cfg: ScenarioConfig,
) -> pd.DataFrame:
    rows = []
    for phase, phase_df in split_by_phase(samples).items():
        scenario = str(phase_df["scenario"].iloc[0])
        if "blink" not in EVALUATION_SCOPE[(scenario, phase)]:
            continue
        if not _is_blink_target_phase(scenario, phase):
            continue
        phase_start_s = float(phase_df["Time_s"].iloc[0])
        duration_s = float(phase_df["Time_s"].iloc[-1] - phase_start_s)
        for beep_index in range(_expected_beep_count(duration_s, scenario_cfg)):
            rows.append(
                {
                    "source_file": str(phase_df["source_file"].iloc[0]),
                    "participant": str(phase_df["participant"].iloc[0]),
                    "scenario": scenario,
                    "phase": phase,
                    "gesture": "blink",
                    "expected_time_s": phase_start_s
                    + scenario_cfg.beep_initial_delay_s
                    + beep_index * scenario_cfg.beep_interval_s,
                    "expected_direction": pd.NA,
                    "expected_direction_deg": np.nan,
                    "expected_amplitude_deg": np.nan,
                }
            )
    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


# greedily match each expected event to the earliest eligible detected event
def match_expected_to_detected(
    expected: pd.DataFrame,
    detected: pd.DataFrame,
    max_latency_s: float,
) -> pd.DataFrame:
    out = expected.sort_values(
        ["source_file", "participant", "phase", "expected_time_s"]
    ).reset_index(drop=True)
    out["detected"] = False
    out["detected_event_index"] = pd.NA
    out["latency_ms"] = np.nan
    out["detected_direction"] = pd.NA
    out["detected_direction_deg"] = np.nan
    out["detected_amplitude_deg"] = np.nan
    out["direction_match"] = pd.Series(pd.NA, index=out.index, dtype="boolean")
    out["direction_error_deg"] = np.nan
    out["amplitude_error_deg"] = np.nan
    used_indices: set[int] = set()
    for index, expected_event in out.iterrows():
        candidates = detected[
            (detected["source_file"] == expected_event["source_file"])
            & (detected["participant"] == expected_event["participant"])
            & (detected["phase"] == expected_event["phase"])
            & (detected["gesture"] == expected_event["gesture"])
            & (detected["start_time_s"] >= expected_event["expected_time_s"])
            & (
                detected["start_time_s"]
                <= expected_event["expected_time_s"] + max_latency_s
            )
            & ~detected.index.isin(used_indices)
        ].sort_values("start_time_s")
        if candidates.empty:
            continue
        detected_index = int(candidates.index[0])
        match = candidates.iloc[0]
        used_indices.add(detected_index)
        out.at[index, "detected"] = True
        out.at[index, "detected_event_index"] = detected_index
        out.at[index, "latency_ms"] = (
            float(match["start_time_s"] - expected_event["expected_time_s"]) * 1000.0
        )
        out.at[index, "detected_direction"] = match["saccade_direction"]
        out.at[index, "detected_direction_deg"] = match["saccade_direction_deg"]
        out.at[index, "detected_amplitude_deg"] = match["saccade_amplitude_deg"]
        if expected_event["gesture"] == "saccade":
            out.at[index, "direction_match"] = bool(
                match["saccade_direction"] == expected_event["expected_direction"]
            )
            out.at[index, "amplitude_error_deg"] = float(
                match["saccade_amplitude_deg"]
                - expected_event["expected_amplitude_deg"]
            )
            out.at[index, "direction_error_deg"] = float(
                (
                    match["saccade_direction_deg"]
                    - expected_event["expected_direction_deg"]
                    + 180.0
                )
                % 360.0
                - 180.0
            )
    return out[MATCH_COLUMNS]


# mark detected events that are evaluated and matched within the explicit scope
def label_detected_events(
    events: pd.DataFrame,
    matches: pd.DataFrame,
    scope: dict[tuple[str, str], tuple[str, ...]],
) -> pd.DataFrame:
    out = events.copy()
    out["evaluated"] = [
        row.gesture in scope[(row.scenario, row.phase)] for row in out.itertuples()
    ]
    matched_indices = matches.loc[matches["detected"], "detected_event_index"].astype(
        int
    )
    out["matched"] = out.index.isin(matched_indices)
    return out


# count metronome beeps played during a recording window
def _expected_beep_count(duration_s: float, cfg: ScenarioConfig) -> int:
    window_s = duration_s - cfg.beep_initial_delay_s - _BEEP_STOP_TOLERANCE_S
    if window_s < 0.0:
        return 0
    return int(window_s // cfg.beep_interval_s) + 1


# identify phases where metronome beeps define expected blinks
def _is_blink_target_phase(scenario: str, phase: str) -> bool:
    return scenario == "blink" or (scenario == "combined" and phase == "Blink")


# wrap Unity euler angles from [0, 360) into [-180, 180)
def _wrap_degrees(angles: np.ndarray) -> np.ndarray:
    return (angles + 180.0) % 360.0 - 180.0
