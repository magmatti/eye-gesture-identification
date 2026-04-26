from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..signal_utils import contiguous_true_segments, segment_duration_ms


@dataclass(slots=True)
class FixationSaccadeConfig:
    fixation_speed_threshold_deg_s: float = 30.0
    fixation_dispersion_threshold_deg: float = 2.5
    fixation_min_duration_ms: float = 100.0
    saccade_speed_threshold_deg_s: float = 120.0
    saccade_min_duration_ms: float = 20.0
    saccade_max_duration_ms: float = 120.0


def detect_fixations_and_saccades(df: pd.DataFrame, cfg: FixationSaccadeConfig | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Detect fixations and saccades from local eye rotations.

    Fixation logic:
        low speed + low dispersion + minimum duration

    Saccade logic:
        high speed burst + plausible duration
    """
    cfg = cfg or FixationSaccadeConfig()
    out = df.copy()

    required = {"Time_ms", "speed_smooth_deg_s", "dispersion_deg", "gaze_speed_deg_s"}
    if not required.issubset(out.columns):
        out["is_fixation_sample"] = False
        out["is_saccade_sample"] = False
        empty = pd.DataFrame(columns=["start_idx", "end_idx", "start_ms", "end_ms", "duration_ms"])
        return out, empty.copy(), empty.copy()

    fixation_candidate = (
        out["speed_smooth_deg_s"].fillna(np.inf) <= cfg.fixation_speed_threshold_deg_s
    ) & (
        out["dispersion_deg"].fillna(np.inf) <= cfg.fixation_dispersion_threshold_deg
    )

    saccade_candidate = out["gaze_speed_deg_s"].fillna(0.0) >= cfg.saccade_speed_threshold_deg_s

    fixation_mask = np.zeros(len(out), dtype=bool)
    saccade_mask = np.zeros(len(out), dtype=bool)
    fixation_events = []
    saccade_events = []

    for start, end in contiguous_true_segments(fixation_candidate.to_numpy()):
        duration_ms = segment_duration_ms(out["Time_ms"], start, end)
        if duration_ms >= cfg.fixation_min_duration_ms:
            fixation_mask[start: end + 1] = True
            fixation_events.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "start_ms": float(out["Time_ms"].iloc[start]),
                    "end_ms": float(out["Time_ms"].iloc[end]),
                    "duration_ms": duration_ms,
                }
            )

    for start, end in contiguous_true_segments(saccade_candidate.to_numpy()):
        duration_ms = segment_duration_ms(out["Time_ms"], start, end)
        if cfg.saccade_min_duration_ms <= duration_ms <= cfg.saccade_max_duration_ms:
            saccade_mask[start: end + 1] = True
            peak_speed = float(out["gaze_speed_deg_s"].iloc[start: end + 1].max())
            saccade_events.append(
                {
                    "start_idx": start,
                    "end_idx": end,
                    "start_ms": float(out["Time_ms"].iloc[start]),
                    "end_ms": float(out["Time_ms"].iloc[end]),
                    "duration_ms": duration_ms,
                    "peak_speed_deg_s": peak_speed,
                }
            )

    out["is_fixation_sample"] = fixation_mask
    out["is_saccade_sample"] = saccade_mask
    return out, pd.DataFrame(fixation_events), pd.DataFrame(saccade_events)


def summarize_fixations_and_saccades(df: pd.DataFrame, fix_events: pd.DataFrame, sac_events: pd.DataFrame) -> dict[str, float]:
    fixation_fraction = float(df.get("is_fixation_sample", pd.Series(dtype=bool)).mean()) if len(df) else 0.0
    saccade_fraction = float(df.get("is_saccade_sample", pd.Series(dtype=bool)).mean()) if len(df) else 0.0
    speed = df.get("gaze_speed_deg_s", pd.Series(dtype=float))
    dispersion = df.get("dispersion_deg", pd.Series(dtype=float))

    return {
        "fixation_event_count": float(len(fix_events)),
        "saccade_event_count": float(len(sac_events)),
        "fixation_sample_fraction": fixation_fraction,
        "saccade_sample_fraction": saccade_fraction,
        "speed_q50": float(speed.quantile(0.50)) if len(speed) else 0.0,
        "speed_q75": float(speed.quantile(0.75)) if len(speed) else 0.0,
        "speed_q90": float(speed.quantile(0.90)) if len(speed) else 0.0,
        "speed_q99": float(speed.quantile(0.99)) if len(speed) else 0.0,
        "dispersion_q95": float(dispersion.quantile(0.95)) if len(dispersion) else 0.0,
        "saccade_peak_mean_deg_s": float(sac_events["peak_speed_deg_s"].mean()) if len(sac_events) else 0.0,
    }
