from __future__ import annotations

import numpy as np
import pandas as pd

from signal_utils import contiguous_true_segments, segment_duration_ms
from models.fixation_saccade_config import FixationSaccadeConfig


BASE_EVENT_COLUMNS = [
    "start_idx",
    "end_idx",
    "start_ms",
    "end_ms",
    "duration_ms",
]

SACCADE_EVENT_COLUMNS = BASE_EVENT_COLUMNS + ["peak_speed_deg_s"]


def detect_fixations_and_saccades(
    df: pd.DataFrame,
    cfg: FixationSaccadeConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cfg = cfg or FixationSaccadeConfig()
    out = df.copy()

    if not _has_fixation_saccade_columns(out):
        return _empty_fixation_saccade_result(out)

    fixation_candidate = _build_fixation_candidate_mask(out, cfg)
    saccade_candidate = _build_saccade_candidate_mask(out, cfg)

    fixation_mask, fixation_events = _extract_fixation_events(
        out,
        fixation_candidate,
        cfg,
    )

    saccade_mask, saccade_events = _extract_saccade_events(
        out,
        saccade_candidate,
        cfg,
    )

    out["is_fixation_sample"] = fixation_mask
    out["is_saccade_sample"] = saccade_mask

    return out, fixation_events, saccade_events


def summarize_fixations_and_saccades(
    df: pd.DataFrame,
    fix_events: pd.DataFrame,
    sac_events: pd.DataFrame,
) -> dict[str, float]:
    fixation_fraction = (
        float(df.get("is_fixation_sample", pd.Series(dtype=bool)).mean())
        if len(df)
        else 0.0
    )

    saccade_fraction = (
        float(df.get("is_saccade_sample", pd.Series(dtype=bool)).mean())
        if len(df)
        else 0.0
    )

    speed = df.get("gaze_speed_deg_s", pd.Series(dtype=float))
    dispersion = df.get("dispersion_deg", pd.Series(dtype=float))

    return {
        "fixation_event_count": float(len(fix_events)),
        "saccade_event_count": float(len(sac_events)),
        "fixation_sample_fraction": fixation_fraction,
        "saccade_sample_fraction": saccade_fraction,
        "high_speed_sample_fraction": (
            float((speed >= 80.0).mean()) if len(speed) else 0.0
        ),
        "speed_q50": float(speed.quantile(0.50)) if len(speed) else 0.0,
        "speed_q75": float(speed.quantile(0.75)) if len(speed) else 0.0,
        "speed_q90": float(speed.quantile(0.90)) if len(speed) else 0.0,
        "speed_q99": float(speed.quantile(0.99)) if len(speed) else 0.0,
        "dispersion_q95": (
            float(dispersion.quantile(0.95)) if len(dispersion) else 0.0
        ),
        "saccade_peak_mean_deg_s": (
            float(sac_events["peak_speed_deg_s"].mean())
            if len(sac_events) and "peak_speed_deg_s" in sac_events.columns
            else 0.0
        ),
    }


def _has_fixation_saccade_columns(df: pd.DataFrame) -> bool:
    required = {
        "Time_ms",
        "speed_smooth_deg_s",
        "dispersion_deg",
        "gaze_speed_deg_s",
    }
    return required.issubset(df.columns)


def _empty_fixation_saccade_result(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    out["is_fixation_sample"] = False
    out["is_saccade_sample"] = False

    fixation_events = pd.DataFrame(columns=BASE_EVENT_COLUMNS)
    saccade_events = pd.DataFrame(columns=SACCADE_EVENT_COLUMNS)

    return out, fixation_events, saccade_events


def _build_fixation_candidate_mask(
    df: pd.DataFrame,
    cfg: FixationSaccadeConfig,
) -> pd.Series:
    is_slow_enough = (
        df["speed_smooth_deg_s"].fillna(np.inf)
        <= cfg.fixation_speed_threshold_deg_s
    )

    is_compact_enough = (
        df["dispersion_deg"].fillna(np.inf)
        <= cfg.fixation_dispersion_threshold_deg
    )

    return is_slow_enough & is_compact_enough


def _build_saccade_candidate_mask(
    df: pd.DataFrame,
    cfg: FixationSaccadeConfig,
) -> pd.Series:
    return (
        df["gaze_speed_deg_s"].fillna(0.0)
        >= cfg.saccade_speed_threshold_deg_s
    )


def _extract_fixation_events(
    df: pd.DataFrame,
    fixation_candidate: pd.Series,
    cfg: FixationSaccadeConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    fixation_mask = np.zeros(len(df), dtype=bool)
    events = []

    for start, end in contiguous_true_segments(fixation_candidate.to_numpy()):
        duration_ms = segment_duration_ms(df["Time_ms"], start, end)

        if duration_ms < cfg.fixation_min_duration_ms:
            continue

        fixation_mask[start : end + 1] = True
        events.append(
            {
                "start_idx": start,
                "end_idx": end,
                "start_ms": float(df["Time_ms"].iloc[start]),
                "end_ms": float(df["Time_ms"].iloc[end]),
                "duration_ms": duration_ms,
            }
        )

    return fixation_mask, pd.DataFrame(events, columns=BASE_EVENT_COLUMNS)


def _extract_saccade_events(
    df: pd.DataFrame,
    saccade_candidate: pd.Series,
    cfg: FixationSaccadeConfig,
) -> tuple[np.ndarray, pd.DataFrame]:
    saccade_mask = np.zeros(len(df), dtype=bool)
    events = []

    for start, end in contiguous_true_segments(saccade_candidate.to_numpy()):
        duration_ms = segment_duration_ms(df["Time_ms"], start, end)

        if not cfg.saccade_min_duration_ms <= duration_ms <= cfg.saccade_max_duration_ms:
            continue

        saccade_mask[start : end + 1] = True

        peak_speed = float(
            df["gaze_speed_deg_s"].iloc[start : end + 1].max()
        )

        events.append(
            {
                "start_idx": start,
                "end_idx": end,
                "start_ms": float(df["Time_ms"].iloc[start]),
                "end_ms": float(df["Time_ms"].iloc[end]),
                "duration_ms": duration_ms,
                "peak_speed_deg_s": peak_speed,
            }
        )

    return saccade_mask, pd.DataFrame(events, columns=SACCADE_EVENT_COLUMNS)
