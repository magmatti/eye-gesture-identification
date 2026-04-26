from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..signal_utils import moving_correlation, moving_gain


@dataclass(slots=True)
class SmoothPursuitConfig:
    min_target_speed_deg_s: float = 10.0
    min_correlation: float = 0.70
    min_gain: float = 0.50
    max_gain: float = 1.60
    max_saccade_fraction_inside_pursuit: float = 0.35
    window_samples: int = 15


def detect_smooth_pursuit(df: pd.DataFrame, cfg: SmoothPursuitConfig | None = None) -> tuple[pd.DataFrame, dict[str, float]]:
    """Detect smooth pursuit using target-aware metrics.

    A sample is marked as pursuit if:
    - the target is moving,
    - gaze and target yaw are strongly correlated in a local window,
    - pursuit gain is inside a plausible range,
    - the sample is not dominated by saccadic bursts.
    """
    cfg = cfg or SmoothPursuitConfig()
    out = df.copy()

    required = {"gaze_yaw_deg", "target_yaw_deg", "target_yaw_speed_deg_s", "gaze_speed_deg_s"}
    if not required.issubset(out.columns):
        out["is_pursuit_sample"] = False
        return out, {
            "pursuit_sample_fraction": 0.0,
            "pursuit_corr_median": 0.0,
            "pursuit_gain_median": 0.0,
            "target_profile": "none",
        }

    corr = moving_correlation(out["gaze_yaw_deg"], out["target_yaw_deg"], window=cfg.window_samples)
    gain = moving_gain(out["gaze_speed_deg_s"], out["target_yaw_speed_deg_s"], window=cfg.window_samples)

    target_moving = out["target_yaw_speed_deg_s"].abs() >= cfg.min_target_speed_deg_s
    saccade_like = out["gaze_speed_deg_s"].fillna(0.0) >= 120.0

    pursuit_mask = (
        target_moving.fillna(False)
        & (corr >= cfg.min_correlation).fillna(False)
        & (gain >= cfg.min_gain).fillna(False)
        & (gain <= cfg.max_gain).fillna(False)
    )

    # Remove pursuit regions that are too contaminated by saccade-like samples.
    contamination = saccade_like.rolling(window=cfg.window_samples, center=True, min_periods=1).mean()
    pursuit_mask &= contamination <= cfg.max_saccade_fraction_inside_pursuit

    out["pursuit_corr"] = corr
    out["pursuit_gain"] = gain
    out["is_pursuit_sample"] = pursuit_mask.fillna(False)

    target_unique_count = int(out.attrs.get("target_unique_count", 0))
    target_profile = "continuous" if target_unique_count > 20 else "jump"

    return out, {
        "pursuit_sample_fraction": float(out["is_pursuit_sample"].mean()),
        "pursuit_corr_median": float(corr.median(skipna=True)) if corr.notna().any() else 0.0,
        "pursuit_gain_median": float(gain.median(skipna=True)) if gain.notna().any() else 0.0,
        "target_profile": target_profile,
        "target_unique_count": float(target_unique_count),
    }
