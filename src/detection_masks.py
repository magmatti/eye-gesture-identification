from __future__ import annotations

import numpy as np
import pandas as pd

from .detection_config import DetectionConfig


# lables samples in the recording as one of 3 gestures or none using selected threshold values
def add_detection_masks(df: pd.DataFrame, cfg: DetectionConfig) -> pd.DataFrame:
    out = df.copy()

    raw_blink = out["blink_avg"].fillna(-np.inf) >= cfg.blink_threshold
    raw_saccade = (
        out["gaze_speed_smooth_deg_s"].fillna(-np.inf)
        >= cfg.saccade_speed_threshold_deg_s
    )
    raw_fixation = (
        out["gaze_speed_smooth_deg_s"].fillna(np.inf)
        <= cfg.fixation_speed_threshold_deg_s
    )

    out["is_blink"] = raw_blink
    out["is_saccade"] = raw_saccade & ~raw_blink
    out["is_fixation"] = raw_fixation & ~raw_blink & ~raw_saccade

    out["detected_gesture"] = np.select(
        [out["is_blink"], out["is_saccade"], out["is_fixation"]],
        ["blink", "saccade", "fixation"],
        default="none",
    )
    
    return out
