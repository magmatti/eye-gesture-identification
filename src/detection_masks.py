from __future__ import annotations

import pandas as pd

from .detection_config import DetectionConfig
from .gesture_specs import BLINK, FIXATION, SACCADE


# labels samples in the recording as one of 3 gestures or none using selected threshold values
def add_detection_masks(df: pd.DataFrame, cfg: DetectionConfig) -> pd.DataFrame:
    out = df.copy()
    raw_blink = out["blink_avg"] >= cfg.blink_threshold
    speed = out["gaze_speed_smooth_deg_s"]
    raw_saccade = speed > cfg.velocity_threshold_deg_s
    raw_fixation = speed <= cfg.velocity_threshold_deg_s
    out[BLINK.mask_column] = raw_blink
    out[SACCADE.mask_column] = raw_saccade & ~raw_blink
    out[FIXATION.mask_column] = raw_fixation & ~raw_blink
    before_trim = out["Time_s"] * 1000.0 < cfg.trim_start_ms
    out.loc[
        before_trim,
        [BLINK.mask_column, SACCADE.mask_column, FIXATION.mask_column],
    ] = False
    return out
