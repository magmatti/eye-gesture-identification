from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class DetectionConfig:
    fixation_speed_threshold_deg_s: float = 30.0
    saccade_speed_threshold_deg_s: float = 120.0
    blink_threshold: float = 0.60
    min_fixation_duration_ms: float = 100.0
    min_saccade_duration_ms: float = 20.0
    max_saccade_duration_ms: float = 150.0
    min_blink_duration_ms: float = 50.0
    max_blink_duration_ms: float = 500.0
    smoothing_window: int = 5
    trim_start_ms: float = 300.0
