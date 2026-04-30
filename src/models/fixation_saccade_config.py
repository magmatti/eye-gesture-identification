from dataclasses import dataclass

@dataclass(slots=True)
class FixationSaccadeConfig:
    fixation_speed_threshold_deg_s: float = 30.0
    fixation_dispersion_threshold_deg: float = 2.5
    fixation_min_duration_ms: float = 100.0
    saccade_speed_threshold_deg_s: float = 120.0
    saccade_min_duration_ms: float = 20.0
    saccade_max_duration_ms: float = 120.0