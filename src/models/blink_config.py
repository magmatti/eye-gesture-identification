from dataclasses import dataclass

@dataclass(slots=True)
class BlinkConfig:
    onset_threshold: float = 0.60
    offset_threshold: float = 0.20
    min_duration_ms: float = 50.0
    max_duration_ms: float = 500.0