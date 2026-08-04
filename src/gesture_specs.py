from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Literal

PeakAggregation = Literal["max", "mean"]


@dataclass(frozen=True, slots=True)
class GestureSpec:
    name: str
    mask_column: str
    min_duration_attr: str
    max_duration_attr: str | None
    peak_column: str
    peak_aggregation: PeakAggregation
    detection_order: int
    report_order: int

    @property
    def count_column(self) -> str:
        return f"{self.name}_count"


BLINK = GestureSpec(
    name="blink",
    mask_column="is_blink",
    min_duration_attr="min_blink_duration_ms",
    max_duration_attr="max_blink_duration_ms",
    peak_column="blink_avg",
    peak_aggregation="max",
    detection_order=0,
    report_order=0,
)

SACCADE = GestureSpec(
    name="saccade",
    mask_column="is_saccade",
    min_duration_attr="min_saccade_duration_ms",
    max_duration_attr="max_saccade_duration_ms",
    peak_column="gaze_speed_smooth_deg_s",
    peak_aggregation="max",
    detection_order=1,
    report_order=2,
)

FIXATION = GestureSpec(
    name="fixation",
    mask_column="is_fixation",
    min_duration_attr="min_fixation_duration_ms",
    max_duration_attr=None,
    peak_column="gaze_speed_smooth_deg_s",
    peak_aggregation="mean",
    detection_order=2,
    report_order=1,
)

GESTURE_SPECS = (BLINK, SACCADE, FIXATION)
DETECTION_GESTURE_SPECS = tuple(
    sorted(GESTURE_SPECS, key=lambda spec: spec.detection_order)
)
REPORT_GESTURE_SPECS = tuple(sorted(GESTURE_SPECS, key=lambda spec: spec.report_order))
GESTURE_SPECS_BY_NAME: Mapping[str, GestureSpec] = MappingProxyType(
    {spec.name: spec for spec in GESTURE_SPECS}
)
