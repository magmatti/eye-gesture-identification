from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GestureSpec:
    name: str
    mask_column: str
    min_duration_attr: str
    max_duration_attr: str | None
    detection_order: int


BLINK = GestureSpec(
    name="blink",
    mask_column="is_blink",
    min_duration_attr="min_blink_duration_ms",
    max_duration_attr="max_blink_duration_ms",
    detection_order=0,
)
SACCADE = GestureSpec(
    name="saccade",
    mask_column="is_saccade",
    min_duration_attr="min_saccade_duration_ms",
    max_duration_attr="max_saccade_duration_ms",
    detection_order=1,
)
FIXATION = GestureSpec(
    name="fixation",
    mask_column="is_fixation",
    min_duration_attr="min_fixation_duration_ms",
    max_duration_attr=None,
    detection_order=2,
)
GESTURE_SPECS = (BLINK, SACCADE, FIXATION)
DETECTION_GESTURE_SPECS = tuple(
    sorted(GESTURE_SPECS, key=lambda spec: spec.detection_order)
)
