from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class GestureSpec:
    name: str
    mask_column: str
    min_duration_attr: str
    max_duration_attr: str | None


BLINK = GestureSpec(
    name="blink",
    mask_column="is_blink",
    min_duration_attr="min_blink_duration_ms",
    max_duration_attr="max_blink_duration_ms",
)
SACCADE = GestureSpec(
    name="saccade",
    mask_column="is_saccade",
    min_duration_attr="min_saccade_duration_ms",
    max_duration_attr="max_saccade_duration_ms",
)
FIXATION = GestureSpec(
    name="fixation",
    mask_column="is_fixation",
    min_duration_attr="min_fixation_duration_ms",
    max_duration_attr=None,
)
GESTURE_SPECS = (BLINK, SACCADE, FIXATION)
