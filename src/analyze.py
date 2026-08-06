from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .blink_signal import add_blink_signal
from .detection_config import DetectionConfig
from .detection_masks import add_detection_masks
from .event_detection import (
    EVENT_COLUMNS,
    detect_all_events,
    summarize_event_counts_by_file,
    summarize_event_counts_by_scenario,
    summarize_events_by_file,
    summarize_events_by_gesture,
    summarize_saccade_directions_by_file,
    summarize_saccade_directions_by_scenario,
)
from .gaze_signal import add_gaze_speed
from .ground_truth import (
    BLINK_WINDOW_COLUMNS,
    TARGET_JUMP_COLUMNS,
    evaluate_blink_targets,
    evaluate_saccade_targets,
    summarize_blink_ground_truth,
    summarize_saccade_ground_truth,
)
from .io_utils import collect_csv_files, load_csv, normalize_time
from .saccade_direction import add_saccade_directions

DATA_DIR = Path("data")


@dataclass(slots=True)
class AnalysisResult:
    events: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=EVENT_COLUMNS)
    )
    saccade_targets: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=TARGET_JUMP_COLUMNS)
    )
    blink_targets: pd.DataFrame = field(
        default_factory=lambda: pd.DataFrame(columns=BLINK_WINDOW_COLUMNS)
    )


# process one recording and return enriched samples plus detected events
def analyze_file(path: Path, cfg: DetectionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_csv(path)
    df = normalize_time(df)
    df = add_gaze_speed(df, smoothing_window=cfg.smoothing_window)
    df = add_blink_signal(df)
    df = add_detection_masks(df, cfg)
    events = detect_all_events(df, cfg)
    events = add_saccade_directions(df, events, cfg)

    return df, events


# analyze every csv file in the data directory, collecting events and
# ground-truth comparisons against the targets shown in the Unity scenarios
def run_analysis(cfg: DetectionConfig) -> AnalysisResult:
    all_events = []
    all_saccade_targets = []
    all_blink_targets = []
    for path in collect_csv_files(DATA_DIR):
        df, events = analyze_file(path, cfg)
        all_events.append(events)
        all_saccade_targets.append(evaluate_saccade_targets(df, events, cfg))
        all_blink_targets.append(evaluate_blink_targets(df, events, cfg))

    if not all_events:
        return AnalysisResult()

    return AnalysisResult(
        events=pd.concat(all_events, ignore_index=True),
        saccade_targets=pd.concat(all_saccade_targets, ignore_index=True),
        blink_targets=pd.concat(all_blink_targets, ignore_index=True),
    )


# compact tables for the default report, optionally extended with per-file
# and per-event detail tables
def build_report_tables(
    result: AnalysisResult,
    verbose: bool = False,
) -> list[tuple[str, pd.DataFrame]]:
    events = result.events
    tables = [
        ("Event counts by scenario", summarize_event_counts_by_scenario(events)),
        (
            "Saccade directions by scenario",
            summarize_saccade_directions_by_scenario(events),
        ),
        (
            "Saccade detection vs target",
            summarize_saccade_ground_truth(result.saccade_targets),
        ),
        ("Event summary by gesture", summarize_events_by_gesture(events)),
    ]
    if not result.blink_targets.empty:
        tables.insert(
            3,
            (
                "Blink detection vs beeps",
                summarize_blink_ground_truth(result.blink_targets),
            ),
        )

    if verbose:
        tables += [
            ("Event counts by file", summarize_event_counts_by_file(events)),
            (
                "Saccade directions by file",
                summarize_saccade_directions_by_file(events),
            ),
            ("Event summary by file", summarize_events_by_file(events)),
            ("Target jumps and matched saccades", result.saccade_targets),
            ("All events", events),
        ]

    return tables


def print_report_table(label: str, table: pd.DataFrame) -> None:
    print(label)
    print()
    if table.empty:
        print("no data")
    else:
        print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print()


# run analysis and print the compact report
def main() -> None:
    result = run_analysis(DetectionConfig())
    for label, table in build_report_tables(result):
        print_report_table(label.lower(), table)


if __name__ == "__main__":
    main()
