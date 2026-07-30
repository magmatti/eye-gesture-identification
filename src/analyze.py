from __future__ import annotations

from pathlib import Path

import pandas as pd

from .blink_signal import add_blink_signal
from .detection_config import DetectionConfig
from .detection_masks import add_detection_masks
from .event_detection import (
    EVENT_COLUMNS,
    detect_all_events,
    summarize_event_counts_by_file,
    summarize_events_by_file,
    summarize_events_by_gesture,
)
from .gaze_signal import add_gaze_speed
from .io_utils import collect_csv_files, load_csv, normalize_time


DATA_DIR = Path("data")


# process one recording and return enriched samples plus detected events
def analyze_file(path: Path, cfg: DetectionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_csv(path)
    df = normalize_time(df)
    df = add_gaze_speed(df, smoothing_window=cfg.smoothing_window)
    df = add_blink_signal(df)
    df = add_detection_masks(df, cfg)
    events = detect_all_events(df, cfg)

    return df, events


# analyze every csv file in the data directory and return all detected events
def run_analysis(cfg: DetectionConfig) -> pd.DataFrame:
    all_events = []
    for path in collect_csv_files(DATA_DIR):
        _, events = analyze_file(path, cfg)
        all_events.append(events)

    if not all_events:
        return pd.DataFrame(columns=EVENT_COLUMNS)
    
    return pd.concat(all_events, ignore_index=True)


def build_report_tables(events: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("Event counts by file", summarize_event_counts_by_file(events)),
        ("Event summary by file", summarize_events_by_file(events)),
        ("Event summary by gesture", summarize_events_by_gesture(events)),
        ("All events", events),
    ]


def print_report_table(label: str, table: pd.DataFrame) -> None:
    print(label)
    print()
    print(table.to_string(index=False))
    print()


# run analysis and print events
def main() -> None:
    events = run_analysis(DetectionConfig())
    for label, table in build_report_tables(events):
        print_report_table(label.lower(), table)


if __name__ == "__main__":
    main()
