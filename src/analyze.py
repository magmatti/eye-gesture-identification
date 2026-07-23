from __future__ import annotations

import shutil
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
REPORTS_DIR = Path("reports")


# process one recording and return enriched samples plus detected events
def analyze_file(path: Path, cfg: DetectionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_csv(path)
    df = normalize_time(df)
    df = add_gaze_speed(df, smoothing_window=cfg.smoothing_window)
    df = add_blink_signal(df)
    df = add_detection_masks(df, cfg)
    events = detect_all_events(df, cfg)
    return df, events


# run the full analysis pipeline for every csv file in the data directory
def run_analysis(cfg: DetectionConfig | None = None) -> pd.DataFrame:
    cfg = cfg or DetectionConfig()
    data_dir = DATA_DIR
    output_dir = REPORTS_DIR

    if output_dir.exists():
        shutil.rmtree(output_dir)

    processed_dir = output_dir / "processed"
    events_dir = output_dir / "events"
    processed_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)

    all_events = []
    for path in collect_csv_files(data_dir):
        processed, events = analyze_file(path, cfg)
        processed.to_csv(processed_dir / f"{path.stem}_processed.csv", index=False)
        all_events.append(events)

    events_df = (
        pd.concat(all_events, ignore_index=True)
        if all_events
        else pd.DataFrame(columns=EVENT_COLUMNS)
    )
    events_df.to_csv(events_dir / "all_events.csv", index=False)
    summarize_event_counts_by_file(events_df).to_csv(
        events_dir / "event_counts_by_file.csv",
        index=False,
    )
    summarize_events_by_file(events_df).to_csv(
        events_dir / "event_summary_by_file.csv",
        index=False,
    )
    summarize_events_by_gesture(events_df).to_csv(
        events_dir / "event_summary_by_gesture.csv",
        index=False,
    )
    return events_df


def print_report_table(label: str, table: pd.DataFrame) -> None:
    print(label)
    print()
    print(table.to_string(index=False))
    print()


# run analysis and print generated event reports
def main() -> None:
    events = run_analysis(DetectionConfig())
    for label, table in [
        ("event counts by file", summarize_event_counts_by_file(events)),
        ("event summary by file", summarize_events_by_file(events)),
        ("event summary by gesture", summarize_events_by_gesture(events)),
        ("all events", events),
    ]:
        print_report_table(label, table)


if __name__ == "__main__":
    main()
