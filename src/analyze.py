from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from .blink_signal import add_blink_signal, has_blink_columns
from .detection_config import DetectionConfig
from .detection_masks import add_detection_masks
from .gesture_events import (
    EVENT_COLUMNS,
    detect_all_events,
    summarize_events_by_file,
    summarize_events_by_gesture,
)
from .gaze_signal import add_gaze_speed, has_gaze_columns
from .io_utils import collect_csv_files, load_csv, normalize_time
from .plots import (
    plot_blink_signal,
    plot_combined_overview,
    plot_detected_events,
    plot_gaze_speed,
)


DATA_DIR = Path("data")
REPORTS_DIR = Path("reports")


# process one recording and return enriched samples plus detected events
def analyze_file(path: Path, cfg: DetectionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_csv(path)
    df = normalize_time(df)

    if has_gaze_columns(df):
        df = add_gaze_speed(df, smoothing_window=cfg.smoothing_window)
    else:
        df = add_gaze_speed(df, smoothing_window=cfg.smoothing_window)

    if has_blink_columns(df):
        df = add_blink_signal(df)
    else:
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
    plots_dir = output_dir / "plots"
    processed_dir.mkdir(parents=True, exist_ok=True)
    events_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    all_events = []
    for path in collect_csv_files(data_dir):
        processed, events = analyze_file(path, cfg)
        stem = path.stem

        processed.to_csv(processed_dir / f"{stem}_processed.csv", index=False)
        all_events.append(events)

        if has_gaze_columns(processed):
            plot_gaze_speed(processed, cfg, plots_dir / f"{stem}_gaze_speed.png")
        if has_blink_columns(processed):
            plot_blink_signal(processed, cfg, plots_dir / f"{stem}_blink_signal.png")
        plot_combined_overview(processed, cfg, plots_dir / f"{stem}_overview.png")
        plot_detected_events(processed, events, cfg, plots_dir / f"{stem}_events.png")

    events_df = (
        pd.concat(all_events, ignore_index=True)
        if all_events
        else pd.DataFrame(columns=EVENT_COLUMNS)
    )
    events_df.to_csv(events_dir / "all_events.csv", index=False)
    summarize_events_by_file(events_df).to_csv(
        events_dir / "event_summary_by_file.csv",
        index=False,
    )
    summarize_events_by_gesture(events_df).to_csv(
        events_dir / "event_summary_by_gesture.csv",
        index=False,
    )
    return events_df


# run analysis and print a short summary
def main() -> pd.DataFrame:
    events = run_analysis(DetectionConfig())
    print(f"Saved processed files and reports to: {REPORTS_DIR}")
    print(f"Detected events: {len(events)}")
    return events


if __name__ == "__main__":
    main()
