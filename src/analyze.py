from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from .blink_signal import add_blink_signal
from .detection_config import DetectionConfig, ScenarioConfig
from .detection_masks import add_detection_masks
from .event_detection import detect_all_events
from .gaze_signal import add_gaze_speed
from .ground_truth import (
    EVALUATION_SCOPE,
    build_expected_blinks,
    build_expected_saccades,
    label_detected_events,
    match_expected_to_detected,
)
from .io_utils import collect_csv_files, load_csv, normalize_time
from .reports import (
    build_blink_evaluation,
    build_saccade_evaluation,
    build_sample_coverage,
)
from .saccade_direction import add_saccade_directions

DATA_DIR = Path("data")


@dataclass(slots=True)
class AnalysisResult:
    samples: dict[str, pd.DataFrame]
    events: pd.DataFrame
    saccade_matches: pd.DataFrame
    blink_matches: pd.DataFrame


# process one recording and return enriched samples plus detected events
def analyze_file(path: Path, cfg: DetectionConfig) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = load_csv(path)
    df = normalize_time(df)
    df = add_gaze_speed(df, smoothing_window=cfg.smoothing_window)
    df = add_blink_signal(df)
    df = add_detection_masks(df, cfg)
    events = detect_all_events(df, cfg)
    events = add_saccade_directions(df, events, cfg)
    events = events[
        (events["gesture"] != "saccade")
        | (events["saccade_amplitude_deg"] >= cfg.min_saccade_amplitude_deg)
    ].reset_index(drop=True)
    return df, events


# analyze every csv file in the data directory, collecting events and
# ground-truth comparisons against the targets shown in the Unity scenarios
def run_analysis(
    detection_cfg: DetectionConfig,
    scenario_cfg: ScenarioConfig,
) -> AnalysisResult:
    samples_by_file = {}
    all_events = []
    all_saccade_matches = []
    all_blink_matches = []
    for path in collect_csv_files(DATA_DIR):
        df, events = analyze_file(path, detection_cfg)
        samples_by_file[str(df["source_file"].iloc[0])] = df
        saccade_matches = match_expected_to_detected(
            build_expected_saccades(df),
            events,
            scenario_cfg.saccade_match_max_latency_ms / 1000.0,
        )
        blink_matches = match_expected_to_detected(
            build_expected_blinks(df, scenario_cfg),
            events,
            scenario_cfg.blink_match_max_latency_ms / 1000.0,
        )
        matches = pd.concat([saccade_matches, blink_matches], ignore_index=True)
        events = label_detected_events(events, matches, EVALUATION_SCOPE)
        if not events.empty:
            all_events.append(events)
        if not saccade_matches.empty:
            all_saccade_matches.append(
                saccade_matches.drop(columns="detected_event_index")
            )
        if not blink_matches.empty:
            all_blink_matches.append(blink_matches.drop(columns="detected_event_index"))
    return AnalysisResult(
        samples=samples_by_file,
        events=pd.concat(all_events, ignore_index=True),
        saccade_matches=pd.concat(all_saccade_matches, ignore_index=True),
        blink_matches=pd.concat(all_blink_matches, ignore_index=True),
    )


# build the three compact tables used by the console smoke test
def build_report_tables(result: AnalysisResult) -> list[tuple[str, pd.DataFrame]]:
    samples = pd.concat(result.samples.values(), ignore_index=True)
    return [
        ("T2 — Sample coverage", build_sample_coverage(samples)),
        (
            "T4 — Saccade evaluation",
            build_saccade_evaluation(result.saccade_matches, result.events),
        ),
        (
            "T6 — Blink evaluation",
            build_blink_evaluation(result.blink_matches, result.events),
        ),
    ]


# print one report table with consistent numeric formatting
def _print_report_table(label: str, table: pd.DataFrame) -> None:
    print(label)
    print(table.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print()


# run the full pipeline and print the three compact smoke-test tables
def main() -> None:
    result = run_analysis(DetectionConfig(), ScenarioConfig())
    for label, table in build_report_tables(result):
        _print_report_table(label, table)


if __name__ == "__main__":
    main()
