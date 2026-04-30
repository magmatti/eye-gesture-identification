from __future__ import annotations

from pathlib import Path

import pandas as pd

from detectors.blink import detect_blinks, summarize_blinks
from detectors.fixation_saccade import detect_fixations_and_saccades, summarize_fixations_and_saccades
from io_utils import collect_csv_files, ensure_time_ms, load_trial_csv
from signal_utils import add_gaze_features


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_REPORT_PATH = PROJECT_ROOT / "reports" / "rule_based_results.csv"


def analyze_trial(path: str | Path) -> dict[str, object]:
    raw = ensure_time_ms(load_trial_csv(path))
    df = add_gaze_features(raw)

    df, blink_events = detect_blinks(df)
    blink_summary = summarize_blinks(df, blink_events)

    df, fix_events, sac_events = detect_fixations_and_saccades(df)
    fix_sac_summary = summarize_fixations_and_saccades(df, fix_events, sac_events)

    predicted = classify_trial_rule_based(df, blink_summary, fix_sac_summary)
    expected = raw.attrs.get("expected_label", "unknown")

    result: dict[str, object] = {
        "file_name": Path(path).name,
        "expected_label": expected,
        "predicted_label": predicted,
        "correct": predicted == expected,
        "n_samples": len(df),
        **blink_summary,
        **fix_sac_summary,
    }
    return result


def classify_trial_rule_based(
    df: pd.DataFrame,
    blink_summary: dict[str, float],
    fix_sac_summary: dict[str, float],
) -> str:
    if blink_summary["blink_peak_max"] >= 0.60 and blink_summary["blink_event_count"] >= 1:
        return "blink"

    if (
        fix_sac_summary["saccade_event_count"] >= 1
        or (
            fix_sac_summary["high_speed_sample_fraction"] >= 0.010
            and fix_sac_summary["dispersion_q95"] >= 2.0
            and fix_sac_summary["speed_q75"] >= 5.6
        )
    ):
        return "saccade"

    if (
        fix_sac_summary["fixation_sample_fraction"] >= 0.45
        and fix_sac_summary["speed_q75"] <= 15.0
        and fix_sac_summary["speed_q90"] <= 35.0
    ):
        return "fixation"

    return "fixation"


def evaluate_dataset(data_dir: str | Path | None = None, report_path: str | Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    report_path = Path(report_path) if report_path is not None else DEFAULT_REPORT_PATH

    rows = [analyze_trial(path) for path in collect_csv_files(data_dir)]
    report = pd.DataFrame(rows).sort_values("file_name").reset_index(drop=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(report_path, index=False)

    accuracy = report["correct"].mean() if len(report) else 0.0
    print(f"Saved report to: {report_path}")
    print(f"Trial-level accuracy on current dataset: {accuracy:.3f}")
    print(report[["file_name", "expected_label", "predicted_label", "correct"]].to_string(index=False))
    return report
