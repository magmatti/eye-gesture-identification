from __future__ import annotations

from pathlib import Path

import pandas as pd

from .io_utils import collect_csv_files, load_trial_csv, ensure_time_ms
from .signal_utils import add_gaze_features, add_target_features
from .detectors.blink import detect_blinks, summarize_blinks
from .detectors.fixation_saccade import detect_fixations_and_saccades, summarize_fixations_and_saccades
from .detectors.smooth_pursuit import detect_smooth_pursuit


DEFAULT_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "raw" / "latest"
DEFAULT_REPORT_PATH = Path(__file__).resolve().parents[2] / "reports" / "rule_based_results.csv"


def analyze_trial(path: str | Path) -> dict[str, object]:
    raw = ensure_time_ms(load_trial_csv(path))
    df = add_target_features(add_gaze_features(raw))

    df, blink_events = detect_blinks(df)
    blink_summary = summarize_blinks(df, blink_events)

    df, fix_events, sac_events = detect_fixations_and_saccades(df)
    fix_sac_summary = summarize_fixations_and_saccades(df, fix_events, sac_events)

    df, pursuit_summary = detect_smooth_pursuit(df)

    predicted = classify_trial_rule_based(df, blink_summary, fix_sac_summary, pursuit_summary)
    expected = raw.attrs.get("expected_label", "unknown")

    result: dict[str, object] = {
        "file_name": Path(path).name,
        "expected_label": expected,
        "predicted_label": predicted,
        "correct": predicted == expected,
        "n_samples": len(df),
        **blink_summary,
        **fix_sac_summary,
        **pursuit_summary,
    }
    return result


def classify_trial_rule_based(
    df: pd.DataFrame,
    blink_summary: dict[str, float],
    fix_sac_summary: dict[str, float],
    pursuit_summary: dict[str, float],
) -> str:
    """Classify an entire controlled trial into one gesture label.

    This is intentionally simple and transparent.
    The rules are chosen so you can explain them in the thesis.
    """
    if blink_summary["blink_peak_max"] >= 0.60 and blink_summary["blink_event_count"] >= 1:
        return "blink"

    # Smooth pursuit: continuous target + large pursuit fraction + good correlation.
    if (
        pursuit_summary.get("target_profile") == "continuous"
        and pursuit_summary["pursuit_sample_fraction"] >= 0.18
        and pursuit_summary["pursuit_corr_median"] >= 0.70
    ):
        return "smooth_pursuit"

    # Saccade: jump target profile OR large saccade peaks with more dispersion than fixation.
    if pursuit_summary.get("target_profile") == "jump":
        return "saccade"

    # Pure fixation trials should have low sustained speed and large fixation fraction.
    if (
        fix_sac_summary["fixation_sample_fraction"] >= 0.45
        and fix_sac_summary["speed_q75"] <= 15.0
        and fix_sac_summary["speed_q90"] <= 35.0
    ):
        return "fixation"

    # Fallbacks for noisy cases.
    if fix_sac_summary["speed_q75"] >= 40.0:
        return "smooth_pursuit"
    if fix_sac_summary["speed_q99"] >= 120.0:
        return "saccade"
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
