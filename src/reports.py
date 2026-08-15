from __future__ import annotations

import numpy as np
import pandas as pd

from .ground_truth import EVALUATION_SCOPE
from .saccade_direction import SACCADE_DIRECTIONS

GROUP_COLUMNS = ["participant", "scenario", "phase"]
GESTURES = ["fixation", "saccade", "blink"]
SIGNAL_BY_GESTURE = {
    "fixation": "mean_speed_deg_s",
    "saccade": "peak_speed_deg_s",
    "blink": "peak_blink_weight",
}


# summarize recording size and sampling characteristics for every source file
def build_dataset_summary(
    samples_by_file: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    rows = []
    for source_file, samples in samples_by_file.items():
        mean_delta_t_s = samples["Time_s"].diff().mean()
        rows.append(
            {
                "participant": str(samples["participant"].iloc[0]),
                "scenario": str(samples["scenario"].iloc[0]),
                "source_file": source_file,
                "duration_variant": source_file.rsplit("_", 1)[-1].removesuffix(".csv"),
                "sample_count": len(samples),
                "duration_s": float(samples["Time_s"].iloc[-1]),
                "mean_delta_t_ms": mean_delta_t_s * 1000.0,
                "effective_hz": 1.0 / mean_delta_t_s,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["participant", "source_file"])
        .reset_index(drop=True)
    )


# report the class distribution and unlabeled share among all samples
def build_sample_coverage(samples: pd.DataFrame) -> pd.DataFrame:
    counts = (
        samples.groupby(["participant", "scenario"])[
            ["is_fixation", "is_saccade", "is_blink"]
        ]
        .sum()
        .rename(
            columns={
                "is_fixation": "fixation_share",
                "is_saccade": "saccade_share",
                "is_blink": "blink_share",
            }
        )
    )
    sample_counts = samples.groupby(["participant", "scenario"]).size()
    shares = counts.div(sample_counts, axis=0)
    shares["unlabeled_share"] = 1.0 - shares.sum(axis=1)
    shares.insert(0, "sample_count", sample_counts)
    return shares.reset_index()


# summarize duration and gesture-specific signal values for each event class
def build_event_characteristics(events: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gesture in GESTURES:
        gesture_events = events[events["gesture"] == gesture]
        row = {
            "gesture": gesture,
            "event_count": len(gesture_events),
            "duration_mean_ms": gesture_events["duration_ms"].mean(),
            "duration_median_ms": gesture_events["duration_ms"].median(),
            "duration_sd_ms": gesture_events["duration_ms"].std(),
            "duration_min_ms": gesture_events["duration_ms"].min(),
            "duration_max_ms": gesture_events["duration_ms"].max(),
            "peak_speed_deg_s": np.nan,
            "mean_speed_deg_s": np.nan,
            "peak_blink_weight": np.nan,
        }
        signal_column = SIGNAL_BY_GESTURE[gesture]
        row[signal_column] = gesture_events[signal_column].mean()
        rows.append(row)
    return pd.DataFrame(rows)


# build participant, scenario, phase and total saccade evaluation metrics
def build_saccade_evaluation(
    matches: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    return _build_evaluation(matches, events, "saccade", include_amplitude=True)


# build the cardinal-direction confusion matrix for matched saccades
def build_direction_confusion(matches: pd.DataFrame) -> pd.DataFrame:
    matched = matches[matches["detected"]]
    return pd.crosstab(
        matched["expected_direction"], matched["detected_direction"]
    ).reindex(
        index=SACCADE_DIRECTIONS[:-1],
        columns=SACCADE_DIRECTIONS,
        fill_value=0,
    )


# summarize direction classification accuracy and continuous angular error
def build_direction_error(matches: pd.DataFrame) -> pd.DataFrame:
    matched = matches[matches["detected"]]
    return pd.DataFrame(
        [
            {
                "direction_accuracy": matched["direction_match"].mean(),
                "direction_error_mean_deg": matched["direction_error_deg"].mean(),
                "direction_error_sd_deg": matched["direction_error_deg"].std(),
            }
        ]
    )


# build participant, scenario, phase and total blink evaluation metrics
def build_blink_evaluation(
    matches: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    return _build_evaluation(matches, events, "blink", include_amplitude=False)


# compare the global detection quality of all three gesture classes
def build_gesture_comparison(
    saccade_matches: pd.DataFrame,
    blink_matches: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        {
            "gesture": "fixation",
            "event_count": int((events["gesture"] == "fixation").sum()),
            "expected_count": np.nan,
            "tp": np.nan,
            "fp": np.nan,
            "fn": np.nan,
        }
    ]
    for gesture, matches in (
        ("saccade", saccade_matches),
        ("blink", blink_matches),
    ):
        gesture_events = events[(events["gesture"] == gesture) & events["evaluated"]]
        tp = int(matches["detected"].sum())
        rows.append(
            {
                "gesture": gesture,
                "event_count": int((events["gesture"] == gesture).sum()),
                "expected_count": len(matches),
                "tp": tp,
                "fp": int((~gesture_events["matched"]).sum()),
                "fn": len(matches) - tp,
            }
        )
    columns = [
        "gesture",
        "event_count",
        "expected_count",
        "tp",
        "fp",
        "fn",
        "precision",
        "recall",
        "f1",
    ]
    return _add_rates(pd.DataFrame(rows))[columns]


# assemble grouped detection counts, quality rates and timing statistics
def _build_evaluation(
    matches: pd.DataFrame,
    events: pd.DataFrame,
    gesture: str,
    include_amplitude: bool,
) -> pd.DataFrame:
    contexts = pd.concat(
        [
            events[["participant", "scenario"]],
            matches[["participant", "scenario"]],
        ]
    ).drop_duplicates()
    groups = pd.DataFrame(
        [
            {
                "participant": context.participant,
                "scenario": context.scenario,
                "phase": phase,
            }
            for context in contexts.itertuples()
            for scenario, phase in EVALUATION_SCOPE
            if scenario == context.scenario
            and gesture in EVALUATION_SCOPE[(scenario, phase)]
        ]
    )
    expected = matches.groupby(GROUP_COLUMNS).size().rename("expected_count")
    tp = matches.groupby(GROUP_COLUMNS)["detected"].sum().rename("tp")
    false_positives = events[
        (events["gesture"] == gesture) & events["evaluated"] & ~events["matched"]
    ]
    fp = false_positives.groupby(GROUP_COLUMNS).size().rename("fp")
    summary = groups.merge(expected, on=GROUP_COLUMNS, how="left")
    summary = summary.merge(tp, on=GROUP_COLUMNS, how="left")
    summary = summary.merge(fp, on=GROUP_COLUMNS, how="left").fillna(
        {"expected_count": 0, "tp": 0, "fp": 0}
    )
    summary[["expected_count", "tp", "fp"]] = summary[
        ["expected_count", "tp", "fp"]
    ].astype(int)
    summary["fn"] = summary["expected_count"] - summary["tp"]
    matched = matches[matches["detected"]]
    aggregations = {
        "latency_median_ms": ("latency_ms", "median"),
        "latency_mean_ms": ("latency_ms", "mean"),
    }
    if include_amplitude:
        aggregations |= {
            "amplitude_error_mean_deg": ("amplitude_error_deg", "mean"),
            "amplitude_error_sd_deg": ("amplitude_error_deg", "std"),
        }
    measurements = matched.groupby(GROUP_COLUMNS).agg(**aggregations).reset_index()
    summary = summary.merge(measurements, on=GROUP_COLUMNS, how="left")
    total = {
        "participant": "TOTAL",
        "scenario": "TOTAL",
        "phase": "TOTAL",
        "expected_count": len(matches),
        "tp": int(matches["detected"].sum()),
        "fp": len(false_positives),
        "fn": int((~matches["detected"]).sum()),
        "latency_median_ms": matched["latency_ms"].median(),
        "latency_mean_ms": matched["latency_ms"].mean(),
    }
    if include_amplitude:
        total |= {
            "amplitude_error_mean_deg": matched["amplitude_error_deg"].mean(),
            "amplitude_error_sd_deg": matched["amplitude_error_deg"].std(),
        }
    summary = pd.concat([summary.sort_values(GROUP_COLUMNS), pd.DataFrame([total])])
    columns = [
        *GROUP_COLUMNS,
        "expected_count",
        "tp",
        "fp",
        "fn",
        "recall",
        "precision",
        "f1",
        "latency_median_ms",
        "latency_mean_ms",
    ]
    if include_amplitude:
        columns += ["amplitude_error_mean_deg", "amplitude_error_sd_deg"]
    return _add_rates(summary)[columns].reset_index(drop=True)


# calculate precision, recall and F1 while preserving undefined zero denominators
def _add_rates(summary: pd.DataFrame) -> pd.DataFrame:
    precision_denominator = summary["tp"] + summary["fp"]
    recall_denominator = summary["tp"] + summary["fn"]
    summary["precision"] = summary["tp"] / precision_denominator.where(
        precision_denominator > 0
    )
    summary["recall"] = summary["tp"] / recall_denominator.where(recall_denominator > 0)
    f1_denominator = summary["precision"] + summary["recall"]
    summary["f1"] = (
        2.0 * summary["precision"] * summary["recall"]
    ) / f1_denominator.where(f1_denominator > 0)
    return summary


# aggregate a per-phase evaluation into a single row per participant
def summarize_evaluation_by_participant(evaluation: pd.DataFrame) -> pd.DataFrame:
    summary = (
        evaluation[evaluation["participant"] != "TOTAL"]
        .groupby("participant", as_index=False)[["expected_count", "tp", "fp", "fn"]]
        .sum()
    )
    return _add_rates(summary)[
        ["participant", "expected_count", "tp", "fp", "fn", "recall", "precision", "f1"]
    ]
