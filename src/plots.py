from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from .analyze import run_analysis
from .blink_signal import LEFT_BLINK_COLUMN, RIGHT_BLINK_COLUMN
from .detection_config import DetectionConfig, ScenarioConfig
from .reports import build_blink_evaluation, build_saccade_evaluation

SACCADE_DIRECTION_COLORS = {
    "left": "tab:blue",
    "right": "tab:orange",
    "up": "tab:green",
    "down": "tab:red",
    "unknown": "tab:gray",
}
EVENT_COLORS = {
    "fixation": "tab:green",
    "saccade": "tab:red",
    "blink": "tab:purple",
}


# plot raw and smoothed gaze speed on shared axes
def plot_gaze_speed(
    df: pd.DataFrame,
    cfg: DetectionConfig,
    show_velocity_threshold: bool = True,
):
    fig, ax = plt.subplots(figsize=(12, 4))
    _plot_series(ax, df, "gaze_speed_deg_s", "Raw gaze speed", alpha=0.45)
    _plot_series(ax, df, "gaze_speed_smooth_deg_s", "Smoothed gaze speed", linewidth=2)
    if show_velocity_threshold:
        ax.axhline(
            cfg.velocity_threshold_deg_s,
            linestyle="--",
            label="Velocity threshold",
        )
    ax.set_title(_title(df, "Gaze speed"))
    _format_gaze_speed_axis(ax)
    return fig, ax


# plot eye blink weights, detection threshold and optional metronome markers
def plot_blink_signal(
    df: pd.DataFrame,
    cfg: DetectionConfig,
    expected_times_s: pd.Series | None = None,
):
    fig, ax = plt.subplots(figsize=(12, 4))
    if LEFT_BLINK_COLUMN in df.columns:
        ax.plot(
            df["Time_s"], df[LEFT_BLINK_COLUMN], label="Left blink weight", alpha=0.5
        )
    if RIGHT_BLINK_COLUMN in df.columns:
        ax.plot(
            df["Time_s"],
            df[RIGHT_BLINK_COLUMN],
            label="Right blink weight",
            alpha=0.5,
        )
    _plot_series(ax, df, "blink_avg", "Blink average", linewidth=2)
    ax.axhline(cfg.blink_threshold, linestyle="--", label="Blink threshold")
    if expected_times_s is not None:
        for index, expected_time_s in enumerate(expected_times_s):
            label = "Metronome signal" if index == 0 else None
            ax.axvline(expected_time_s, color="black", alpha=0.45, label=label)
    ax.set_title(_title(df, "Blink signal"))
    ax.set(xlabel="Time [s]", ylabel="Blink weight", ylim=(-0.05, 1.05))
    ax.grid(alpha=0.3)
    _legend_if_needed(ax)
    return fig, ax


# plot the log-scale distribution of smoothed gaze speed over the dataset
def plot_speed_histogram(samples: pd.DataFrame, cfg: DetectionConfig):
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(samples["gaze_speed_smooth_deg_s"].dropna(), bins=100)
    ax.axvline(
        cfg.velocity_threshold_deg_s,
        color="tab:red",
        linestyle="--",
        label="Velocity threshold",
    )
    ax.set(
        title="Smoothed gaze speed distribution",
        xlabel="Speed [deg/s]",
        ylabel="Sample count",
        yscale="log",
    )
    ax.grid(alpha=0.3)
    ax.legend()
    return fig, ax


# plot gaze speed with detected-event spans and optional stimulus markers
def plot_detected_events(
    samples: pd.DataFrame,
    events: pd.DataFrame,
    cfg: DetectionConfig,
    expected_times_s: pd.Series | None = None,
    show_detected_events: bool = True,
    show_velocity_threshold: bool = True,
):
    fig, ax = plot_gaze_speed(samples, cfg, show_velocity_threshold)
    if show_detected_events:
        for event in events.itertuples():
            ax.axvspan(
                event.start_time_s,
                event.end_time_s,
                color=EVENT_COLORS[event.gesture],
                alpha=0.18,
                label=f"Detected {event.gesture}",
            )
    if expected_times_s is not None:
        for expected_time_s in expected_times_s:
            ax.axvline(expected_time_s, color="black", linewidth=1, alpha=0.55)
    ax.set_title(_title(samples, "Detected eye gestures"))
    _deduplicate_legend(ax)
    return fig, ax


# plot continuous horizontal and vertical displacements by cardinal direction
def plot_saccade_directions(events: pd.DataFrame, title: str):
    fig, ax = plt.subplots(figsize=(7, 7))
    saccades = events[
        (events["gesture"] == "saccade")
        & events["saccade_delta_horizontal_deg"].notna()
        & events["saccade_delta_vertical_deg"].notna()
    ]
    for direction, color in SACCADE_DIRECTION_COLORS.items():
        points = saccades[saccades["saccade_direction"] == direction]
        if points.empty:
            continue
        ax.scatter(
            points["saccade_delta_horizontal_deg"],
            points["saccade_delta_vertical_deg"],
            label=direction,
            color=color,
            alpha=0.75,
        )
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.axvline(0.0, color="black", linewidth=0.8)
    ax.set(
        title=title,
        xlabel="Horizontal gaze displacement [deg]",
        ylabel="Vertical gaze displacement [deg]",
    )
    ax.grid(alpha=0.3)
    _legend_if_needed(ax)
    return fig, ax


# plot saccade recall and precision across velocity thresholds, per participant
def plot_threshold_sweep(sweep: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for metric, ax in zip(("recall", "precision"), axes, strict=True):
        for participant, rows in sweep.groupby("participant"):
            ax.plot(
                rows["velocity_threshold_deg_s"],
                rows[metric],
                marker="o",
                label=participant,
            )
        ax.set(
            title=f"Saccade {metric} versus velocity threshold",
            xlabel="Velocity threshold [deg/s]",
            ylabel="Score",
            ylim=(0.0, 1.05),
        )
        ax.grid(alpha=0.3)
    axes[0].legend()
    return fig, axes


# show smoothing window effect on saccade detection
def plot_smoothing_window_effect(
    detection_cfg: DetectionConfig,
    scenario_cfg: ScenarioConfig,
    windows: Sequence[int] = (1, 3, 5, 7, 9, 11, 15),
):
    metrics = _prepare_smoothing_window_metrics(detection_cfg, scenario_cfg, windows)
    fig, ax_f1 = plt.subplots(figsize=(9, 5))
    ax_f1.plot(
        metrics["window"],
        metrics["f1"],
        marker="o",
        color="tab:blue",
        label="Saccade detection F1",
    )
    ax_f1.axvline(
        detection_cfg.smoothing_window,
        color="tab:green",
        linestyle="--",
        linewidth=1.2,
        label="Selected window width",
    )
    ax_f1.set_xlabel("Smoothing window width [samples]")
    ax_f1.set_ylabel("Saccade detection F1", color="tab:blue")
    ax_f1.set_ylim(0, 1)
    ax_speed = ax_f1.twinx()
    ax_speed.plot(
        metrics["window"],
        metrics["mean_peak_speed_deg_s"],
        marker="s",
        color="tab:orange",
        label="Mean peak speed",
    )
    ax_speed.axhline(
        detection_cfg.velocity_threshold_deg_s,
        color="tab:red",
        linestyle=":",
        linewidth=1.2,
        label=f"Threshold {detection_cfg.velocity_threshold_deg_s:g} deg/s",
    )
    ax_speed.set_ylabel("Mean peak speed [deg/s]", color="tab:orange")
    handles, labels = ax_f1.get_legend_handles_labels()
    speed_handles, speed_labels = ax_speed.get_legend_handles_labels()
    ax_f1.legend(handles + speed_handles, labels + speed_labels, loc="upper right")
    ax_f1.set_title("Effect of smoothing window width on saccade detection")
    fig.tight_layout()
    return fig, (ax_f1, ax_speed)


# plot latency distributions in saccade and blink scenarios
def plot_latency_distributions(
    detection_cfg: DetectionConfig,
    scenario_cfg: ScenarioConfig,
    max_window_s: float = 1.5,
    bin_width_ms: float = 50,
):
    if max_window_s <= 0 or bin_width_ms <= 0:
        raise ValueError("Matching window and histogram bin width must be positive.")
    latencies = _prepare_latency_distributions(
        detection_cfg, scenario_cfg, max_window_s
    )
    bins = np.arange(0, max_window_s * 1000 + bin_width_ms, bin_width_ms)
    panels = (
        ("saccade", scenario_cfg.saccade_match_max_latency_ms),
        ("blink", scenario_cfg.blink_match_max_latency_ms),
    )
    figures = {}
    for gesture, window_ms in panels:
        fig, ax = plt.subplots(figsize=(6.3, 3.2))
        ax.hist(latencies[gesture], bins=bins, color="tab:blue")
        ax.axvline(
            window_ms,
            color="tab:green",
            linestyle="--",
            linewidth=1.2,
            label="Selected matching window",
        )
        ax.set_xlim(0, max_window_s * 1000)
        ax.set_xlabel("Latency relative to the expected event [ms]")
        ax.set_ylabel("Number of matched detections")
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(axis="y", alpha=0.3)
        ax.legend(loc="upper right")
        fig.tight_layout()
        figures[gesture] = (fig, ax)
    return figures


# rerun detection across thresholds for selected blink recordings
def plot_blink_threshold_metrics(
    detection_cfg: DetectionConfig,
    scenario_cfg: ScenarioConfig,
    participants: Sequence[str],
    thresholds: Sequence[float] = tuple(value / 100 for value in range(10, 91, 5)),
):
    metrics = _prepare_blink_threshold_metrics(
        detection_cfg, scenario_cfg, thresholds, participants
    )
    fig, ax = plt.subplots(figsize=(9, 5))
    for metric, color, label in (
        ("recall", "tab:blue", "Recall"),
        ("precision", "tab:orange", "Precision"),
        ("f1", "tab:green", "F1"),
    ):
        ax.plot(
            metrics["threshold"], metrics[metric], marker="o", color=color, label=label
        )
    ax.axvline(
        detection_cfg.blink_threshold,
        color="tab:red",
        linestyle="--",
        linewidth=1.2,
        label="Selected threshold",
    )
    ax.set(
        title="Effect of eyelid closure threshold on blink detection",
        xlabel="Eyelid closure threshold",
        ylabel="Metric value",
        xlim=(metrics["threshold"].min() - 0.02, metrics["threshold"].max() + 0.02),
        ylim=(0.0, 1.05),
    )
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


# pool saccade f1 score and average one maximum speed per recording
def _prepare_smoothing_window_metrics(
    detection_cfg: DetectionConfig,
    scenario_cfg: ScenarioConfig,
    windows: Sequence[int],
) -> pd.DataFrame:
    rows = []
    for window in windows:
        cfg = replace(detection_cfg, smoothing_window=window)
        result = run_analysis(cfg, scenario_cfg)
        evaluation = build_saccade_evaluation(result.saccade_matches, result.events)
        selected = evaluation[evaluation["participant"] != "TOTAL"]
        tp, fp, fn = selected[["tp", "fp", "fn"]].sum()
        recall = tp / (tp + fn) if tp + fn else 0.0
        precision = tp / (tp + fp) if tp + fp else 0.0
        peaks = [
            samples["gaze_speed_smooth_deg_s"].max()
            for filename, samples in result.samples.items()
            if filename.startswith("SaccadeData")
        ]
        rows.append(
            {
                "window": window,
                "f1": 2 * recall * precision / (recall + precision)
                if recall + precision
                else 0.0,
                "mean_peak_speed_deg_s": float(np.mean(peaks)),
            }
        )
    return pd.DataFrame(rows)


def _prepare_latency_distributions(
    detection_cfg: DetectionConfig,
    scenario_cfg: ScenarioConfig,
    max_window_s: float,
) -> dict[str, pd.Series]:
    cfg = replace(
        scenario_cfg,
        saccade_match_max_latency_ms=max_window_s * 1000,
        blink_match_max_latency_ms=max_window_s * 1000,
    )
    result = run_analysis(detection_cfg, cfg)
    return {
        gesture: matches.loc[
            (matches["scenario"] != "combined") & matches["detected"], "latency_ms"
        ]
        for gesture, matches in (
            ("saccade", result.saccade_matches),
            ("blink", result.blink_matches),
        )
    }


def _prepare_blink_threshold_metrics(
    detection_cfg: DetectionConfig,
    scenario_cfg: ScenarioConfig,
    thresholds: Sequence[float],
    participants: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        cfg = replace(detection_cfg, blink_threshold=threshold)
        result = run_analysis(cfg, scenario_cfg)
        evaluation = build_blink_evaluation(result.blink_matches, result.events)
        selected = evaluation[
            (evaluation["scenario"] == "blink")
            & evaluation["participant"].isin(participants)
        ]
        tp, fp, fn = selected[["tp", "fp", "fn"]].sum()
        rows.append(
            {
                "threshold": threshold,
                "tp": int(tp),
                "fp": int(fp),
                "fn": int(fn),
                "recall": tp / (tp + fn) if tp + fn else float("nan"),
                "precision": tp / (tp + fp) if tp + fp else float("nan"),
                "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else float("nan"),
            }
        )
    return pd.DataFrame(rows)


# apply shared labels and legend to a gaze-speed axis
def _format_gaze_speed_axis(ax) -> None:
    ax.set(xlabel="Time [s]", ylabel="Speed [deg/s]")
    ax.grid(alpha=0.3)
    _legend_if_needed(ax)


# plot one time series when it contains measured values
def _plot_series(ax, df: pd.DataFrame, column: str, label: str, **kwargs) -> None:
    if df[column].dropna().empty:
        return
    ax.plot(df["Time_s"], df[column], label=label, **kwargs)


# show a legend only when the axis contains labeled artists
def _legend_if_needed(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend()


# remove repeated event labels after drawing multiple spans
def _deduplicate_legend(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    unique = dict(zip(labels, handles, strict=False))
    ax.legend(unique.values(), unique.keys())


# compose a plot title from its description and source filename
def _title(df: pd.DataFrame, prefix: str) -> str:
    return f"{prefix} - {df['source_file'].iloc[0]}"
