from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .blink_signal import LEFT_BLINK_COLUMN, RIGHT_BLINK_COLUMN
from .detection_config import DetectionConfig

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
