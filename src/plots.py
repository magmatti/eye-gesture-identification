from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .blink_signal import LEFT_BLINK_COLUMN, RIGHT_BLINK_COLUMN
from .detection_config import DetectionConfig
from .saccade_direction import SACCADE_DIRECTIONS


SACCADE_DIRECTION_COLORS = dict(
    zip(SACCADE_DIRECTIONS, ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:gray"])
)


def plot_gaze_speed(df: pd.DataFrame, cfg: DetectionConfig):
    fig, ax = plt.subplots(figsize=(12, 4))
    _plot_series(ax, df, "gaze_speed_deg_s", "Gaze speed", linewidth=1.5)
    _add_gaze_thresholds(ax, cfg)
    ax.set_title(_title(df, "Gaze speed"))
    _format_gaze_speed_axis(ax)
    
    return fig, ax


def plot_smoothed_gaze_speed(df: pd.DataFrame, cfg: DetectionConfig):
    fig, ax = plt.subplots(figsize=(12, 4))
    _plot_series(
        ax,
        df,
        "gaze_speed_smooth_deg_s",
        "Smoothed gaze speed",
        linewidth=2,
    )
    _add_gaze_thresholds(ax, cfg)
    ax.set_title(_title(df, "Smoothed gaze speed"))
    _format_gaze_speed_axis(ax)

    return fig, ax


def plot_blink_signal(df: pd.DataFrame, cfg: DetectionConfig):
    fig, ax = plt.subplots(figsize=(12, 4))
    if LEFT_BLINK_COLUMN in df.columns:
        ax.plot(
            df["Time_s"],
            df[LEFT_BLINK_COLUMN],
            label="Left blink weight",
            alpha=0.5,
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
    ax.set_title(_title(df, "Blink signal"))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Blink weight")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    _legend_if_needed(ax)

    return fig, ax


def plot_saccade_directions(events: pd.DataFrame):
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
    ax.set_title("Head-relative saccade directions")
    ax.set_xlabel("Horizontal gaze displacement [deg]")
    ax.set_ylabel("Vertical gaze displacement [deg]")
    ax.grid(alpha=0.3)
    _legend_if_needed(ax)

    return fig, ax


def _add_gaze_thresholds(ax, cfg: DetectionConfig) -> None:
    ax.axhline(
        cfg.fixation_speed_threshold_deg_s,
        linestyle="--",
        label="Fixation threshold",
    )
    ax.axhline(
        cfg.saccade_speed_threshold_deg_s,
        linestyle="--",
        label="Saccade threshold",
    )


def _format_gaze_speed_axis(ax) -> None:
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Speed [deg/s]")
    ax.grid(alpha=0.3)
    _legend_if_needed(ax)


def _plot_series(ax, df: pd.DataFrame, column: str, label: str, **kwargs) -> None:
    if column not in df.columns or df[column].dropna().empty:
        return
    ax.plot(df["Time_s"], df[column], label=label, **kwargs)


def _legend_if_needed(ax) -> None:
    handles, labels = ax.get_legend_handles_labels()
    if handles and labels:
        ax.legend()


def _title(df: pd.DataFrame, prefix: str) -> str:
    if "source_file" in df.columns and len(df):
        return f"{prefix} - {df['source_file'].iloc[0]}"
    
    return prefix
