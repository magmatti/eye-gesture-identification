from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from .detection_config import DetectionConfig


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
    if "LeftBlinkWeight" in df.columns:
        ax.plot(df["Time_s"], df["LeftBlinkWeight"], label="Left blink weight", alpha=0.5)
    if "RightBlinkWeight" in df.columns:
        ax.plot(df["Time_s"], df["RightBlinkWeight"], label="Right blink weight", alpha=0.5)
    _plot_series(ax, df, "blink_avg", "Blink average", linewidth=2)
    ax.axhline(cfg.blink_threshold, linestyle="--", label="Blink threshold")
    ax.set_title(_title(df, "Blink signal"))
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Blink weight")
    ax.set_ylim(-0.05, 1.05)
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
