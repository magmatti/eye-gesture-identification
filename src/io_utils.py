from __future__ import annotations

from pathlib import Path

import pandas as pd


def collect_csv_files(data_dir: Path) -> list[Path]:
    data_dir = Path(data_dir)

    return sorted(data_dir.rglob("*.csv"))


def load_csv(path: Path) -> pd.DataFrame:
    path = Path(path)
    df = pd.read_csv(path)
    df["source_file"] = path.name

    return df


# normalizes time (deletes null values), makes sure always starts at 0 seconds
# converts time from ms to seconds
def normalize_time(df: pd.DataFrame) -> pd.DataFrame:
    if "Time_ms" not in df.columns:
        raise ValueError("CSV file does not contain the required Time_ms column.")

    out = df.copy()
    out["Time_ms"] = pd.to_numeric(out["Time_ms"], errors="coerce")
    out = out.dropna(subset=["Time_ms"]).reset_index(drop=True)

    if out.empty:
        out["Time_s"] = pd.Series(dtype=float)
        return out

    start_ms = float(out["Time_ms"].iloc[0])
    out["Time_s"] = (out["Time_ms"].astype(float) - start_ms) / 1000.0

    return out


# used for labeling phases in combined gestures .csv files
# if .csv contains recording of only one gesture it just add recording
def split_by_phase(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if "Phase" in df.columns:
        return {
            str(phase): part.copy().reset_index(drop=True)
            for phase, part in df.groupby("Phase", sort=False, dropna=False)
        }

    return {"recording": df.copy().reset_index(drop=True)}
