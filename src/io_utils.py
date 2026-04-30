from __future__ import annotations

from pathlib import Path
import pandas as pd

SUPPORTED_LABELS = {"blink", "fixation", "saccade"}


def expected_label_from_filename(path: str | Path) -> str:
    name = Path(path).name.lower()
    if "blinkdata" in name:
        return "blink"
    if "fixationdata" in name:
        return "fixation"
    if "saccadedata" in name:
        return "saccade"
    return "unknown"


def collect_csv_files(data_dir: str | Path, include_legacy: bool = False) -> list[Path]:
    """Collect CSV files from the project data directory."""
    data_dir = Path(data_dir)
    files = sorted(data_dir.rglob("*.csv"))
    if not include_legacy:
        files = [p for p in files if "legacy" not in p.parts]
    return [p for p in files if expected_label_from_filename(p) in SUPPORTED_LABELS]


def load_trial_csv(path: str | Path) -> pd.DataFrame:
    """Load one CSV file and attach the source path in attrs.

    The function keeps the raw columns intact because different gestures use different schemas.
    """
    path = Path(path)
    df = pd.read_csv(path)
    df.attrs["source_path"] = str(path)
    df.attrs["expected_label"] = expected_label_from_filename(path)
    return df


def ensure_time_ms(df: pd.DataFrame) -> pd.DataFrame:
    """Guarantee that a numeric Time_ms column exists."""
    if "Time_ms" not in df.columns:
        raise ValueError("CSV file does not contain the required Time_ms column.")
    out = df.copy()
    out["Time_ms"] = pd.to_numeric(out["Time_ms"], errors="coerce")
    out = out.dropna(subset=["Time_ms"]).reset_index(drop=True)
    return out
