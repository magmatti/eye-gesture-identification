from __future__ import annotations

import numpy as np
import pandas as pd


BLINK_COLUMNS = ["LeftBlinkWeight", "RightBlinkWeight"]


def has_blink_columns(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in BLINK_COLUMNS)


# checks if df has blink columns if not returns NaN
# averages LeftBlinkWeight and RightBlinkWeight fields and returns it 
def add_blink_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if not has_blink_columns(out):
        out["blink_avg"] = np.nan
        return out

    left = pd.to_numeric(out["LeftBlinkWeight"], errors="coerce")
    right = pd.to_numeric(out["RightBlinkWeight"], errors="coerce")
    out["blink_avg"] = (left + right) / 2.0
    return out
