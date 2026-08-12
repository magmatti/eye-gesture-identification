from __future__ import annotations

import numpy as np
import pandas as pd

LEFT_BLINK_COLUMN = "LeftBlinkWeight"
RIGHT_BLINK_COLUMN = "RightBlinkWeight"
BLINK_COLUMNS = [LEFT_BLINK_COLUMN, RIGHT_BLINK_COLUMN]


def has_blink_columns(df: pd.DataFrame) -> bool:
    return all(column in df.columns for column in BLINK_COLUMNS)


# checks if df has blink columns if not returns NaN
# averages LeftBlinkWeight and RightBlinkWeight fields and returns it
def add_blink_signal(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if not has_blink_columns(out):
        out["blink_avg"] = np.nan
        return out

    out["blink_avg"] = (out[LEFT_BLINK_COLUMN] + out[RIGHT_BLINK_COLUMN]) / 2.0

    return out
