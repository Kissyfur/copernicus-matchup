from __future__ import annotations

import re
from typing import Any

import pandas as pd


def dms_to_decimal(value: Any) -> Any:
    """Convert degree-minute-second text to decimal degrees.

    Values that are already numeric, missing, or not recognized are returned unchanged.
    """
    if pd.isna(value) or isinstance(value, (int, float)):
        return value

    text = (
        str(value)
        .replace("º", "°")
        .replace("Âº", "°")
        .replace("Â°", "°")
        .replace("’", "'")
        .replace("‘", "'")
        .replace("′", "'")
        .replace("″", '"')
        .replace("“", '"')
        .replace("”", '"')
        .replace("''", '"')
        .strip()
    )
    match = re.match(r"(\d+(?:\.\d+)?)°\s*(\d+(?:\.\d+)?)'?\s*([\d.]+)?\"?\s*([NSEW])?", text)
    if not match:
        return value

    degrees, minutes, seconds, direction = match.groups()
    decimal = float(degrees) + float(minutes) / 60.0 + float(seconds or 0.0) / 3600.0
    if direction in {"S", "W"}:
        decimal = -decimal
    return decimal
