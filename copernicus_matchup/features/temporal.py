from __future__ import annotations

import numpy as np


def day_to_circle_x(day, period: int = 365):
    angle = 2 * np.pi * day / period
    return np.cos(angle)


def day_to_circle_y(day, period: int = 365):
    angle = 2 * np.pi * day / period
    return np.sin(angle)
