from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

FEATURE_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "rate_of_change",
    "acceleration",
    "rolling_mean_5",
    "rolling_std_5",
]


def make_synthetic_series(cfg: dict[str, Any]) -> pd.Series:
    """Random walk with drift for direction-classification demo."""
    data_cfg = cfg.get("data") or {}
    seed = int(data_cfg.get("seed", 42))
    n_samples = int(data_cfg.get("n_samples", 500))

    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.standard_normal(n_samples) + 0.1))


def build_feature_frame(series: pd.Series) -> pd.DataFrame:
    """Lag and rolling features; target is next-step up/down (no leakage)."""
    df = pd.DataFrame(
        {
            "value": series,
            "lag_1": series.shift(1),
            "lag_2": series.shift(2),
            "lag_3": series.shift(3),
            "rate_of_change": series.diff(),
            "acceleration": series.diff().diff(),
            "rolling_mean_5": series.shift(1).rolling(window=5).mean(),
            "rolling_std_5": series.shift(1).rolling(window=5).std(),
        }
    )
    df["target"] = (series.shift(-1) > series).astype(int)
    return df.dropna()


def temporal_train_test_split(
    df: pd.DataFrame, train_ratio: float
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    split_idx = int(len(df) * train_ratio)
    return df.iloc[:split_idx], df.iloc[split_idx:], split_idx
