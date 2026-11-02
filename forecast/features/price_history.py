"""Continuous daily realtime prices, selected only from the legal information set."""
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .daily import MissingHistory


@dataclass(frozen=True)
class PriceHistory:
    origin: pd.Timestamp
    dates: pd.DatetimeIndex
    values: np.ndarray

    def copy(self):
        return PriceHistory(self.origin, self.dates.copy(), self.values.copy())


def build(info, context_length=512):
    dates = pd.date_range(info.origin-pd.Timedelta(days=context_length), periods=context_length)
    values = info.daily["实时电价"].reindex(dates).to_numpy(dtype=float)
    invalid = ~np.isfinite(values)
    if invalid.any():
        missing = ", ".join(dates[invalid].strftime("%Y-%m-%d"))
        raise MissingHistory(f"TimesFM 需要连续 {context_length} 天实时日均价，缺失或非有限: {missing}")
    assert dates[-1] == info.origin-pd.Timedelta(days=1)
    policy = info.checks["daily"]["实时电价"]
    evidence = {"predict_time": str(info.origin.date()), "field": "实时电价", "source": "daily",
                "context_length": context_length, "start_date": str(dates[0].date()),
                "end_date": str(dates[-1].date()), "missing_policy": "fail_without_imputation",
                "records": [{"business_date": str(d.date()), "value": float(v)} for d, v in zip(dates, values)],
                "basis": policy["basis"], "upper_bound": policy["upper_bound"],
                "availability_policy": info.checks["availability_policy"], "publication_time_check": "not_run"}
    return PriceHistory(info.origin, dates, values), evidence
