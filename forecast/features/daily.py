"""Minimal features; future labels are not accepted by this API."""
import pandas as pd


class MissingHistory(ValueError):
    pass


def build(info):
    known = info.daily["实时电价"].dropna()
    if known.empty:
        raise MissingHistory("业务边界内没有有效实时日均价")
    source_day = known.index[-1]
    value = float(known.iloc[-1])
    rows = [{"predict_time": info.origin, "horizon": h,
             "target_date": info.origin + pd.Timedelta(days=h),
             "last_price": value, "source_date": source_day,
             "data_age_days": (info.origin-source_day).days} for h in range(2, 8)]
    evidence = {"predict_time": str(info.origin.date()), "field": "实时电价",
                "source": "daily", "source_date": str(source_day.date()),
                "source_value": value, "data_age_days": (info.origin-source_day).days,
                "basis": info.checks["daily"]["实时电价"]["basis"],
                "upper_bound": info.checks["daily"]["实时电价"]["upper_bound"],
                "availability_policy": info.checks["availability_policy"],
                "publication_time_check": info.checks["publication_time_check"]}
    return pd.DataFrame(rows), evidence
