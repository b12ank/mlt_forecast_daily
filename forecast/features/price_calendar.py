"""Fixed origin-relative price features. Receives only the legal information set."""
import numpy as np
import pandas as pd

from .daily import build as reference_features

LAGS = (1, 2, 3, 7, 14)
WINDOWS = (3, 7, 14)
CATEGORICAL = ("weekday", "horizon")
NUMERIC = ("last_price", "data_age_days", "weekend",
           *(f"price_lag{n}" for n in LAGS),
           *(f"price_{stat}{n}" for n in WINDOWS for stat in ("mean", "std", "count")),
           "price_change1", "price_deviation7", "same_weekday_mean28")


def build(info):
    frame, latest = reference_features(info)
    d = info.origin
    series = info.daily["实时电价"].loc[:d-pd.Timedelta(days=1)]
    dates = pd.date_range(d-pd.Timedelta(days=28), d-pd.Timedelta(days=1))
    history = series.reindex(dates)
    values, definitions = {}, {}
    for n in LAGS:
        name = f"price_lag{n}"
        values[name] = history.loc[d-pd.Timedelta(days=n)]
        definitions[name] = {"operation": "exact_date", "dates": [str((d-pd.Timedelta(days=n)).date())]}
    for n in WINDOWS:
        window = history.iloc[-n:]
        count = int(window.notna().sum())
        for stat, value in (("mean", window.mean() if count else np.nan),
                            ("std", window.std(ddof=0) if count >= 2 else np.nan),
                            ("count", count)):
            name = f"price_{stat}{n}"
            values[name] = value
            definitions[name] = {"operation": stat, "ddof": 0 if stat == "std" else None,
                                 "dates": [str(v.date()) for v in window.index]}
    values["price_change1"] = values["price_lag1"]-values["price_lag2"]
    values["price_deviation7"] = values["price_lag1"]-values["price_mean7"]
    definitions["price_change1"] = {"components": ["price_lag1", "price_lag2"], "operation": "subtract"}
    definitions["price_deviation7"] = {"components": ["price_lag1", "price_mean7"], "operation": "subtract"}
    for key, value in values.items():
        frame[key] = value
    frame["weekday"] = frame["target_date"].dt.dayofweek
    frame["weekend"] = (frame["weekday"] >= 5).astype(int)
    same_days = {}
    means = []
    for target in frame["target_date"]:
        window = history[history.index.dayofweek == target.dayofweek]
        means.append(window.mean() if window.notna().any() else np.nan)
        same_days[str(target.date())] = [str(v.date()) for v in window.index]
    frame["same_weekday_mean28"] = means
    definitions["same_weekday_mean28"] = {"operation": "mean", "target_windows": same_days}
    definitions.update({"last_price": latest, "data_age_days": {"operation": "origin_minus_source_date"},
                        "weekday": {"operation": "target_dayofweek", "basis": "calendar"},
                        "weekend": {"operation": "weekday >= 5", "basis": "calendar"},
                        "horizon": {"operation": "target_minus_origin", "basis": "task"}})
    for key in NUMERIC:
        frame[key+"__missing"] = frame[key].isna().astype(int)
        definitions[key+"__missing"] = {"operation": "is_missing", "components": [key]}
    evidence = {"predict_time": str(d.date()), "group": "price_calendar", "type": "observation",
                "source_field": "实时电价", "latest": latest, "definitions": definitions,
                "raw_records": [{"date": str(t.date()), "value": None if pd.isna(v) else float(v),
                                 "age_days": (d-t).days} for t, v in history.items()],
                "missing_policy": "keep NaN; rolling mean needs >=1, population std >=2; no backfill",
                "upper_bound": str((d-pd.Timedelta(days=1)).date()), "basis": latest["basis"]}
    return frame, evidence
