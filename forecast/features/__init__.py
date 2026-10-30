import pandas as pd

from . import daily, hourly, weather, price_calendar
from .schema import validate_groups, input_schema


def build_features(info, groups, weather_data=None, weather_source="history"):
    if groups != ["daily"] and groups not in (["hourly"], ["weather"]):
        validate_groups(groups)
        if "price_calendar" in groups:
            frame, evidence = price_calendar.build(info)
            records = [evidence]
        else:
            # Keep the same eligibility as F2, without constructing price inputs.
            if not info.daily["实时电价"].notna().any():
                raise daily.MissingHistory("业务边界内没有有效实时日均价；不满足共同样本资格")
            frame = pd.DataFrame({"predict_time": [info.origin]*6, "horizon": list(range(2, 8)),
                                  "target_date": pd.date_range(info.origin+pd.Timedelta(days=2), periods=6)})
            records = []
        weather_group = next((g for g in groups if g.startswith("weather_")), None)
        if weather_group:
            if weather_data is None:
                raise ValueError("天气实验缺少天气宽表")
            extra, source = weather_data.features(info.origin, weather_group, source=weather_source)
            frame = frame.merge(extra, on="target_date", how="left", validate="one_to_one", sort=False)
            records.append(source)
        return frame, {"predict_time": str(info.origin.date()), "groups": records,
                       "constructed_schema": input_schema(groups)}
    if groups != ["daily"]:
        if "hourly" in groups:
            hourly.build(info)
        if "weather" in groups:
            weather.build(info)
        raise ValueError(f"未支持的特征组合: {groups}")
    return daily.build(info)
