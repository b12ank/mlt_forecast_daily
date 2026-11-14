"""Shared origin validation and weather preflight for operational forecasts."""
import pandas as pd

from .availability import day
from .weather_data import WeatherMissing


def normalize_origins(origins):
    origins = [day(d) for d in origins]
    if not origins or origins != list(pd.date_range(origins[0], origins[-1])):
        raise ValueError("预测起点必须为非空、连续且升序的日期")
    return origins


def choose_weather(weather, origins, allow_history, first_day=2):
    sources, details = {}, []
    for origin in origins:
        source, missing = "forecast", []
        if weather is not None:
            dates = pd.date_range(origin + pd.Timedelta(days=first_day), origin + pd.Timedelta(days=7))
            missing = dates.difference(weather.tables["forecast"].index).strftime("%Y-%m-%d").tolist()
            if missing and allow_history:
                source = "history"
            try:
                weather.select_range(dates[0], dates[-1], source=source)
            except WeatherMissing as exc:
                raise ValueError(f"预测起点 {origin.date()}: {exc}") from exc
        sources[origin] = source
        details.append({"forecast_date": str(origin.date()), "weather_source": source if weather else "none",
                        "weather_note": ("理想信息条件下的历史补算" if source == "history" else
                                         "历史天气训练、预报天气推理；不代表复原各历史起点预报信息集") if weather else "模型未使用天气输入",
                        "weather_required_start": str((origin + pd.Timedelta(days=first_day)).date()) if weather else None,
                        "weather_required_end": str((origin + pd.Timedelta(days=7)).date()) if weather else None,
                        "missing_forecast_dates": missing,
                        "fallback_reason": "预报缺少必需日期，整个推理天气区间使用历史天气" if missing and allow_history else None})
    return sources, details


def preflight_weather(config, weather, origins, allow_history=False):
    is_xreg = config["model"]["name"] == "timesfm_xreg"
    sources, details = choose_weather(weather, origins, allow_history, first_day=0 if is_xreg else 2)
    if weather is not None and is_xreg:
        context = config["model"]["params"].get("context_length", 512)
        for origin in origins:
            weather.select_range(origin-pd.Timedelta(days=context), origin-pd.Timedelta(days=1), "history")
    return sources, details
