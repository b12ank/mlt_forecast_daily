"""User-supplied daily wide tables: history research and forecast inference."""
import csv

import numpy as np
import pandas as pd

from .availability import day
from .data import resolve_path, digest
from .features.daily import MissingHistory

PROTOCOL = "wide_daily_weather_research_v1"
FIELDS = tuple("weather_code temperature_2m_max temperature_2m_mean temperature_2m_min "
               "apparent_temperature_max apparent_temperature_mean apparent_temperature_min sunrise sunset "
               "daylight_duration sunshine_duration precipitation_sum rain_sum snowfall_sum precipitation_hours "
               "wind_speed_10m_max wind_gusts_10m_max wind_direction_10m_dominant "
               "shortwave_radiation_sum et0_fao_evapotranspiration".split())
SITES = tuple([f"wind_{n}" for n in range(3, 10)] + [f"solar_{n}" for n in (1, 2, 3, 4, 6, 7)])
COLUMNS = {f"{field}_{site}": (field, site) for site in SITES for field in FIELDS}
AGGREGATES = (("wind_speed", "wind_speed_10m_max", "wind_"),
              ("solar_radiation", "shortwave_radiation_sum", "solar_"),
              ("solar_sunshine", "sunshine_duration", "solar_"),
              ("temperature", "temperature_2m_mean", ""))


def full_column(field, site):
    prefix = "wx_cat__" if field == "weather_code" else "wx_num__"
    return f"{prefix}{site}__{field}" + ("_minute" if field in {"sunrise", "sunset"} else "")


def weather_columns(group):
    if group == "weather_aggregate":
        columns = [f"wx_num__{name}_{stat}" for name, _, _ in AGGREGATES for stat in ("mean", "std", "count")]
    elif group in {"weather_full", "weather_full_no_missing"}:
        columns = [full_column(field, site) for field, site in COLUMNS.values()]
    else:
        raise ValueError(f"未知天气组: {group}")
    if group != "weather_full_no_missing":
        columns += ["wx_num__missing__"+column for column in list(columns)]
    return columns


SCENARIOS = {
    "weather_selection_pending": "尚未完成逐起点天气检查；具体发布时间及历史版本校验均未执行。",
    "history_train_history_inference": "历史天气训练、历史天气推理：理想信息条件下的历史补算，不代表历史预报回测精度；具体发布时间及历史版本校验均未执行。",
    "history_train_mixed_inference": "训练使用历史天气；各起点分别使用预报或整组回退历史天气。历史回退属于理想信息条件下的历史补算，不代表历史预报回测精度；具体发布时间及历史版本校验均未执行。",
    "none": "未使用天气输入。",
    "history_target_weather_research": "理想信息条件下的开发实验：使用历史目标日天气，不是历史预报回测精度，不能据此宣称预报天气的实际增益。",
    "history_train_forecast_inference": "历史天气训练、预报天气推理：使用用户提供的D日起点预报文件；未执行具体发布时间及历史版本校验。",
    "history_train_forecast_backtest": "历史天气训练、预报天气推理：使用用户提供的预报天气宽表进行开发回测；未执行具体发布时间及历史版本校验，不代表已复原各历史起点的预报信息集。",
}


class WeatherBlocked(ValueError):
    """Missing or malformed table; never silently substitute another source."""


class WeatherMissing(MissingHistory):
    """At least one whole target date is missing."""


class WeatherData:
    def __init__(self, config, load_forecast=False):
        self.config = dict(config)
        self.tables, self.raw, self.sources, self.rows = {}, {}, {}, {}
        self.cache = {}
        self._load("history")
        if load_forecast:
            self._load("forecast")

    def _load(self, source):
        path = resolve_path(self.config[source])
        try:
            with path.open(encoding="utf-8-sig", newline="") as stream:
                header = next(csv.reader(stream), [])
            if not header or header[0] != "time" or len(header) != len(set(header)):
                raise WeatherBlocked(f"天气表首列须为 time，列名不可重复: {path}")
            if set(header[1:]) != set(COLUMNS):
                raise WeatherBlocked(f"天气字段不匹配（20类×13站点）: {path}; "
                                     f"缺少={sorted(set(COLUMNS)-set(header[1:]))}; 多余={sorted(set(header[1:])-set(COLUMNS))}")
            if source == "forecast" and header[1:] != list(self.raw["history"].columns):
                raise WeatherBlocked("历史天气与预报天气的列名或顺序不一致")
            raw = pd.read_csv(path, encoding="utf-8-sig", dtype=str)
            if raw.empty:
                raise WeatherBlocked(f"天气表为空: {path}")
            times = raw.pop("time")
            dates = pd.DatetimeIndex(pd.to_datetime(times, format="%Y-%m-%d %H:%M:%S", errors="raise"))
            if dates.hasnans or dates.has_duplicates or not (dates == dates.normalize()).all():
                raise WeatherBlocked(f"天气日期必须为唯一的每日零点: {path}")
            if not (dates.strftime("%Y-%m-%d %H:%M:%S") == times.to_numpy()).all():
                raise WeatherBlocked(f"天气 time 格式必须为 YYYY-MM-DD 00:00:00: {path}")
            raw.index = dates
            raw = raw.replace(r"^\s*$", np.nan, regex=True)
            parsed = {}
            for column in raw:
                field, _ = COLUMNS[column]
                if field in {"sunrise", "sunset"}:
                    stamps = pd.to_datetime(raw[column], format="%d/%m/%Y %H:%M:%S", errors="raise")
                    valid = stamps.notna()
                    if not (stamps.loc[valid].dt.normalize().to_numpy() == dates[valid.to_numpy()].to_numpy()).all():
                        raise WeatherBlocked(f"{column} 的日期与所在天气行不一致")
                    parsed[column] = stamps.dt.hour*60+stamps.dt.minute+stamps.dt.second/60
                else:
                    value = pd.to_numeric(raw[column], errors="raise").astype(float)
                    if np.isinf(value.to_numpy()).any():
                        raise WeatherBlocked(f"{column} 包含无限值")
                    if field == "weather_code" and not value.dropna().eq(np.floor(value.dropna())).all():
                        raise WeatherBlocked(f"{column} 天气代码必须为整数或缺失")
                    parsed[column] = value
            numeric = pd.DataFrame(parsed, index=dates)
            self.rows[source] = {date: row for row, date in enumerate(dates, 2)}
            self.raw[source], self.tables[source] = raw.sort_index(), numeric.sort_index()
            self.sources["weather_"+source] = {"path": str(path), "sha256": digest(path), "rows": len(raw)}
        except WeatherBlocked:
            raise
        except (OSError, ValueError, TypeError) as exc:
            raise WeatherBlocked(f"{source} 天气宽表无法读取: {exc}") from exc

    def check_forecast_coverage(self, interval):
        """Require every test target date before any forecast-weather backtest fit."""
        if "forecast" not in self.tables:
            raise WeatherBlocked("未加载 forecast 天气表；不能回退使用其他来源")
        start, end = map(day, interval)
        dates = self.tables["forecast"].index
        missing = pd.date_range(start, end).difference(dates)
        coverage = {"status": "passed", "test_start": str(start.date()), "test_end": str(end.date()),
                    "forecast_start": str(dates.min().date()), "forecast_end": str(dates.max().date()),
                    "source_file": self.sources["weather_forecast"], "missing_dates": []}
        if len(missing):
            raise WeatherBlocked(
                f"forecast 天气未完整覆盖 backtest.test {start.date()}～{end.date()}；"
                f"预报覆盖范围 {dates.min().date()}～{dates.max().date()}；"
                f"缺失日期: {', '.join(missing.strftime('%Y-%m-%d'))}；"
                "请调整 test 区间或提供完整预报文件，不能用历史天气补齐")
        return coverage

    def select_range(self, start, end, source="history"):
        """Explicit inclusive dates; no source substitution or market-hour shift."""
        if source not in {"history", "forecast"} or source not in self.tables:
            raise WeatherBlocked(f"未加载 {source} 天气表")
        dates = pd.date_range(day(start), day(end))
        missing = dates.difference(self.tables[source].index)
        if len(missing):
            raise WeatherMissing(f"{source} 天气缺少完整日期: {', '.join(missing.strftime('%Y-%m-%d'))}")
        return self.tables[source].loc[dates].copy()

    def select(self, origin, source="history"):
        if source not in {"history", "forecast"}:
            raise ValueError(f"未知天气来源: {source}")
        if source not in self.tables:
            raise WeatherBlocked(f"未加载 {source} 天气表；不能回退使用其他来源")
        d = day(origin)
        dates = pd.date_range(d+pd.Timedelta(days=2), d+pd.Timedelta(days=7))
        missing = dates.difference(self.tables[source].index)
        if len(missing):
            raise WeatherMissing(f"{source} 天气缺少完整目标日期: {', '.join(missing.strftime('%Y-%m-%d'))}")
        return self.tables[source].loc[dates].copy()

    def features(self, origin, group, source="history"):
        d = day(origin)
        key = (source, d, group)
        if key in self.cache:
            frame, evidence = self.cache[key]
            return frame.copy(), evidence
        selected = self.select(d, source)
        values, definitions = {}, {}
        if group == "weather_aggregate":
            for name, field, prefix in AGGREGATES:
                columns = [f"{field}_{site}" for site in SITES if site.startswith(prefix)]
                block = selected[columns]
                count = block.notna().sum(axis=1)
                for stat, value in (("mean", block.mean(axis=1)),
                                    ("std", block.std(axis=1, ddof=0).where(count >= 2)), ("count", count)):
                    output = f"wx_num__{name}_{stat}"
                    values[output] = value
                    definitions[output] = {"source_columns": columns, "operation": "equal_weight_"+stat,
                                           "ddof": 0 if stat == "std" else None}
        elif group in {"weather_full", "weather_full_no_missing"}:
            for column in selected:
                field, site = COLUMNS[column]
                output = full_column(field, site)
                values[output] = selected[column]
                definitions[output] = {"source_columns": [column],
                                       "operation": "file_clock_minute" if field in {"sunrise", "sunset"} else "original_value"}
        else:
            raise ValueError(f"未知天气组: {group}")
        for column in list(values) if group != "weather_full_no_missing" else []:
            flag = "wx_num__missing__"+column
            values[flag] = values[column].isna().astype(int)
            definitions[flag] = {"components": [column], "operation": "is_missing"}
        frame = pd.DataFrame(values, index=selected.index).rename_axis("target_date").reset_index()
        raw_records = []
        for date, row in self.raw[source].loc[selected.index].iterrows():
            raw_records.append({"date": str(date.date()), "csv_row": self.rows[source][date],
                                "values": {column: None if pd.isna(value) else value for column, value in row.items()}})
        evidence = {"predict_time": str(d.date()), "group": group, "source_type": source,
                    "type": "historical_target_weather" if source == "history" else "user_supplied_forecast",
                    "source_file": self.sources["weather_"+source], "protocol": PROTOCOL,
                    "definitions": definitions, "raw_records": raw_records,
                    "basis": "user-confirmed weather research exception, 2026-10-27; forecast backtests allowed 2026-10-29; no publication/version audit claimed",
                    "date_policy": "file weather date equals target date; no market-hour shift or timezone conversion",
                    "missing_policy": "keep cell NaN; missing whole dates fail; never cross-source fill",
                    "weather_missing_indicators": group != "weather_full_no_missing",
                    "weather_publication_time_check": "not_run", "weather_historical_revision_audit": "not_run"}
        self.cache[key] = (frame, evidence)
        return frame.copy(), evidence
