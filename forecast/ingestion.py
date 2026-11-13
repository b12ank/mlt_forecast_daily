"""Read MySQL snapshots and publish exactly three model-input daily CSVs."""
from decimal import Decimal, InvalidOperation
from pathlib import Path
import os
import uuid

import numpy as np
import pandas as pd

from .data import DEVELOP, MarketData, check_integrity, read_table
from .snapshot import FILES, JOURNAL, dataset_lock, publish, cleanup_temporary
from .weather_data import COLUMNS, FIELDS, SITES, WeatherData

GRID_MAPPING = dict(zip(
    "local_plant tie_line wind solar nuclear self_owned test_unit direct_load bidding_space new_energy".split(),
    "地方电厂总加 联络线受电负荷 风电总加 光伏总加 核电总加 自备机组总加 试验机组总加 直调负荷 竞价空间 新能源总加".split()))
MARKET_MAPPING = {"price_dayahead": "日前电价", "price_realtime": "实时电价"}
for prefix, suffix in (("fcast", "预测值"), ("actual", "实际值")):
    MARKET_MAPPING.update({f"{prefix}_{key}": value + suffix for key, value in GRID_MAPPING.items()})
COORDINATES = dict(zip(SITES, (
    ("38.172889", "118.729113"), ("37.91136", "119.38029"),
    ("37.428042", "119.450849"), ("37.855671", "121.415637"),
    ("37.688373", "121.924003"), ("36.57044", "121.696484"),
    ("36.393549", "121.361539"), ("35.306205", "115.87577"),
    ("35.787843", "116.718906"), ("35.451592", "117.120295"),
    ("35.019488", "117.107963"), ("37.929328", "118.489965"),
    ("36.989386", "119.2992259"))))
WEATHER_META = "site_id site_name energy_type energy_base latitude longitude data_kind model valid_time first_fetched_at last_fetched_at".split()
WEATHER_EXTRA = "showers_sum precipitation_probability_max uv_index_max uv_index_clear_sky_max".split()
MARKET_META = ["id", "data_time", "is_rt_temporary", "create_time", "update_time"]


class EmptyDatabaseResult(ValueError):
    """Safe, locally constructed query failure (contains no driver message)."""


def require(condition, message):
    if not condition:
        raise ValueError(message)


def database_settings(env_path=None):
    from dotenv import dotenv_values
    values = dotenv_values(env_path or DEVELOP / ".env", interpolate=False)
    def value(key, default=None):
        return os.environ[key] if key in os.environ else values.get(key, default)
    settings = {"host": value("DB_HOST"), "database": value("DB"),
                "user": value("DB_USER"), "password": value("DB_PWD")}
    require(all(settings[k] for k in ("host", "database", "user")), "缺少 DB_HOST、DB 或 DB_USER")
    require(settings["password"] is not None, "缺少 DB_PWD（可显式设为空）")
    try:
        settings["port"] = int(value("DB_PORT", "3306"))
    except (ValueError, TypeError):
        raise ValueError("DB_PORT 必须是端口整数") from None
    require(1 <= settings["port"] <= 65535, "DB_PORT 超出端口范围")
    return settings


def fetch_database(env_path=None, connect=None):
    settings = database_settings(env_path)
    if connect is None:
        import pymysql
        connect = pymysql.connect
    connection = None
    try:
        connection = connect(**settings, charset="utf8mb4", autocommit=False,
                             connect_timeout=15, read_timeout=120, write_timeout=30)
        with connection.cursor() as cursor:
            cursor.execute("SET SESSION TRANSACTION ISOLATION LEVEL REPEATABLE READ")
            cursor.execute("START TRANSACTION WITH CONSISTENT SNAPSHOT, READ ONLY")
            def query(table, columns, where="", params=None):
                cursor.execute("SELECT " + ", ".join(f"`{c}`" for c in columns) +
                               f" FROM `{table}`" + where, params)
                rows = cursor.fetchall()
                if not rows:
                    raise EmptyDatabaseResult(f"数据库表 {table} 查询为空")
                return pd.DataFrame(rows, columns=columns)
            market = query("epf_market_data", MARKET_META + list(MARKET_MAPPING), " ORDER BY data_time")
            weather = query("new_energy_base_meteorological_data_daily", WEATHER_META + list(FIELDS) + WEATHER_EXTRA,
                            " WHERE data_kind IN (%s, %s) ORDER BY valid_time, site_id, data_kind, model",
                            ("historical", "forecast"))
        connection.rollback()  # End the read-only snapshot, never write to the DB.
        return market, weather
    except EmptyDatabaseResult:
        raise
    except Exception:
        # Driver messages can embed connection credentials; do not propagate them.
        raise ValueError("数据库读取失败，请检查连接配置、账号权限、网络与表结构（凭据不输出）") from None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def numeric(series, name):
    try:
        result = pd.to_numeric(series, errors="raise").astype(float)
    except (ValueError, TypeError):
        raise ValueError(f"{name} 含非法数值") from None
    require(not np.isinf(result.to_numpy()).any(), f"{name} 含无限值")
    return result


def convert_market(raw):
    required = {"id", "data_time", "is_rt_temporary", *MARKET_MAPPING}
    require(not raw.empty and required <= set(raw), "电价查询为空或缺少必要字段")
    require(raw.id.notna().all() and not raw.id.duplicated().any(), "电价主键缺失或重复")
    times = pd.DatetimeIndex(pd.to_datetime(raw.data_time, errors="raise"))
    require(times.tz is None and not times.hasnans and not times.has_duplicates,
            "电价时间必须为唯一的北京时间（不含时区）")
    require((times == times.floor("h")).all(), "电价时间必须为整点，不能按15分钟数据处理")
    values = pd.DataFrame({target: numeric(raw[source], source).to_numpy()
                           for source, target in MARKET_MAPPING.items()}, index=times)
    temporary = numeric(raw.is_rt_temporary, "is_rt_temporary").fillna(1)
    require(temporary.isin([0, 1]).all(), "is_rt_temporary 必须为0或1")
    values.loc[temporary.to_numpy() == 1, "实时电价"] = np.nan
    business = (times - pd.Timedelta(hours=1)).normalize()
    dates = pd.date_range(business.min(), business.max(), freq="D")
    hours = pd.date_range(dates[0] + pd.Timedelta(hours=1), dates[-1] + pd.Timedelta(days=1), freq="h")
    values = values.reindex(hours)  # Only positions, never values, are filled.
    grouped = values.groupby((hours - pd.Timedelta(hours=1)).normalize())
    counts = grouped.count()
    daily = grouped.mean().where(counts.eq(24))
    daily.index.name = "时刻"
    # Independent 24-slot calculation validates aggregation and missingness.
    blocks = values.to_numpy().reshape(len(dates), 24, len(MARKET_MAPPING))
    expected = blocks.sum(axis=1) / 24
    np.testing.assert_allclose(daily.to_numpy(), expected, rtol=1e-12, atol=1e-9, equal_nan=True)
    counts.index.name = "时刻"
    evidence = raw[[c for c in MARKET_META if c in raw]].copy()
    evidence["raw_csv_row"] = np.arange(len(raw)) + 2
    evidence["business_date"] = business.strftime("%Y-%m-%d")
    evidence["realtime_excluded"] = temporary.to_numpy() == 1
    stats = {"status": "passed", "raw_rows": len(raw), "days": len(daily),
             "added_hour_positions": len(hours) - len(raw),
             "excluded_temporary_or_unknown_rows": int(temporary.eq(1).sum()),
             "incomplete_field_days": int(counts.lt(24).sum().sum()),
             "valid_labels": int(daily["实时电价"].notna().sum()),
             "start": str(dates[0].date()), "end": str(dates[-1].date()),
             "aggregation": "24 nonempty hours per field; business_date=date(data_time-1h)",
             "publication_time_check": "not_run", "historical_revision_audit": "not_run"}
    return daily, counts, evidence, stats


def coordinate(value):
    try:
        number = Decimal(str(value))
        require(number.is_finite(), "天气坐标非有限值")
        rounded = number.quantize(Decimal("0.0000001"))
        require(number == rounded, "天气坐标超过数据库七位小数精度")
        return rounded
    except (InvalidOperation, TypeError):
        raise ValueError("天气坐标非法") from None


def convert_weather(raw, kind):
    require(set(WEATHER_META + list(FIELDS)) <= set(raw), "天气查询缺少必要字段")
    data = raw.loc[raw.data_kind == kind].copy()
    require(not data.empty, f"{kind} 天气查询为空")
    data["raw_csv_row"] = np.flatnonzero((raw.data_kind == kind).to_numpy()) + 2
    required_keys = ["site_id", "model", "valid_time", "first_fetched_at", "last_fetched_at"]
    require(data[required_keys].notna().all().all(), "天气主键或抓取时间缺失")
    require(not data.duplicated(["site_id", "model", "valid_time"]).any(), "天气主键重复")
    times = pd.DatetimeIndex(pd.to_datetime(data.valid_time, errors="raise"))
    require(times.tz is None and not times.hasnans and (times == times.normalize()).all(),
            "天气 valid_time 必须为北京时间每日零点")
    data["valid_time"] = times
    for name in ("first_fetched_at", "last_fetched_at"):
        data[name] = pd.to_datetime(data[name], errors="raise", utc=True)
    mapping = {tuple(map(coordinate, coords)): site for site, coords in COORDINATES.items()}
    data["canonical_site"] = [mapping.get((coordinate(lat), coordinate(lon)))
                              for lat, lon in zip(data.latitude, data.longitude)]
    ignored = int(data.canonical_site.isna().sum())
    data = data.loc[data.canonical_site.notna()]
    require(set(data.canonical_site) == set(SITES), f"{kind} 天气缺少必需站点")
    before = len(data)
    data = data.sort_values(["last_fetched_at", "first_fetched_at", "site_id", "model"],
                            ascending=[False, False, True, True], kind="stable")
    data = data.drop_duplicates(["canonical_site", "valid_time"], keep="first")
    data = data.sort_values(["valid_time", "canonical_site"]).reset_index(drop=True)
    for name in FIELDS:
        if name in {"sunrise", "sunset"}:
            stamps = pd.to_datetime(data[name], errors="raise")
            require(stamps.dt.tz is None, f"{name} 应为源表本地时间")
            valid = stamps.notna()
            require((stamps[valid].dt.normalize() == data.loc[valid, "valid_time"]).all(),
                    f"{name} 与天气日期不一致")
            data[name] = stamps.dt.strftime("%d/%m/%Y %H:%M:%S")
        else:
            data[name] = numeric(data[name], name)
            if name == "weather_code":
                require(data[name].dropna().eq(np.floor(data[name].dropna())).all(), "天气代码必须为整数")
    wide = data.pivot(index="valid_time", columns="canonical_site", values=list(FIELDS)).sort_index()
    wide = wide.reindex(columns=pd.MultiIndex.from_tuples(list(COLUMNS.values())))
    wide.columns = list(COLUMNS)
    wide.index.name = "time"
    evidence = data[WEATHER_META + ["canonical_site", "raw_csv_row"]]
    stats = {"raw_rows": int((raw.data_kind == kind).sum()), "selected_rows": len(data),
             "ignored_coordinates": ignored, "duplicates_removed": before - len(data),
             "days": len(wide), "missing_cells": int(wide.isna().sum().sum()),
             "missing_site_days": len(wide)*len(SITES) - len(data),
             "start": str(wide.index.min().date()), "end": str(wide.index.max().date()),
             "publication_time_check": "not_run", "historical_revision_audit": "not_run"}
    return wide, evidence, stats


def refresh_data(*, data_dir=None, fetcher=None):
    """Refresh three daily files; no reports, raw exports or outputs directories."""
    directory = Path(data_dir or DEVELOP / "data").resolve()
    with dataset_lock(directory):
        temporary = directory / (".load-data-tmp-" + uuid.uuid4().hex)
        stage = temporary / "staged"
        stage.mkdir(parents=True)
        try:
            print("正在读取数据库…", flush=True)
            market, weather = (fetcher or fetch_database)()
            print(f"正在更新 {FILES[0]}…", flush=True)
            daily, _, _, _ = convert_market(market)
            daily.to_csv(stage / FILES[0], encoding="gb18030", date_format="%Y-%m-%d")
            history, forecast = None, None
            for kind, name in zip(("historical", "forecast"), FILES[1:]):
                print(f"正在更新 {name}…", flush=True)
                wide, _, _ = convert_weather(weather, kind)
                wide.to_csv(stage / name, encoding="utf-8-sig", date_format="%Y-%m-%d %H:%M:%S")
                if kind == "historical":
                    history = wide
                else:
                    forecast = wide
            # Validate every exported value before replacing any input file.
            actual, _ = read_table(stage / FILES[0], "gb18030")
            pd.testing.assert_frame_equal(actual, daily, check_names=False, check_freq=False, rtol=1e-12, atol=1e-9)
            check_integrity(MarketData(actual, actual.iloc[:0], market_input_mode="daily_only"))
            WeatherData({"history": str(stage / FILES[1]), "forecast": str(stage / FILES[2])}, load_forecast=True)
            for expected, name in zip((history, forecast), FILES[1:]):
                actual = pd.read_csv(stage / name, encoding="utf-8-sig", index_col="time", float_precision="round_trip")
                actual.index = pd.to_datetime(actual.index)
                pd.testing.assert_frame_equal(actual, expected.infer_objects(), check_dtype=False,
                                              check_freq=False, check_exact=True)
            publish(directory, stage)
        except AssertionError:
            raise ValueError("导出回读或日均聚合校验失败，未发布新数据") from None
        finally:
            # Failed recovery must keep its backups until the next attempt.
            if not (directory / JOURNAL).exists():
                cleanup_temporary(directory, temporary)
        print("数据更新完成。", flush=True)
