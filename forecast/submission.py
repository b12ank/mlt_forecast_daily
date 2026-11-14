"""In-memory daily forecasts and atomic MySQL upserts; no result artifacts."""
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP

import pandas as pd

from .availability import AVAILABILITY_POLICY, day
from .data import load_market, check_integrity
from .configuration import submit_dates, apply_submit_settings
from .ingestion import database_settings
from .pipeline import needs_weather, predict_origins
from .snapshot import guarded_run
from .weather_data import WeatherData
from .inference import choose_weather, preflight_weather, normalize_origins


SQL = """INSERT INTO `epf_mlt_forecast_daily_result`
(`forecast_date`, `target_date`, `forecast_value`, `product_type`, `horizon_days`, `model_name`, `remark`)
VALUES (%s, %s, %s, %s, %s, %s, %s)
ON DUPLICATE KEY UPDATE
`forecast_value`=VALUES(`forecast_value`), `horizon_days`=VALUES(`horizon_days`),
`remark`=VALUES(`remark`), `update_time`=CURRENT_TIMESTAMP"""
FIELDS = ("forecast_date", "target_date", "forecast_value", "product_type", "horizon_days", "model_name", "remark")
MAX_VALUE = Decimal("9999999999.9999")


def submit_config(config):
    """Compatibility wrapper; shared configuration owns all submit data settings."""
    return apply_submit_settings(config)


def forecast_decimal(value):
    try:
        number = Decimal(str(value))
        if not number.is_finite() or abs(number) > MAX_VALUE:
            raise ValueError("预测值必须为 decimal(14,4) 范围内的有限数值")
        return number.quantize(Decimal("0.0001"), rounding=ROUND_HALF_UP)
    except (InvalidOperation, TypeError):
        raise ValueError("预测值无法转换为 decimal(14,4)") from None


def result_rows(predictions, origins, model_name, details):
    expected = [str(d.date()) for d in origins]
    if predictions["predict_time"].tolist() != expected or len(details) != len(expected):
        raise ValueError("提交结果的预测起点不完整或顺序错误")
    if not isinstance(model_name, str) or not 1 <= len(model_name) <= 50:
        raise ValueError("model_name 必须为1～50字符")
    rows = []
    for prediction, origin, detail in zip(predictions.to_dict("records"), origins, details):
        source = detail["weather_source"]
        if source not in {"history", "forecast", "none"}:
            raise ValueError(f"未知提交天气来源: {source}")
        remark = f"weather_source={source}"
        for horizon in range(2, 8):
            rows.append(dict(forecast_date=origin.date(), target_date=(origin + pd.Timedelta(days=horizon)).date(),
                             forecast_value=forecast_decimal(prediction[f"pred_D+{horizon}"]),
                             product_type="realtime", horizon_days=horizon, model_name=model_name, remark=remark))
    return rows


@guarded_run
def prepare_submission(config, origins, allow_history=False, status_callback=None):
    """Hold the snapshot lock throughout loading/training; return only memory objects."""
    if status_callback:
        status_callback("preparing")
    market = load_market(config["data"])
    integrity = check_integrity(market, config["data"]["mean_tolerance"])
    weather = WeatherData(config["weather"], load_forecast=True) if needs_weather(config) else None
    sources, details = preflight_weather(config, weather, origins, allow_history)
    predictions, checks, provenance = predict_origins(market, origins, config, weather_data=weather,
                                                     weather_sources=sources, status_callback=status_callback)
    failures = [f"{r['predict_time']}: {r['reason']}" for r in checks if r["status"] != "passed"]
    if failures:
        raise ValueError("整批取消提交；预测失败: " + "; ".join(failures))
    rows = result_rows(predictions, origins, config["model"]["name"], details)
    manifest = dict(market.sources)
    if weather is not None:
        manifest.update(weather.sources)
    return {"origin_count": len(origins), "row_count": len(rows), "model": config["model"],
            "sources": manifest, "origins": details, "checks": checks, "integrity": integrity,
            "availability_policy": AVAILABILITY_POLICY, "publication_time_check": "not_run",
            "historical_revision_audit": "not_run", "rows": rows, "provenance": provenance}


def upsert_results(rows, env_path=None, connect=None):
    """All six-day sets in one transaction, including repeated identical forecasts."""
    if not rows:
        raise ValueError("无可提交结果")
    params = [tuple(row[field] for field in FIELDS) for row in rows]
    settings = database_settings(env_path)
    if connect is None:
        import pymysql
        connect = pymysql.connect
    connection = None
    try:
        connection = connect(**settings, charset="utf8mb4", autocommit=False,
                             connect_timeout=15, read_timeout=120, write_timeout=30)
        with connection.cursor() as cursor:
            cursor.executemany(SQL, params)
        connection.commit()
    except BaseException as exc:
        if connection is not None:
            try:
                connection.rollback()
            except Exception:
                pass
        if not isinstance(exc, Exception):
            raise
        raise ValueError("数据库提交失败，请检查连接、写入权限及表结构；可按相同四列唯一键重试（凭据不输出）") from None
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass


def run_submit(config, origins, allow_history=False, dry_run=False, status_callback=None):
    origins = normalize_origins(origins)
    result = prepare_submission(submit_config(config), origins, allow_history, status_callback=status_callback)
    if not dry_run:
        if status_callback:
            status_callback("submitting", row_count=result["row_count"])
        upsert_results(result["rows"])
    result.update(status="dry_run" if dry_run else "submitted", database="not_connected" if dry_run else "committed")
    return result
