"""Small orchestration layer shared by backtests and explicit-origin predictions."""
from copy import deepcopy
import importlib.metadata
import json
from pathlib import Path
import platform
import uuid

import numpy as np
import pandas as pd
import yaml

from .availability import AVAILABILITY_POLICY, day, information_set
from .data import DEVELOP, PROJECT, load_market, check_integrity, digest
from .features import build_features
from .features.schema import validate_groups, input_schema
from .features.daily import MissingHistory
from .models import create_model
from .models.regression import validate_model
from .weather_data import WeatherData, PROTOCOL as WEATHER_PROTOCOL, SCENARIOS
from .evaluation import DATE_COLUMNS, PRED_COLUMNS, TRUE_COLUMNS, arrays, evaluate, write_report


def merge_config(base, override, path=""):
    if not isinstance(override, dict):
        raise ValueError("配置必须为映射")
    result = deepcopy(base)
    for key, value in override.items():
        if key not in base:
            raise ValueError(f"未知配置项: {key}")
        key_path = f"{path}.{key}" if path else key
        if key_path == "model.params":
            if not isinstance(value, dict):
                raise ValueError("model.params 必须为映射")
            result[key] = deepcopy(value)
        elif isinstance(base[key], dict):
            result[key] = merge_config(base[key], value, key_path)
        else:
            if type(value) is not type(base[key]):
                raise ValueError(f"配置类型不匹配: {key}")
            result[key] = value
    return result


def load_config(path=None):
    base = yaml.safe_load((DEVELOP/"configs/default.yaml").read_text(encoding="utf-8"))
    if path:
        resolved = Path(path)
        if not resolved.is_absolute():
            resolved = DEVELOP/resolved
        base = merge_config(base, yaml.safe_load(resolved.read_text(encoding="utf-8")))
    return validate_config(base)


def validate_config(base):
    if base["task"] != {"horizons": [2, 3, 4, 5, 6, 7], "timezone": "Asia/Shanghai"}:
        raise ValueError("首版任务固定为北京时间 D+2～D+7")
    groups = base["features"]["groups"]
    validate_groups(groups)
    validate_model(base["model"])
    if base["model"]["name"] != "reference" and groups == ["daily"]:
        raise ValueError("新增模型必须使用 price_calendar 特征")
    if groups in (["weather_full"], ["weather_full_no_missing"]) and base["model"]["name"] in {"reference", "recent_mean", "weekday_mean"}:
        raise ValueError("纯天气特征必须使用可训练模型")
    if type(base["weather"]["match_origins"]) is not bool:
        raise ValueError("weather.match_origins 必须为布尔值")
    base["weather"].setdefault("weather_source", "history")
    prediction_weather_source(base, "backtest")
    for field in ("history", "forecast"):
        if not isinstance(base["weather"][field], str) or not base["weather"][field].strip():
            raise ValueError(f"weather.{field} 必须为非空路径")
    if "validation" in base["backtest"]:
        raise ValueError("未知配置项: validation")
    if base["evaluation"] != {"protocol": "evaluation_v1", "equal_tolerance": 1e-8}:
        raise ValueError("修改评价定义须实现新的协议版本")
    if base["backtest"]["window"] != "expanding" or base["backtest"]["retrain_days"] != 1:
        raise ValueError("首版仅支持扩展窗口、每日重训")
    target_origins(base)
    b = base["backtest"]
    if not day(b["training_target_start"]) < day(b["test"][0]):
        raise ValueError("训练起始日必须早于回测目标区间")
    if base["data"]["mean_tolerance"] <= 0:
        raise ValueError("均价容差必须大于零")
    day(base["data"]["snapshot_origin"])
    return base


def target_origins(config):
    interval = config["backtest"]["test"]
    if len(interval) != 2:
        raise ValueError("目标区间必须包含开始和结束日期")
    start, end = map(day, interval)
    if (end-start).days < 5:
        raise ValueError("目标区间不足六天")
    return pd.date_range(start-pd.Timedelta(days=2), end-pd.Timedelta(days=7))


def needs_weather(config):
    return config["weather"]["match_origins"] or any(g.startswith("weather_") for g in config["features"]["groups"])


def prediction_weather_source(config, mode):
    source = config["weather"].get("weather_source", "history")
    if not isinstance(source, str) or source not in {"history", "forecast"}:
        raise ValueError("weather.weather_source 必须为 history 或 forecast")
    if mode not in {"backtest", "predict"}:
        raise ValueError(f"未知运行模式: {mode}")
    return "forecast" if mode == "predict" else source


def weather_context(config, mode):
    used = any(g.startswith("weather_") for g in config["features"]["groups"])
    source = prediction_weather_source(config, mode)
    scenario = ("history_train_forecast_inference" if mode == "predict" else
                "history_target_weather_research" if source == "history" else
                "history_train_forecast_backtest") if used else "none"
    return {"weather_protocol": WEATHER_PROTOCOL if needs_weather(config) else "none",
            "weather_scenario": scenario, "weather_note": SCENARIOS[scenario],
            "weather_training_source": "history" if used else "none",
            "weather_prediction_source": source if used else "none",
            "weather_matching_source": source if config["weather"]["match_origins"] else "none",
            "weather_publication_time_check": "not_run", "weather_historical_revision_audit": "not_run"}


def build_at(market, origin, groups, weather_data=None, match_origins=False, weather_source="history"):
    # Integrity checks cover both tables; daily features only receive daily records.
    info = information_set(market.daily, market.hourly.iloc[:0], origin)
    if match_origins:
        if weather_data is None:
            raise ValueError("天气共同样本限制缺少宽表")
        weather_data.select(origin, source=weather_source)
    frame, evidence = build_features(info, groups, weather_data, weather_source)
    if match_origins:
        evidence["sample_basis"] = f"six target dates present in {weather_source} weather wide table; no publication/version audit"
    return frame, evidence


def training_set(market, origin, config, cache=None, weather_data=None):
    """Fit parameters only on mature labels; cached features use their own origins."""
    cache = {} if cache is None else cache
    start = max(day(config["backtest"]["training_target_start"])-pd.Timedelta(days=2), market.daily.index.min())
    end = day(origin)-pd.Timedelta(days=8)
    chunks, targets = [], []
    # Labels use the same D-1 business boundary as realtime-price features.
    known_labels = information_set(market.daily, market.hourly.iloc[:0], origin).daily["实时电价"]
    for d in pd.date_range(start, end):
        dates = pd.date_range(d+pd.Timedelta(days=2), d+pd.Timedelta(days=7))
        label = known_labels.reindex(dates)
        if label.isna().any():
            continue
        history_version = weather_data.sources["weather_history"]["sha256"] if weather_data is not None else None
        key = (d, tuple(config["features"]["groups"]), config["weather"]["match_origins"], "history", history_version)
        if key not in cache:
            try:
                cache[key] = build_at(market, d, config["features"]["groups"], weather_data,
                                      config["weather"]["match_origins"], weather_source="history")
            except MissingHistory:
                cache[key] = None
        if cache[key] is not None:
            chunks.append(cache[key][0])
            targets.extend(label.to_numpy(dtype=float))
    X = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    y = pd.Series(targets, dtype=float, name="label")
    if not X.empty:
        assert (X["target_date"] <= day(origin)-pd.Timedelta(days=1)).all()
        if "price_calendar" in config["features"]["groups"] or config["features"]["groups"] == ["daily"]:
            assert (X["source_date"] < X["predict_time"]).all()
        else:
            assert "source_date" not in X
    return X, y


def predict_origins(market, origins, config, model_factory=None, weather_data=None, progress=None, weather_source="history"):
    rows, records, provenance, cache = [], [], [], {}
    factory = model_factory or (lambda: create_model(config["model"], config["features"]["groups"]))
    for index, origin in enumerate(origins, 1):
        d = day(origin)
        row = {"predict_time": str(d.date()), "D+2": str((d+pd.Timedelta(days=2)).date()),
               "D+7": str((d+pd.Timedelta(days=7)).date())}
        row.update({f"pred_D+{h}": np.nan for h in range(2, 8)})
        try:
            X, evidence = build_at(market, d, config["features"]["groups"], weather_data,
                                   config["weather"]["match_origins"], weather_source)
            model = factory()
            if model.requires_training:
                train_X, train_y = training_set(market, d, config, cache, weather_data)
                if train_X.empty:
                    raise MissingHistory("无六天标签已成熟且信息集可用的训练样本")
            else:
                train_X, train_y = pd.DataFrame(), pd.Series(dtype=float)
            model.fit(train_X.copy(), train_y.copy())
            pred = np.asarray(model.predict(X.copy()), dtype=float)
            if pred.shape != (6,) or not np.isfinite(pred).all():
                raise ValueError("模型必须按顺序返回六个有限预测值")
            row.update({f"pred_D+{h}": float(pred[h-2]) for h in range(2, 8)})
            provenance.append(evidence)
            records.append({"predict_time": row["predict_time"], "status": "passed", "training_rows": len(train_X),
                            "training_origins": len(train_X)//6})
        except MissingHistory as exc:
            records.append({"predict_time": row["predict_time"], "status": "failed", "reason": str(exc)})
        rows.append(row)
        if progress:
            progress(index, len(origins), records[-1])
    for value in cache.values():
        if value is not None:
            provenance.append({"role": "historical_training_features", **value[1]})
    return pd.DataFrame(rows, columns=PRED_COLUMNS), records, provenance


def actual_table(predictions, labels, blank=False):
    result = predictions[DATE_COLUMNS].copy()
    for h in range(2, 8):
        dates = pd.to_datetime(result["predict_time"]) + pd.Timedelta(days=h)
        result[f"true_D+{h}"] = np.nan if blank else labels.reindex(dates).to_numpy()
    return result[TRUE_COLUMNS]


def new_output(kind):
    name = pd.Timestamp.now(tz="Asia/Shanghai").strftime("%Y%m%d_%H%M%S") + "_" + kind + "_" + uuid.uuid4().hex[:8]
    directory = DEVELOP/"outputs"/name
    directory.mkdir(parents=True, exist_ok=False)
    return directory


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def metadata(config, market, mode, weather_data=None):
    versions = {}
    for package in ("pandas", "numpy", "PyYAML", "matplotlib", "pytest", "scikit-learn", "scipy", "joblib", "threadpoolctl", "catboost", "lightgbm", "tzdata"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    code = {str(p.relative_to(DEVELOP)): digest(p) for p in sorted((DEVELOP/"forecast").rglob("*.py"))}
    code["run.py"] = digest(DEVELOP/"run.py")
    sources = dict(market.sources)
    if weather_data is not None:
        # History backtests do not depend on a forecast file loaded by another caller.
        source = prediction_weather_source(config, mode)
        sources.update({k: v for k, v in weather_data.sources.items() if source == "forecast" or k == "weather_history"})
    return {"status": "running", "mode": mode, "availability_policy": AVAILABILITY_POLICY, "timezone": "Asia/Shanghai",
            "evaluation": config["evaluation"], "sources": sources, "code_sha256": code,
            "rule_sha256": digest(PROJECT/"DATA_AVAILABILITY.md"),
            "python": platform.python_version(), "executable": __import__("sys").executable,
            "dependencies": versions, "model": config["model"], "features": config["features"],
            "input_schema": input_schema(config["features"]["groups"], config["model"]["name"],
                                         config["model"]["strategy"] == "independent"),
            "weather_sample_matching": config["weather"]["match_origins"],
            **weather_context(config, mode),
            "availability_basis": "user-confirmed business boundaries; no invented publication timestamps",
            "historical_revision_audit": "not_run", "publication_time_check": "not_run"}


def run_experiment(config, mode, origin=None, output_dir=None, market=None, weather_data=None, progress=None):
    # Validate the command and date range before creating output artifacts.
    if mode not in {"backtest", "predict"}:
        raise ValueError(f"未知运行模式: {mode}")
    origins = target_origins(config) if mode == "backtest" else [day(origin)]
    validate_config(config)
    weather_source = prediction_weather_source(config, mode)
    directory = new_output(mode) if output_dir is None else Path(output_dir)
    if output_dir is not None:
        directory.mkdir(parents=True, exist_ok=False)
    (directory/"config.yaml").write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding="utf-8")
    meta = {"status": "running", "mode": mode, **weather_context(config, mode)}
    try:
        market = load_market(config["data"]) if market is None else market
        if needs_weather(config) and weather_data is None:
            weather_data = WeatherData(config["weather"], load_forecast=weather_source == "forecast")
        if not needs_weather(config):
            weather_data = None
        meta = metadata(config, market, mode, weather_data)
        forecast_coverage = None
        if mode == "backtest" and weather_data is not None and weather_source == "forecast":
            forecast_coverage = weather_data.check_forecast_coverage(config["backtest"]["test"])
        if mode == "predict" and weather_data is not None:
            weather_data.select(day(origin), source="forecast")
        integrity = check_integrity(market, config["data"]["mean_tolerance"])
        predictions, records, features = predict_origins(market, origins, config, weather_data=weather_data, progress=progress,
                                                       weather_source=weather_source)
        # Future truth can only be joined after the prediction loop has finished.
        actuals = actual_table(predictions, market.labels, blank=mode == "predict")
        metrics = evaluate(predictions, actuals, config["evaluation"]["equal_tolerance"])
        for name, table in [("predictions", predictions), ("actuals", actuals), ("metrics", metrics)]:
            table.to_csv(directory/f"{name}.csv", index=False, encoding="utf-8-sig")
        write_json(directory/"features.json", {"sources": meta["sources"],
            "input_schema": meta["input_schema"],
            **weather_context(config, mode),
            "groups": config["features"]["groups"],
            "extended_definitions": "Each origin record includes exact raw records, transformations, missing policy and availability basis.",
            "definitions": {**({
                "last_price": "latest nonempty daily realtime price in the legal information set",
                "source_date": "business date of last_price; raw date is not shifted",
                "data_age_days": "predict_time minus source_date in calendar days",
                } if "price_calendar" in config["features"]["groups"] or config["features"]["groups"] == ["daily"] else {}),
                "horizon": "known forecast lead, one of 2 through 7",
                "target_date": "predict_time plus horizon days",
                "predict_time": "explicit forecast origin, never inferred from file end"},
            "records": features})
        write_json(directory/"checks.json", {"integrity": integrity, "origins": records,
                   "availability_policy": AVAILABILITY_POLICY, "publication_time_check": "not_run",
                   "automated_tests": "not_run", "hourly_features": "not_run",
                   "weather": {**weather_context(config, mode),
                               "forecast_test_coverage": forecast_coverage,
                               "date_coverage_check": "enforced_per_origin" if weather_data is not None else "not_run"}})
        failures = sum(r["status"] != "passed" for r in records)
        meta.update({"origin_count": len(predictions), "failed_origins": failures,
                     "origin_start": str(pd.Timestamp(origins[0]).date()), "origin_end": str(pd.Timestamp(origins[-1]).date()),
                     "target_start": str((pd.Timestamp(origins[0])+pd.Timedelta(days=2)).date()),
                     "target_end": str((pd.Timestamp(origins[-1])+pd.Timedelta(days=7)).date()),
                     "purpose": "development_tuning" if mode == "backtest" else "forecast"})
        purpose = "开发调优回测" if mode == "backtest" else "未来预测"
        write_report(directory/"report.html", predictions, actuals, metrics, f"六天电价预测：{config['model']['name']}",
                     f"{purpose}；目标区间 {meta['target_start']}～{meta['target_end']}；"
                     f"预测起点 {meta['origin_start']}～{meta['origin_end']}。"
                     f"模型 {config['model']['name']} / {config['model']['strategy']}；特征 {config['features']['groups']}。"
                     "按分字段业务边界取数；电价及电网"
                     "未执行具体发布时间校验；历史修订版本未独立核验。"+meta["weather_note"])
        meta["status"] = "partial" if meta.get("failed_origins", 0) else "passed"
        write_json(directory/"metadata.json", meta)
        return directory
    except Exception as exc:
        meta.update({"status": "failed", "error": str(exc)})
        write_json(directory/"metadata.json", meta)
        write_json(directory/"checks.json", {"status": "failed", "error": str(exc), **weather_context(config, mode)})
        raise


def evaluate_runs(paths, output_dir=None):
    runs = []
    for path in paths:
        path = Path(path)
        if not path.is_absolute():
            path = DEVELOP/"outputs"/path
        meta = json.loads((path/"metadata.json").read_text(encoding="utf-8"))
        predictions = pd.read_csv(path/"predictions.csv", encoding="utf-8-sig")
        actuals = pd.read_csv(path/"actuals.csv", encoding="utf-8-sig")
        arrays(predictions, actuals)
        if meta.get("status") not in {"passed", "partial"}:
            raise ValueError("不能评价未完成的运行")
        runs.append((path, meta, predictions, actuals))
    if not runs:
        raise ValueError("至少提供一个结果目录")
    base = runs[0][1]
    weather_versions = {}
    weather_scenarios = set()
    for _, meta, _, _ in runs:
        if meta.get("availability_policy") != AVAILABILITY_POLICY:
            raise ValueError("实验不可比: 数据边界策略不兼容；旧运行请使用当前流程重新生成")
        for field in ("evaluation", "rule_sha256", "availability_policy", "timezone"):
            if meta[field] != base[field]:
                raise ValueError(f"实验不可比: {field} 不一致")
        if {k: v["sha256"] for k, v in meta["sources"].items() if not k.startswith("weather_")} != {k: v["sha256"] for k, v in base["sources"].items() if not k.startswith("weather_")}:
            raise ValueError("实验不可比: 数据版本不一致")
        for key, value in meta["sources"].items():
            if key.startswith("weather_"):
                if meta.get("weather_protocol") != WEATHER_PROTOCOL:
                    raise ValueError("天气接口或实验口径不同，不能混合旧档案实验与宽表研究实验")
                if key in weather_versions and weather_versions[key] != value["sha256"]:
                    raise ValueError("实验不可比: 天气数据版本不一致")
                weather_versions[key] = value["sha256"]
        scenario = meta.get("weather_scenario", "none")
        if scenario != "none":
            if scenario not in SCENARIOS:
                raise ValueError("未知天气实验口径")
            weather_scenarios.add(scenario)
    if len(weather_scenarios) > 1:
        raise ValueError("不能混合历史天气研究回测与预报天气推理成绩")
    weather_note = SCENARIOS[next(iter(weather_scenarios))] if weather_scenarios else SCENARIOS["none"]
    dates = sorted(set.intersection(*(set(p["predict_time"]) for _, _, p, _ in runs)))
    if not dates:
        raise ValueError("没有共同预测起点")
    selected, shared = [], np.ones((len(dates), 6), dtype=bool)
    truth = None
    for path, meta, pred, actual in runs:
        p = pred.set_index("predict_time").loc[dates].reset_index()
        a = actual.set_index("predict_time").loc[dates].reset_index()
        pv, av = arrays(p, a)
        if truth is not None and not np.array_equal(truth, av, equal_nan=True):
            raise ValueError("共同日期的真实值不一致")
        truth = av
        shared &= np.isfinite(pv) & np.isfinite(av)
        selected.append((path, p, a))
    tables = []
    for (path, meta, p, a), (_, sp, sa) in zip(runs, selected):
        for sample, metrics in [("own", evaluate(p, a)), ("common", evaluate(sp, sa, common_mask=shared))]:
            metrics.insert(0, "sample", sample)
            metrics.insert(0, "run", path.name)
            tables.append(metrics)
    metrics = pd.concat(tables, ignore_index=True)
    directory = new_output("evaluation") if output_dir is None else Path(output_dir)
    if output_dir is not None:
        directory.mkdir(parents=True, exist_ok=False)
    metrics.to_csv(directory/"metrics.csv", index=False, encoding="utf-8-sig")
    write_json(directory/"metadata.json", {"status": "passed", "evaluation": base["evaluation"],
               "availability_policy": AVAILABILITY_POLICY, "publication_time_check": "not_run",
               "runs": [str(p.resolve()) for p, _, _, _ in runs], "common_origins": len(dates),
               "weather_scenarios": sorted(weather_scenarios), "weather_note": weather_note,
               "weather_protocol": WEATHER_PROTOCOL if weather_versions else "none",
               "common_scored_values": int(shared.sum())})
    write_report(directory/"report.html", runs[0][2], runs[0][3], metrics, "实验评价与对比",
                 "曲线展示首个实验；指标同时展示各自样本与共同成功样本。"+weather_note)
    return directory
