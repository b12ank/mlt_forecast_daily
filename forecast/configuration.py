"""Common/model/submit composition with backward-compatible runtime dictionaries."""
from copy import deepcopy
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd
import yaml

from .availability import day
from .data import DEVELOP, resolve_path

COMMON_CONFIG = DEVELOP / "configs/common/default.yaml"
REFERENCE_CONFIG = DEVELOP / "configs/experiments/reference.yaml"
SUBMIT_COMMON = DEVELOP / "configs/common/submit.yaml"
MODEL_DIRECTORY = DEVELOP / "configs/model"
MODEL_NAMES = ("catboost", "lightgbm", "timesfm")
MODEL_KEYS = ("name", "strategy", "seed", "threads", "params")
COMMON_KEYS = {
    "data": "market_input_mode daily hourly encoding mean_tolerance snapshot_origin".split(),
    "task": "horizons timezone".split(),
    "weather": "weather_source history forecast match_origins".split(),
    "backtest": "training_target_start test window retrain_days".split(),
    "evaluation": "protocol equal_tolerance".split(),
}
COMMON_TYPES = {
    "data": (str, str, str, str, float, str),
    "task": (list, str),
    "weather": (str, str, str, bool),
    "backtest": (str, list, str, int),
    "evaluation": (str, float),
}


class RequestConfigError(ValueError):
    """Invalid prediction/submission settings or command arguments."""


SubmitConfigError = RequestConfigError


def resolve_config(path):
    resolved = Path(path)
    return (resolved if resolved.is_absolute() else DEVELOP / resolved).resolve()


def read_yaml(path):
    try:
        value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"YAML 格式错误: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"配置必须为映射: {path}")
    return value


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
                raise ValueError(f"配置类型不匹配: {key_path}")
            result[key] = deepcopy(value)
    return result


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} 必须恰好包含 {sorted(keys)}")


def runtime_layers(path=None):
    """Use reference defaults only for default or partial configurations."""
    selected_path = resolve_config(REFERENCE_CONFIG if path is None else path)
    selected = read_yaml(selected_path)
    if "submit" in selected or "predict" in selected:
        raise ValueError("请求配置仅用于对应的 submit 或 predict 接口；回测请指定模型配置")
    model, features = selected.get("model"), selected.get("features")
    complete = (isinstance(model, dict) and set(MODEL_KEYS).issubset(model)
                and isinstance(features, dict) and "groups" in features)
    layers = []
    if not complete and selected_path != REFERENCE_CONFIG.resolve():
        layers.append((REFERENCE_CONFIG.resolve(), read_yaml(REFERENCE_CONFIG)))
    return [*layers, (selected_path, selected)]


def config_files(path=None):
    """Files actually used by the runtime loader, including shared defaults."""
    paths = [COMMON_CONFIG.resolve(), *(p for p, _ in runtime_layers(path))]
    return list(dict.fromkeys(paths))


def load_config(path=None):
    # Local import preserves the long-standing pipeline.load_config API without
    # introducing a configuration -> pipeline -> configuration import cycle.
    from .pipeline import validate_config
    common = read_yaml(COMMON_CONFIG)
    exact_keys(common, COMMON_KEYS, "通用配置")
    for key, fields in COMMON_KEYS.items():
        exact_keys(common[key], fields, f"通用配置 {key}")
        for field, expected in zip(fields, COMMON_TYPES[key]):
            if type(common[key][field]) is not expected:
                raise ValueError(f"配置类型不匹配: {key}.{field}")
    layers = runtime_layers(path)
    defaults = layers[0][1]
    exact_keys(defaults.get("model"), MODEL_KEYS, "模型配置")
    exact_keys(defaults.get("features"), ("groups",), "特征配置")
    for field, expected in {"name": str, "strategy": str, "seed": int, "threads": int, "params": dict}.items():
        if type(defaults["model"][field]) is not expected:
            raise ValueError(f"配置类型不匹配: model.{field}")
    if not isinstance(defaults["features"]["groups"], list):
        raise ValueError("配置类型不匹配: features.groups")
    base = {**common, "model": defaults["model"], "features": defaults["features"]}
    for _, selected in layers:
        base = merge_config(base, selected)
    return validate_config(base)


def apply_live_settings(config):
    """Share daily inputs for predict/submit, independent of checkout name."""
    from .pipeline import validate_config
    overlay = read_yaml(SUBMIT_COMMON)
    exact_keys(overlay, ("data", "weather"), "提交通用配置")
    exact_keys(overlay["data"], ("market_input_mode", "daily", "hourly", "encoding"), "提交数据配置")
    exact_keys(overlay["weather"], ("history", "forecast"), "提交天气路径配置")
    for section, field in (("data", "daily"), ("weather", "history"), ("weather", "forecast")):
        value = overlay[section][field]
        if isinstance(value, str) and value:
            overlay[section][field] = str(resolve_path(value, root=DEVELOP))
    result = merge_config(config, overlay)
    if result["data"]["market_input_mode"] != "daily_only" or result["data"]["hourly"]:
        raise ValueError("提交必须使用 daily_only，且 hourly 为空")
    return validate_config(result)


apply_submit_settings = apply_live_settings


def request_dates(values=None):
    """Normalize CLI or YAML dates; two endpoints explicitly enable range mode."""
    if values is None:
        values = [datetime.now(ZoneInfo("Asia/Shanghai")).date().isoformat()]
    elif isinstance(values, str) or type(values) is date:
        values = [values]
    if not isinstance(values, list) or len(values) not in (1, 2):
        raise RequestConfigError("日期必须为一个日期或包含一至两个日期的列表")
    try:
        if any(not isinstance(v, str) and type(v) is not date for v in values):
            raise ValueError("日期必须为 YYYY-MM-DD，不接受数字、空值或时刻")
        dates = [day(v) for v in values]
        if dates[-1] < dates[0]:
            raise ValueError("结束日期不得早于开始日期")
        day(dates[-1] + pd.Timedelta(days=7))
        return list(pd.date_range(dates[0], dates[-1])), len(values) == 2
    except (ValueError, OverflowError) as exc:
        raise RequestConfigError(str(exc)) from exc


submit_dates = request_dates


@dataclass
class ForecastRequest:
    config: dict
    origins: list
    allow_history: bool
    config_files: list


SubmitRequest = ForecastRequest


def load_request(kind, path=None, dates=None):
    if kind not in {"submit", "predict"}:
        raise ValueError(f"未知请求类型: {kind}")
    if path is not None and dates is not None:
        raise RequestConfigError(f"--config 与 --date 不能同时使用；请在 {kind} 配置中设置日期")
    if path is None:
        origins, allow_history = submit_dates(dates)
        model_path = MODEL_DIRECTORY / "lightgbm.yaml"
        paths = config_files(model_path)
    else:
        resolved = resolve_config(path)
        selected = read_yaml(resolved)
        if "submit" in selected or "predict" in selected:
            try:
                exact_keys(selected, (kind,), f"{kind} 配置")
                settings = selected[kind]
                if not isinstance(settings, dict) or "model" not in settings or set(settings)-{"model", "date"}:
                    raise ValueError(f"{kind} 只接受 model、date，且 model 必须指定")
                if not isinstance(settings["model"], str) or settings["model"] not in MODEL_NAMES:
                    raise ValueError(f"{kind}.model 必须为 catboost、lightgbm 或 timesfm")
                origins, allow_history = submit_dates(settings.get("date"))
            except ValueError as exc:
                raise RequestConfigError(str(exc)) from exc
            model_path = MODEL_DIRECTORY / f"{settings['model']}.yaml"
            paths = [*config_files(model_path), resolved]
        else:
            # Old flat model configurations remain accepted; no date means today.
            model_path = resolved
            origins, allow_history = submit_dates()
            paths = config_files(resolved)
    return ForecastRequest(load_config(model_path), origins, allow_history,
                         list(dict.fromkeys([*paths, SUBMIT_COMMON.resolve()])))


def load_submit_config(path=None, dates=None):
    return load_request("submit", path, dates)


def load_predict_config(path=None, dates=None):
    return load_request("predict", path, dates)
