"""Serial, user-directed experiment suites with immutable run fingerprints."""
from copy import deepcopy
from datetime import datetime
from zoneinfo import ZoneInfo
from html import escape
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
import uuid

import pandas as pd
import yaml

from .data import DEVELOP, PROJECT, digest, load_market, check_integrity
from .models.regression import DEFAULTS, PACKAGES, check_dependency
from .pipeline import (load_config, merge_config, validate_config, run_experiment, evaluate_runs,
                       new_output, needs_weather, target_origins, prediction_weather_source, weather_context)
from .weather_data import WeatherData, WeatherBlocked, WeatherMissing, PROTOCOL as WEATHER_PROTOCOL, SCENARIOS
from .features.schema import FEATURE_SETS

ARTIFACTS = ("config.yaml", "metadata.json", "predictions.csv", "actuals.csv", "metrics.csv",
             "checks.json", "features.json", "report.html")


def now():
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def atomic_json(path, value):
    temporary = path.with_suffix(path.suffix+".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_yaml(path):
    value = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"配置必须为映射: {path}")
    return value


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"{label} 必须恰好包含 {sorted(keys)}")


def expand_suite(name, selections_path=None):
    if name in {"timesfm1", "timesfm2"}:
        from .timesfm_suite import expand
        return expand(selections_path, name)
    if name not in {"stage1", "stage2", "stage3", "stage4"}:
        raise ValueError("suite 必须为 stage1/stage2/stage3/stage4")
    suite_path = DEVELOP/"configs/suites"/f"{name}.yaml"
    suite = read_yaml(suite_path)
    expected = {"stage1": {"stage", "seed", "threads", "models"},
                "stage2": {"stage", "seed", "threads", "feature_sets"},
                "stage3": {"stage", "seed", "threads", "feature_sets"},
                "stage4": {"stage", "seed", "threads", "strategies", "grids"}}
    exact_keys(suite, expected[name], name)
    if suite["stage"] != name:
        raise ValueError("批次文件与 stage 不匹配")
    baseline_path = DEVELOP/"configs/suites/stage1.yaml"
    baseline = read_yaml(baseline_path)
    exact_keys(baseline, expected["stage1"], "stage1")
    if set(baseline["models"]) != set(DEFAULTS):
        raise ValueError("第一批必须包含六种指定模型")
    paths = [suite_path, baseline_path, DEVELOP/"configs/default.yaml"]
    selection = {"stage2_models": [], "stage3_models": [], "stage4_combinations": []}
    if name != "stage1":
        path = Path(selections_path) if selections_path else DEVELOP/"configs/selections.yaml"
        if not path.is_absolute():
            path = DEVELOP/path
        selection = read_yaml(path)
        if "stage3_combinations" in selection:
            raise ValueError("选择配置需要迁移：将 stage3_combinations 改名为 stage4_combinations，并新增 stage3_models；原参数搜索现为 stage4")
        exact_keys(selection, {"stage2_models", "stage3_models", "stage4_combinations"}, "候选选择配置")
        paths.append(path)
    base, experiments = load_config(), []

    def add(identifier, model_name, feature_set="F0", strategy="shared", params=None):
        if model_name not in DEFAULTS or feature_set not in FEATURE_SETS:
            raise ValueError("未知模型或特征组合")
        model = {"name": model_name, "strategy": strategy, "seed": suite["seed"],
                 "threads": suite["threads"], "params": deepcopy(baseline["models"][model_name] if params is None else params)}
        config = merge_config(base, {"features": {"groups": FEATURE_SETS[feature_set]}, "model": model})
        validate_config(config)
        experiments.append({"id": identifier, "model": model_name, "feature_set": feature_set,
                            "strategy": strategy, "config": config})

    if name == "stage1":
        for model in baseline["models"]:
            add(model, model)
    elif name == "stage2":
        selected = selection["stage2_models"]
        if (not isinstance(selected, list) or len(selected) not in (2, 3)
                or any(not isinstance(m, str) for m in selected)
                or len(set(selected)) != len(selected) or not set(selected).issubset(PACKAGES)):
            raise ValueError("请在 selections.yaml 的 stage2_models 填写两个或三个不同候选：ridge/catboost/lightgbm")
        if suite["feature_sets"] != ["F0", "F1", "F2"]:
            raise ValueError("第二批固定比较 F0/F1/F2")
        for model in selected:
            for feature in suite["feature_sets"]:
                add(f"{model}_{feature}", model, feature)
    elif name == "stage3":
        selected = selection["stage3_models"]
        if (not isinstance(selected, list) or not 1 <= len(selected) <= 3
                or any(not isinstance(m, str) for m in selected)
                or len(set(selected)) != len(selected) or not set(selected).issubset(PACKAGES)):
            raise ValueError("请在 selections.yaml 的 stage3_models 填写一至三个不同候选：ridge/catboost/lightgbm")
        if suite["feature_sets"] != ["F3", "F4", "F5"]:
            raise ValueError("第三批固定比较 F3/F4/F5，不重跑 F2")
        for model in selected:
            for feature in suite["feature_sets"]:
                add(f"{model}_{feature}", model, feature)
    else:
        selected = selection["stage4_combinations"]
        if not isinstance(selected, list) or not selected:
            raise ValueError("请在 selections.yaml 的 stage4_combinations 填写模型与特征组合")
        if suite["strategies"] != ["shared", "independent"] or set(suite["grids"]) != set(PACKAGES):
            raise ValueError("第四批必须配置两种结构及三种模型的参数清单")
        for model, grid in suite["grids"].items():
            if not isinstance(grid, list) or len(grid) != 4 or any(not isinstance(p, dict) for p in grid):
                raise ValueError(f"{model} 必须恰好配置四组参数")
        seen = set()
        for item in selected:
            exact_keys(item, {"model", "feature_set"}, "第四批选择")
            model, feature = item["model"], item["feature_set"]
            if model not in PACKAGES or feature not in FEATURE_SETS or (model, feature) in seen:
                raise ValueError("第四批选择未知或重复的模型/特征组合")
            seen.add((model, feature))
            for index, params in enumerate(suite["grids"][model], 1):
                for strategy in suite["strategies"]:
                    add(f"{model}_{feature}_p{index}_{strategy}", model, feature, strategy, params)
    return experiments, selection, list(dict.fromkeys(paths))


def prepare(name, selections_path=None):
    if name in {"timesfm1", "timesfm2"}:
        from .timesfm_suite import prepare as prepare_timesfm
        return prepare_timesfm(selections_path, name)
    experiments, selection, paths = expand_suite(name, selections_path)
    # Reporting is a shared requirement, unlike optional per-model dependencies.
    importlib.import_module("matplotlib")
    base = experiments[0]["config"]
    weather_source = prediction_weather_source(base, "backtest")
    # Forecast suites must not let matched baselines bypass failed coverage checks.
    if name in {"stage2", "stage3"} and weather_source == "forecast":
        for e in experiments:
            e["config"]["weather"]["match_origins"] = True
    market = load_market(base["data"])
    integrity = check_integrity(market, base["data"]["mean_tolerance"])
    weather, weather_error, weather_coverage, forecast_coverage = None, None, None, None
    if any(needs_weather(e["config"]) for e in experiments):
        try:
            weather = WeatherData(base["weather"], load_forecast=weather_source == "forecast")
            if weather_source == "forecast":
                forecast_coverage = weather.check_forecast_coverage(base["backtest"]["test"])
        except WeatherBlocked as exc:
            weather_error = str(exc)
            weather = None
        if weather is not None:
            origins = target_origins(base)
            history_start = max(pd.Timestamp(base["backtest"]["training_target_start"])-pd.Timedelta(days=2), market.daily.index.min())
            legal, training_reasons = set(), {}
            for d in pd.date_range(history_start, origins[-1]):
                try:
                    weather.select(d, source="history")
                    legal.add(d)
                except WeatherMissing as exc:
                    training_reasons[str(d.date())] = str(exc)
            inference_legal, reasons = set(), {}
            for d in origins:
                try:
                    weather.select(d, source=weather_source)
                    inference_legal.add(d)
                except WeatherMissing as exc:
                    reasons[str(d.date())] = str(exc)
            labels = market.labels
            mature = pd.DatetimeIndex(sorted(t for t in legal
                       if labels.reindex(pd.date_range(t+pd.Timedelta(days=2), periods=6)).notna().all()
                       and labels.loc[:t-pd.Timedelta(days=1)].notna().any()))
            mature_counts = [int(mature.searchsorted(d-pd.Timedelta(days=8), side="right")) for d in origins]
            weather_coverage = {"forecast_origins": len(inference_legal),
                                "training_source": "history", "prediction_source": weather_source,
                                "training_origins_min": min(mature_counts), "training_origins_max": max(mature_counts),
                                "excluded_origins": reasons, "excluded_training_origins": training_reasons,
                                "forecast_test_coverage": forecast_coverage}
            if not any(d in inference_legal and count > 0 for d, count in zip(origins, mature_counts)):
                weather_error = f"没有同时满足历史天气成熟训练标签及 {weather_source} 六天目标天气覆盖的回测起点"
                weather = None
    dependencies = {}
    for model in dict.fromkeys(e["model"] for e in experiments):
        try:
            check_dependency(model)
            dependencies[model] = None
        except Exception as exc:
            dependencies[model] = f"依赖不可用: {type(exc).__name__}: {exc}"
    for e in experiments:
        config = e["config"]
        if name in {"stage2", "stage3"} and weather is not None:
            config["weather"]["match_origins"] = True
        error = weather_error if needs_weather(config) and weather is None else dependencies[e["model"]]
        e["preflight_status"] = "blocked" if error else "ready"
        e["preflight_reason"] = error
    versions = {}
    for package in ("numpy", "pandas", "PyYAML", "matplotlib", "scikit-learn", "scipy", "joblib", "threadpoolctl", "catboost", "lightgbm", "tzdata", "pytest"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not_installed"
    source_hashes = {key: value["sha256"] for key, value in market.sources.items()}
    if any(needs_weather(e["config"]) for e in experiments):
        for key in (("history", "forecast") if weather_source == "forecast" else ("history",)):
            path = PROJECT/base["weather"][key]
            try:
                source_hashes["weather_"+key] = digest(path) if path.is_file() else "missing"
            except OSError as exc:
                # A missing optional history table must not prevent unmatched F0.
                source_hashes["weather_"+key] = "unreadable:"+type(exc).__name__
    code_paths = list((DEVELOP/"forecast").rglob("*.py")) + [DEVELOP/"run.py"]
    identity = {"suite": name, "experiments": experiments, "selections": selection, "sources": source_hashes,
                "config_files": {str(p.relative_to(DEVELOP)) if p.is_relative_to(DEVELOP) else str(p): digest(p) for p in paths},
                "code": {str(p.relative_to(DEVELOP)): digest(p) for p in sorted(code_paths)},
                "rules": {p: digest(PROJECT/p) for p in ("DATA_AVAILABILITY.md", "AGENTS.md")},
                "dependencies": versions, "python": platform.python_version(), "executable": sys.executable}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")).hexdigest()
    context_config = next((e["config"] for e in experiments
                           if any(g.startswith("weather_") for g in e["config"]["features"]["groups"])), base)
    inspection = {"suite": name, "experiment_count": len(experiments), "integrity": integrity,
                  **weather_context(context_config, "backtest"),
                  "weather_protocol": WEATHER_PROTOCOL,
                  "weather_status": "ready" if weather is not None else "blocked" if weather_error else "not_requested",
                  "weather_reason": weather_error, "weather_coverage": weather_coverage,
                  "experiments": [{k: e[k] for k in ("id", "model", "feature_set", "strategy", "preflight_status", "preflight_reason")} for e in experiments]}
    return experiments, market, weather, inspection, identity, fingerprint


def completed(entry):
    if entry.get("status") != "passed" or not entry.get("artifact_hashes"):
        return False
    path = Path(entry["output"])
    if not all((path/name).is_file() and digest(path/name) == entry["artifact_hashes"].get(name) for name in ARTIFACTS):
        return False
    return json.loads((path/"metadata.json").read_text(encoding="utf-8"))["status"] == "passed"


def summarize(directory, state):
    rows, paths = [], []
    for entry in state["experiments"]:
        row = {key: entry.get(key) for key in ("id", "model", "feature_set", "strategy", "status", "started_at", "ended_at", "elapsed_seconds", "output", "reason")}
        row["params"] = json.dumps(entry.get("config", {}).get("model", {}).get("params", {}), ensure_ascii=False, sort_keys=True)
        if entry.get("status") in {"passed", "partial"}:
            path = Path(entry["output"])
            paths.append(path)
            checks = json.loads((path/"checks.json").read_text(encoding="utf-8"))
            counts = [r["training_origins"] for r in checks["origins"] if r["status"] == "passed"]
            row.update({"training_origins_min": min(counts) if counts else None,
                        "training_origins_max": max(counts) if counts else None,
                        "failed_origins": sum(r["status"] != "passed" for r in checks["origins"])})
        rows.append(row)
    status_table = pd.DataFrame(rows)
    status_table.to_csv(directory/"experiments.csv", index=False, encoding="utf-8-sig")
    metrics = pd.DataFrame(columns=["run", "sample", "scope", "metric", "value", "count"])
    if paths:
        comparison = directory/("comparison_"+uuid.uuid4().hex[:8])
        evaluate_runs(paths, output_dir=comparison)
        metrics = pd.read_csv(comparison/"metrics.csv", encoding="utf-8-sig")
        lookup = {Path(e["output"]).name: e["id"] for e in state["experiments"] if e.get("output")}
        metrics.insert(0, "experiment", metrics["run"].map(lookup))
        state["comparison"] = str(comparison)
    metrics.to_csv(directory/"metrics.csv", index=False, encoding="utf-8-sig")
    note = "开发调优结果；未自动选模。电价及电网具体发布时间校验未执行。天气状态："+state["preflight"]["weather_status"]
    note += "。"+SCENARIOS[state["preflight"].get("weather_scenario", "none")]
    if state["preflight"].get("weather_reason"):
        note += "。"+state["preflight"]["weather_reason"]+"；不得据此声称天气增益。"
    else:
        note += "。共同样本指标按成功预测交集计算；请同时检查覆盖率和训练样本数量。"
    links = "".join(f'<li>{escape(e["id"])}：<a href="{escape(Path(e["output"]).relative_to(directory).as_posix(), quote=True)}/report.html">实验报告</a></li>'
                    for e in state["experiments"] if e.get("output") and (Path(e["output"])/"report.html").is_file())
    page = f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>批次汇总</title>
<style>body{{font:15px system-ui;margin:30px}}table{{border-collapse:collapse}}td,th{{padding:7px;border:1px solid #ddd}}.scroll{{overflow:auto}}</style>
<h1>{escape(state['suite'])} 批次汇总</h1><p>{escape(note)}</p><ul>{links}</ul>
<h2>实验状态与耗时</h2><div class="scroll">{status_table.to_html(index=False, na_rep='—', escape=True)}</div>
<h2>自身与共同成功样本指标</h2><div class="scroll">{metrics.to_html(index=False, na_rep='—', escape=True)}</div></html>'''
    if state['suite'] == 'timesfm2':
        from .timesfm_suite import window_summary
        page = page.replace('</html>', window_summary(directory, state)+'</html>')
    (directory/"report.html").write_text(page, encoding="utf-8")


def run_batch(name, check_only=False, resume=None, selections_path=None):
    experiments, market, weather, inspection, identity, fingerprint = prepare(name, selections_path)
    if check_only:
        print(json.dumps(inspection, ensure_ascii=False, indent=2))
        return None, 1 if any(e["preflight_status"] == "blocked" for e in experiments) else 0
    if resume:
        directory = Path(resume)
        if not directory.is_absolute():
            directory = DEVELOP/"outputs"/directory
        directory = directory.resolve()
        state = json.loads((directory/"batch.json").read_text(encoding="utf-8"))
        if state["fingerprint"] != fingerprint:
            raise ValueError("续跑版本不一致（配置/代码/数据/规则/依赖），请启动新批次")
    else:
        directory = new_output("batch_"+name).resolve()
        state = {"suite": name, "fingerprint": fingerprint, "identity": identity, "preflight": inspection,
                 "started_at": now(), "status": "running", "experiments": [
                     {"id": e["id"], "model": e["model"], "feature_set": e["feature_set"],
                      "strategy": e["strategy"], "config": e["config"], "status": "pending", "attempts": []} for e in experiments]}
    if [e["id"] for e in state["experiments"]] != [e["id"] for e in experiments]:
        raise ValueError("续跑清单顺序不一致")
    lock = directory/"batch.lock"
    try:
        with lock.open("x", encoding="utf-8") as stream:
            stream.write(f"started_at={now()}\n")
    except FileExistsError as exc:
        raise ValueError(f"批次正在运行或遗留锁文件：{lock}；确认原进程停止后手动删除此锁") from exc
    try:
        state["status"] = "running"
        state.pop("summary_error", None)
        print(f"批次目录：{directory}", flush=True)
        atomic_json(directory/"batch.json", state)
        backend_cache = {}
        for index, (experiment, entry) in enumerate(zip(experiments, state["experiments"]), 1):
            print(f"[{index}/{len(experiments)}] {entry['id']}", flush=True)
            if completed(entry):
                print("  已完成且产物校验一致，跳过", flush=True)
                continue
            if experiment["preflight_status"] == "blocked":
                entry.update(status="blocked", reason=experiment["preflight_reason"], ended_at=now())
                print(f"  阻塞：{entry['reason']}", flush=True)
                atomic_json(directory/"batch.json", state)
                continue
            attempt = directory/(entry["id"]+"_"+uuid.uuid4().hex[:8])
            entry.update(status="running", started_at=now(), output=str(attempt), reason=None)
            atomic_json(directory/"batch.json", state)
            start = time.perf_counter()
            interrupted = False
            try:
                def progress(current, total, record):
                    print(f"  起点 {current}/{total} {record['predict_time']} {record['status']}", flush=True)
                run_experiment(experiment["config"], "backtest", output_dir=attempt, market=market,
                               weather_data=weather, progress=progress, **({"backend_cache": backend_cache} if name in {"timesfm1", "timesfm2"} else {}))
                meta = json.loads((attempt/"metadata.json").read_text(encoding="utf-8"))
                entry["status"] = meta["status"]
                entry["artifact_hashes"] = {file: digest(attempt/file) for file in ARTIFACTS}
                if meta["status"] == "partial":
                    entry["reason"] = f"{meta['failed_origins']} 个预测起点失败；详见 checks.json"
            except KeyboardInterrupt:
                entry.update(status="interrupted", reason="用户中断；续跑将重新执行本实验")
                interrupted = True
            except Exception as exc:
                entry.update(status="failed", reason=f"{type(exc).__name__}: {exc}")
            finally:
                entry.update(ended_at=now(), elapsed_seconds=round(time.perf_counter()-start, 3))
                entry["attempts"].append({k: entry.get(k) for k in ("output", "status", "started_at", "ended_at", "elapsed_seconds", "reason")})
                atomic_json(directory/"batch.json", state)
                print(f"  {entry['status']}，耗时 {entry['elapsed_seconds']} 秒"+
                      (f"；{entry['reason']}" if entry.get("reason") else ""), flush=True)
            if interrupted:
                state["status"] = "interrupted"
                break
        if state["status"] != "interrupted":
            state["status"] = "passed" if all(e["status"] == "passed" for e in state["experiments"]) else "partial"
        state["ended_at"] = now()
        try:
            summarize(directory, state)
        except Exception as exc:
            state.update(status="partial", summary_error=f"{type(exc).__name__}: {exc}")
        atomic_json(directory/"batch.json", state)
        return directory, 0 if state["status"] == "passed" else 1
    finally:
        lock.unlink(missing_ok=True)
