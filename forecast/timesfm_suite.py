"""Fixed TimesFM ablation/window suites using the shared batch machinery."""
from copy import deepcopy
import hashlib
import importlib.metadata
import json
import platform
import sys
from pathlib import Path

from .data import DEVELOP, PROJECT, digest, load_market
from .configuration import config_files
from .evaluation import smape_values
from .features.xreg import SETS
from .pipeline import load_config, validate_config, check_backtest, weather_context
from .weather_data import WeatherData


LENGTHS = [64, 128, 256, 512, 768, 1024, 1536]


def expand(selections_path=None, name='timesfm1'):
    from .batch import read_yaml, exact_keys
    if selections_path:
        raise ValueError(f'{name} 不使用树形模型 selections')
    if name not in ('timesfm1', 'timesfm2'):
        raise ValueError('未知 TimesFM suite')
    path = DEVELOP/f'configs/suites/{name}.yaml'
    suite = read_yaml(path)
    exact_keys(suite, {'stage', 'config', 'experiments' if name == 'timesfm1' else 'lengths'}, name)
    if suite['stage'] != name:
        raise ValueError('批次文件与 stage 不匹配')
    if name == 'timesfm1' and suite['experiments'] != list(SETS):
        raise ValueError('timesfm1 固定为全特征及五个逐组消融')
    if name == 'timesfm2' and suite['lengths'] != LENGTHS:
        raise ValueError(f'timesfm2 固定长度为 {LENGTHS}')
    config_path = DEVELOP/suite['config']
    base = load_config(config_path)
    if base['model']['name'] != 'timesfm_xreg' or not base['weather']['match_origins']:
        raise ValueError(f'{name} 必须使用 timesfm_xreg 并启用共同样本资格')
    if base['model']['params']['context_length'] != 512:
        raise ValueError('TimesFM suite 基础配置须保留 512 天对照')
    if name == 'timesfm2' and base['features']['groups'] != SETS['no_actual']:
        raise ValueError('timesfm2 固定使用 no_actual 特征')
    experiments = []
    variants = [(identifier, groups, 512) for identifier, groups in SETS.items()] if name == 'timesfm1' else [
        (f'L{length}', SETS['no_actual'], length) for length in LENGTHS]
    for identifier, groups, length in variants:
        config = deepcopy(base)
        config['features']['groups'] = list(groups)
        config['model']['params']['context_length'] = length
        validate_config(config)
        experiments.append({'id': identifier, 'model': 'timesfm_xreg', 'feature_set': identifier if name == 'timesfm1' else 'no_actual',
                            'strategy': 'independent', 'config': config})
    return experiments, {}, [path, *config_files(config_path)]


def prepare(selections_path=None, name='timesfm1'):
    experiments, selection, paths = expand(selections_path, name)
    base = experiments[0]['config']
    market = load_market(base['data'])
    source = base['weather']['weather_source']
    weather = WeatherData(base['weather'], load_forecast=source == 'forecast')
    # Check every length before any fitting. All experiments share eligibility/status.
    inspections = {e['id']: check_backtest(e['config'], market, weather)
                   for e in (experiments if name == 'timesfm2' else experiments[:1])}
    inspection = deepcopy(next(iter(inspections.values())))
    status = 'ready' if all(i['status'] == 'passed' for i in inspections.values()) else 'blocked'
    reason = None if status == 'ready' else '完整窗口/特征检查失败，详见 preflight'
    if name == 'timesfm2':
        inspection['window_checks'] = inspections
        inspection['status'] = 'passed' if status == 'ready' else 'failed'
    for e in experiments:
        e.update(preflight_status=status, preflight_reason=reason)
    versions = {}
    for package in ('numpy', 'pandas', 'PyYAML', 'matplotlib', 'scikit-learn', 'scipy', 'torch',
                    'safetensors', 'jax', 'jaxlib', 'ml_dtypes', 'opt_einsum', 'tzdata'):
        versions[package] = importlib.metadata.version(package)
    identity = {'suite': name, 'experiments': experiments, 'selections': selection,
                'sources': {k: v['sha256'] for k, v in {**market.sources, **weather.sources}.items()},
                'config_files': {str(p.relative_to(DEVELOP)): digest(p) for p in paths},
                'code': {str(p.relative_to(DEVELOP)): digest(p) for p in
                         sorted((DEVELOP/'forecast').rglob('*.py'))+[DEVELOP/'run.py']},
                'rules': {p: digest(PROJECT/p) for p in ('DATA_AVAILABILITY.md', 'AGENTS.md')},
                'weights': {n: digest(Path(base['model']['params']['checkpoint_dir'])/n)
                            for n in ('config.json', 'model.safetensors')},
                'dependencies': versions, 'python': platform.python_version(), 'executable': sys.executable}
    fingerprint = hashlib.sha256(json.dumps(identity, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    inspection.update(suite=name, experiment_count=len(experiments), **weather_context(base, 'backtest'))
    inspection.update(weather_status='ready', weather_reason=None)
    inspection['experiments'] = [{k: e[k] for k in ('id', 'model', 'feature_set', 'strategy', 'preflight_status', 'preflight_reason')}
                                 for e in experiments]
    return experiments, market, weather, inspection, identity, fingerprint


def window_summary(directory, state):
    """Additional window diagnostics, with months grouped by target date."""
    import numpy as np
    import pandas as pd
    rows, monthly = [], []
    for entry in state['experiments']:
        if entry['status'] != 'passed':
            continue
        path = Path(entry['output'])
        pred = pd.read_csv(path/'predictions.csv')
        truth = pd.read_csv(path/'actuals.csv')
        keys = ['predict_time', 'D+2', 'D+7']
        joined = pred.merge(truth, on=keys, validate='one_to_one')
        if len(joined) != len(pred) or len(joined) != len(truth):
            raise ValueError('窗口汇总预测与标签日期不一致')
        errors = np.column_stack([joined[f'pred_D+{h}']-joined[f'true_D+{h}'] for h in range(2, 8)])
        if not np.isfinite(errors).all():
            raise ValueError('已完成实验含非有限误差')
        square = np.square(errors).ravel()
        smape = smape_values(joined[[f'pred_D+{h}' for h in range(2, 8)]].to_numpy(),
                             joined[[f'true_D+{h}' for h in range(2, 8)]].to_numpy()).ravel()
        metrics = pd.read_csv(path/'metrics.csv')
        row = {'experiment': entry['id'], 'context_length': entry['config']['model']['params']['context_length'],
               'origins': len(joined), 'elapsed_seconds': entry.get('elapsed_seconds'),
               'max_absolute_error': float(np.abs(errors).max()),
               'top5_squared_error_share': float(np.sort(square)[-5:].sum()/square.sum()) if square.sum() else 0.0}
        for scope, metric in [('overall', 'MAE'), ('overall', 'MSE'), ('overall', 'prediction_coverage'),
                              ('curve', 'ordering_accuracy'), ('curve', 'direction_accuracy'), ('curve', 'turn_F1')]:
            row[metric] = float(metrics.loc[(metrics.scope == scope) & (metrics.metric == metric), 'value'].iloc[0])
        # Compute from saved pairs so legacy metrics.csv files remain readable.
        row['SMAPE'] = float(smape.mean())
        rows.append(row)
        dates = np.column_stack([(pd.to_datetime(joined.predict_time)+pd.Timedelta(days=h)).dt.strftime('%Y-%m').to_numpy()
                                 for h in range(2, 8)]).ravel()
        for month in sorted(set(dates)):
            e = errors.ravel()[dates == month]
            monthly.append({'experiment': entry['id'], 'target_month': month, 'count': len(e),
                            'MAE': float(np.abs(e).mean()), 'MSE': float(np.square(e).mean()),
                            'SMAPE': float(smape[dates == month].mean())})
    summary = pd.DataFrame(rows)
    if len(summary) and 'L512' in set(summary.experiment):
        baseline = summary.set_index('experiment').loc['L512']
        for metric in ('MAE', 'MSE', 'SMAPE'):
            summary[metric+'_delta_vs_L512'] = summary[metric]-baseline[metric]
            summary[metric+'_change_percent_vs_L512'] = (summary[metric]/baseline[metric]-1)*100 if baseline[metric] else np.nan
    months = pd.DataFrame(monthly)
    summary.to_csv(directory/'window_summary.csv', index=False, encoding='utf-8-sig')
    months.to_csv(directory/'monthly_metrics.csv', index=False, encoding='utf-8-sig')
    return ('<h2>历史窗口长度对比</h2><p>月份按目标日期统计；每个起点与预测距离为一个评价点。'
            '长度同时改变 XReg 拟合段与 TimesFM 残差上下文，未自动替换默认模型。'
            'SMAPE 单位为 %，相对 L512 的差值为百分点；相对变化以基线为分母，基线为 0 时为空。</p>'
            +summary.to_html(index=False, escape=True)+'<h2>分月误差</h2>'+months.to_html(index=False, escape=True))
