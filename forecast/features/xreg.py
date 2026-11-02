"""Origin-safe, lead-specific dynamic covariates for TimesFM XReg."""
from copy import deepcopy
from dataclasses import dataclass

import numpy as np
import pandas as pd

from ..availability import GRID_FIELDS, OFFSETS
from ..weather_data import COLUMNS
from .daily import MissingHistory
from .price_history import build as price_history

GROUPS = ['xreg_actual', 'xreg_dayahead', 'xreg_grid_forecast', 'weather_full', 'xreg_calendar']
FULL = ['price_history', *GROUPS]
SETS = {'full': FULL, **{'no_'+name: [g for g in FULL if g != group] for name, group in
        zip(('actual', 'dayahead', 'grid_forecast', 'weather', 'calendar'), GROUPS)}}


def is_xreg(groups):
    return groups in SETS.values()


def schema(groups):
    numerical, categorical = [], []
    for group, fields in market_groups().items():
        if group in groups:
            numerical.extend(fields)
    if 'weather_full' in groups:
        for col, (field, _) in COLUMNS.items():
            (categorical if field == 'weather_code' else numerical).append(col)
    missing = [c+'__missing' for c in numerical+categorical]
    numerical.extend(missing)
    if 'xreg_calendar' in groups:
        numerical.extend(['weekend', 'year_sin', 'year_cos'])
        categorical.append('weekday')
    return {'numeric': numerical, 'categorical': categorical, 'columns': numerical+categorical,
            'column_count': len(numerical)+len(categorical), 'input_type': 'XRegHistory',
            'dimension_basis': 'dynamic_covariates_before_one_hot_encoding',
            'weather_missing_indicators': 'weather_full' in groups}


def market_groups():
    return {'xreg_actual': [f+'实际值' for f in GRID_FIELDS], 'xreg_dayahead': ['日前电价'],
            'xreg_grid_forecast': [f+'预测值' for f in GRID_FIELDS]}


@dataclass
class XRegHistory:
    origin: pd.Timestamp
    dates: pd.DatetimeIndex
    values: np.ndarray
    tasks: dict

    def copy(self):
        return deepcopy(self)


def build(info, groups, weather, source='history', length=512, match_full=False):
    history, price_evidence = price_history(info, length)
    required = FULL if match_full else groups
    # Validate complete dates before extracting cells; missing cells have a separate policy.
    wx = None
    if 'weather_full' in required:
        if weather is None:
            raise ValueError('XReg 天气/共同样本检查缺少天气表')
        past = weather.select_range(history.dates[0], history.dates[-1], 'history')
        future = weather.select_range(info.origin, info.origin+pd.Timedelta(days=7), source)
        wx = pd.concat([past, future])
    tasks, maps = {}, {}
    for h in range(2, 8):
        dates = pd.date_range(history.dates[0], info.origin+pd.Timedelta(days=h))
        numeric, categorical, mapping = {}, {}, {}
        for group, fields in market_groups().items():
            if group not in required:
                continue
            lag = {'xreg_actual': h+1, 'xreg_dayahead': h, 'xreg_grid_forecast': h-1}[group]
            raw_dates = dates-pd.Timedelta(days=lag)
            missing_dates = raw_dates.difference(info.daily.index)
            if len(missing_dates):
                raise MissingHistory(f'{group} 缺少原始日期: {missing_dates.strftime("%Y-%m-%d").tolist()}')
            for field in fields:
                assert raw_dates[-1] <= info.origin+pd.Timedelta(days=OFFSETS[field])
            if group in groups:
                block = info.daily.loc[raw_dates, fields]
                numeric.update({c: block[c].to_numpy(dtype=float) for c in fields})
                mapping[group] = {'columns': fields, 'lag_days': lag,
                                  'source_dates': raw_dates.strftime('%Y-%m-%d').tolist(),
                                  'last_source_date': str(raw_dates[-1].date())}
        if 'weather_full' in groups:
            block = wx.loc[dates]
            for col, (field, _) in COLUMNS.items():
                (categorical if field == 'weather_code' else numeric)[col] = block[col].to_numpy(dtype=float)
        # Fixed missing flags for every raw field, including categories; no future-based pruning.
        for col, values in list(numeric.items())+list(categorical.items()):
            numeric[col+'__missing'] = pd.isna(values).astype(float)
        if 'xreg_calendar' in groups:
            phase = 2*np.pi*(dates.dayofyear.to_numpy()-1)/np.where(dates.is_leap_year, 366, 365)
            numeric.update(weekend=(dates.dayofweek.to_numpy() >= 5).astype(float),
                           year_sin=np.sin(phase), year_cos=np.cos(phase))
            categorical['weekday'] = dates.dayofweek.to_numpy()
        tasks[h] = {'dates': dates, 'numeric': numeric, 'categorical': categorical}
        maps[str(h)] = {'forecast_dates': dates[length:].strftime('%Y-%m-%d').tolist(),
                        'covariate_length': len(dates), 'output_index': h, 'market': mapping,
                        'fit_label_count': length, 'fit_label_last_date': str(history.dates[-1].date())}
    evidence = {'predict_time': str(info.origin.date()), 'training_mode': 'in_context_xreg',
                'price_history': price_evidence, 'horizons': maps,
                'groups': groups, 'eligibility_groups': required,
                'calendar': 'weekday, weekend, sin/cos of zero-based day-of-year / 365 or 366',
                'missing_policy': 'numeric history median (all empty=0); category missing sentinel; fixed flags; no bfill',
                'publication_time_check': 'not_run'}
    if 'weather_full' in groups:
        evidence['weather'] = {'columns': list(COLUMNS), 'sources': {
            s: weather.sources['weather_'+s] for s in dict.fromkeys(['history', source])},
            'history_dates': history.dates.strftime('%Y-%m-%d').tolist(),
            'history_csv_rows': [weather.rows['history'][d] for d in history.dates],
            'future_dates': future.index.strftime('%Y-%m-%d').tolist(),
            'future_csv_rows': [weather.rows[source][d] for d in future.index], 'future_source': source,
            'transformations': 'sunrise/sunset: file-clock minutes; weather_code: categorical; others: original numeric',
            'publication_time_check': 'not_run', 'historical_revision_audit': 'not_run'}
    return XRegHistory(history.origin, history.dates, history.values, tasks), evidence
