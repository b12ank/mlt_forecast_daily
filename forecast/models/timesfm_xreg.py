"""Six independent in-context XReg fits with a shared frozen TimesFM backend."""
import importlib.metadata
from dataclasses import replace
import math

import numpy as np
import pandas as pd

from .timesfm import TimesFMModel, validate_params as price_params, DEFAULTS as PRICE_DEFAULTS
from ..features.xreg import XRegHistory
from ..data import resolve_path

DEFAULTS = {**PRICE_DEFAULTS, 'ridge': 0.01}


def validate_params(params):
    if not isinstance(params, dict) or set(params)-set(DEFAULTS):
        raise ValueError(f'未知 TimesFM XReg 参数: {params}')
    result = {**DEFAULTS, **params}
    price_params({k: result[k] for k in PRICE_DEFAULTS})
    if type(result['ridge']) not in (float, int) or result['ridge'] != 0.01:
        raise ValueError('TimesFM XReg 固定 ridge=0.01')
    return result


def check_dependencies():
    for name in ('jax', 'jaxlib', 'scikit-learn'):
        importlib.metadata.version(name)
    import jax
    import jax.numpy as jnp
    from sklearn.preprocessing import OneHotEncoder
    return {'jax': jax.__version__, 'cpu_available': bool(jax.devices('cpu'))}


def preprocess(task, length):
    """Only the historical prefix contributes imputation parameters."""
    numerical, categorical, medians = {}, {}, {}
    for col, raw in task['numeric'].items():
        values = np.asarray(raw, dtype=float)
        if np.isinf(values).any():
            raise ValueError(f'协变量包含无穷值: {col}')
        known = values[:length][np.isfinite(values[:length])]
        median = float(np.median(known)) if len(known) else 0.0
        filled = np.where(np.isnan(values), median, values)
        numerical[col] = [filled]
        medians[col] = median
    for col, raw in task['categorical'].items():
        if np.isinf(np.asarray(raw, dtype=float)).any():
            raise ValueError(f'类别包含无穷值: {col}')
        categorical[col] = [['__missing__' if pd.isna(v) else str(int(v)) for v in raw]]
    return numerical, categorical, medians


class TimesFMXRegModel(TimesFMModel):
    return_backcast = True
    requires_training = False  # No expanding-window supervised dataset; fit happens inside XReg.

    def __init__(self, config):
        params = validate_params(config['params'])
        super().__init__({**config, 'params': {k: params[k] for k in PRICE_DEFAULTS}})
        self.params = params
        self.backend_cache = None
        self.last_diagnostics = {}
        self.runtime_metadata.update(training_mode='in_context_xreg', xreg_mode='xreg + timesfm',
                                     horizon_strategy='six_independent_calls', ridge=params['ridge'])

    def _load(self):
        check_dependencies()
        key = (str(resolve_path(self.params['checkpoint_dir'])), self.seed, self.threads)
        if self.backend_cache is not None and key in self.backend_cache:
            self.backend, saved = self.backend_cache[key]
            self.runtime_metadata.update(saved)
        else:
            super()._load()
            self.runtime_metadata.update(forecast_horizons=list(range(3, 9)), output_indices=list(range(2, 8)))
            self.runtime_metadata.pop('output_slice', None)
            if self.backend_cache is not None:
                self.backend_cache[key] = (self.backend, dict(self.runtime_metadata))
        self._configure_context()

    def _configure_context(self):
        """Reconfigure one loaded backend; never retain a model per window size."""
        length = self.params['context_length']
        compiled = math.ceil(length/32)*32
        if self.backend.forecast_config.max_context != compiled:
            self.backend.compile(replace(self.backend.forecast_config, max_context=compiled))
        self.runtime_metadata.update(context_length=length, compiled_context_length=compiled)

    def predict(self, X):
        if not isinstance(X, XRegHistory):
            raise ValueError('TimesFM XReg 仅接受 XRegHistory')
        length = self.params['context_length']
        expected = pd.date_range(X.origin-pd.Timedelta(days=length), periods=length)
        if not X.dates.equals(expected) or X.values.shape != (length,) or not np.isfinite(X.values).all():
            raise ValueError(f'XReg 输入必须为截至 D-1 的{length}天连续有限价格')
        values = X.values.astype(np.float32)
        if not np.isfinite(values).all() or set(X.tasks) != set(range(2, 8)):
            raise ValueError('XReg 输入溢出或预测距离不完整')
        if self.backend is None:
            self._load()
        elif self.backend_cache is not None:
            self._configure_context()
        result, diagnostics = [], {}
        for h in range(2, 8):
            task = X.tasks[h]
            dates = pd.date_range(expected[0], X.origin+pd.Timedelta(days=h))
            if not task['dates'].equals(dates) or any(len(v) != length+h+1 for kind in ('numeric', 'categorical') for v in task[kind].values()):
                raise ValueError(f'h={h} 的动态协变量形状/日期不匹配')
            num, cat, medians = preprocess(task, length)
            point, _ = self.backend.forecast_with_covariates(
                inputs=[values.copy()], dynamic_numerical_covariates=num,
                dynamic_categorical_covariates=cat or None, xreg_mode='xreg + timesfm',
                ridge=self.params['ridge'], normalize_xreg_target_per_input=True,
                max_rows_per_col=0, force_on_cpu=True)
            prediction = np.asarray(point, dtype=float)
            if prediction.shape != (1, h+1) or not np.isfinite(prediction).all():
                raise ValueError(f'h={h} 必须返回 (1,{h+1}) 有限预测，不允许补齐')
            result.append(prediction[0, h])
            diagnostics[str(h)] = {'fit_label_count': length, 'fit_label_start': str(expected[0].date()),
                                  'fit_label_last': str(expected[-1].date()), 'numeric_medians': medians,
                                  'numeric_count': len(num), 'categorical_count': len(cat),
                                  'forecast_count': h+1, 'selected_index': h}
        self.last_diagnostics = diagnostics
        return np.asarray(result)
