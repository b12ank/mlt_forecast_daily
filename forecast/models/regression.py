"""Explicit feature schemas and train-only preprocessing for scalar regressors."""
import importlib

import numpy as np
import pandas as pd

from ..features.daily import MissingHistory
from ..features.schema import input_schema

DEFAULTS = {
    "reference": {}, "recent_mean": {}, "weekday_mean": {},
    "ridge": {"alpha": 10.0},
    "catboost": {"depth": 5, "iterations": 300, "learning_rate": 0.05, "loss_function": "RMSE"},
    "lightgbm": {"max_depth": 5, "num_leaves": 15, "min_child_samples": 30,
                 "n_estimators": 300, "learning_rate": 0.05, "objective": "regression"},
}
PACKAGES = {"ridge": "sklearn", "catboost": "catboost", "lightgbm": "lightgbm"}


def validate_model(config):
    name = config["name"]
    if name not in DEFAULTS:
        raise ValueError(f"模型未实现: {name}")
    if config["strategy"] not in {"shared", "independent"}:
        raise ValueError("strategy 必须为 shared 或 independent")
    if name not in PACKAGES and config["strategy"] != "shared":
        raise ValueError("简单基线无需独立训练")
    if type(config["seed"]) is not int or config["seed"] < 0:
        raise ValueError("seed 必须为非负整数")
    if type(config["threads"]) is not int or config["threads"] < 1:
        raise ValueError("threads 必须为正整数")
    params = config["params"]
    if not isinstance(params, dict) or set(params)-set(DEFAULTS[name]):
        raise ValueError(f"未知模型参数: {params}")
    for key, value in params.items():
        default = DEFAULTS[name][key]
        if isinstance(default, str):
            allowed = {"loss_function": {"RMSE", "MAE"}, "objective": {"regression", "regression_l1"}}
            if value not in allowed[key]:
                raise ValueError(f"不支持 {key}={value}")
        elif isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0:
            raise ValueError(f"参数 {key} 必须为有限正数")
        elif isinstance(default, int) and type(value) is not int:
            raise ValueError(f"参数 {key} 必须为整数")


def check_dependency(name):
    if name in PACKAGES:
        importlib.import_module(PACKAGES[name])


class MeanModel:
    requires_training = False

    def __init__(self, name):
        self.column = "price_mean7" if name == "recent_mean" else "same_weekday_mean28"

    def fit(self, X, y):
        return self

    def predict(self, X):
        result = X[self.column].to_numpy(dtype=float)
        if not np.isfinite(result).all():
            raise MissingHistory(f"简单基线无有效历史值: {self.column}")
        return result


class ScalarRegressor:
    requires_training = True

    def __init__(self, config, omit_horizon=False, feature_groups=None):
        self.config = config
        self.omit_horizon = omit_horizon
        self.feature_groups = ["price_calendar"] if feature_groups is None else list(feature_groups)

    def _frame(self, X):
        missing = [c for c in self.columns if c not in X]
        if missing:
            raise ValueError(f"模型输入缺少声明列: {missing}")
        out = X[self.columns].copy()
        for col in self.categorical:
            out[col] = out[col].map(lambda v: "missing" if pd.isna(v) else str(int(v)))
        for col in self.numeric:
            out[col] = pd.to_numeric(out[col], errors="raise").astype(float)
        return out

    def fit(self, X, y):
        if X.empty or len(X) != len(y) or not np.isfinite(np.asarray(y, dtype=float)).all():
            raise ValueError("训练矩阵和标签为空、长度不匹配或标签非有限")
        schema = input_schema(self.feature_groups, independent=self.omit_horizon)
        self.numeric, self.categorical, self.columns = schema["numeric"], schema["categorical"], schema["columns"]
        data = self._frame(X)
        name, seed, threads = self.config["name"], self.config["seed"], self.config["threads"]
        params = {**DEFAULTS[name], **self.config["params"]}
        if name == "ridge":
            from sklearn.linear_model import Ridge
            from sklearn.preprocessing import OneHotEncoder, StandardScaler
            # Explicit all-empty-column handling is stable across sklearn releases.
            self.medians = data[self.numeric].median().fillna(0.0)
            self.scaler = StandardScaler().fit(data[self.numeric].fillna(self.medians))
            self.encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(data[self.categorical])
            self.estimator = Ridge(**params)
        elif name == "catboost":
            from catboost import CatBoostRegressor
            self.estimator = CatBoostRegressor(**params, random_seed=seed, thread_count=threads,
                                               task_type="CPU", verbose=False, allow_writing_files=False)
        else:
            from lightgbm import LGBMRegressor
            self.categories = {c: sorted(data[c].unique().tolist()) for c in self.categorical}
            self.estimator = LGBMRegressor(**params, random_state=seed, n_jobs=threads, verbosity=-1,
                                           device_type="cpu", deterministic=True, force_col_wise=True)
        matrix = self._transform(data)
        if name == "catboost":
            self.estimator.fit(matrix, np.asarray(y), cat_features=self.categorical)
        else:
            self.estimator.fit(matrix, np.asarray(y))
        return self

    def _transform(self, data):
        if self.config["name"] == "ridge":
            return np.column_stack([self.scaler.transform(data[self.numeric].fillna(self.medians)),
                                    self.encoder.transform(data[self.categorical])])
        if self.config["name"] == "lightgbm":
            data = data.copy()
            for col, levels in self.categories.items():
                data[col] = pd.Categorical(data[col], categories=levels)
        return data

    def predict(self, X):
        return self.estimator.predict(self._transform(self._frame(X)))


class IndependentRegressor:
    requires_training = True

    def __init__(self, config, feature_groups=None):
        self.config = config
        self.feature_groups = feature_groups

    def fit(self, X, y):
        counts = X.groupby("predict_time")["horizon"].agg(list)
        if not counts.map(lambda values: sorted(values) == list(range(2, 8))).all():
            raise ValueError("独立模型也必须使用完整六步训练样本")
        self.models = {}
        target = np.asarray(y)
        for h in range(2, 8):
            mask = X["horizon"].eq(h).to_numpy()
            self.models[h] = ScalarRegressor(self.config, omit_horizon=True,
                                            feature_groups=self.feature_groups).fit(X.loc[mask], target[mask])
        return self

    def predict(self, X):
        result = np.full(len(X), np.nan)
        for h, model in self.models.items():
            mask = X["horizon"].eq(h).to_numpy()
            if mask.any():
                result[mask] = model.predict(X.loc[mask])
        return result
