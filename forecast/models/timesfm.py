"""Zero-shot TimesFM adapter. Importing this module never imports torch or loads weights."""
import ctypes
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys

import numpy as np
import pandas as pd

from ..data import PROJECT, digest
from ..features.price_history import PriceHistory

DEFAULTS = {"context_length": 512, "checkpoint_dir": "E:/pycharmproject/epf/models/timesFM"}


def validate_params(params):
    if not isinstance(params, dict) or set(params)-set(DEFAULTS):
        raise ValueError(f"未知 TimesFM 参数: {params}")
    result = {**DEFAULTS, **params}
    length = result["context_length"]
    if type(length) is not int or length <= 0 or math.ceil(length/32)*32+128 > 16384:
        raise ValueError("TimesFM context_length 必须为正整数，且按 32 对齐后加 128 不超过 16384")
    if not isinstance(result["checkpoint_dir"], str) or not result["checkpoint_dir"].strip():
        raise ValueError("TimesFM checkpoint_dir 必须为非空本地路径")
    return result


def available_memory():
    if os.name == "nt":
        class MemoryStatus(ctypes.Structure):
            _fields_ = [("length", ctypes.c_ulong), ("load", ctypes.c_ulong)] + [
                (name, ctypes.c_ulonglong) for name in
                ("total", "available", "page_total", "page_available", "virtual_total", "virtual_available", "extended")]
        status = MemoryStatus()
        status.length = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError("无法检查可用内存")
        return status.available
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_AVPHYS_PAGES")
    except (AttributeError, ValueError, OSError) as exc:
        raise OSError("无法检查可用内存，停止加载 TimesFM") from exc


def preflight(checkpoint):
    """Check resources and dependencies before importing the private model implementation."""
    if sys.version_info < (3, 11):
        raise ValueError("TimesFM 需要 Python 3.11 或更新版本")
    for name in ("config.json", "model.safetensors"):
        if not (checkpoint/name).is_file():
            raise FileNotFoundError(f"TimesFM 本地文件不存在（不会自动下载）: {checkpoint/name}")
    ram = available_memory()
    if ram < 2 * 1024**3:
        raise ValueError("TimesFM 加载前可用 RAM 不足 2 GiB，已停止")
    versions = {}
    for package in ("torch", "safetensors", "numpy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError as exc:
            raise ImportError("TimesFM 缺少依赖，请安装 develop/requirements.txt") from exc
    import torch
    device, free_vram = "cpu", None
    if torch.cuda.is_available():
        free_vram, _ = torch.cuda.mem_get_info(0)
        if free_vram >= 2 * 1024**3:
            device = "cuda:0"
    return {"status": "passed", "available_ram_bytes": ram, "free_vram_bytes": free_vram,
            "memory_warning": "available RAM below recommended 4 GiB" if ram < 4*1024**3 else None,
            "disk_free_bytes": shutil.disk_usage(checkpoint).free, "download_required": False,
            "device": device, "dependencies": versions}


def check_checkpoint(params):
    """Read local file structure only; never instantiate a model or materialize tensors."""
    params = validate_params(params)
    checkpoint = Path(params["checkpoint_dir"])
    if not checkpoint.is_absolute():
        checkpoint = PROJECT/checkpoint
    checkpoint = checkpoint.resolve()
    resources = preflight(checkpoint)
    config_path = checkpoint/"config.json"
    model_config = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(model_config, dict) or not model_config:
        raise ValueError(f"TimesFM config.json 必须是非空 JSON 对象: {config_path}")
    from safetensors import safe_open, SafetensorError
    weights_path = checkpoint/"model.safetensors"
    try:
        # safe_open validates header/offsets/file length. keys() reads metadata,
        # unlike get_tensor(), which would materialize the pretrained weights.
        with safe_open(str(weights_path), framework="numpy") as weights:
            tensor_count = len(weights.keys())
    except SafetensorError as exc:
        raise ValueError(f"TimesFM 权重格式无效: {weights_path}: {exc}") from exc
    if not tensor_count:
        raise ValueError(f"TimesFM 权重文件不包含张量: {weights_path}")
    return {"status": "passed", "checkpoint_dir": str(checkpoint), "preflight": resources,
            "config_json_check": "passed", "safetensors_structure_check": "passed",
            "tensor_count": tensor_count, "weights_bytes": weights_path.stat().st_size,
            "model_compatibility_check": "not_run", "tensor_values_check": "not_run",
            "model_load": "not_run", "prediction": "not_run"}


class TimesFMModel:
    requires_training = False

    def __init__(self, config):
        self.params = validate_params(config["params"])
        self.seed, self.threads = config["seed"], config["threads"]
        self.backend = None
        self.runtime_metadata = {"model_version": "TimesFM 2.5 200M", "training_mode": "zero_shot",
                                 "context_length": self.params["context_length"], "load_status": "not_loaded"}

    def fit(self, X, y):
        return self

    def _load(self):
        checkpoint = Path(self.params["checkpoint_dir"])
        if not checkpoint.is_absolute():
            checkpoint = PROJECT/checkpoint
        checkpoint = checkpoint.resolve()
        checked = preflight(checkpoint)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        import torch
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.set_num_threads(self.threads)
        torch.use_deterministic_algorithms(True)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.set_float32_matmul_precision("high")
        from .._vendor.timesfm.configs import ForecastConfig
        from .._vendor.timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch
        backend = TimesFM_2p5_200M_torch.from_local_checkpoint(checkpoint, device=checked["device"])
        backend.compile(ForecastConfig(
            max_context=math.ceil(self.params["context_length"]/32)*32, max_horizon=128,
            normalize_inputs=True, per_core_batch_size=1, use_continuous_quantile_head=False,
            force_flip_invariance=True, infer_is_positive=False, fix_quantile_crossing=False,
            return_backcast=getattr(self, "return_backcast", False)))
        self.backend = backend
        self.runtime_metadata.update({"load_status": "loaded", "device": checked["device"],
            "preflight": checked, "checkpoint_dir": str(checkpoint),
            "checkpoint_sha256": {name: digest(checkpoint/name) for name in ("config.json", "model.safetensors")},
            "compiled_context_length": backend.forecast_config.max_context, "compiled_max_horizon": 128,
            "forecast_horizon": 8, "output_slice": [2, 8], "torch_compile": False})

    def predict(self, X):
        if not isinstance(X, PriceHistory):
            raise ValueError("TimesFM 仅接受 PriceHistory 输入")
        length = self.params["context_length"]
        expected = pd.date_range(X.origin-pd.Timedelta(days=length), periods=length)
        if not X.dates.equals(expected) or X.values.shape != (length,) or not np.isfinite(X.values).all():
            raise ValueError("TimesFM 输入必须为截至 D-1 的完整连续历史序列")
        # Conversion must not silently overflow otherwise finite raw prices.
        with np.errstate(over="ignore"):
            values = X.values.astype(np.float32)
        if not np.isfinite(values).all():
            raise ValueError("TimesFM 输入超出 float32 数值范围")
        if self.backend is None:
            self._load()
        try:
            point, _ = self.backend.forecast(horizon=8, inputs=[values])
        except RuntimeError as exc:
            raise ValueError(f"TimesFM 推理失败: {exc}") from exc
        point = np.asarray(point, dtype=float)
        if point.shape != (1, 8) or not np.isfinite(point).all():
            raise ValueError("TimesFM 必须返回形状 (1, 8) 的有限点预测，不允许补齐")
        return point[0, 2:8].copy()
