"""Read-only source loading and label integrity checks."""
from dataclasses import dataclass, field
from pathlib import Path
import hashlib

import numpy as np
import pandas as pd

from .availability import OFFSETS

DEVELOP = Path(__file__).resolve().parents[1]
PROJECT = DEVELOP.parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


@dataclass
class MarketData:
    daily: pd.DataFrame
    hourly: pd.DataFrame
    availability: dict = field(default_factory=dict)
    sources: dict = field(default_factory=dict)
    daily_rows: dict = field(default_factory=dict)

    @property
    def labels(self):
        return self.daily["实时电价"].copy()


def read_table(path, encoding, hourly=False):
    raw = pd.read_csv(path, encoding=encoding)
    if "时刻" not in raw:
        raise ValueError(f"缺少时刻列: {path}")
    fmt = "%Y-%m-%d %H:%M:%S" if hourly else "%Y-%m-%d"
    index = pd.DatetimeIndex(pd.to_datetime(raw.pop("时刻"), format=fmt, errors="raise"))
    if index.hasnans or index.has_duplicates:
        raise ValueError(f"空时间或重复时间: {path}")
    if (hourly and not (index == index.floor("h")).all()) or (not hourly and not (index == index.normalize()).all()):
        raise ValueError(f"时间不符合{'整点' if hourly else '业务日'}要求: {path}")
    raw.index = index
    available = {}
    for col in list(raw):
        if col.endswith("__available_at"):
            field_name = col.removesuffix("__available_at")
            if field_name not in OFFSETS:
                raise ValueError(f"未知可用时间字段: {col}")
            values = raw.pop(col)
            parsed = []
            for value in values:
                if pd.isna(value):
                    parsed.append(pd.NaT)
                else:
                    stamp = pd.Timestamp(value)
                    if stamp.tzinfo is None:
                        raise ValueError(f"{col} 必须含时区")
                    parsed.append(stamp.tz_convert("UTC"))
            available[field_name] = pd.Series(pd.to_datetime(parsed, utc=True), index=index)
    if set(raw) != set(OFFSETS):
        raise ValueError(f"字段不匹配: 缺少={set(OFFSETS)-set(raw)}, 未分类={set(raw)-set(OFFSETS)}")
    for col in raw:
        raw[col] = pd.to_numeric(raw[col], errors="raise")
        if np.isinf(raw[col].to_numpy(dtype=float)).any():
            raise ValueError(f"无限数值: {col}")
    return raw.sort_index(), available


def load_market(config):
    tables, times, sources = {}, {}, {}
    for freq in ("daily", "hourly"):
        path = (PROJECT / config[freq]).resolve()
        tables[freq], times[freq] = read_table(path, config["encoding"], freq == "hourly")
        sources[freq] = {"path": str(path), "sha256": digest(path), "rows": len(tables[freq])}
    raw_dates = pd.read_csv(PROJECT/config["daily"], encoding=config["encoding"], usecols=["时刻"])["时刻"]
    daily_rows = {str(pd.Timestamp(d).date()): i+2 for i, d in enumerate(raw_dates)}
    return MarketData(tables["daily"], tables["hourly"], times, sources, daily_rows)


def check_integrity(market, tolerance=0.001):
    d, h = market.daily, market.hourly
    if d.empty or h.empty:
        raise ValueError("日表和小时表不得为空")
    expected = pd.date_range(d.index.min(), d.index.max(), freq="D")
    if not d.index.equals(expected):
        raise ValueError("日表日期不连续")
    business = (h.index - pd.Timedelta(hours=1)).normalize()
    grouped = h["实时电价"].groupby(business)
    stats = grouped.agg(["size", "count", "mean"])
    if not stats.index.equals(d.index):
        raise ValueError("小时表与日表业务日不一致")
    label = d["实时电价"]
    invalid = ((stats["size"] != 24) | (~stats["count"].isin([0, 24])) |
               (label.notna() != stats["count"].eq(24)) |
               ((label - stats["mean"]).abs() > tolerance))
    if invalid.any():
        raise ValueError("日均标签完整性失败: " + ", ".join(d.index[invalid].strftime("%Y-%m-%d")))
    return {"status": "passed", "business_days": len(d), "verified_labels": int(label.notna().sum()),
            "unpublished_days": int(label.isna().sum()), "mean_tolerance": tolerance,
            "max_mean_difference": float((label-stats["mean"]).abs().max())}
