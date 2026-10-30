"""Business availability rules. No guessed publication timestamps."""
from dataclasses import dataclass
import re

import pandas as pd

GRID_FIELDS = (
    "地方电厂总加", "联络线受电负荷", "风电总加", "光伏总加", "核电总加",
    "自备机组总加", "试验机组总加", "直调负荷", "竞价空间", "新能源总加",
)
OFFSETS = {"实时电价": -1, "日前电价": 0}
OFFSETS.update({name + suffix: offset for name in GRID_FIELDS
                for suffix, offset in [("实际值", -1), ("预测值", 1)]})
RULE = "DATA_AVAILABILITY.md §1.2; business-boundary-only policy confirmed 2026-10-27"
AVAILABILITY_POLICY = "business_boundaries_only_v1"


def day(value):
    if isinstance(value, str) and not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        raise ValueError("日期必须为 YYYY-MM-DD")
    stamp = pd.Timestamp(value)
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError("日期必须是不含时区与时刻的业务日")
    return stamp


@dataclass
class InformationSet:
    origin: pd.Timestamp
    daily: pd.DataFrame
    hourly: pd.DataFrame
    checks: dict


def slice_fields(frame, origin, frequency):
    """Mask each column by its confirmed business boundary, without a run clock."""
    unknown = set(frame.columns) - set(OFFSETS)
    if unknown:
        raise ValueError(f"未分类字段，禁止进入信息集: {sorted(unknown)}")
    extra_day = 1 if frequency == "hourly" else 0
    if frequency not in {"hourly", "daily"}:
        raise ValueError("未知数据频率")
    result = frame.loc[frame.index <= origin + pd.Timedelta(days=1 + extra_day)].copy()
    evidence = {}
    for col in result:
        bound = origin + pd.Timedelta(days=OFFSETS[col] + extra_day)
        allowed = pd.Series(result.index <= bound, index=result.index)
        result.loc[~allowed, col] = float("nan")
        assert result.loc[result.index > bound, col].isna().all()
        evidence[col] = {"upper_bound": bound.isoformat(), "basis": RULE,
                         "nonempty": int(result[col].notna().sum())}
    return result, evidence


def information_set(daily, hourly, origin):
    origin = day(origin)
    d, dc = slice_fields(daily, origin, "daily")
    h, hc = slice_fields(hourly, origin, "hourly")
    return InformationSet(origin, d, h,
                          {"status": "passed", "availability_policy": AVAILABILITY_POLICY,
                           "publication_time_check": "not_run", "daily": dc, "hourly": hc})
