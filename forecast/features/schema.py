"""Explicit model inputs; alignment columns never become features by accident."""
from .price_calendar import NUMERIC, CATEGORICAL

FEATURE_SETS = {
    "F0": ["price_calendar"],
    "F1": ["price_calendar", "weather_aggregate"],
    "F2": ["price_calendar", "weather_full"],
    "F3": ["weather_full"],
    "F4": ["weather_full_no_missing"],
    "F5": ["price_calendar", "weather_full_no_missing"],
}


def validate_groups(groups):
    if groups != ["daily"] and groups not in FEATURE_SETS.values():
        raise ValueError(f"不支持的特征组: {groups}")


def input_schema(groups, model_name=None, independent=False):
    from ..weather_data import weather_columns

    validate_groups(groups)
    simple = {"reference": "last_price", "recent_mean": "price_mean7",
              "weekday_mean": "same_weekday_mean28"}
    if model_name in simple:
        numeric, categorical = [simple[model_name]], []
    elif groups == ["daily"]:
        numeric, categorical = ["last_price"], []
    else:
        numeric, categorical = [], []
        if "price_calendar" in groups:
            numeric = list(NUMERIC) + [c+"__missing" for c in NUMERIC]
            categorical = [c for c in CATEGORICAL if not (independent and c == "horizon")]
        for group in groups:
            if group.startswith("weather_"):
                columns = weather_columns(group)
                numeric.extend(c for c in columns if c.startswith("wx_num__"))
                categorical.extend(c for c in columns if c.startswith("wx_cat__"))
    weather = next((g for g in groups if g.startswith("weather_")), None)
    columns = numeric + categorical
    return {"numeric": numeric, "categorical": categorical, "columns": columns,
            "column_count": len(columns), "dimension_basis": "before_category_encoding_per_estimator",
            "weather_missing_indicators": weather is not None and weather != "weather_full_no_missing"}
