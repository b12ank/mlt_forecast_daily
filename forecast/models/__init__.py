from .reference import ReferenceModel
from .regression import MeanModel, ScalarRegressor, IndependentRegressor, validate_model


def create_model(config, feature_groups=None):
    if isinstance(config, str):
        config = {"name": config, "strategy": "shared", "params": {}, "seed": 42, "threads": 4}
    validate_model(config)
    if config["name"] == "timesfm_xreg":
        from ..features.xreg import is_xreg
        if not is_xreg(feature_groups):
            raise ValueError("TimesFM XReg 特征组不匹配")
        from .timesfm_xreg import TimesFMXRegModel
        return TimesFMXRegModel(config)
    if config["name"] == "timesfm":
        if feature_groups != ["price_history"]:
            raise ValueError("TimesFM 必须使用 price_history 特征")
        from .timesfm import TimesFMModel
        return TimesFMModel(config)
    if config["name"] == "reference":
        return ReferenceModel()
    if config["name"] in {"recent_mean", "weekday_mean"}:
        return MeanModel(config["name"])
    if config["strategy"] == "independent":
        return IndependentRegressor(config, feature_groups=feature_groups)
    return ScalarRegressor(config, feature_groups=feature_groups)
