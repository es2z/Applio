"""HPA-RMVPE (PhamHuynhAnh16/HPA-RMVPE) with the published 76000 / 112000 weights."""

from .predictor import HPARMVPEPredictor, get_offline_predictor

__all__ = ["HPARMVPEPredictor", "get_offline_predictor"]
