from __future__ import annotations

from src.data import FieldDims
from src.models.eta import ETAModel
from src.models.hyformer import HyFormerModel
from src.models.hyformer_event import EventAwareHyFormerModel
from src.models.hyformer_dynamic import DynamicHyFormerModel
from src.models.hyformer_hierarchical import HierarchicalHyFormerModel
from src.models.hyformer_multigrain import MultiGranularityHyFormerModel
from src.models.hyformer_optimized import OptimizedHyFormerModel
from src.models.hyformer_session import SessionAwareHyFormerModel
from src.models.hyformer_static import StaticFeatureHyFormerModel
from src.models.hyformer_time import TimeAwareHyFormerModel
from src.models.lstm import LSTMBaseModel
from src.models.sim import SIMModel
from src.models.twin import TWINModel
from src.models.twin_lite import TWINModel as TWINLiteModel
from src.utils import Config


def build_model(cfg: Config, field_dims: FieldDims):
    name = cfg.model.lower()
    if name in {"base", "lstm"}:
        return LSTMBaseModel(cfg, field_dims)
    if name == "sim":
        return SIMModel(cfg, field_dims)
    if name == "eta":
        return ETAModel(cfg, field_dims)
    if name == "twin":
        return TWINModel(cfg, field_dims)
    if name in {"twin_lite", "twin_old"}:
        return TWINLiteModel(cfg, field_dims)
    if name in {"hyformer", "hybrid_transformer"}:
        return HyFormerModel(cfg, field_dims)
    if name in {"hyformer_opt", "hyformer_optimized"}:
        return OptimizedHyFormerModel(cfg, field_dims)
    if name in {"hyformer_time", "hyformer_temporal"}:
        return TimeAwareHyFormerModel(cfg, field_dims)
    if name in {"hyformer_event", "hyformer_btag"}:
        return EventAwareHyFormerModel(cfg, field_dims)
    if name in {"hyformer_multigrain", "hyformer_multi"}:
        return MultiGranularityHyFormerModel(cfg, field_dims)
    if name in {"hyformer_session", "hyformer_sessional"}:
        return SessionAwareHyFormerModel(cfg, field_dims)
    if name in {"hyformer_static", "hyformer_profile"}:
        return StaticFeatureHyFormerModel(cfg, field_dims)
    if name in {"hyformer_hier", "hyformer_hierarchical"}:
        return HierarchicalHyFormerModel(cfg, field_dims)
    if name in {"hyformer_dynamic", "hyformer_dyn"}:
        return DynamicHyFormerModel(cfg, field_dims)
    raise ValueError(f"未知模型: {cfg.model}")
