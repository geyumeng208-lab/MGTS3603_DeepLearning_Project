from __future__ import annotations

from src.data import FieldDims
from src.utils import Config


def build_model(cfg: Config, field_dims: FieldDims):
    name = cfg.model.lower()
    if name in {"base", "lstm"}:
        from src.models.lstm import LSTMBaseModel

        return LSTMBaseModel(cfg, field_dims)
    if name == "sim":
        from src.models.sim import SIMModel

        return SIMModel(cfg, field_dims)
    if name == "eta":
        from src.models.eta import ETAModel

        return ETAModel(cfg, field_dims)
    if name == "twin":
        from src.models.twin import TWINModel

        return TWINModel(cfg, field_dims)
    if name in {"twin_lite", "twin_old"}:
        from src.models.twin_lite import TWINModel as TWINLiteModel

        return TWINLiteModel(cfg, field_dims)
    try:
        if name in {"hyformer", "hybrid_transformer"}:
            from src.models.hyformer import HyFormerModel

            return HyFormerModel(cfg, field_dims)
        if name in {"hyformer_opt", "hyformer_optimized"}:
            from src.models.hyformer_optimized import OptimizedHyFormerModel

            return OptimizedHyFormerModel(cfg, field_dims)
        if name in {"hyformer_time", "hyformer_temporal"}:
            from src.models.hyformer_time import TimeAwareHyFormerModel

            return TimeAwareHyFormerModel(cfg, field_dims)
        if name in {"hyformer_event", "hyformer_btag"}:
            from src.models.hyformer_event import EventAwareHyFormerModel

            return EventAwareHyFormerModel(cfg, field_dims)
        if name in {"hyformer_multigrain", "hyformer_multi"}:
            from src.models.hyformer_multigrain import MultiGranularityHyFormerModel

            return MultiGranularityHyFormerModel(cfg, field_dims)
        if name in {"hyformer_session", "hyformer_sessional"}:
            from src.models.hyformer_session import SessionAwareHyFormerModel

            return SessionAwareHyFormerModel(cfg, field_dims)
        if name in {"hyformer_static", "hyformer_profile"}:
            from src.models.hyformer_static import StaticFeatureHyFormerModel

            return StaticFeatureHyFormerModel(cfg, field_dims)
        if name in {"hyformer_hier", "hyformer_hierarchical"}:
            from src.models.hyformer_hierarchical import HierarchicalHyFormerModel

            return HierarchicalHyFormerModel(cfg, field_dims)
        if name in {"hyformer_dynamic", "hyformer_dyn"}:
            from src.models.hyformer_dynamic import DynamicHyFormerModel

            return DynamicHyFormerModel(cfg, field_dims)
        if name in {"hyformer_topk", "hyformer_filter"}:
            from src.models.hyformer_topk import TopKFilteredHyFormerModel

            return TopKFilteredHyFormerModel(cfg, field_dims)
        if name in {"hyformer_offline_long", "hyformer_cached_long"}:
            from src.models.hyformer_offline_long import OfflineLongTermHyFormerModel

            return OfflineLongTermHyFormerModel(cfg, field_dims)
    except ModuleNotFoundError as exc:
        if exc.name == "main_pytorch":
            raise ModuleNotFoundError(
                "HyFormer models require external/Hyformer_Pytorch/main_pytorch.py. "
                "Run: git clone https://github.com/WestbrookLong/Hyformer_Pytorch.git "
                "external/Hyformer_Pytorch"
            ) from exc
        raise
    raise ValueError(f"未知模型: {cfg.model}")
