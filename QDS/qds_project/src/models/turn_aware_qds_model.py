"""Turn-aware QDS model for AIS trajectories. See src/models/README.md."""

from src.models.attention_qds_model_base import AttentionQDSModelBase


class TurnAwareQDSModel(AttentionQDSModelBase):
    """Turn-aware QDS model for AIS trajectory data."""

    #: Number of point features expected by this model.
    POINT_FEATURES: int = 8

    def __init__(self, embed_dim: int = 64, num_heads: int = 4) -> None:
        super().__init__(
            point_features=self.POINT_FEATURES,
            embed_dim=embed_dim,
            num_heads=num_heads,
        )
