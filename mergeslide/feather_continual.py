"""FEATHER model adapters for continual-learning baselines."""

import torch
import torch.nn as nn

from mergeslide.feather_models import FeatherMILWrapper


class FeatherGlobalClassifier(FeatherMILWrapper):
    """Expose a FEATHER global classifier through the baseline model interface."""

    def __init__(self, feather_model: nn.Module, num_classes: int) -> None:
        super().__init__(feather_model, num_classes=num_classes)

    @property
    def backbone(self) -> nn.Module:
        """Return the underlying FEATHER model for optional backbone freezing."""
        return self.model

    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        patch_size: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(features, coords, patch_size)
