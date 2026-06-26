"""FEATHER model adapters for continual-learning baselines."""

import torch
import torch.nn as nn

from mergeslide.feather_models import FeatherMILWrapper, get_feather_classifier_module


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

    def forward_with_embedding(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        patch_size: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Return logits and the tensor entering FEATHER's final classifier."""
        classifier = get_feather_classifier_module(self.model, int(self.num_classes))
        captured: dict[str, torch.Tensor] = {}

        def capture_input(_module, inputs):
            if inputs and isinstance(inputs[0], torch.Tensor):
                captured["cls_token"] = inputs[0]

        handle = classifier.register_forward_pre_hook(capture_input)
        try:
            logits = self.forward(features, coords, patch_size)
        finally:
            handle.remove()

        if "cls_token" not in captured:
            raise RuntimeError("Could not capture FEATHER classifier input for LWSR.")

        cls_token = captured["cls_token"]
        if cls_token.dim() == 1:
            cls_token = cls_token.unsqueeze(0)
        elif cls_token.dim() > 2:
            cls_token = cls_token.reshape(-1, cls_token.shape[-1])
        if cls_token.shape[0] != logits.shape[0]:
            raise RuntimeError(
                f"Captured FEATHER embedding batch {cls_token.shape[0]} does not match "
                f"logits batch {logits.shape[0]}."
            )

        return {"logits": logits, "cls_token": cls_token.float()}
