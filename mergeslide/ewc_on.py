"""Online EWC components for FEATHER continual WSI learning."""

from typing import Dict, Optional

import torch
import torch.nn as nn

from mergeslide.continual_model import ContinualModel
from mergeslide.feather_continual import FeatherGlobalClassifier


class EwcOn(ContinualModel):
    """Online EWC trainer for a FEATHER global classifier.

    The diagonal Fisher follows Mammoth's oEWC implementation: each squared
    score gradient is weighted by the conditional probability of its observed
    label, then accumulated online. The stored parameter checkpoint and Fisher
    only cover trainable parameters, so this also works with a frozen backbone.
    """

    NAME = "ewc_on"
    COMPATIBILITY = ("class-il", "domain-il", "task-il")

    def __init__(
        self,
        model: FeatherGlobalClassifier,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        e_lambda: float,
        gamma: float,
        patch_size: int = 512,
    ) -> None:
        super().__init__(model=model, optimizer=optimizer, device=device)
        if e_lambda < 0:
            raise ValueError("e_lambda must be non-negative.")
        if not 0.0 <= gamma <= 1.0:
            raise ValueError("gamma must be in [0, 1].")

        self.e_lambda = float(e_lambda)
        self.gamma = float(gamma)
        self.patch_size_value = int(patch_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.checkpoint: Optional[Dict[str, torch.Tensor]] = None
        self.fisher: Optional[Dict[str, torch.Tensor]] = None
        # Keep the checkpoint schema shared with replay baselines.
        self.buffer = ()

    @property
    def patch_size(self) -> torch.Tensor:
        return torch.tensor(self.patch_size_value, dtype=torch.int32, device=self.device)

    def _named_trainable_parameters(self):
        return [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]

    def penalty(self) -> torch.Tensor:
        """Return the weighted EWC penalty for the current parameters."""
        if self.checkpoint is None or self.fisher is None:
            return torch.zeros((), device=self.device)

        penalty = torch.zeros((), device=self.device)
        for name, param in self._named_trainable_parameters():
            if name not in self.checkpoint or name not in self.fisher:
                continue
            penalty = penalty + (
                self.fisher[name] * (param - self.checkpoint[name]).pow(2)
            ).sum()
        return self.e_lambda * penalty

    @staticmethod
    def _sample_bag(
        features: torch.Tensor,
        coords: torch.Tensor,
        k: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if k > 0 and features.shape[0] > k:
            indices = torch.randperm(features.shape[0])[:k]
            features = features[indices]
            coords = coords[indices]
        return features, coords

    def observe(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Run one stream update with the online EWC constraint."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(features, coords, self.patch_size)
        loss_ce = self.loss_fn(logits, labels)
        loss_ewc = self.penalty()
        loss = loss_ce + loss_ewc
        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite FEATHER EWC loss.")
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "loss_ce": float(loss_ce.detach().cpu()),
            "loss_ewc": float(loss_ewc.detach().cpu()),
            "buffer_size": 0.0,
        }

    def end_task(
        self,
        train_loader,
        *,
        label_offset: int = 0,
        k: int = 0,
    ) -> int:
        """Estimate the empirical Fisher on the completed FEATHER task."""
        named_params = self._named_trainable_parameters()
        fisher = {name: torch.zeros_like(param, device=self.device) for name, param in named_params}
        num_batches = 0
        for features, coords, labels in train_loader:
            features, coords = self._sample_bag(features, coords, int(k))
            features = features.to(self.device, non_blocking=True)
            coords = coords.long().to(self.device, non_blocking=True)
            labels = labels.to(self.device, non_blocking=True).long() + int(label_offset)

            self.optimizer.zero_grad(set_to_none=True)
            logits = self.model(features, coords, self.patch_size)
            nll = nn.functional.cross_entropy(logits, labels, reduction="none")
            conditional_probability = torch.exp(-nll.detach()).mean()
            # Mammoth differentiates log p(y | x); the sign is irrelevant once
            # the gradient is squared for the Fisher diagonal.
            (-nll.mean()).backward()
            for name, param in named_params:
                if param.grad is not None:
                    fisher[name].add_(conditional_probability * param.grad.detach().pow(2))
            num_batches += 1

        self.optimizer.zero_grad(set_to_none=True)
        if not num_batches:
            raise RuntimeError("Cannot estimate EWC Fisher from an empty task loader.")

        fisher = {name: value / num_batches for name, value in fisher.items()}
        if self.fisher is None:
            self.fisher = fisher
        else:
            self.fisher = {
                name: self.gamma * self.fisher.get(name, torch.zeros_like(value)) + value
                for name, value in fisher.items()
            }
        self.checkpoint = {
            name: param.detach().clone()
            for name, param in self._named_trainable_parameters()
        }

        super().end_task()
        return num_batches
