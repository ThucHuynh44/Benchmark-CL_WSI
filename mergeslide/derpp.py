"""
DER++ components for TITAN-based continual WSI learning.

This module intentionally does not use TITAN text prompts. TITAN is used only
as a slide encoder, followed by a randomly initialized global classifier head.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TitanGlobalClassifier(nn.Module):
    """Wrap TITAN's vision encoder with one global classification head."""

    def __init__(self, titan_model: nn.Module, num_classes: int, embed_dim: int = 768):
        super().__init__()
        self.backbone = titan_model.vision_encoder
        self.classifier = nn.Linear(embed_dim, num_classes)
        nn.init.normal_(self.classifier.weight, mean=0.0, std=0.01)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, features: torch.Tensor, coords: torch.Tensor, patch_size: torch.Tensor) -> torch.Tensor:
        slide_embed = self.backbone(features, coords, patch_size)
        return self.classifier(slide_embed.float())


class ReservoirBuffer:
    """Slide-level reservoir buffer for variable-length WSI feature bags."""

    def __init__(self, buffer_size: int, seed: int = 0):
        self.buffer_size = int(buffer_size)
        self.num_seen_examples = 0
        self.rng = np.random.default_rng(seed)
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.labels: List[torch.Tensor] = []
        self.logits: List[torch.Tensor] = []

    def __len__(self) -> int:
        return len(self.examples)

    def is_empty(self) -> bool:
        return len(self.examples) == 0

    def _reservoir_index(self) -> int:
        if self.buffer_size <= 0:
            return -1
        if self.num_seen_examples < self.buffer_size:
            return self.num_seen_examples
        index = int(self.rng.integers(0, self.num_seen_examples + 1))
        return index if index < self.buffer_size else -1

    def add_data(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        labels: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        """Store one slide sample and its pre-update logits on the current device."""
        if self.buffer_size <= 0:
            return

        index = self._reservoir_index()
        self.num_seen_examples += 1
        if index < 0:
            return

        example = (features.detach(), coords.detach().long())
        label = labels.detach().long()
        logit = logits.detach().float()

        if index == len(self.examples):
            self.examples.append(example)
            self.labels.append(label)
            self.logits.append(logit)
        else:
            self.examples[index] = example
            self.labels[index] = label
            self.logits[index] = logit

    def get_data(self, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Sample one slide bag on ``device``."""
        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty buffer.")

        indices = self.rng.choice(len(self.examples), size=1, replace=False)
        batch = []
        for raw_index in np.atleast_1d(indices):
            index = int(raw_index)
            features, coords = self.examples[index]
            batch.append(
                (
                    features.to(device, non_blocking=True),
                    coords.to(device, non_blocking=True),
                    self.labels[index].to(device, non_blocking=True),
                    self.logits[index].to(device, non_blocking=True),
                )
            )
        return batch


class DerppTITAN:
    """DER++ trainer for a TITAN global classifier."""

    def __init__(
        self,
        model: TitanGlobalClassifier,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        buffer_size: int,
        alpha: float,
        beta: float,
        patch_size: int = 1024,
        seed: int = 0,
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.buffer = ReservoirBuffer(buffer_size, seed=seed)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.patch_size_value = int(patch_size)
        self.loss_fn = nn.CrossEntropyLoss()

    @property
    def patch_size(self) -> torch.Tensor:
        return torch.tensor(self.patch_size_value, dtype=torch.int32, device=self.device)

    def _replay_mse(self) -> torch.Tensor:
        losses = []
        for features, coords, _, logits in self.buffer.get_data(self.device):
            outputs = self.model(features, coords, self.patch_size)
            losses.append(F.mse_loss(outputs, logits))
        return torch.stack(losses).mean()

    def _replay_ce(self) -> torch.Tensor:
        losses = []
        for features, coords, labels, _ in self.buffer.get_data(self.device):
            outputs = self.model(features, coords, self.patch_size)
            losses.append(self.loss_fn(outputs, labels))
        return torch.stack(losses).mean()

    def observe(self, features: torch.Tensor, coords: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Run one DER++ optimization step and add the current slide to memory."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(features, coords, self.patch_size)
        loss_ce = self.loss_fn(outputs, labels)
        loss = loss_ce
        loss_mse = torch.zeros((), device=self.device)
        loss_replay_ce = torch.zeros((), device=self.device)

        if not self.buffer.is_empty():
            loss_mse = self.alpha * self._replay_mse()
            loss_replay_ce = self.beta * self._replay_ce()
            loss = loss + loss_mse + loss_replay_ce

        loss.backward()
        self.optimizer.step()

        self.buffer.add_data(features=features, coords=coords, labels=labels, logits=outputs)

        return {
            "loss": float(loss.detach().cpu()),
            "loss_ce": float(loss_ce.detach().cpu()),
            "loss_mse": float(loss_mse.detach().cpu()),
            "loss_replay_ce": float(loss_replay_ce.detach().cpu()),
            "buffer_size": float(len(self.buffer)),
        }
