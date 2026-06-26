"""Learning without Forgetting components for FEATHER continual WSI learning."""

from copy import deepcopy
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD

from mergeslide.continual_model import ContinualModel
from mergeslide.feather_continual import FeatherGlobalClassifier
from mergeslide.feather_models import get_feather_classifier_module


class Lwf(ContinualModel):
    """LwF trainer for a FEATHER global classifier.

    This follows Mammoth's LwF sequence: warm up the new classifier rows with
    SGD, then retain old-class predictions while learning the full task.
    """

    NAME = "lwf"
    COMPATIBILITY = ("class-il", "task-il")

    def __init__(
        self,
        model: FeatherGlobalClassifier,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        alpha: float,
        softmax_temp: float,
        patch_size: int = 512,
    ) -> None:
        super().__init__(model=model, optimizer=optimizer, device=device)
        if alpha < 0:
            raise ValueError("alpha must be non-negative.")
        if softmax_temp <= 0:
            raise ValueError("softmax_temp must be positive.")

        self.alpha = float(alpha)
        self.softmax_temp = float(softmax_temp)
        self.patch_size_value = int(patch_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.old_model: Optional[FeatherGlobalClassifier] = None
        self.past_classes = 0
        self.seen_classes: Optional[int] = None
        # Keep the checkpoint schema shared with replay baselines.
        self.buffer = ()

    @property
    def patch_size(self) -> torch.Tensor:
        return torch.tensor(self.patch_size_value, dtype=torch.int32, device=self.device)

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

    def _warmup_new_classifier(
        self,
        train_loader,
        *,
        label_offset: int,
        warmup_epochs: int,
        warmup_lr: float,
        k: int,
    ) -> None:
        """Mirror Mammoth's SGD warm-up for the classifier rows of a new task."""
        if self.past_classes == 0 or warmup_epochs <= 0:
            return
        total_classes = int(self.model.num_classes)
        classifier = get_feather_classifier_module(self.model.model, total_classes)
        original_requires_grad = {
            name: parameter.requires_grad
            for name, parameter in self.model.named_parameters()
        }
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        for parameter in classifier.parameters():
            parameter.requires_grad_(True)

        optimizer = SGD(classifier.parameters(), lr=warmup_lr)
        self.model.eval()
        for _ in range(warmup_epochs):
            for features, coords, labels in train_loader:
                features, coords = self._sample_bag(features, coords, int(k))
                features = features.to(self.device, non_blocking=True)
                coords = coords.long().to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True).long() + int(label_offset)

                optimizer.zero_grad(set_to_none=True)
                logits = self.model(features, coords, self.patch_size)
                loss = self.loss_fn(
                    logits[:, self.past_classes:self.seen_classes],
                    labels - self.past_classes,
                )
                loss.backward()
                optimizer.step()

        for name, parameter in self.model.named_parameters():
            parameter.requires_grad_(original_requires_grad[name])

    def begin_task(
        self,
        train_loader,
        *,
        label_offset: int,
        past_classes: int,
        seen_classes: int,
        warmup_epochs: int,
        warmup_lr: float,
        k: int,
    ) -> None:
        """Warm up new rows and set class boundaries for the upcoming task."""
        if past_classes < 0 or seen_classes <= 0 or past_classes > seen_classes:
            raise ValueError("Invalid LwF class boundaries.")
        self.past_classes = int(past_classes)
        self.seen_classes = int(seen_classes)
        if self.past_classes and self.old_model is None:
            raise RuntimeError("LwF requires an old-model snapshot after the first task.")
        self._warmup_new_classifier(
            train_loader,
            label_offset=label_offset,
            warmup_epochs=warmup_epochs,
            warmup_lr=warmup_lr,
            k=k,
        )
        self.model.train()

    def _distillation_loss(
        self,
        logits: torch.Tensor,
        features: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        if self.old_model is None or self.past_classes == 0:
            return torch.zeros((), device=self.device)

        with torch.no_grad():
            old_logits = self.old_model(features, coords, self.patch_size)
        temperature = self.softmax_temp
        old_probabilities = F.softmax(old_logits[:, : self.past_classes] / temperature, dim=1)
        new_log_probabilities = F.log_softmax(logits[:, : self.past_classes] / temperature, dim=1)
        return -(old_probabilities * new_log_probabilities).sum(dim=1).mean()

    def observe(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Run one LwF stream update for a FEATHER WSI bag."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        logits = self.model(features, coords, self.patch_size)
        seen_classes = self.seen_classes or logits.shape[1]
        if seen_classes > logits.shape[1]:
            raise ValueError("LwF seen_classes exceeds FEATHER classifier dimension.")
        loss_ce = self.loss_fn(logits[:, :seen_classes], labels)
        loss_distill = self._distillation_loss(logits, features, coords)
        loss = loss_ce + self.alpha * loss_distill
        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite FEATHER LwF loss.")
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "loss_ce": float(loss_ce.detach().cpu()),
            "loss_distill": float(loss_distill.detach().cpu()),
            "buffer_size": 0.0,
        }

    def end_task(self) -> None:
        """Freeze the current FEATHER model as the next distillation teacher."""
        self.old_model = deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad_(False)
        super().end_task()
