"""
A-GEM components for TITAN-based continual WSI learning.

This module uses TITAN only as a slide encoder with a randomly initialized
global classifier head. It does not use prompts or TITAN's text encoder.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from mergeslide.continual_model import ContinualModel
from mergeslide.derpp import TitanGlobalClassifier


class AgemBuffer:
    """Reservoir memory of previous-task slide bags for A-GEM."""

    def __init__(self, buffer_size: int, seed: int = 0):
        self.buffer_size = int(buffer_size)
        self.num_seen_examples = 0
        self.rng = np.random.default_rng(seed)
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.labels: List[torch.Tensor] = []

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

    def add_data(self, features: torch.Tensor, coords: torch.Tensor, labels: torch.Tensor) -> None:
        if self.buffer_size <= 0:
            return

        index = self._reservoir_index()
        self.num_seen_examples += 1
        if index < 0:
            return

        example = (features.detach(), coords.detach().long())
        label = labels.detach().long()

        if index == len(self.examples):
            self.examples.append(example)
            self.labels.append(label)
        else:
            self.examples[index] = example
            self.labels[index] = label

    def get_data(self, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty A-GEM buffer.")

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
                )
            )
        return batch


class AgemTITAN(ContinualModel):
    """A-GEM trainer for a TITAN global classifier."""

    NAME = "agem"
    COMPATIBILITY = ("class-il", "domain-il", "task-il")

    def __init__(
        self,
        model: TitanGlobalClassifier,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        buffer_size: int,
        patch_size: int = 1024,
        seed: int = 0,
    ):
        super().__init__(model=model, optimizer=optimizer, device=device)
        self.buffer = AgemBuffer(buffer_size, seed=seed)
        self.patch_size_value = int(patch_size)
        self.loss_fn = nn.CrossEntropyLoss()
        self.params = [p for p in self.model.parameters() if p.requires_grad]

    @property
    def patch_size(self) -> torch.Tensor:
        return torch.tensor(self.patch_size_value, dtype=torch.int32, device=self.device)

    def _grad_vector(self) -> torch.Tensor:
        chunks = []
        for param in self.params:
            if param.grad is None:
                chunks.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
            else:
                chunks.append(param.grad.detach().reshape(-1).float())
        return torch.cat(chunks)

    def _overwrite_grad(self, grad_vector: torch.Tensor) -> None:
        pointer = 0
        for param in self.params:
            numel = param.numel()
            grad = grad_vector[pointer:pointer + numel].view_as(param).to(param.dtype)
            if param.grad is None:
                param.grad = grad.clone()
            else:
                param.grad.detach().copy_(grad)
            pointer += numel

    def _reference_loss(self) -> torch.Tensor:
        losses = []
        for features, coords, labels in self.buffer.get_data(self.device):
            outputs = self.model(features, coords, self.patch_size)
            losses.append(self.loss_fn(outputs, labels))
        return torch.stack(losses).mean()

    def observe(self, features: torch.Tensor, coords: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Run one A-GEM optimization step.

        The current gradient is projected only when it conflicts with the
        replay-memory gradient, i.e. dot(current, reference) < 0.
        """
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(features, coords, self.patch_size)
        loss = self.loss_fn(outputs, labels)
        loss.backward()
        current_grad = self._grad_vector()

        dot_product = torch.zeros((), device=self.device)
        projected = False
        reference_loss = torch.zeros((), device=self.device)

        if not self.buffer.is_empty():
            self.optimizer.zero_grad(set_to_none=True)
            reference_loss = self._reference_loss()
            reference_loss.backward()
            reference_grad = self._grad_vector()

            dot_product = torch.dot(current_grad, reference_grad)
            denom = torch.dot(reference_grad, reference_grad)
            if dot_product.item() < 0 and denom.item() > 0:
                current_grad = current_grad - (dot_product / denom) * reference_grad
                projected = True

            self._overwrite_grad(current_grad)

        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "reference_loss": float(reference_loss.detach().cpu()),
            "dot_product": float(dot_product.detach().cpu()),
            "projected": float(projected),
            "buffer_size": float(len(self.buffer)),
        }

    def add_to_buffer(self, features: torch.Tensor, coords: torch.Tensor, labels: torch.Tensor) -> None:
        self.buffer.add_data(features=features, coords=coords, labels=labels)

    def end_task(
        self,
        train_loader,
        label_offset: int = 0,
        k: int = 0,
        samples_per_task: int = 1,
    ) -> int:
        """Store a task quota of WSI bags in memory without changing A-GEM loss."""
        samples_to_add = max(0, int(samples_per_task))
        added = 0

        for features, coords, labels in train_loader:
            if added >= samples_to_add:
                break

            if k is not None:
                k = int(k)
                if k > 0 and features.shape[0] > k:
                    indices = torch.randperm(features.shape[0])[:k]
                    features = features[indices]
                    coords = coords[indices]

            features = features.to(self.device, non_blocking=True)
            coords = coords.long().to(self.device, non_blocking=True)
            global_labels = labels.to(self.device, non_blocking=True).long() + int(label_offset)
            self.add_to_buffer(features, coords, global_labels)
            added += 1

        super().end_task()
        return added
