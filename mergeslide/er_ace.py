"""
ER-ACE components for TITAN-based continual WSI learning.

TITAN is used only as a slide encoder with a randomly initialized global
classifier head. No class-aware prompts or TITAN text encoder are used.
"""

from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn

from mergeslide.continual_model import ContinualModel
from mergeslide.derpp import TitanGlobalClassifier


class ErAceBuffer:
    """Reservoir memory of slide bags and labels for ER-ACE replay."""

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
            raise RuntimeError("Cannot sample from an empty ER-ACE buffer.")

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


class ErAceTITAN(ContinualModel):
    """ER-ACE trainer for a TITAN global classifier."""

    NAME = "er_ace"
    COMPATIBILITY = ("class-il", "task-il")

    def __init__(
        self,
        model: TitanGlobalClassifier,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        buffer_size: int,
        patch_size: int = 1024,
        seed: int = 0,
        use_amp: bool = True,
        task_free: bool = False,
    ):
        super().__init__(model=model, optimizer=optimizer, device=device)
        self.buffer = ErAceBuffer(buffer_size, seed=seed)
        self.patch_size_value = int(patch_size)
        self.loss_fn = nn.CrossEntropyLoss()
        _ = use_amp
        self.task_free = bool(task_free)
        self.seen_so_far = torch.tensor([], dtype=torch.long, device=device)

    @property
    def patch_size(self) -> torch.Tensor:
        return torch.tensor(self.patch_size_value, dtype=torch.int32, device=self.device)

    def _ace_logits(self, logits: torch.Tensor, present: torch.Tensor) -> torch.Tensor:
        mask = torch.zeros_like(logits)
        mask[:, present] = 1
        if self.seen_so_far.numel() > 0 and self.seen_so_far.max() < logits.shape[1] - 1:
            mask[:, self.seen_so_far.max():] = 1
        return logits.masked_fill(mask == 0, torch.finfo(logits.dtype).min)

    def _replay_loss(self) -> torch.Tensor:
        losses = []
        for features, coords, labels in self.buffer.get_data(self.device):
            outputs = self.model(features, coords, self.patch_size)
            losses.append(self.loss_fn(outputs, labels))
        return torch.stack(losses).mean()

    def observe(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        labels: torch.Tensor,
    ) -> Dict[str, float]:
        """Run one ER-ACE step and store the current slide in memory."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        outputs = self.model(features, coords, self.patch_size)
        present = labels.unique()
        self.seen_so_far = torch.cat([self.seen_so_far, present]).unique()

        stream_logits = outputs
        if self.current_task > 0 or self.task_free:
            stream_logits = self._ace_logits(outputs, present)
        loss_stream = self.loss_fn(stream_logits, labels)
        loss_replay = torch.zeros((), device=self.device)
        loss = loss_stream

        if not self.buffer.is_empty() and (self.current_task > 0 or self.task_free):
            loss_replay = self._replay_loss()
            loss = loss + loss_replay

        loss.backward()
        self.optimizer.step()

        self.buffer.add_data(features=features, coords=coords, labels=labels)

        return {
            "loss": float(loss.detach().cpu()),
            "loss_stream": float(loss_stream.detach().cpu()),
            "loss_replay": float(loss_replay.detach().cpu()),
            "buffer_size": float(len(self.buffer)),
        }
