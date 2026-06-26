"""MICIL components adapted for FEATHER continual WSI learning."""

from copy import deepcopy
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from mergeslide.continual_model import ContinualModel
from mergeslide.feather_continual import FeatherGlobalClassifier
from mergeslide.feather_models import get_feather_classifier_module


class MicilReplayBuffer:
    """Reservoir memory of variable-length WSI bags and global labels."""

    def __init__(self, buffer_size: int, device: torch.device, seed: int = 0) -> None:
        self.buffer_size = int(buffer_size)
        self.device = device
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

        example = (
            features.detach().to(self.device, non_blocking=True),
            coords.detach().long().to(self.device, non_blocking=True),
        )
        label = labels.detach().long().to(self.device, non_blocking=True)
        if index == len(self.examples):
            self.examples.append(example)
            self.labels.append(label)
        else:
            self.examples[index] = example
            self.labels[index] = label

    def get_data(self, size: int, device: torch.device) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty MICIL buffer.")

        size = min(int(size), len(self.examples))
        indices = self.rng.choice(len(self.examples), size=size, replace=False)
        batch = []
        for raw_index in np.atleast_1d(indices):
            index = int(raw_index)
            features, coords = self.examples[index]
            batch.append(
                (
                    features.to(device, non_blocking=True),
                    coords.to(device, non_blocking=True),
                    self.labels[index].to(device, non_blocking=True).long().reshape(-1),
                )
            )
        return batch


class MicilFEATHER(ContinualModel):
    """MICIL trainer for a FEATHER global classifier.

    The original MICIL pipeline used TransMIL and local NPY embeddings.  This
    adapter keeps only the MICIL loss structure and applies it to MergeSlide's
    FEATHER model, H5 feature bags, and global-label task stream.
    """

    NAME = "micil"
    COMPATIBILITY = ("class-il", "task-il")

    def __init__(
        self,
        model: FeatherGlobalClassifier,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        buffer_size: int,
        minibatch_size: int,
        ce_loss_weight: float,
        kd_loss_weight: float,
        em_loss_weight: float,
        weight_norm: bool,
        num_classes: int,
        patch_size: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__(model=model, optimizer=optimizer, device=device)
        if ce_loss_weight < 0 or kd_loss_weight < 0 or em_loss_weight < 0:
            raise ValueError("MICIL loss weights must be non-negative.")
        if num_classes <= 0:
            raise ValueError("num_classes must be positive.")

        self.buffer = MicilReplayBuffer(buffer_size, device=device, seed=seed)
        self.minibatch_size = max(1, int(minibatch_size))
        self.ce_loss_weight = float(ce_loss_weight)
        self.kd_loss_weight = float(kd_loss_weight)
        self.em_loss_weight = float(em_loss_weight)
        self.weight_norm = bool(weight_norm)
        self.num_classes = int(num_classes)
        self.patch_size_value = int(patch_size)
        self.temperature = 2.0
        self.old_model: Optional[FeatherGlobalClassifier] = None
        self.old_classes = 0
        self.seen_classes = int(num_classes)
        self._warned_weight_norm = False

    @property
    def patch_size(self) -> torch.Tensor:
        return torch.tensor(self.patch_size_value, dtype=torch.int32, device=self.device)

    @staticmethod
    def _sample_bag(features: torch.Tensor, coords: torch.Tensor, k: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if k > 0 and features.shape[0] > k:
            indices = torch.randperm(features.shape[0])[:k]
            features = features[indices]
            coords = coords[indices]
        return features, coords

    def begin_task(self, *, past_classes: int, seen_classes: int) -> None:
        """Set class boundaries and validate the frozen teacher for the task."""
        if past_classes < 0 or seen_classes <= 0 or past_classes > seen_classes:
            raise ValueError("Invalid MICIL class boundaries.")
        if seen_classes > self.num_classes:
            raise ValueError("MICIL seen_classes exceeds classifier dimension.")

        self.old_classes = int(past_classes)
        self.seen_classes = int(seen_classes)
        if self.old_classes == 0:
            return
        if self.old_model is None:
            raise RuntimeError("MICIL requires a frozen teacher snapshot after task 0.")

        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad_(False)

    def _forward(self, features: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model.forward_with_embedding(features, coords, self.patch_size)

    def _teacher_forward(self, features: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        if self.old_model is None:
            raise RuntimeError("MICIL teacher is not initialized.")
        with torch.no_grad():
            return self.old_model.forward_with_embedding(features, coords, self.patch_size)

    def _distillation_loss(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        if self.old_classes == 0:
            return torch.zeros((), device=self.device)
        if student_logits.shape[1] < self.old_classes or teacher_logits.shape[1] < self.old_classes:
            raise ValueError("MICIL KD class slice exceeds logits dimension.")

        temperature = self.temperature
        teacher_probabilities = F.softmax(teacher_logits[:, : self.old_classes] / temperature, dim=1)
        student_log_probabilities = F.log_softmax(student_logits[:, : self.old_classes] / temperature, dim=1)
        return F.kl_div(student_log_probabilities, teacher_probabilities, reduction="batchmean")

    def _embedding_matching_loss(
        self,
        student_embeddings: torch.Tensor,
        teacher_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        if self.old_classes == 0:
            return torch.zeros((), device=self.device)
        if student_embeddings.shape != teacher_embeddings.shape:
            raise RuntimeError(
                "MICIL embedding shapes do not match: "
                f"student={tuple(student_embeddings.shape)} teacher={tuple(teacher_embeddings.shape)}"
            )
        return F.mse_loss(student_embeddings.float(), teacher_embeddings.float())

    def _normalize_classifier_weights(self) -> bool:
        if not self.weight_norm or self.current_task == 0:
            return False
        try:
            classifier = get_feather_classifier_module(self.model.model, self.num_classes)
        except RuntimeError as exc:
            if not self._warned_weight_norm:
                warnings.warn(f"Skipping MICIL classifier weight normalization: {exc}")
                self._warned_weight_norm = True
            return False

        if not isinstance(classifier, nn.Linear) or classifier.weight.dim() != 2:
            if not self._warned_weight_norm:
                warnings.warn("Skipping MICIL classifier weight normalization: incompatible classifier head.")
                self._warned_weight_norm = True
            return False

        with torch.no_grad():
            classifier.weight.copy_(F.normalize(classifier.weight, dim=1))
        return True

    def _validate_label_range(
        self,
        labels: torch.Tensor,
        *,
        min_label: int,
        max_label: int,
        context: str,
    ) -> torch.Tensor:
        labels = labels.reshape(-1)
        if labels.numel() == 0:
            raise ValueError(f"MICIL received empty {context} labels.")
        if torch.is_floating_point(labels) and not torch.allclose(labels, labels.round()):
            raise ValueError(f"MICIL {context} labels must be integer class ids.")

        labels = labels.long()
        if max_label <= min_label:
            raise ValueError(
                f"Invalid MICIL {context} label interval [{min_label}, {max_label})."
            )
        label_min = int(labels.min().item())
        label_max = int(labels.max().item())
        if label_min < min_label or label_max >= max_label:
            raise ValueError(
                f"MICIL {context} labels must be global-contiguous in "
                f"[{min_label}, {max_label}). Got min={label_min}, max={label_max}."
            )
        return labels

    def _make_training_batch(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        labels: torch.Tensor,
    ) -> List[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        stream_labels = self._validate_label_range(
            labels,
            min_label=self.old_classes,
            max_label=self.seen_classes,
            context="stream",
        )
        batch = [(features, coords, stream_labels)]
        if self.current_task > 0 and not self.buffer.is_empty():
            for buf_features, buf_coords, buf_labels in self.buffer.get_data(self.minibatch_size, self.device):
                replay_labels = self._validate_label_range(
                    buf_labels,
                    min_label=0,
                    max_label=self.old_classes,
                    context="replay",
                )
                batch.append((buf_features, buf_coords, replay_labels))
        return batch

    def observe(self, features: torch.Tensor, coords: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Run one MICIL update with CE, old-class KD, embedding matching, and replay."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        student_logits, student_embeddings, batch_labels = [], [], []
        teacher_logits, teacher_embeddings = [], []
        training_batch = self._make_training_batch(features, coords, labels)

        for bag_features, bag_coords, bag_labels in training_batch:
            outputs = self._forward(bag_features, bag_coords)
            student_logits.append(outputs["logits"])
            student_embeddings.append(outputs["cls_token"])
            batch_labels.append(bag_labels.long())
            if self.current_task > 0 and self.old_classes > 0:
                teacher_outputs = self._teacher_forward(bag_features, bag_coords)
                teacher_logits.append(teacher_outputs["logits"])
                teacher_embeddings.append(teacher_outputs["cls_token"])

        logits = torch.cat(student_logits, dim=0).float()
        embeddings = torch.cat(student_embeddings, dim=0).float()
        target_labels = torch.cat(batch_labels, dim=0).long()
        if target_labels.min().item() < 0 or target_labels.max().item() >= self.seen_classes:
            raise ValueError(
                "MICIL labels must be global labels within the seen class range. "
                f"label_min={int(target_labels.min())} label_max={int(target_labels.max())} "
                f"seen_classes={self.seen_classes}"
            )

        loss_ce_raw = F.cross_entropy(logits[:, : self.seen_classes], target_labels)
        loss_kd_raw = torch.zeros((), device=self.device)
        loss_em_raw = torch.zeros((), device=self.device)
        if self.current_task > 0 and self.old_classes > 0:
            teacher_logits_tensor = torch.cat(teacher_logits, dim=0).float()
            teacher_embeddings_tensor = torch.cat(teacher_embeddings, dim=0).float()
            loss_kd_raw = self._distillation_loss(logits, teacher_logits_tensor)
            loss_em_raw = self._embedding_matching_loss(embeddings, teacher_embeddings_tensor)

        loss_ce = self.ce_loss_weight * loss_ce_raw
        loss_kd = self.kd_loss_weight * loss_kd_raw
        loss_em = self.em_loss_weight * loss_em_raw
        loss = loss_ce + loss_kd + loss_em
        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite FEATHER MICIL loss.")

        loss.backward()
        self.optimizer.step()
        normalized = self._normalize_classifier_weights()

        return {
            "loss": float(loss.detach().cpu()),
            "total_loss": float(loss.detach().cpu()),
            "ce_loss": float(loss_ce.detach().cpu()),
            "kd_loss": float(loss_kd.detach().cpu()),
            "em_loss": float(loss_em.detach().cpu()),
            "ce_loss_raw": float(loss_ce_raw.detach().cpu()),
            "kd_loss_raw": float(loss_kd_raw.detach().cpu()),
            "em_loss_raw": float(loss_em_raw.detach().cpu()),
            "buffer_size": float(len(self.buffer)),
            "buffer_len": float(len(self.buffer)),
            "old_classes": float(self.old_classes),
            "seen_classes": float(self.seen_classes),
            "logits_batch": float(logits.shape[0]),
            "logits_dim": float(logits.shape[1]),
            "embedding_batch": float(embeddings.shape[0]),
            "embedding_dim": float(embeddings.shape[1]),
            "weight_norm_applied": float(normalized),
        }

    def end_task(
        self,
        train_loader,
        *,
        label_offset: int = 0,
        k: int = 0,
    ) -> int:
        """Store completed-task bags and snapshot the student as frozen teacher."""
        added = 0
        for features, coords, labels in train_loader:
            features, coords = self._sample_bag(features, coords, int(k))
            global_labels = (labels.long() + int(label_offset)).reshape(-1)
            global_labels = self._validate_label_range(
                global_labels,
                min_label=self.old_classes,
                max_label=self.seen_classes,
                context="buffer-add",
            )
            self.buffer.add_data(features=features, coords=coords, labels=global_labels)
            added += 1

        self.old_model = deepcopy(self.model).to(self.device)
        self.old_model.eval()
        for param in self.old_model.parameters():
            param.requires_grad_(False)

        super().end_task()
        return added
