"""LWSR components for FEATHER continual WSI learning."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from mergeslide.continual_model import ContinualModel
from mergeslide.feather_continual import FeatherGlobalClassifier


class LwsrReplayBuffer:
    """Reservoir memory of variable-length WSI bags for LWSR replay."""

    def __init__(self, buffer_size: int, seed: int = 0) -> None:
        self.buffer_size = int(buffer_size)
        self.num_seen_examples = 0
        self.rng = np.random.default_rng(seed)
        self.examples: List[Tuple[torch.Tensor, torch.Tensor]] = []
        self.labels: List[torch.Tensor] = []
        self.previous_dist_matrix: Optional[torch.Tensor] = None

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

        example = (features.detach().cpu(), coords.detach().long().cpu())
        label = labels.detach().long().cpu()

        if index == len(self.examples):
            self.examples.append(example)
            self.labels.append(label)
        else:
            self.examples[index] = example
            self.labels[index] = label

    def get_data(self, size: int, device: torch.device) -> List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        if self.is_empty():
            raise RuntimeError("Cannot sample from an empty LWSR buffer.")

        size = min(int(size), len(self.examples))
        indices = self.rng.choice(len(self.examples), size=size, replace=False)
        batch = []
        for raw_index in np.atleast_1d(indices):
            index = int(raw_index)
            features, coords = self.examples[index]
            batch.append(
                (
                    index,
                    features.to(device, non_blocking=True),
                    coords.to(device, non_blocking=True),
                    self.labels[index].to(device, non_blocking=True),
                )
            )
        return batch

    def iter_data(self, device: torch.device) -> List[Tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]]:
        return [
            (
                index,
                features.to(device, non_blocking=True),
                coords.to(device, non_blocking=True),
                self.labels[index].to(device, non_blocking=True),
            )
            for index, (features, coords) in enumerate(self.examples)
        ]

    def set_dist_matrix(self, dist_matrix: Optional[torch.Tensor]) -> None:
        self.previous_dist_matrix = None if dist_matrix is None else dist_matrix.detach().cpu()


class LwsrRetrievalLoss:
    """LWSR pair, classification, and distance-consistency losses."""

    def __init__(
        self,
        pair_loss_weight: float,
        ce_loss_weight: float,
        dc_loss_weight: float,
        num_classes: int,
    ) -> None:
        self.pair_loss_weight = float(pair_loss_weight)
        self.ce_loss_weight = float(ce_loss_weight)
        self.dc_loss_weight = float(dc_loss_weight)
        self.num_classes = int(num_classes)

    def pair_loss(self, logits: torch.Tensor, labels: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        _ = logits
        features = features.float()
        labels = labels.long()
        num = features.shape[0]
        feature_num = features.shape[1]
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float()
        label_similarity = 2 * torch.matmul(one_hot, one_hot.t()) - 1
        similarity_error = torch.matmul(features, features.t()) / feature_num - label_similarity
        regularizer = torch.sum(torch.abs(torch.abs(features) - 1)) / max(num * feature_num, 1)
        denominator = num * max(num - 1, 1)
        return torch.sum(similarity_error.pow(2)) / denominator + 0.3 * regularizer

    @staticmethod
    def ce_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits.float(), labels.long())

    @staticmethod
    def dist_consistency_loss(
        previous_dist_mat: Optional[torch.Tensor],
        current_dist_mat: torch.Tensor,
        return_idx: torch.Tensor,
    ) -> torch.Tensor:
        if previous_dist_mat is None or return_idx.numel() == 0:
            return torch.zeros((), device=current_dist_mat.device)
        indices = return_idx.detach().cpu()
        original = previous_dist_mat[indices][:, indices].to(current_dist_mat.device)
        return torch.mean((current_dist_mat - original) ** 2)

    def total(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        features: torch.Tensor,
        previous_dist_mat: Optional[torch.Tensor] = None,
        current_dist_mat: Optional[torch.Tensor] = None,
        return_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        loss_pair = self.pair_loss(logits, labels, features)
        loss_ce = self.ce_loss(logits, labels)
        loss_dcl = torch.zeros((), device=logits.device)
        if current_dist_mat is not None and return_idx is not None:
            loss_dcl = self.dist_consistency_loss(previous_dist_mat, current_dist_mat, return_idx)
        loss = (
            self.pair_loss_weight * loss_pair
            + self.ce_loss_weight * loss_ce
            + self.dc_loss_weight * loss_dcl
        )
        return loss, loss_pair, loss_ce, loss_dcl


class LwsrFEATHER(ContinualModel):
    """LWSR trainer for a FEATHER global classifier."""

    NAME = "lwsr"
    COMPATIBILITY = ("class-il",)

    def __init__(
        self,
        model: FeatherGlobalClassifier,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        buffer_size: int,
        minibatch_size: int,
        pair_loss_weight: float,
        ce_loss_weight: float,
        dc_loss_weight: float,
        num_classes: int,
        patch_size: int = 256,
        seed: int = 0,
    ) -> None:
        super().__init__(model=model, optimizer=optimizer, device=device)
        self.buffer = LwsrReplayBuffer(buffer_size, seed=seed)
        self.minibatch_size = max(1, int(minibatch_size))
        self.patch_size_value = int(patch_size)
        self.loss_fn = LwsrRetrievalLoss(
            pair_loss_weight=pair_loss_weight,
            ce_loss_weight=ce_loss_weight,
            dc_loss_weight=dc_loss_weight,
            num_classes=num_classes,
        )

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

    @staticmethod
    def calc_dist(features: torch.Tensor) -> torch.Tensor:
        return torch.cdist(features.float(), features.float(), p=2)

    def _forward(self, features: torch.Tensor, coords: torch.Tensor) -> Dict[str, torch.Tensor]:
        return self.model.forward_with_embedding(features, coords, self.patch_size)

    def observe(self, features: torch.Tensor, coords: torch.Tensor, labels: torch.Tensor) -> Dict[str, float]:
        """Run one LWSR stream update with replay and distance consistency."""
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        stream_outputs = self._forward(features, coords)
        loss_logits = stream_outputs["logits"]
        loss_features = stream_outputs["cls_token"]
        loss_labels = labels.long()
        return_idx = None
        current_dist = None

        if not self.buffer.is_empty() and self.current_task > 0:
            replay_logits, replay_features, replay_labels, replay_indices = [], [], [], []
            for index, buf_features, buf_coords, buf_labels in self.buffer.get_data(self.minibatch_size, self.device):
                outputs = self._forward(buf_features, buf_coords)
                replay_logits.append(outputs["logits"])
                replay_features.append(outputs["cls_token"])
                replay_labels.append(buf_labels)
                replay_indices.append(index)

            replay_logits_tensor = torch.cat(replay_logits, dim=0)
            replay_features_tensor = torch.cat(replay_features, dim=0)
            replay_labels_tensor = torch.cat(replay_labels, dim=0)
            loss_logits = torch.cat((loss_logits, replay_logits_tensor), dim=0)
            loss_features = torch.cat((loss_features, replay_features_tensor), dim=0)
            loss_labels = torch.cat((loss_labels, replay_labels_tensor), dim=0)
            return_idx = torch.tensor(replay_indices, dtype=torch.long)
            current_dist = self.calc_dist(replay_features_tensor)

        loss, loss_pair, loss_ce, loss_dcl = self.loss_fn.total(
            logits=loss_logits,
            labels=loss_labels,
            features=loss_features,
            previous_dist_mat=self.buffer.previous_dist_matrix,
            current_dist_mat=current_dist,
            return_idx=return_idx,
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Encountered a non-finite LWSR loss.")
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "loss_pair": float(loss_pair.detach().cpu()),
            "loss_ce": float(loss_ce.detach().cpu()),
            "loss_dcl": float(loss_dcl.detach().cpu()),
            "buffer_size": float(len(self.buffer)),
        }

    def _refresh_dist_matrix(self) -> None:
        if self.buffer.is_empty():
            self.buffer.set_dist_matrix(None)
            return

        self.model.eval()
        features_list = []
        with torch.no_grad():
            for _, features, coords, _ in self.buffer.iter_data(self.device):
                outputs = self._forward(features, coords)
                features_list.append(outputs["cls_token"])
        buffer_features = torch.cat(features_list, dim=0)
        self.buffer.set_dist_matrix(self.calc_dist(buffer_features))

    def end_task(
        self,
        train_loader,
        *,
        label_offset: int = 0,
        k: int = 0,
    ) -> int:
        """Store the completed task in memory and snapshot replay distances."""
        added = 0
        for features, coords, labels in train_loader:
            features, coords = self._sample_bag(features, coords, int(k))
            labels = labels.long() + int(label_offset)
            self.buffer.add_data(features=features, coords=coords, labels=labels)
            added += 1

        self._refresh_dist_matrix()
        super().end_task()
        return added
