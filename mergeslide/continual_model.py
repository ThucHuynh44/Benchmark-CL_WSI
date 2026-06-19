"""
Lightweight continual-learning lifecycle base for MergeSlide baselines.

This is intentionally smaller than ConSlide/Mammoth's framework class.  The
training scripts still own the loop, while method classes expose familiar
``begin_task``/``end_task`` hooks for task-boundary behavior.
"""

from typing import Any

import torch
import torch.nn as nn


class ContinualModel:
    """Minimal base class for continual method wrappers."""

    NAME = "continual_model"
    COMPATIBILITY = ()

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.current_task = 0

    def begin_task(self, *args: Any, **kwargs: Any) -> None:
        """Hook called before a task starts."""
        _ = args, kwargs

    def end_task(self, *args: Any, **kwargs: Any) -> None:
        """Hook called after a task finishes."""
        _ = args, kwargs
        self.current_task += 1
