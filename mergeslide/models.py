"""
mergeslide/models.py
Shared model components used across training and evaluation scripts.
"""

import numpy as np
import torch
import torch.nn as nn


class CustomSequential(nn.Module):
    """Wraps TITAN's vision encoder with an MLP classification head."""

    def __init__(self, model, mlp):
        super().__init__()
        self.backbone = model.vision_encoder
        self.mlp = mlp

    def forward(self, features, coords, ps):
        x = self.backbone(features, coords, ps)
        x = self.mlp(x)
        return x


class EarlyStopping:
    """Stop training when validation loss stops improving.

    Args:
        patience (int): Number of epochs to wait after last improvement.
        min_delta (float): Minimum change to qualify as an improvement.
        verbose (bool): If True, prints a message for each patience check.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = False):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_model_weights = None

    def __call__(self, val_loss: float, model: nn.Module):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.best_model_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}


def cosine_lr(optimizer, base_lr: float, warmup_length: int, steps: int):
    """Cosine LR schedule with linear warmup.

    Copied from:
    https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/scheduler.py
    """

    def _warmup_lr(base_lr, warmup_length, step):
        return base_lr * (step + 1) / warmup_length

    def _assign_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = new_lr * param_group["lr_scale"]
            else:
                param_group["lr"] = new_lr

    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        _assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster


def create_mlp(
    in_dim: int,
    out_dim: int,
    hid_dims: list = [],
    act: nn.Module = None,
    dropout: float = 0.0,
    end_with_fc: bool = True,
) -> nn.Sequential:
    """Build an MLP with optional hidden layers.

    Args:
        in_dim: Input feature dimension.
        out_dim: Output feature dimension.
        hid_dims: List of hidden layer sizes. Empty means single linear layer.
        act: Activation module. Defaults to ReLU.
        dropout: Dropout probability applied after each hidden layer.
        end_with_fc: If False, append activation + dropout after final layer.

    Returns:
        nn.Sequential MLP.
    """
    if act is None:
        act = nn.ReLU()
    layers = []
    for hid_dim in hid_dims:
        layers.extend([nn.Linear(in_dim, hid_dim), act, nn.Dropout(dropout)])
        in_dim = hid_dim
    layers.append(nn.Linear(in_dim, out_dim))
    if not end_with_fc:
        layers.extend([act, nn.Dropout(dropout)])
    return nn.Sequential(*layers)


def pad_numpy_arrays(arrays, pad_value: float = 0.0):
    """Pad a list of arrays with varying shapes to the same shape and stack.

    Args:
        arrays: List of NumPy arrays with potentially different shapes.
        pad_value: Constant value used for padding.

    Returns:
        np.ndarray of shape (len(arrays), *max_shape).
    """
    max_dim = max(arr.ndim for arr in arrays)
    arrays = [arr.reshape((1,) * (max_dim - arr.ndim) + arr.shape) for arr in arrays]
    max_shape = np.max([arr.shape for arr in arrays], axis=0)
    padded = []
    for arr in arrays:
        pad_width = [(0, max_shape[i] - arr.shape[i]) for i in range(max_dim)]
        padded.append(np.pad(arr, pad_width=pad_width, mode='constant', constant_values=pad_value))
    return np.stack(padded)
