"""
mergeslide/utils.py
Utility functions: metrics, seeding, and model-merging helpers.
"""

import copy
import os
import random
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    cohen_kappa_score,
    log_loss,
    roc_auc_score,
)
from tqdm import tqdm

# Type alias for model state dictionaries
StateDictType = Dict[str, torch.Tensor]
Tensor = torch.Tensor


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

def get_eval_metrics(
    targets_all: Union[List[int], np.ndarray],
    preds_all: Union[List[int], np.ndarray],
    probs_all: Optional[Union[List[float], np.ndarray]] = None,
    unique_classes: Optional[List[int]] = None,
    get_report: bool = True,
    prefix: str = "",
    roc_kwargs: Dict[str, Any] = {},
) -> Dict[str, Any]:
    """Calculate evaluation metrics.

    Args:
        targets_all: True target values.
        preds_all: Predicted target values.
        probs_all: Predicted probabilities for each class. Optional.
        unique_classes: Class labels to consider. Defaults to unique(targets_all).
        get_report: Whether to include the classification report.
        prefix: Prefix added to result keys.
        roc_kwargs: Additional keyword arguments for roc_auc_score.

    Returns:
        dict: Evaluation metrics keyed by '{prefix}/{metric}'.
    """
    unique_classes = unique_classes if unique_classes is not None else np.unique(targets_all)
    bacc = balanced_accuracy_score(targets_all, preds_all) if len(targets_all) > 1 else 0
    kappa = cohen_kappa_score(targets_all, preds_all, weights="quadratic")
    nw_kappa = cohen_kappa_score(targets_all, preds_all, weights="linear")
    acc = accuracy_score(targets_all, preds_all)
    cls_rep = classification_report(
        targets_all, preds_all, output_dict=True, zero_division=0, labels=unique_classes
    )

    eval_metrics = {
        f"{prefix}/acc": acc,
        f"{prefix}/bacc": bacc,
        f"{prefix}/kappa": kappa,
        f"{prefix}/nw_kappa": nw_kappa,
        f"{prefix}/weighted_f1": cls_rep["weighted avg"]["f1-score"],
    }

    if probs_all is not None and len(np.unique(targets_all)) > 1:
        try:
            loss = log_loss(targets_all, probs_all, labels=unique_classes)
            roc_auc = roc_auc_score(targets_all, probs_all, labels=unique_classes, **roc_kwargs)
        except ValueError:
            roc_auc = -1
            loss = -1
        eval_metrics[f"{prefix}/loss"] = loss
        eval_metrics[f"{prefix}/auroc"] = roc_auc

    return eval_metrics


# ---------------------------------------------------------------------------
# Seeding
# ---------------------------------------------------------------------------

def seed_torch(device: torch.device, seed: int = 0):
    """Set all random seeds for reproducibility.

    References:
        HIPT: https://github.com/mahmoodlab/HIPT/blob/master/2-Weakly-Supervised-Subtyping/main.py
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Bootstrap confidence intervals
# ---------------------------------------------------------------------------

def _merge_dict(dict1: dict, dict2: dict) -> dict:
    """Append values from dict2 into lists in dict1."""
    for k, v in dict2.items():
        if k in dict1:
            dict1[k].append(v)
        else:
            dict1[k] = [v]
    return dict1


def bootstrap(
    results_dict=None,
    preds_all=None,
    targets_all=None,
    probs_all=None,
    n: int = 1000,
    alpha: float = 0.95,
    format_as_str: bool = False,
) -> Tuple[dict, dict]:
    """Compute bootstrap confidence intervals for evaluation metrics.

    Args:
        results_dict: Optional pre-computed results dict with 'targets', 'preds'/'probs' keys.
        preds_all: Predicted labels (used if results_dict is None).
        targets_all: True labels (used if results_dict is None).
        probs_all: Predicted probabilities (used if results_dict is None).
        n: Number of bootstrap iterations.
        alpha: Confidence level.
        format_as_str: Unused; reserved for future string formatting.

    Returns:
        (mean_dict, std_dict): Mean and standard deviation of each metric.
    """
    if results_dict is not None:
        targets_all = results_dict['targets']
        probs_key = 'logits' if 'logits' in results_dict else 'probs'
        probs_all = results_dict.get(probs_key)
        preds_all = results_dict.get('preds')
        if preds_all is None:
            preds_all = np.argmax(probs_all, axis=1)

    num_classes = len(np.unique(targets_all))
    if probs_all is not None and len(probs_all.shape) == 2:
        probs_all = probs_all[:, 1] if num_classes == 2 else probs_all
    roc_kwargs = {'average': 'macro', 'multi_class': 'ovo'} if num_classes > 2 else {}

    all_scores: dict = {}
    for seed in tqdm(range(n)):
        np.random.seed(seed)
        bootstrap_ind = list(
            pd.Series(targets_all).sample(n=len(targets_all), replace=True, random_state=seed).index
        )
        collision = 0
        while len(np.unique(targets_all[bootstrap_ind])) != num_classes:
            bootstrap_ind = list(
                pd.Series(targets_all).sample(
                    n=len(targets_all), replace=True, random_state=seed + collision + n
                ).index
            )
            collision += 1
            if collision % 100 == 0:
                print(collision)

        results = get_eval_metrics(
            probs_all=probs_all[bootstrap_ind] if probs_all is not None else None,
            preds_all=preds_all[bootstrap_ind] if preds_all is not None else None,
            targets_all=targets_all[bootstrap_ind],
            roc_kwargs=roc_kwargs,
        )
        _merge_dict(all_scores, results)

    mean_dict = {k: np.array(v).mean() for k, v in all_scores.items()}
    std_dict = {k: np.array(v).std() for k, v in all_scores.items()}
    return mean_dict, std_dict


# ---------------------------------------------------------------------------
# State-dict / task-vector helpers
# ---------------------------------------------------------------------------

def state_dict_sub(
    a: StateDictType, b: StateDictType, strict: bool = True, device=None
) -> StateDictType:
    """Compute the element-wise difference a - b between two state dicts.

    Args:
        a: First state dict.
        b: Second state dict.
        strict: If True, assert that both dicts have the same keys.
        device: Optional device to move result tensors to.

    Returns:
        StateDictType: Element-wise difference.
    """
    if strict:
        assert set(a.keys()) == set(b.keys())
    diff: StateDictType = OrderedDict()
    for k in a:
        if k in b:
            diff[k] = a[k] - b[k]
            if device is not None:
                diff[k] = diff[k].to(device, non_blocking=True)
    return diff


def state_dict_to_vector(
    state_dict: StateDictType,
    remove_keys: Optional[List[str]] = None,
) -> Tensor:
    """Flatten a state dict into a single 1-D vector.

    Args:
        state_dict: The state dictionary to convert.
        remove_keys: Keys to exclude before flattening.

    Returns:
        torch.Tensor: Flattened parameter vector.
    """
    remove_keys = remove_keys or []
    shared_state_dict = copy.deepcopy(state_dict)
    for key in remove_keys:
        shared_state_dict.pop(key, None)
    sorted_state_dict = OrderedDict(sorted(shared_state_dict.items()))
    return nn.utils.parameters_to_vector(
        [v.reshape(-1) for v in sorted_state_dict.values()]
    )


def _svd(w: Tensor, full_matrices: bool = True) -> Tuple[Tensor, Tensor, Tensor]:
    """SVD wrapper that selects the LAPACK driver automatically.

    Args:
        w: Input tensor.
        full_matrices: Whether to compute full-sized U and V.

    Returns:
        (U, S, V) matrices.
    """
    u, s, vh = torch.linalg.svd(
        w, full_matrices=full_matrices, driver="gesvd" if w.is_cuda else None
    )
    return u, s, vh.T


def svd(
    w: Tensor, full_matrices: bool = True, accelerator=None
) -> Tuple[Tensor, Tensor, Tensor]:
    """SVD with optional device offloading.

    Args:
        w: Input tensor.
        full_matrices: Whether to compute full-sized U and V.
        accelerator: Device string to offload computation (e.g. 'cuda').

    Returns:
        (U, S, V) matrices on the original device.
    """
    if accelerator is None:
        return _svd(w, full_matrices=full_matrices)
    original_device = w.device
    u, s, v = _svd(w.to(accelerator))
    return u.to(original_device), s.to(original_device), v.to(original_device)


def frobenius_inner_product(w1: Tensor, w2: Tensor) -> Tensor:
    """Compute the Frobenius inner product tr(w1.T @ w2)."""
    return torch.trace(w1.T @ w2)


def is_leaf_module(module: nn.Module) -> bool:
    """Return True if the module has no child modules."""
    return len(list(module.children())) == 0


def get_task_vector_norm(model: StateDictType, pretrained_model: StateDictType) -> Tensor:
    """Compute the L2 norm of the task vector (model - pretrained_model).

    Args:
        model: Fine-tuned model state dict.
        pretrained_model: Pre-trained base model state dict.

    Returns:
        Scalar tensor: L2 norm of the task vector.
    """
    return torch.linalg.norm(
        state_dict_to_vector(state_dict_sub(model, pretrained_model))
    )


def get_task_vector_state_dict(
    model: StateDictType, pretrained_model: StateDictType
) -> Tensor:
    """Return the task vector as a flattened 1-D tensor.

    Args:
        model: Fine-tuned model state dict.
        pretrained_model: Pre-trained base model state dict.

    Returns:
        torch.Tensor: Flattened task vector.
    """
    return state_dict_to_vector(state_dict_sub(model, pretrained_model))
