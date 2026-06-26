"""
Train FEATHER continual-learning baselines on the MergeSlide task stream.

Examples:
    python scripts/train_feather_continual.py --method derpp
    python scripts/train_feather_continual.py --method agem --num_folds 1
    python scripts/train_feather_continual.py --method er_ace --num_tasks 3
"""

import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from tqdm import tqdm

from mergeslide.agem import AgemTITAN
from mergeslide.datasets import Sequential_Generic_MIL_Dataset
from mergeslide.derpp import DerppTITAN
from mergeslide.er_ace import ErAceTITAN
from mergeslide.ewc_on import EwcOn
from mergeslide.feather_continual import FeatherGlobalClassifier
from mergeslide.feather_models import (
    DEFAULT_FEATURE_DIM,
    DEFAULT_FEATHER_MODEL_NAME,
    create_feather_model,
    freeze_feather_backbone,
    prepare_hf_token_env,
)
from mergeslide.lwsr import LwsrFEATHER
from mergeslide.micil import MicilFEATHER
from mergeslide.models import cosine_lr
from mergeslide.lwf import Lwf
from mergeslide.utils import seed_torch


REPO_ROOT = Path(__file__).resolve().parent.parent
FEATHER_CONFIG = REPO_ROOT / "configs" / "feather.yaml"
CONTINUAL_CONFIG = REPO_ROOT / "configs" / "feather_continual.yaml"
METHODS = ("derpp", "agem", "er_ace", "ewc_on", "lwf", "lwsr", "micil")


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as handle:
        return yaml.safe_load(handle) or {}


def _cfg_value(args, cfg: dict, name: str, default):
    value = getattr(args, name)
    return value if value is not None else cfg.get(name, default)


def _class_offsets(num_classes: List[int]) -> List[int]:
    offsets, total = [], 0
    for n_classes in num_classes:
        offsets.append(total)
        total += int(n_classes)
    return offsets


def _memory_samples_for_task(buffer_size: int, num_tasks: int, task_id: int) -> int:
    if buffer_size <= 0 or num_tasks <= 0:
        return 0
    base = buffer_size // num_tasks
    remainder = buffer_size % num_tasks
    return base + int(task_id < remainder)


def _sample_patches(
    features: torch.Tensor,
    coords: torch.Tensor,
    device: torch.device,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k > 0 and features.shape[0] > k:
        indices = torch.randperm(features.shape[0])[:k]
        features = features[indices]
        coords = coords[indices]
    return (
        features.to(device, non_blocking=True),
        coords.long().to(device, non_blocking=True),
    )


def _build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float):
    named_parameters = [(name, param) for name, param in model.named_parameters() if param.requires_grad]
    if not named_parameters:
        raise RuntimeError("FEATHER continual model has no trainable parameters.")
    no_decay = lambda name, param: param.ndim < 2 or any(
        token in name.lower() for token in ("bn", "ln", "bias")
    )
    return torch.optim.AdamW(
        [
            {"params": [p for n, p in named_parameters if no_decay(n, p)], "weight_decay": 0.0},
            {"params": [p for n, p in named_parameters if not no_decay(n, p)], "weight_decay": weight_decay},
        ],
        lr=lr,
    )


def _train_one_task(
    trainer,
    train_loader,
    *,
    method_name: str,
    task_id: int,
    label_offset: int,
    num_epochs: int,
    scheduler,
    device: torch.device,
    k: int,
    use_wandb: bool,
) -> Dict[str, float]:
    step = 0
    last_stats: Dict[str, float] = {}
    for epoch in tqdm(range(num_epochs), desc=f"task {task_id}", leave=False):
        epoch_loss = 0.0
        projected_steps = 0.0
        for features, coords, labels in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            scheduler(step)
            features, coords = _sample_patches(features, coords, device, k)
            global_labels = labels.to(device, non_blocking=True).long() + label_offset
            last_stats = trainer.observe(features, coords, global_labels)
            epoch_loss += last_stats["loss"]
            projected_steps += last_stats.get("projected", 0.0)
            step += 1

        avg_loss = epoch_loss / max(len(train_loader), 1)
        log_values = {
            "train/task_id": task_id,
            "train/epoch": epoch,
            "train/avg_loss": avg_loss,
            "train/lr": float(trainer.optimizer.param_groups[0]["lr"]),
            **{f"train/{key}": value for key, value in last_stats.items()},
        }
        if method_name == "feather_agem":
            log_values["train/projection_rate"] = projected_steps / max(len(train_loader), 1)
        if method_name == "feather_micil":
            tqdm.write(
                f"[FEATHER {method_name}] task={task_id} epoch={epoch} "
                f"avg_loss={avg_loss:.4f} "
                f"total_loss={float(last_stats.get('total_loss', avg_loss)):.4f} "
                f"ce_loss={float(last_stats.get('ce_loss', 0.0)):.4f} "
                f"kd_loss={float(last_stats.get('kd_loss', 0.0)):.4f} "
                f"em_loss={float(last_stats.get('em_loss', 0.0)):.4f} "
                f"buffer_len={int(last_stats.get('buffer_len', last_stats.get('buffer_size', 0)))} "
                f"old_classes={int(last_stats.get('old_classes', 0))} "
                f"seen_classes={int(last_stats.get('seen_classes', 0))} "
                f"logits_shape=({int(last_stats.get('logits_batch', 0))}, "
                f"{int(last_stats.get('logits_dim', 0))}) "
                f"embedding_shape=({int(last_stats.get('embedding_batch', 0))}, "
                f"{int(last_stats.get('embedding_dim', 0))})"
            )
        else:
            tqdm.write(
                f"[FEATHER {method_name}] task={task_id} epoch={epoch} "
                f"loss={avg_loss:.4f} buffer={int(last_stats.get('buffer_size', 0))}"
            )
        if use_wandb:
            import wandb
            wandb.log(log_values)
    return last_stats


def _balanced_accuracy(targets: np.ndarray, preds: np.ndarray) -> float:
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
        return float(balanced_accuracy_score(targets, preds))


def _safe_auroc(targets: np.ndarray, probabilities: np.ndarray) -> float:
    """Compute task-local AUROC without assuming every seen class is present."""
    present_classes = np.unique(targets).astype(int)
    if len(present_classes) < 2:
        return np.nan
    task_probabilities = probabilities[:, present_classes]
    normalizer = task_probabilities.sum(axis=1, keepdims=True)
    task_probabilities = np.divide(
        task_probabilities,
        normalizer,
        out=np.zeros_like(task_probabilities),
        where=normalizer > 0,
    )
    try:
        if len(present_classes) == 2:
            return float(
                roc_auc_score(
                    (targets == present_classes[1]).astype(np.int64),
                    task_probabilities[:, 1],
                )
            )
        return float(
            roc_auc_score(
                targets,
                task_probabilities,
                labels=present_classes,
                multi_class="ovo",
                average="macro",
            )
        )
    except ValueError:
        return np.nan


def _evaluation_metrics(
    targets: np.ndarray,
    preds: np.ndarray,
    probabilities: np.ndarray,
    loss: float,
) -> Dict[str, float]:
    return {
        "loss": float(loss),
        "acc": float(accuracy_score(targets, preds)),
        "bacc": _balanced_accuracy(targets, preds),
        "macro_f1": float(f1_score(targets, preds, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, preds, average="weighted", zero_division=0)),
        "auroc": _safe_auroc(targets, probabilities),
        "kappa": float(cohen_kappa_score(targets, preds)),
        "n": int(len(targets)),
    }


def _feature_path(dataset, slide_id: str, row=None) -> str:
    if row is not None:
        for column in ("feature_path", "features_path", "h5_path"):
            if column in row and pd.notna(row[column]) and str(row[column]).strip():
                return str(row[column])

    data_dir = str(getattr(dataset, "data_dir", ""))
    if not data_dir:
        return ""
    stem = str(slide_id).strip()
    if stem.endswith(".svs"):
        stem = stem[:-4]
    candidates = [
        os.path.join(data_dir, "h5_files", f"{stem}.h5"),
        os.path.join(data_dir, "features_conch_v15", f"{stem}.h5"),
        os.path.join(data_dir, f"{stem}.h5"),
        os.path.join(data_dir, "pt_files", f"{stem}.pt"),
        os.path.join(data_dir, f"{stem}.pt"),
    ]
    return next((path for path in candidates if os.path.exists(path)), candidates[0])


def _slide_metadata(dataset, sample_index: int) -> Dict[str, str]:
    """Recover test-slide metadata without changing the existing MIL loaders."""
    if hasattr(dataset, "slide_data"):
        row = dataset.slide_data.iloc[int(sample_index)]
        slide_id = str(row.get("slide_id", ""))
        patient_id = row.get("case_id", row.get("patient_id", ""))
        patient_id = "" if pd.isna(patient_id) else str(patient_id)
        return {
            "slide_id": slide_id,
            "patient_id": patient_id,
            "feature_path": _feature_path(dataset, slide_id, row=row),
        }

    if hasattr(dataset, "data"):
        slide_id = str(dataset.data[int(sample_index)])
        return {
            "slide_id": slide_id,
            "patient_id": "",
            "feature_path": _feature_path(dataset, slide_id),
        }
    return {"slide_id": "", "patient_id": "", "feature_path": ""}


def _evaluate_task(
    model: FeatherGlobalClassifier,
    test_loader,
    *,
    method_name: str,
    fold_id: int,
    after_task: int,
    eval_task: int,
    task_name: str,
    label_offset: int,
    task_num_classes: int,
    seen_classes: int,
    total_classes: int,
    device: torch.device,
    k: int,
    patch_size: int,
    mask_unseen: bool,
    seed: int,
    confusion_matrix_dir: str,
) -> Tuple[Dict[str, float], List[dict]]:
    model.eval()
    start = time.perf_counter()
    loss_sum = 0.0
    preds_all, targets_all, probs_all = [], [], []
    prediction_rows: List[dict] = []
    patch_size_tensor = torch.tensor(patch_size, dtype=torch.int32, device=device)
    with torch.no_grad():
        for sample_index, (features, coords, labels) in enumerate(tqdm(test_loader, desc="eval", leave=False)):
            num_patches_total = int(features.shape[0])
            features, coords = _sample_patches(features, coords, device, k)
            num_patches_used = int(features.shape[0])
            targets = labels.to(device, non_blocking=True).long() + label_offset
            raw_logits = model(features, coords, patch_size_tensor).float()
            logits = raw_logits
            if mask_unseen:
                logits = logits[:, :seen_classes]
            loss_sum += float(F.cross_entropy(logits, targets, reduction="sum").cpu())
            probabilities = F.softmax(logits, dim=1)
            preds = logits.argmax(1)
            preds_all.append(preds.cpu().numpy())
            targets_all.append(targets.cpu().numpy())
            probs_all.append(probabilities.cpu().numpy())

            metadata = _slide_metadata(test_loader.dataset, sample_index)
            raw_logits_values = raw_logits[0].cpu().numpy()
            probability_values = probabilities[0].cpu().numpy()
            target_global = int(targets[0].item())
            pred_global = int(preds[0].item())
            is_local_prediction = label_offset <= pred_global < label_offset + task_num_classes
            prediction_row = {
                "method": method_name,
                "fold": fold_id,
                "mode": "class-il-seen" if mask_unseen else "class-il-all",
                "after_task": after_task,
                "eval_task": eval_task,
                **metadata,
                "task_name": task_name,
                "split": "test",
                "y_true_local": target_global - label_offset,
                "y_true_global": target_global,
                "y_pred_global": pred_global,
                "y_pred_local": pred_global - label_offset if is_local_prediction else "",
                "correct": int(pred_global == target_global),
                "num_patches_total": num_patches_total,
                "num_patches_used": num_patches_used,
                "k": k,
                "seed": seed,
            }
            for class_id in range(total_classes):
                prediction_row[f"prob_{class_id}"] = (
                    float(probability_values[class_id]) if class_id < len(probability_values) else 0.0
                )
                prediction_row[f"logit_{class_id}"] = float(raw_logits_values[class_id])
            prediction_rows.append(prediction_row)

    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    probabilities = np.concatenate(probs_all)
    matrix_labels = np.arange(seen_classes if mask_unseen else total_classes)
    matrix_path = os.path.join(
        confusion_matrix_dir,
        f"fold_{fold_id}_after_{after_task}_eval_{eval_task}.csv",
    )
    os.makedirs(confusion_matrix_dir, exist_ok=True)
    matrix = confusion_matrix(targets, preds, labels=matrix_labels)
    pd.DataFrame(
        matrix,
        index=[f"true_{label}" for label in matrix_labels],
        columns=[f"pred_{label}" for label in matrix_labels],
    ).to_csv(matrix_path)
    metrics = _evaluation_metrics(
        targets,
        preds,
        probabilities,
        loss=loss_sum / max(len(targets), 1),
    )
    metrics["confusion_matrix_path"] = matrix_path
    metrics["eval_time_sec"] = time.perf_counter() - start
    return metrics, prediction_rows


def _save_checkpoint(
    path: str,
    model: FeatherGlobalClassifier,
    trainer,
    *,
    method: str,
    model_name: str,
    fold_id: int,
    task_id: int,
    num_classes: List[int],
    run_config: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "model_name": model_name,
            "method": method,
            "fold": fold_id,
            "task_id": task_id,
            "num_classes": num_classes,
            "total_classes": sum(num_classes),
            "buffer_size": len(trainer.buffer),
            "forward_mode": model._forward_mode,
            "run_config": run_config,
        },
        path,
    )


def _write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _prepare_csv(path: str, fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def _append_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    if not rows:
        return
    with open(path, "a", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writerows(rows)


def _finite_mean(values: List[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    return float(np.mean(finite_values)) if finite_values else np.nan


def _finite_std(values: List[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(value)]
    return float(np.std(finite_values)) if finite_values else np.nan


def _continual_scores(final_rows: List[dict], all_rows: List[dict]) -> Tuple[float, float]:
    if len(final_rows) <= 1:
        return np.nan, np.nan
    final_after_task = int(final_rows[0]["after_task"])
    final_by_task = {int(row["eval_task"]): float(row["acc"]) for row in final_rows}
    diagonal = {
        int(row["eval_task"]): float(row["acc"])
        for row in all_rows
        if int(row["after_task"]) == int(row["eval_task"])
    }
    bwt_values = [
        final_by_task[task_id] - diagonal[task_id]
        for task_id in range(final_after_task)
        if task_id in final_by_task and task_id in diagonal
    ]
    forgetting_values = []
    for task_id in range(final_after_task):
        trajectory = [
            float(row["acc"])
            for row in all_rows
            if int(row["eval_task"]) == task_id and int(row["after_task"]) >= task_id
        ]
        if trajectory and task_id in final_by_task:
            forgetting_values.append(max(trajectory) - final_by_task[task_id])
    return _finite_mean(bwt_values), _finite_mean(forgetting_values)


def _build_fold_summary(
    method_name: str,
    rows: List[dict],
    training_times: Dict[int, float],
) -> List[dict]:
    summaries = []
    for fold_id in sorted({int(row["fold"]) for row in rows}):
        fold_rows = [row for row in rows if int(row["fold"]) == fold_id]
        final_after_task = max(int(row["after_task"]) for row in fold_rows)
        final_rows = [row for row in fold_rows if int(row["after_task"]) == final_after_task]
        bwt, fgt = _continual_scores(final_rows, fold_rows)
        total_eval_time = float(sum(float(row["eval_time_sec"]) for row in fold_rows))
        summaries.append({
            "method": method_name,
            "fold": fold_id,
            "mode": final_rows[0]["mode"],
            "final_acc": _finite_mean([row["acc"] for row in final_rows]),
            "final_bacc": _finite_mean([row["bacc"] for row in final_rows]),
            "macro_f1": _finite_mean([row["macro_f1"] for row in final_rows]),
            "weighted_f1": _finite_mean([row["weighted_f1"] for row in final_rows]),
            "auroc": _finite_mean([row["auroc"] for row in final_rows]),
            "mACC": _finite_mean([row["acc"] for row in final_rows]),
            "BWT": bwt,
            "FGT": fgt,
            "training_time": float(training_times.get(fold_id, 0.0)),
            "total_eval_time": total_eval_time,
            "inference_time_per_task": total_eval_time / max(len(fold_rows), 1),
        })
    return summaries


def _with_fold_aggregates(rows: List[dict], fieldnames: List[str]) -> List[dict]:
    if not rows:
        return rows
    numeric_fields = [field for field in fieldnames if field not in ("method", "fold", "mode")]
    aggregate_rows = []
    for label, reducer in (("mean", _finite_mean), ("std", _finite_std)):
        aggregate_rows.append({
            "method": rows[0]["method"],
            "fold": label,
            "mode": rows[0]["mode"],
            **{field: reducer([row[field] for row in rows]) for field in numeric_fields},
        })
    return rows + aggregate_rows


def _build_task_summary(method_name: str, rows: List[dict]) -> List[dict]:
    summaries = []
    for task_id in sorted({int(row["eval_task"]) for row in rows}):
        task_rows = [row for row in rows if int(row["eval_task"]) == task_id]
        final_rows = [
            row for row in task_rows
            if int(row["after_task"]) == max(
                int(candidate["after_task"])
                for candidate in rows
                if int(candidate["fold"]) == int(row["fold"])
            )
        ]
        if not final_rows:
            continue
        task_name = final_rows[0]["task_name"]
        summaries.append({
            "method": method_name,
            "task": task_id,
            "task_name": task_name,
            "mean_acc": _finite_mean([row["acc"] for row in final_rows]),
            "std_acc": _finite_std([row["acc"] for row in final_rows]),
            "mean_bacc": _finite_mean([row["bacc"] for row in final_rows]),
            "std_bacc": _finite_std([row["bacc"] for row in final_rows]),
            "mean_macro_f1": _finite_mean([row["macro_f1"] for row in final_rows]),
            "std_macro_f1": _finite_std([row["macro_f1"] for row in final_rows]),
            "mean_auroc": _finite_mean([row["auroc"] for row in final_rows]),
            "std_auroc": _finite_std([row["auroc"] for row in final_rows]),
            "n_test": int(sum(int(row["n"]) for row in final_rows)),
        })
    return summaries


def _repo_commit() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser(description="FEATHER continual-learning baselines")
    parser.add_argument("--method", choices=METHODS, required=True)
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--buffer_size", type=int, default=None)
    parser.add_argument("--alpha", type=float, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--e_lambda", type=float, default=None)
    parser.add_argument("--gamma", type=float, default=None)
    parser.add_argument("--softmax_temp", type=float, default=None)
    parser.add_argument("--minibatch_size", type=int, default=None)
    parser.add_argument("--pair_loss_weight", type=float, default=None)
    parser.add_argument("--ce_loss_weight", type=float, default=None)
    parser.add_argument("--dc_loss_weight", type=float, default=None)
    parser.add_argument("--kd_loss_weight", type=float, default=None)
    parser.add_argument("--em_loss_weight", type=float, default=None)
    parser.add_argument("--weight_norm", dest="weight_norm", action="store_true", default=None)
    parser.add_argument("--no_weight_norm", dest="weight_norm", action="store_false", default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--task_free", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--no_eval_after_task", action="store_true")
    parser.add_argument("--no_mask_unseen_eval", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    feather_cfg = _load_yaml(FEATHER_CONFIG).get("feather", {})
    method_cfg = _load_yaml(CONTINUAL_CONFIG).get(args.method, {})
    model_name = str(args.model_name or feather_cfg.get("model_name", DEFAULT_FEATHER_MODEL_NAME))
    num_epochs = int(_cfg_value(args, method_cfg, "num_epochs", 10))
    lr = float(_cfg_value(args, method_cfg, "lr", 1e-5))
    weight_decay = float(_cfg_value(args, method_cfg, "weight_decay", 1e-4))
    buffer_size = int(_cfg_value(args, method_cfg, "buffer_size", 30))
    alpha = float(_cfg_value(args, method_cfg, "alpha", 0.2))
    beta = float(_cfg_value(args, method_cfg, "beta", 0.2))
    e_lambda = float(_cfg_value(args, method_cfg, "e_lambda", 1.0))
    gamma = float(_cfg_value(args, method_cfg, "gamma", 0.9))
    softmax_temp = float(_cfg_value(args, method_cfg, "softmax_temp", 2.0))
    minibatch_size = int(_cfg_value(args, method_cfg, "minibatch_size", 1))
    pair_loss_weight = float(_cfg_value(args, method_cfg, "pair_loss_weight", 0.5))
    ce_loss_weight = float(_cfg_value(args, method_cfg, "ce_loss_weight", 0.5))
    dc_loss_weight = float(_cfg_value(args, method_cfg, "dc_loss_weight", 0.5))
    kd_loss_weight = float(_cfg_value(args, method_cfg, "kd_loss_weight", 10.0))
    em_loss_weight = float(_cfg_value(args, method_cfg, "em_loss_weight", 1.0))
    save_dir = str(_cfg_value(args, method_cfg, "save_dir", f"./checkpoints/feather_{args.method}"))
    num_folds = int(_cfg_value(args, method_cfg, "num_folds", 10))
    num_workers = int(_cfg_value(args, method_cfg, "num_workers", feather_cfg.get("num_workers", 0)))
    k = int(_cfg_value(args, method_cfg, "k", feather_cfg.get("k", 400)))
    patch_size = int(args.patch_size if args.patch_size is not None else feather_cfg.get("patch_size", 256))
    seed = int(_cfg_value(args, method_cfg, "seed", 0))
    from_pretrained = bool(feather_cfg.get("from_pretrained", True)) and not args.no_pretrained
    freeze_backbone = bool(args.freeze_backbone or method_cfg.get("freeze_backbone", False))
    task_free = bool(args.task_free or method_cfg.get("task_free", False))
    weight_norm = bool(args.weight_norm if args.weight_norm is not None else method_cfg.get("weight_norm", False))
    eval_after_task = bool((not args.no_eval_after_task) and method_cfg.get("eval_after_task", True))
    mask_unseen_eval = bool((not args.no_mask_unseen_eval) and method_cfg.get("mask_unseen_eval", True))
    use_wandb = (args.use_wandb or feather_cfg.get("use_wandb", False)) and not args.disable_wandb
    if use_wandb:
        try:
            import wandb  # noqa: F401
        except ImportError:
            warnings.warn("wandb package not found. Disabling wandb tracking.")
            use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, seed)
    prepare_hf_token_env(feather_cfg.get("hf_token"))
    os.makedirs(save_dir, exist_ok=True)

    seq_dataset = Sequential_Generic_MIL_Dataset(config_path=str(FEATHER_CONFIG))
    configured_tasks = int(method_cfg.get("num_tasks", len(seq_dataset.num_classes)))
    num_tasks = int(args.num_tasks if args.num_tasks is not None else configured_tasks)
    num_classes = seq_dataset.num_classes[:num_tasks]
    offsets = _class_offsets(num_classes)
    total_classes = sum(num_classes)
    prefix = f"feather_{args.method}"
    method_name = prefix
    task_names = list(seq_dataset.task_order[:num_tasks])
    run_config = {
        "method": args.method,
        "model_name": model_name,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "buffer_size": buffer_size,
        "alpha": alpha if args.method in ("derpp", "lwf") else None,
        "beta": beta if args.method == "derpp" else None,
        "e_lambda": e_lambda if args.method == "ewc_on" else None,
        "gamma": gamma if args.method == "ewc_on" else None,
        "softmax_temp": softmax_temp if args.method == "lwf" else None,
        "minibatch_size": minibatch_size if args.method in ("lwsr", "micil") else None,
        "pair_loss_weight": pair_loss_weight if args.method == "lwsr" else None,
        "ce_loss_weight": ce_loss_weight if args.method in ("lwsr", "micil") else None,
        "dc_loss_weight": dc_loss_weight if args.method == "lwsr" else None,
        "kd_loss_weight": kd_loss_weight if args.method == "micil" else None,
        "em_loss_weight": em_loss_weight if args.method == "micil" else None,
        "weight_norm": weight_norm if args.method == "micil" else None,
        "num_tasks": num_tasks,
        "k": k,
        "patch_size": patch_size,
        "from_pretrained": from_pretrained,
        "freeze_backbone": freeze_backbone,
        "task_free": task_free if args.method == "er_ace" else None,
        "mask_unseen_eval": mask_unseen_eval,
    }
    eval_matrix_fields = [
        "method", "fold", "mode", "after_task", "eval_task", "task_name", "acc", "bacc",
        "macro_f1", "weighted_f1", "auroc", "kappa", "loss", "n", "eval_time_sec",
        "confusion_matrix_path",
    ]
    prediction_fields = [
        "method", "fold", "mode", "after_task", "eval_task", "slide_id", "patient_id",
        "task_name", "split", "feature_path", "y_true_local", "y_true_global", "y_pred_global",
        "y_pred_local", "correct",
    ]
    prediction_fields += [f"prob_{class_id}" for class_id in range(total_classes)]
    prediction_fields += [f"logit_{class_id}" for class_id in range(total_classes)]
    prediction_fields += ["num_patches_total", "num_patches_used", "k", "seed"]
    per_fold_fields = [
        "method", "fold", "mode", "final_acc", "final_bacc", "macro_f1", "weighted_f1", "auroc",
        "mACC", "BWT", "FGT", "training_time", "total_eval_time", "inference_time_per_task",
    ]
    per_task_fields = [
        "method", "task", "task_name", "mean_acc", "std_acc", "mean_bacc", "std_bacc",
        "mean_macro_f1", "std_macro_f1", "mean_auroc", "std_auroc", "n_test",
    ]
    eval_matrix_path = os.path.join(save_dir, "eval_matrix.csv")
    predictions_path = os.path.join(save_dir, "per_slide_predictions.csv")
    confusion_matrix_dir = os.path.join(save_dir, "confusion_matrices")
    _prepare_csv(eval_matrix_path, eval_matrix_fields)
    _prepare_csv(predictions_path, prediction_fields)
    manifest = {
        "method": method_name,
        "backbone": "FEATHER",
        "model_name": model_name,
        "feature_type": "CONCH_v1.5",
        "feature_dim": DEFAULT_FEATURE_DIM,
        "num_tasks": num_tasks,
        "num_folds": num_folds,
        "task_order": task_names,
        "num_classes_per_task": [int(value) for value in num_classes],
        "k": k,
        "patch_size": patch_size,
        "magnification": int(feather_cfg.get("magnification", 20)),
        "seed": seed,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "freeze_backbone": freeze_backbone,
        "from_pretrained": from_pretrained,
        "repo_commit": _repo_commit(),
        "command": "python " + shlex.join([sys.argv[0], *sys.argv[1:]]),
        "method_config": run_config,
    }
    with open(os.path.join(save_dir, "run_manifest.json"), "w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)

    eval_rows: List[dict] = []
    fold_training_times: Dict[int, float] = {}

    for fold_id in tqdm(range(num_folds), desc="folds"):
        fold_dir = os.path.join(save_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        fold_training_time = 0.0
        if use_wandb:
            import wandb
            wandb.init(
                project=feather_cfg.get("wandb_project", "MergeSlide-FEATHER"),
                entity=feather_cfg.get("wandb_entity"),
                group=f"feather_{args.method}",
                job_type="continual_train",
                name=f"feather_{args.method}_fold_{fold_id}",
                config={**run_config, "fold": fold_id},
                reinit=True,
            )

        base_model = create_feather_model(
            model_name,
            num_classes=total_classes,
            from_pretrained=from_pretrained,
        )
        if freeze_backbone:
            frozen, trainable = freeze_feather_backbone(base_model, num_classes=total_classes)
            print(f"[FEATHER] frozen_params={frozen} trainable_params={trainable}")
        model = FeatherGlobalClassifier(base_model, num_classes=total_classes).to(device)
        optimizer = _build_optimizer(model, lr=lr, weight_decay=weight_decay)

        if args.method == "derpp":
            trainer = DerppTITAN(
                model=model,
                optimizer=optimizer,
                device=device,
                buffer_size=buffer_size,
                alpha=alpha,
                beta=beta,
                patch_size=patch_size,
                seed=seed + fold_id,
            )
        elif args.method == "agem":
            trainer = AgemTITAN(
                model=model,
                optimizer=optimizer,
                device=device,
                buffer_size=buffer_size,
                patch_size=patch_size,
                seed=seed + fold_id,
            )
        elif args.method == "er_ace":
            trainer = ErAceTITAN(
                model=model,
                optimizer=optimizer,
                device=device,
                buffer_size=buffer_size,
                patch_size=patch_size,
                seed=seed + fold_id,
                use_amp=False,
                task_free=task_free,
            )
        elif args.method == "ewc_on":
            trainer = EwcOn(
                model=model,
                optimizer=optimizer,
                device=device,
                e_lambda=e_lambda,
                gamma=gamma,
                patch_size=patch_size,
            )
        elif args.method == "lwf":
            trainer = Lwf(
                model=model,
                optimizer=optimizer,
                device=device,
                alpha=alpha,
                softmax_temp=softmax_temp,
                patch_size=patch_size,
            )
        elif args.method == "lwsr":
            trainer = LwsrFEATHER(
                model=model,
                optimizer=optimizer,
                device=device,
                buffer_size=buffer_size,
                minibatch_size=minibatch_size,
                pair_loss_weight=pair_loss_weight,
                ce_loss_weight=ce_loss_weight,
                dc_loss_weight=dc_loss_weight,
                num_classes=total_classes,
                patch_size=patch_size,
                seed=seed + fold_id,
            )
        elif args.method == "micil":
            trainer = MicilFEATHER(
                model=model,
                optimizer=optimizer,
                device=device,
                buffer_size=buffer_size,
                minibatch_size=minibatch_size,
                ce_loss_weight=ce_loss_weight,
                kd_loss_weight=kd_loss_weight,
                em_loss_weight=em_loss_weight,
                weight_norm=weight_norm,
                num_classes=total_classes,
                patch_size=patch_size,
                seed=seed + fold_id,
            )
        else:
            raise ValueError(f"Unsupported FEATHER continual method: {args.method}")

        for task_id in range(num_tasks):
            start = time.time()
            train_loader, _, _ = seq_dataset.get_data_loaders(
                fold_id,
                task_id,
                num_workers=num_workers,
            )
            seen_classes = sum(num_classes[:task_id + 1])
            if args.method == "lwf":
                trainer.begin_task(
                    train_loader,
                    label_offset=offsets[task_id],
                    past_classes=offsets[task_id],
                    seen_classes=seen_classes,
                    warmup_epochs=num_epochs,
                    warmup_lr=lr,
                    k=k,
                )
            elif args.method == "micil":
                trainer.begin_task(
                    past_classes=offsets[task_id],
                    seen_classes=seen_classes,
                )
            steps = max(1, len(train_loader) * num_epochs)
            scheduler = cosine_lr(
                optimizer=optimizer,
                base_lr=lr,
                warmup_length=max(1, int(steps * 0.1)),
                steps=steps,
            )
            _train_one_task(
                trainer,
                train_loader,
                method_name=method_name,
                task_id=task_id,
                label_offset=offsets[task_id],
                num_epochs=num_epochs,
                scheduler=scheduler,
                device=device,
                k=k,
                use_wandb=use_wandb,
            )

            if args.method == "agem":
                quota = _memory_samples_for_task(buffer_size, num_tasks, task_id)
                added = trainer.end_task(
                    train_loader=train_loader,
                    label_offset=offsets[task_id],
                    k=k,
                    samples_per_task=quota,
                )
                print(f"[FEATHER agem] added {added}/{quota} WSI to replay buffer")
            elif args.method == "er_ace":
                trainer.end_task()
            elif args.method == "ewc_on":
                fisher_batches = trainer.end_task(
                    train_loader,
                    label_offset=offsets[task_id],
                    k=k,
                )
                print(f"[FEATHER ewc_on] Fisher estimated from {fisher_batches} WSI bags")
            elif args.method == "lwf":
                trainer.end_task()
            elif args.method == "lwsr" and task_id < num_tasks - 1:
                added = trainer.end_task(
                    train_loader,
                    label_offset=offsets[task_id],
                    k=k,
                )
                print(f"[FEATHER lwsr] added {added} WSI to replay buffer")
            elif args.method == "micil":
                added = trainer.end_task(
                    train_loader,
                    label_offset=offsets[task_id],
                    k=k,
                )
                print(f"[FEATHER micil] added {added} WSI to replay buffer")

            elapsed = time.time() - start
            fold_training_time += elapsed
            print(f"[FEATHER {args.method}] fold={fold_id} task={task_id} took {elapsed:.1f}s")
            checkpoint_path = os.path.join(fold_dir, f"{prefix}_after_task_{task_id}.pt")
            _save_checkpoint(
                checkpoint_path,
                model,
                trainer,
                method=args.method,
                model_name=model_name,
                fold_id=fold_id,
                task_id=task_id,
                num_classes=num_classes,
                run_config=run_config,
            )

            if eval_after_task:
                for eval_task_id in range(task_id + 1):
                    _, _, test_loader = seq_dataset.get_data_loaders(
                        fold_id,
                        eval_task_id,
                        num_workers=num_workers,
                    )
                    metrics, prediction_rows = _evaluate_task(
                        model,
                        test_loader,
                        method_name=method_name,
                        fold_id=fold_id,
                        after_task=task_id,
                        eval_task=eval_task_id,
                        task_name=task_names[eval_task_id],
                        label_offset=offsets[eval_task_id],
                        task_num_classes=num_classes[eval_task_id],
                        seen_classes=seen_classes,
                        total_classes=total_classes,
                        device=device,
                        k=k,
                        patch_size=patch_size,
                        mask_unseen=mask_unseen_eval,
                        seed=seed,
                        confusion_matrix_dir=confusion_matrix_dir,
                    )
                    row = {
                        "method": method_name,
                        "fold": fold_id,
                        "mode": "class-il-seen" if mask_unseen_eval else "class-il-all",
                        "after_task": task_id,
                        "eval_task": eval_task_id,
                        "task_name": task_names[eval_task_id],
                        **metrics,
                    }
                    eval_rows.append(row)
                    _append_csv(eval_matrix_path, [row], eval_matrix_fields)
                    _append_csv(predictions_path, prediction_rows, prediction_fields)
                    print(
                        f"[FEATHER {args.method}] eval fold={fold_id} after={task_id} "
                        f"task={eval_task_id} acc={metrics['acc']:.4f} bacc={metrics['bacc']:.4f}"
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "eval/after_task": task_id,
                            "eval/task_id": eval_task_id,
                            **{f"eval/{key}": value for key, value in metrics.items()},
                        })

        final_path = os.path.join(fold_dir, f"{prefix}_final.pt")
        _save_checkpoint(
            final_path,
            model,
            trainer,
            method=args.method,
            model_name=model_name,
            fold_id=fold_id,
            task_id=num_tasks - 1,
            num_classes=num_classes,
            run_config=run_config,
        )
        if use_wandb:
            import wandb
            wandb.log({"train/final_buffer_size": len(trainer.buffer)})
            wandb.finish()
        fold_training_times[fold_id] = fold_training_time

    if eval_rows:
        fold_summary_rows = _build_fold_summary(method_name, eval_rows, fold_training_times)
        fold_summary_rows = _with_fold_aggregates(fold_summary_rows, per_fold_fields)
        task_summary_rows = _build_task_summary(method_name, eval_rows)
        _write_csv(
            os.path.join(save_dir, f"{prefix}_eval.csv"),
            eval_rows,
            eval_matrix_fields,
        )
        _write_csv(
            os.path.join(save_dir, "per_fold_summary.csv"),
            fold_summary_rows,
            per_fold_fields,
        )
        _write_csv(os.path.join(save_dir, "per_task_summary.csv"), task_summary_rows, per_task_fields)
        _write_csv(
            os.path.join(save_dir, f"{prefix}_eval_summary_per_fold.csv"),
            fold_summary_rows,
            per_fold_fields,
        )


if __name__ == "__main__":
    main()
