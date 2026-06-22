"""Calibrate a global FEATHER classifier on top of a merged backbone.

Unlike naive CLASS-IL, this script does not concatenate independently trained
task heads. It freezes the merged FEATHER backbone and fits one classifier in
the global class space using task training splits, selecting the best head on
the corresponding validation splits.
"""

import argparse
import copy
import os
import sys
import warnings
from pathlib import Path
from typing import Iterable, Iterator, List, Sequence, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from mergeslide.datasets import Sequential_Generic_MIL_Dataset, get_num_classes
from mergeslide.feather_models import (
    DEFAULT_FEATHER_MODEL_NAME,
    FeatherMILWrapper,
    create_feather_model,
    freeze_feather_backbone,
    prepare_hf_token_env,
    sample_patch_bag,
    split_feather_state_dict,
)
from mergeslide.utils import seed_torch


REPO_ROOT = Path(__file__).resolve().parent.parent
FEATHER_CONFIG = REPO_ROOT / "configs" / "feather.yaml"
FEATHER_MERGE_CONFIG = REPO_ROOT / "configs" / "merge_feather.yaml"
FEATHER_EVAL_CONFIG = REPO_ROOT / "configs" / "eval_feather.yaml"


def _load_section(path: Path, section: str) -> dict:
    with open(path, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get(section, {})


def _torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _merged_backbone_path(merge_model_path: str, fold: str, num_tasks: int) -> str:
    if not os.path.isdir(merge_model_path):
        return merge_model_path
    return os.path.join(
        merge_model_path,
        f"_{fold}",
        f"merged_backbone_feather_opcm_{fold}_task_{num_tasks - 1}.pth",
    )


def _build_calibration_model(
    *,
    model_name: str,
    total_classes: int,
    merged_backbone_path: str,
    device: torch.device,
) -> Tuple[FeatherMILWrapper, Tuple[str, ...]]:
    base_model = create_feather_model(
        model_name,
        num_classes=total_classes,
        from_pretrained=False,
    )
    merged_backbone = _torch_load(merged_backbone_path, map_location="cpu")
    missing, unexpected = base_model.load_state_dict(merged_backbone, strict=False)
    if unexpected:
        raise KeyError(f"Merged FEATHER backbone has unexpected keys: {unexpected[:5]}")

    _, _, classifier_keys = split_feather_state_dict(base_model, num_classes=total_classes)
    unexpected_missing = sorted(set(missing) - set(classifier_keys))
    if unexpected_missing:
        raise KeyError(
            "Merged FEATHER backbone is incomplete; missing non-classifier keys: "
            f"{unexpected_missing[:5]}"
        )
    freeze_feather_backbone(base_model, num_classes=total_classes)
    return FeatherMILWrapper(base_model, num_classes=total_classes).to(device), classifier_keys


def _global_class_weights(
    train_loaders: Sequence,
    offsets: Sequence[int],
    total_classes: int,
) -> torch.Tensor:
    counts = np.zeros(total_classes, dtype=np.float64)
    for loader, offset in zip(train_loaders, offsets):
        labels = loader.dataset.slide_data["label"].to_numpy(dtype=np.int64)
        np.add.at(counts, labels + offset, 1)
    if np.any(counts == 0):
        missing = np.flatnonzero(counts == 0).tolist()
        raise ValueError(f"Calibration train split is missing global classes: {missing}")
    weights = counts.sum() / (total_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


def _round_robin_batches(
    loaders: Sequence,
    offsets: Sequence[int],
) -> Iterator[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]]:
    iterators = [iter(loader) for loader in loaders]
    active = list(range(len(iterators)))
    while active:
        for index in active.copy():
            try:
                features, coords, labels = next(iterators[index])
            except StopIteration:
                active.remove(index)
                continue
            yield features, coords, labels, offsets[index]


def _evaluate_global_head(
    model: FeatherMILWrapper,
    val_loaders: Sequence,
    offsets: Sequence[int],
    *,
    device: torch.device,
    k: int,
    patch_size: int,
) -> Tuple[float, float]:
    model.eval()
    patch_size_tensor = torch.tensor(patch_size, dtype=torch.int32, device=device)
    preds_all: List[np.ndarray] = []
    targets_all: List[np.ndarray] = []
    with torch.no_grad():
        for loader, offset in zip(val_loaders, offsets):
            for features, coords, labels in loader:
                features, coords = sample_patch_bag(features, coords, k)
                logits = model(
                    features.to(device, non_blocking=True),
                    coords.long().to(device, non_blocking=True),
                    patch_size_tensor,
                ).float()
                preds_all.append(logits.argmax(1).cpu().numpy())
                targets_all.append((labels.long() + offset).cpu().numpy())
    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    return float((preds == targets).mean()), float(balanced_accuracy_score(targets, preds))


def _classifier_state(model: FeatherMILWrapper, classifier_keys: Iterable[str]) -> dict:
    state = model.model.state_dict()
    return {key: state[key].detach().cpu().clone() for key in classifier_keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a global FEATHER CLASS-IL classifier")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None, help="Per-task checkpoint directory for provenance")
    parser.add_argument("--merge_model_path", type=str, default=None)
    parser.add_argument("--calibration_dir", type=str, default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--patience", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--fold_start", type=int, default=0)
    parser.add_argument("--fold_end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    feather_cfg = _load_section(FEATHER_CONFIG, "feather")
    merge_cfg = _load_section(FEATHER_MERGE_CONFIG, "feather_merging")
    eval_cfg = _load_section(FEATHER_EVAL_CONFIG, "feather_evaluation")
    model_name = str(args.model_name or feather_cfg.get("model_name", DEFAULT_FEATHER_MODEL_NAME))
    save_dir = str(args.save_dir or eval_cfg.get("save_dir", feather_cfg.get("save_dir")))
    merge_model_path = str(
        args.merge_model_path
        or eval_cfg.get("merge_model_path")
        or merge_cfg.get("des_merged_checkpoints", "./checkpoints/feather_merged")
    )
    calibration_dir = str(
        args.calibration_dir
        or eval_cfg.get("classil_calibration_dir", "./checkpoints/feather_classil_calibrated")
    )
    num_folds = int(args.num_folds if args.num_folds is not None else eval_cfg.get("num_folds", 10))
    num_workers = int(args.num_workers if args.num_workers is not None else eval_cfg.get("num_workers", 0))
    num_epochs = int(args.num_epochs if args.num_epochs is not None else eval_cfg.get("classil_calibration_epochs", 20))
    lr = float(args.lr if args.lr is not None else eval_cfg.get("classil_calibration_lr", 1e-3))
    weight_decay = float(
        args.weight_decay
        if args.weight_decay is not None
        else eval_cfg.get("classil_calibration_weight_decay", 1e-4)
    )
    patience = int(args.patience if args.patience is not None else eval_cfg.get("classil_calibration_patience", 5))
    k = int(args.k if args.k is not None else eval_cfg.get("k", feather_cfg.get("k", 400)))
    patch_size = int(
        args.patch_size
        if args.patch_size is not None
        else eval_cfg.get("patch_size", feather_cfg.get("patch_size", 512))
    )
    use_wandb = (args.use_wandb or eval_cfg.get("use_wandb", False)) and not args.disable_wandb
    if use_wandb:
        try:
            import wandb  # noqa: F401
        except ImportError:
            warnings.warn("wandb package not found. Disabling wandb tracking.")
            use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, args.seed)
    prepare_hf_token_env(feather_cfg.get("hf_token"))
    dataset = Sequential_Generic_MIL_Dataset(config_path=str(FEATHER_CONFIG))
    num_tasks = int(args.num_tasks if args.num_tasks is not None else eval_cfg.get("num_tasks", len(dataset.num_classes)))
    num_classes = dataset.num_classes[:num_tasks]
    total_classes = sum(num_classes)
    offsets = np.cumsum([0, *num_classes[:-1]]).tolist()
    fold_end = int(args.fold_end if args.fold_end is not None else num_folds)

    for fold_id in range(args.fold_start, fold_end):
        fold = f"fold_{fold_id}"
        merged_path = _merged_backbone_path(merge_model_path, fold, num_tasks)
        if not os.path.exists(merged_path):
            raise FileNotFoundError(f"Missing merged FEATHER backbone: {merged_path}")
        train_loaders, val_loaders = [], []
        for task_id in range(num_tasks):
            train_loader, val_loader, _ = dataset.get_data_loaders(
                fold_id,
                task_id,
                num_workers=num_workers,
            )
            if val_loader is None:
                raise ValueError(f"Task {task_id} fold {fold_id} has no validation split for calibration.")
            train_loaders.append(train_loader)
            val_loaders.append(val_loader)

        model, classifier_keys = _build_calibration_model(
            model_name=model_name,
            total_classes=total_classes,
            merged_backbone_path=merged_path,
            device=device,
        )
        class_weights = _global_class_weights(train_loaders, offsets, total_classes).to(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
        loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        patch_size_tensor = torch.tensor(patch_size, dtype=torch.int32, device=device)
        best_state, best_bacc, stale_epochs = None, -float("inf"), 0

        if use_wandb:
            import wandb
            wandb.init(
                project=eval_cfg.get("wandb_project", "MergeSlide-FEATHER"),
                entity=eval_cfg.get("wandb_entity"),
                group="feather_classil_calibration",
                job_type="calibrate_classil",
                name=f"feather_classil_calibration_{fold}",
                config={
                    "fold": fold_id,
                    "model_name": model_name,
                    "num_tasks": num_tasks,
                    "total_classes": total_classes,
                    "k": k,
                    "patch_size": patch_size,
                    "num_epochs": num_epochs,
                    "lr": lr,
                    "weight_decay": weight_decay,
                },
                reinit=True,
            )

        for epoch in range(num_epochs):
            model.eval()
            total_loss, total_examples = 0.0, 0
            for features, coords, labels, offset in tqdm(
                _round_robin_batches(train_loaders, offsets),
                desc=f"[FEATHER calibration] fold={fold_id} epoch={epoch}",
                leave=False,
            ):
                features, coords = sample_patch_bag(features, coords, k)
                targets = labels.long().to(device, non_blocking=True) + offset
                optimizer.zero_grad(set_to_none=True)
                logits = model(
                    features.to(device, non_blocking=True),
                    coords.long().to(device, non_blocking=True),
                    patch_size_tensor,
                ).float()
                loss = loss_fn(logits, targets)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach().cpu()) * targets.numel()
                total_examples += targets.numel()

            val_acc, val_bacc = _evaluate_global_head(
                model,
                val_loaders,
                offsets,
                device=device,
                k=k,
                patch_size=patch_size,
            )
            train_loss = total_loss / max(total_examples, 1)
            print(
                f"[FEATHER calibration] fold={fold_id} epoch={epoch} "
                f"train_loss={train_loss:.4f} val_acc={val_acc:.4f} val_bacc={val_bacc:.4f}"
            )
            if use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_loss,
                    "val/acc": val_acc,
                    "val/bacc": val_bacc,
                })

            if val_bacc > best_bacc:
                best_bacc = val_bacc
                best_state = copy.deepcopy(_classifier_state(model, classifier_keys))
                stale_epochs = 0
            else:
                stale_epochs += 1
                if stale_epochs >= patience:
                    print(f"[FEATHER calibration] early stop at epoch={epoch}")
                    break

        if best_state is None:
            raise RuntimeError("Calibration finished without a classifier state.")
        output_dir = Path(calibration_dir) / fold
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"feather_global_classifier_{fold}.pt"
        torch.save(
            {
                "classifier_state_dict": best_state,
                "classifier_state_keys": classifier_keys,
                "model_name": model_name,
                "num_classes": total_classes,
                "fold_id": fold_id,
                "merged_backbone_path": merged_path,
                "source_task_checkpoints": save_dir,
                "k": k,
                "patch_size": patch_size,
                "best_val_bacc": best_bacc,
            },
            output_path,
        )
        print(f"[FEATHER calibration] saved: {output_path} best_val_bacc={best_bacc:.4f}")
        if use_wandb:
            import wandb
            wandb.log({"val/best_bacc": best_bacc})
            wandb.finish()


if __name__ == "__main__":
    main()
