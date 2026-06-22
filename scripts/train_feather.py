"""
Per-task finetuning for FEATHER on MergeSlide WSI tasks.

This script is separate from the TITAN finetuning path.
"""

import argparse
import os
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
import torch.nn as nn
import yaml
from typing import Optional
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from mergeslide.datasets import Sequential_Generic_MIL_Dataset
from mergeslide.models import EarlyStopping
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


def _load_feather_cfg() -> dict:
    with open(FEATHER_CONFIG, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get("feather", {})


def _cfg_value(args, cfg: dict, name: str, default):
    value = getattr(args, name)
    return value if value is not None else cfg.get(name, default)


def _build_optimizer(model: nn.Module, lr: float, weight_decay: float):
    named_parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if not named_parameters:
        raise RuntimeError("FEATHER model has no trainable parameters.")
    no_decay = lambda n, p: p.ndim < 2 or any(token in n.lower() for token in ("bn", "ln", "bias"))
    gain_or_bias_params = [p for n, p in named_parameters if no_decay(n, p)]
    rest_params = [p for n, p in named_parameters if not no_decay(n, p)]
    return torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": weight_decay},
        ],
        lr=lr,
    )


def _run_eval_loader(
    loader,
    model: nn.Module,
    loss_fn,
    num_classes: int,
    device: torch.device,
    k: int,
    patch_size: int,
):
    model.eval()
    preds_all, targets_all = [], []
    total_loss = 0.0
    patch_size_tensor = torch.tensor(patch_size, dtype=torch.int32, device=device)
    amp_enabled = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        for features, coords, labels in loader:
            features, coords = sample_patch_bag(features, coords, k)
            features = features.to(device, non_blocking=True)
            coords = coords.long().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features, coords, patch_size_tensor).float()
            loss = loss_fn(logits, labels)
            total_loss += loss.item()
            preds_all.append(logits.argmax(1).cpu().numpy())
            targets_all.append(labels.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    return {
        "loss": total_loss / max(len(loader), 1),
        "bacc": balanced_accuracy_score(targets_all, preds_all),
        "acc": float((preds_all == targets_all).mean()),
    }


def train_one_task(
    train_loader,
    val_loader,
    model: nn.Module,
    *,
    num_classes: int,
    num_epochs: int,
    lr: float,
    weight_decay: float,
    device: torch.device,
    k: int,
    patch_size: int,
    use_wandb: bool = False,
    debug_batches: bool = False,
    patience: Optional[int] = None,
):
    optimizer = _build_optimizer(model, lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    amp_enabled = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    patch_size_tensor = torch.tensor(patch_size, dtype=torch.int32, device=device)
    early_stopping = EarlyStopping(patience=patience, verbose=True) if patience is not None and patience > 0 else None

    for epoch in range(num_epochs):
        model.train()
        preds_all, targets_all = [], []
        total_loss = 0.0
        start_epoch = time.time()

        for batch_idx, (features, coords, labels) in enumerate(tqdm(train_loader, desc=f"epoch {epoch}", leave=False)):
            features, coords = sample_patch_bag(features, coords, k)
            features = features.to(device, non_blocking=True)
            coords = coords.long().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
                logits = model(features, coords, patch_size_tensor)
                loss = loss_fn(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item()
            preds_all.append(logits.argmax(1).detach().cpu().numpy())
            targets_all.append(labels.detach().cpu().numpy())

            if debug_batches:
                tqdm.write(
                    f"[train] epoch={epoch} batch={batch_idx + 1}/{len(train_loader)} "
                    f"features={tuple(features.shape)} loss={loss.item():.4f}"
                )

        preds_all = np.concatenate(preds_all)
        targets_all = np.concatenate(targets_all)
        train_metrics = {
            "loss": total_loss / max(len(train_loader), 1),
            "bacc": balanced_accuracy_score(targets_all, preds_all),
            "acc": float((preds_all == targets_all).mean()),
        }
        val_metrics = _run_eval_loader(
            val_loader,
            model,
            loss_fn,
            num_classes=num_classes,
            device=device,
            k=k,
            patch_size=patch_size,
        )

        tqdm.write(
            f"epoch {epoch}, train_loss={train_metrics['loss']:.4f}, "
            f"train_bacc={train_metrics['bacc']:.4f}, val_loss={val_metrics['loss']:.4f}, "
            f"val_bacc={val_metrics['bacc']:.4f}, elapsed={time.time() - start_epoch:.1f}s"
        )
        if use_wandb:
            import wandb
            wandb.log({
                "epoch": epoch,
                "train/loss": train_metrics["loss"],
                "train/bacc": train_metrics["bacc"],
                "train/acc": train_metrics["acc"],
                "val/loss": val_metrics["loss"],
                "val/bacc": val_metrics["bacc"],
                "val/acc": val_metrics["acc"],
            })

        if early_stopping is not None:
            early_stopping(val_metrics["loss"], model)
            if early_stopping.early_stop:
                tqdm.write("Early stopping triggered")
                break

    if early_stopping is not None and early_stopping.best_model_weights is not None:
        model.load_state_dict(early_stopping.best_model_weights)

    model.eval()
    return model


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser(description="Per-task finetuning with FEATHER")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--fold_start", type=int, default=0)
    parser.add_argument("--fold_end", type=int, default=None)
    parser.add_argument("--only_task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--no_pretrained", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    parser.add_argument("--debug_batches", action="store_true")
    parser.add_argument("--patience", type=int, default=None)
    args = parser.parse_args()

    feather_cfg = _load_feather_cfg()
    model_name = str(_cfg_value(args, feather_cfg, "model_name", DEFAULT_FEATHER_MODEL_NAME))
    num_epochs = int(_cfg_value(args, feather_cfg, "num_epochs", 10))
    lr = float(_cfg_value(args, feather_cfg, "lr", 1e-5))
    weight_decay = float(_cfg_value(args, feather_cfg, "weight_decay", 1e-4))
    save_dir = str(_cfg_value(args, feather_cfg, "save_dir", "./checkpoints/feather_finetuned"))
    num_folds = int(_cfg_value(args, feather_cfg, "num_folds", 10))
    num_workers = int(_cfg_value(args, feather_cfg, "num_workers", 0))
    k = int(_cfg_value(args, feather_cfg, "k", 400))
    patch_size = int(_cfg_value(args, feather_cfg, "patch_size", 512))
    from_pretrained = bool(feather_cfg.get("from_pretrained", True)) and not args.no_pretrained
    freeze_backbone = bool(args.freeze_backbone or feather_cfg.get("freeze_backbone", False))
    use_wandb = (args.use_wandb or feather_cfg.get("use_wandb", False)) and not args.disable_wandb
    patience_val = _cfg_value(args, feather_cfg, "patience", None)
    patience = int(patience_val) if patience_val is not None else None

    if use_wandb:
        try:
            import wandb  # noqa: F401
        except ImportError:
            warnings.warn("wandb package not found. Disabling wandb tracking.")
            use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, args.seed)
    prepare_hf_token_env(feather_cfg.get("hf_token"))

    seq_dataset = Sequential_Generic_MIL_Dataset(config_path=str(FEATHER_CONFIG))
    num_classes = seq_dataset.num_classes
    num_tasks = len(num_classes)
    task_ids = [args.only_task] if args.only_task is not None else list(range(num_tasks))
    fold_end = int(args.fold_end if args.fold_end is not None else num_folds)

    for fold_id in range(args.fold_start, fold_end):
        fold_dir = os.path.join(save_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        for task_id in task_ids:
            train_loader, val_loader, _ = seq_dataset.get_data_loaders(
                fold_id,
                task_id,
                num_workers=num_workers,
            )

            if use_wandb:
                import wandb
                wandb.init(
                    project=feather_cfg.get("wandb_project", "MergeSlide-FEATHER"),
                    entity=feather_cfg.get("wandb_entity"),
                    name=f"feather_fold_{fold_id}_task_{task_id}",
                    config={
                        "fold": fold_id,
                        "task": task_id,
                        "model_name": model_name,
                        "num_epochs": num_epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                        "num_workers": num_workers,
                        "k": k,
                        "patch_size": patch_size,
                        "from_pretrained": from_pretrained,
                        "freeze_backbone": freeze_backbone,
                    },
                    reinit=True,
                )

            print(f"[FEATHER] fold={fold_id} task={task_id} classes={num_classes[task_id]}")
            base_model = create_feather_model(
                model_name,
                num_classes=num_classes[task_id],
                from_pretrained=from_pretrained,
            )
            if freeze_backbone:
                frozen, trainable = freeze_feather_backbone(
                    base_model,
                    num_classes=num_classes[task_id],
                )
                print(f"[FEATHER] freeze_backbone frozen_params={frozen} trainable_params={trainable}")
            model = FeatherMILWrapper(base_model, num_classes=num_classes[task_id]).to(device)

            start = time.time()
            model = train_one_task(
                train_loader,
                val_loader,
                model,
                num_classes=num_classes[task_id],
                num_epochs=num_epochs,
                lr=lr,
                weight_decay=weight_decay,
                device=device,
                k=k,
                patch_size=patch_size,
                use_wandb=use_wandb,
                debug_batches=args.debug_batches,
                patience=patience,
            )
            print(f"[FEATHER] fold={fold_id} task={task_id} training took {time.time() - start:.1f}s")

            ckpt_path = os.path.join(fold_dir, f"feather_task_{task_id}.pt")
            backbone_state, head_state, classifier_keys = split_feather_state_dict(
                model.model,
                num_classes=num_classes[task_id],
            )
            torch.save(
                {
                    "model_state_dict": model.model.state_dict(),
                    "backbone_state_dict": backbone_state,
                    "head_state_dict": head_state,
                    "classifier_state_keys": classifier_keys,
                    "model_name": model_name,
                    "num_classes": num_classes[task_id],
                    "task_id": task_id,
                    "fold_id": fold_id,
                    "forward_mode": model._forward_mode,
                    "feature_dim": model.feature_dim,
                    "patch_size": patch_size,
                    "k": k,
                },
                ckpt_path,
            )
            print(f"[FEATHER] saved checkpoint: {ckpt_path}")

            if use_wandb:
                import wandb
                wandb.finish()


if __name__ == "__main__":
    main()
