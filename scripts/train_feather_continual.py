"""
Train FEATHER continual-learning baselines on the MergeSlide task stream.

Examples:
    python scripts/train_feather_continual.py --method derpp
    python scripts/train_feather_continual.py --method agem --num_folds 1
    python scripts/train_feather_continual.py --method er_ace --num_tasks 3
"""

import argparse
import csv
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import torch
import yaml
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm

from mergeslide.agem import AgemTITAN
from mergeslide.datasets import Sequential_Generic_MIL_Dataset
from mergeslide.derpp import DerppTITAN
from mergeslide.er_ace import ErAceTITAN
from mergeslide.feather_continual import FeatherGlobalClassifier
from mergeslide.feather_models import (
    DEFAULT_FEATHER_MODEL_NAME,
    create_feather_model,
    freeze_feather_backbone,
    prepare_hf_token_env,
)
from mergeslide.models import cosine_lr
from mergeslide.utils import seed_torch


REPO_ROOT = Path(__file__).resolve().parent.parent
FEATHER_CONFIG = REPO_ROOT / "configs" / "feather.yaml"
CONTINUAL_CONFIG = REPO_ROOT / "configs" / "feather_continual.yaml"
METHODS = ("derpp", "agem", "er_ace")


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
    method: str,
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
            **{f"train/{key}": value for key, value in last_stats.items()},
        }
        if method == "agem":
            log_values["train/projection_rate"] = projected_steps / max(len(train_loader), 1)
        tqdm.write(
            f"[FEATHER {method}] task={task_id} epoch={epoch} "
            f"loss={avg_loss:.4f} buffer={int(last_stats.get('buffer_size', 0))}"
        )
        if use_wandb:
            import wandb
            wandb.log(log_values)
    return last_stats


def _evaluate_task(
    model: FeatherGlobalClassifier,
    test_loader,
    *,
    label_offset: int,
    seen_classes: int,
    device: torch.device,
    k: int,
    patch_size: int,
    mask_unseen: bool,
) -> Dict[str, float]:
    model.eval()
    preds_all, targets_all = [], []
    patch_size_tensor = torch.tensor(patch_size, dtype=torch.int32, device=device)
    with torch.no_grad():
        for features, coords, labels in tqdm(test_loader, desc="eval", leave=False):
            features, coords = _sample_patches(features, coords, device, k)
            targets = labels.to(device, non_blocking=True).long() + label_offset
            logits = model(features, coords, patch_size_tensor).float()
            if mask_unseen:
                logits = logits[:, :seen_classes]
            preds_all.append(logits.argmax(1).cpu().numpy())
            targets_all.append(targets.cpu().numpy())

    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    return {
        "acc": float(accuracy_score(targets, preds)),
        "bacc": float(balanced_accuracy_score(targets, preds)),
        "n": float(len(targets)),
    }


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


def _summarize_fold(rows: List[dict], fold_id: int) -> dict:
    final_task = max(int(row["after_task"]) for row in rows)
    final_rows = [row for row in rows if int(row["after_task"]) == final_task]
    return {
        "fold": fold_id,
        "num_tasks": final_task + 1,
        "final_acc": float(np.mean([row["acc"] for row in final_rows])),
        "final_bacc": float(np.mean([row["bacc"] for row in final_rows])),
    }


def _write_summary(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    fold_rows = [
        _summarize_fold([row for row in rows if int(row["fold"]) == fold_id], fold_id)
        for fold_id in sorted({int(row["fold"]) for row in rows})
    ]
    mean_row = {
        "fold": "mean",
        "num_tasks": float(np.mean([row["num_tasks"] for row in fold_rows])),
        "final_acc": float(np.mean([row["final_acc"] for row in fold_rows])),
        "final_bacc": float(np.mean([row["final_bacc"] for row in fold_rows])),
    }
    std_row = {
        "fold": "std",
        "num_tasks": float(np.std([row["num_tasks"] for row in fold_rows])),
        "final_acc": float(np.std([row["final_acc"] for row in fold_rows])),
        "final_bacc": float(np.std([row["final_bacc"] for row in fold_rows])),
    }
    _write_csv(path, fold_rows + [mean_row, std_row], ["fold", "num_tasks", "final_acc", "final_bacc"])


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
    save_dir = str(_cfg_value(args, method_cfg, "save_dir", f"./checkpoints/feather_{args.method}"))
    num_folds = int(_cfg_value(args, method_cfg, "num_folds", 10))
    num_workers = int(_cfg_value(args, method_cfg, "num_workers", feather_cfg.get("num_workers", 0)))
    k = int(_cfg_value(args, method_cfg, "k", feather_cfg.get("k", 400)))
    patch_size = int(args.patch_size if args.patch_size is not None else feather_cfg.get("patch_size", 256))
    seed = int(_cfg_value(args, method_cfg, "seed", 0))
    from_pretrained = bool(feather_cfg.get("from_pretrained", True)) and not args.no_pretrained
    freeze_backbone = bool(args.freeze_backbone or method_cfg.get("freeze_backbone", False))
    task_free = bool(args.task_free or method_cfg.get("task_free", False))
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
    run_config = {
        "method": args.method,
        "model_name": model_name,
        "num_epochs": num_epochs,
        "lr": lr,
        "weight_decay": weight_decay,
        "buffer_size": buffer_size,
        "alpha": alpha if args.method == "derpp" else None,
        "beta": beta if args.method == "derpp" else None,
        "num_tasks": num_tasks,
        "k": k,
        "patch_size": patch_size,
        "from_pretrained": from_pretrained,
        "freeze_backbone": freeze_backbone,
        "task_free": task_free if args.method == "er_ace" else None,
        "mask_unseen_eval": mask_unseen_eval,
    }
    eval_rows: List[dict] = []

    for fold_id in tqdm(range(num_folds), desc="folds"):
        fold_dir = os.path.join(save_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
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
        else:
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

        for task_id in range(num_tasks):
            start = time.time()
            train_loader, _, _ = seq_dataset.get_data_loaders(fold_id, task_id, num_workers=num_workers)
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
                method=args.method,
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

            elapsed = time.time() - start
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
                seen_classes = sum(num_classes[:task_id + 1])
                for eval_task_id in range(task_id + 1):
                    _, _, test_loader = seq_dataset.get_data_loaders(
                        fold_id,
                        eval_task_id,
                        num_workers=num_workers,
                    )
                    metrics = _evaluate_task(
                        model,
                        test_loader,
                        label_offset=offsets[eval_task_id],
                        seen_classes=seen_classes,
                        device=device,
                        k=k,
                        patch_size=patch_size,
                        mask_unseen=mask_unseen_eval,
                    )
                    row = {
                        "fold": fold_id,
                        "after_task": task_id,
                        "eval_task": eval_task_id,
                        **metrics,
                    }
                    eval_rows.append(row)
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

    if eval_rows:
        _write_csv(
            os.path.join(save_dir, f"{prefix}_eval.csv"),
            eval_rows,
            ["fold", "after_task", "eval_task", "acc", "bacc", "n"],
        )
        _write_summary(
            os.path.join(save_dir, f"{prefix}_eval_summary_per_fold.csv"),
            eval_rows,
        )


if __name__ == "__main__":
    main()
