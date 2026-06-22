"""
Train an A-GEM continual-learning baseline with TITAN as the WSI backbone.

This script does not use class-aware prompts. It trains one TITAN vision encoder
plus one global randomly initialized classifier across the task stream.

Example:
    python scripts/train_agem.py --save_dir ./checkpoints/agem_titan
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import csv
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModel

from configs.loader import load_config
from mergeslide.agem import AgemTITAN
from mergeslide.datasets import Sequential_Generic_MIL_Dataset
from mergeslide.derpp import TitanGlobalClassifier
from mergeslide.models import cosine_lr
from mergeslide.utils import seed_torch


def _cfg_value(args, cfg: dict, key: str, default):
    value = getattr(args, key)
    return value if value is not None else cfg.get(key, default)


def _class_offsets(num_classes: List[int]) -> List[int]:
    offsets, total = [], 0
    for n_classes in num_classes:
        offsets.append(total)
        total += int(n_classes)
    return offsets


def _memory_samples_for_task(buffer_size: int, num_tasks: int, task_id: int) -> int:
    """Split A-GEM memory capacity across tasks as evenly as possible."""
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
    """Randomly keep up to ``k`` patches from one WSI, then move to device."""
    if k is not None:
        k = int(k)
        if k > 0 and features.shape[0] > k:
            indices = torch.randperm(features.shape[0])[:k]
            features = features[indices]
            coords = coords[indices]

    features = features.to(device, non_blocking=True)
    coords = coords.long().to(device, non_blocking=True)
    return features, coords


def _build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.Optimizer:
    named_parameters = list(model.named_parameters())
    exclude = lambda name, param: param.ndim < 2 or any(x in name for x in ("bn", "ln", "bias", "logit_scale"))
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if not exclude(n, p) and p.requires_grad]
    return torch.optim.AdamW(
        [
            {"params": gain_or_bias_params, "weight_decay": 0.0},
            {"params": rest_params, "weight_decay": weight_decay},
        ],
        lr=lr,
    )


def train_one_task(
    trainer: AgemTITAN,
    train_loader,
    task_id: int,
    label_offset: int,
    num_epochs: int,
    lr_scheduler,
    device: torch.device,
    k: int,
    use_wandb: bool = False,
) -> Dict[str, float]:
    step = 0
    last_stats: Dict[str, float] = {}
    for epoch in tqdm(range(num_epochs), desc=f"task {task_id}", leave=False):
        epoch_loss = 0.0
        projected_steps = 0.0
        for features, coords, labels in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            if lr_scheduler is not None:
                lr_scheduler(step)
            features, coords = _sample_patches(features, coords, device, k)
            global_labels = labels.to(device, non_blocking=True).long() + label_offset
            last_stats = trainer.observe(features, coords, global_labels)
            epoch_loss += last_stats["loss"]
            projected_steps += last_stats["projected"]
            step += 1

        avg_loss = epoch_loss / max(1, len(train_loader))
        proj_rate = projected_steps / max(1, len(train_loader))
        tqdm.write(
            f"task {task_id} epoch {epoch}: loss={avg_loss:.4f} "
            f"proj_rate={proj_rate:.3f} buffer={int(last_stats.get('buffer_size', 0))}"
        )
        if use_wandb:
            import wandb
            wandb.log({
                "train/task_id": task_id,
                "train/epoch": epoch,
                "train/avg_loss": avg_loss,
                "train/projection_rate": proj_rate,
                **{f"train/{key}": value for key, value in last_stats.items()},
            })
    return last_stats


def evaluate_task(
    model: TitanGlobalClassifier,
    test_loader,
    task_id: int,
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
        for features, coords, labels in tqdm(test_loader, desc=f"eval task {task_id}", leave=False):
            features, coords = _sample_patches(features, coords, device, k)
            targets = labels.to(device, non_blocking=True).long() + label_offset
            logits = model(features, coords, patch_size_tensor).float()
            if mask_unseen:
                logits = logits[:, :seen_classes]
            preds_all.append(logits.argmax(1).detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())

    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    return {
        "acc": float(accuracy_score(targets, preds)),
        "bacc": float(balanced_accuracy_score(targets, preds)),
        "n": float(len(targets)),
    }


def save_checkpoint(
    path: str,
    model: TitanGlobalClassifier,
    trainer: AgemTITAN,
    fold_id: int,
    task_id: int,
    num_classes: List[int],
    args_dict: dict,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "fold": fold_id,
            "task_id": task_id,
            "num_classes": num_classes,
            "total_classes": sum(num_classes),
            "buffer_size": len(trainer.buffer),
            "args": args_dict,
        },
        path,
    )


def write_eval_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["fold", "after_task", "eval_task", "acc", "bacc", "n"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _weighted_seen_accuracy(rows: List[dict], metric_key: str = "acc") -> float:
    total_n = float(sum(float(row["n"]) for row in rows))
    if total_n <= 0:
        return float("nan")
    return float(sum(float(row[metric_key]) * float(row["n"]) for row in rows) / total_n)


def _forgetting(results: List[List[float]]) -> float:
    n_tasks = len(results)
    if n_tasks <= 1:
        return float("nan")
    padded = [row + [0.0] * (n_tasks - len(row)) for row in results]
    np_res = np.array(padded, dtype=float)
    max_per_task = np.max(np_res, axis=0)
    return float(np.mean([max_per_task[i] - padded[-1][i] for i in range(n_tasks - 1)]))


def _backward_transfer(results: List[List[float]]) -> float:
    n_tasks = len(results)
    if n_tasks <= 1:
        return float("nan")
    return float(np.mean([results[-1][i] - results[i][i] for i in range(n_tasks - 1)]))


def _summarize_fold(rows: List[dict], fold_id: int) -> dict:
    max_task = max(int(row["after_task"]) for row in rows)
    results_by_seq: List[List[float]] = []
    acc_all_seqs = []

    for seq_task in range(max_task + 1):
        seq_rows = sorted(
            [row for row in rows if int(row["after_task"]) == seq_task],
            key=lambda row: int(row["eval_task"]),
        )
        expected_tasks = list(range(seq_task + 1))
        present_tasks = [int(row["eval_task"]) for row in seq_rows]
        if present_tasks != expected_tasks:
            raise ValueError(
                f"Fold {fold_id}, after_task {seq_task}: expected eval_task "
                f"{expected_tasks}, got {present_tasks}"
            )

        results_by_seq.append([float(row["acc"]) for row in seq_rows])
        acc_all_seqs.append(_weighted_seen_accuracy(seq_rows, metric_key="acc"))

    final_rows = [row for row in rows if int(row["after_task"]) == max_task]
    return {
        "fold": fold_id,
        "num_tasks": max_task + 1,
        "final_bacc": float(np.mean([float(row["bacc"]) for row in final_rows])),
        "mACC": float(np.mean(acc_all_seqs)),
        "BWT": _backward_transfer(results_by_seq),
        "FGT": _forgetting(results_by_seq),
    }


def _append_mean_std(rows: List[dict], metric_keys: List[str]) -> List[dict]:
    output_rows = list(rows)
    mean_row = {"fold": "mean", "num_tasks": float(np.mean([float(row["num_tasks"]) for row in rows]))}
    std_row = {"fold": "std", "num_tasks": float(np.std([float(row["num_tasks"]) for row in rows], ddof=0))}
    for key in metric_keys:
        values = np.array([float(row[key]) for row in rows], dtype=float)
        mean_row[key] = float(np.nanmean(values))
        std_row[key] = float(np.nanstd(values, ddof=0))
    output_rows.extend([mean_row, std_row])
    return output_rows


def write_eval_summary_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return

    fold_rows = []
    for fold_id in sorted({int(row["fold"]) for row in rows}):
        fold_rows.append(_summarize_fold([row for row in rows if int(row["fold"]) == fold_id], fold_id))

    output_rows = _append_mean_std(fold_rows, ["final_bacc", "mACC", "BWT", "FGT"])
    fieldnames = ["fold", "num_tasks", "final_bacc", "mACC", "BWT", "FGT"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)


def main():
    parser = argparse.ArgumentParser(description="A-GEM with TITAN backbone and no prompts")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--buffer_size", type=int, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--k", type=int, default=None, help="Patch budget per WSI; <=0 uses full slide bags")
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--no_eval_after_task", action="store_true")
    parser.add_argument("--no_mask_unseen_eval", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    cfg = load_config(default_filename="agem.yaml")
    agem_cfg = cfg.get("agem", {})

    num_epochs = int(_cfg_value(args, agem_cfg, "num_epochs", 10))
    lr = float(_cfg_value(args, agem_cfg, "lr", 1e-5))
    weight_decay = float(_cfg_value(args, agem_cfg, "weight_decay", 1e-4))
    buffer_size = int(_cfg_value(args, agem_cfg, "buffer_size", 500))
    save_dir = str(_cfg_value(args, agem_cfg, "save_dir", "./checkpoints/agem_titan"))
    num_folds = int(_cfg_value(args, agem_cfg, "num_folds", 10))
    num_workers = int(_cfg_value(args, agem_cfg, "num_workers", 4))
    k = int(_cfg_value(args, agem_cfg, "k", 400))
    patch_size = int(_cfg_value(args, agem_cfg, "patch_size", 1024))
    seed = int(_cfg_value(args, agem_cfg, "seed", 0))

    freeze_backbone = bool(args.freeze_backbone or agem_cfg.get("freeze_backbone", False))
    eval_after_task = bool((not args.no_eval_after_task) and agem_cfg.get("eval_after_task", True))
    mask_unseen_eval = bool((not args.no_mask_unseen_eval) and agem_cfg.get("mask_unseen_eval", True))
    use_wandb = (args.use_wandb or agem_cfg.get("use_wandb", False)) and not args.disable_wandb
    if use_wandb:
        try:
            import wandb  # noqa: F401
        except ImportError:
            warnings.warn("wandb package not found. Disabling wandb tracking.")
            use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, seed)
    os.makedirs(save_dir, exist_ok=True)

    seq_dataset = Sequential_Generic_MIL_Dataset()
    all_num_classes = seq_dataset.num_classes
    if args.num_tasks is not None:
        num_tasks = int(args.num_tasks)
    else:
        cfg_num_tasks = agem_cfg.get("num_tasks")
        num_tasks = int(cfg_num_tasks) if cfg_num_tasks is not None else len(all_num_classes)
    num_classes = all_num_classes[:num_tasks]
    offsets = _class_offsets(num_classes)
    total_classes = sum(num_classes)

    eval_rows: List[dict] = []
    run_args = vars(args).copy()
    run_args.update(
        {
            "num_epochs": num_epochs,
            "lr": lr,
            "weight_decay": weight_decay,
            "buffer_size": buffer_size,
            "num_tasks": num_tasks,
            "k": k,
            "patch_size": patch_size,
            "freeze_backbone": freeze_backbone,
            "mask_unseen_eval": mask_unseen_eval,
        }
    )

    for fold_id in tqdm(range(num_folds), desc="folds"):
        fold_dir = os.path.join(save_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)
        if use_wandb:
            import wandb
            wandb.init(
                project=agem_cfg.get("wandb_project", "MergeSlide-AGEM"),
                entity=agem_cfg.get("wandb_entity"),
                group="agem_train",
                job_type="train",
                name=f"agem_fold_{fold_id}",
                config={**run_args, "fold": fold_id},
                reinit=True,
            )

        base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)
        model = TitanGlobalClassifier(base_model, total_classes).to(device)
        if freeze_backbone:
            for param in model.backbone.parameters():
                param.requires_grad = False

        optimizer = _build_optimizer(model, lr=lr, weight_decay=weight_decay)
        trainer = AgemTITAN(
            model=model,
            optimizer=optimizer,
            device=device,
            buffer_size=buffer_size,
            patch_size=patch_size,
            seed=seed + fold_id,
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
            train_one_task(
                trainer=trainer,
                train_loader=train_loader,
                task_id=task_id,
                label_offset=offsets[task_id],
                num_epochs=num_epochs,
                lr_scheduler=scheduler,
                device=device,
                k=k,
                use_wandb=use_wandb,
            )
            memory_samples = _memory_samples_for_task(buffer_size, num_tasks, task_id)
            added_to_buffer = trainer.end_task(
                train_loader=train_loader,
                label_offset=offsets[task_id],
                k=k,
                samples_per_task=memory_samples,
            )
            elapsed = time.time() - start
            print(
                f"Fold {fold_id}, task {task_id}: A-GEM training took {elapsed:.1f}s "
                f"(added {added_to_buffer}/{memory_samples} WSI to buffer, "
                f"buffer={len(trainer.buffer)}/{buffer_size})"
            )

            ckpt_path = os.path.join(fold_dir, f"agem_titan_after_task_{task_id}.pt")
            save_checkpoint(ckpt_path, model, trainer, fold_id, task_id, num_classes, run_args)
            print(f"Saved checkpoint: {ckpt_path}")

            if eval_after_task:
                seen_classes = sum(num_classes[:task_id + 1])
                for eval_task_id in range(task_id + 1):
                    _, _, test_loader = seq_dataset.get_data_loaders(fold_id, eval_task_id, num_workers=num_workers)
                    metrics = evaluate_task(
                        model=model,
                        test_loader=test_loader,
                        task_id=eval_task_id,
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
                        f"eval fold={fold_id} after_task={task_id} task={eval_task_id}: "
                        f"acc={metrics['acc']:.4f} bacc={metrics['bacc']:.4f}"
                    )
                    if use_wandb:
                        import wandb
                        wandb.log({
                            "eval/after_task": task_id,
                            "eval/task_id": eval_task_id,
                            **{f"eval/{key}": value for key, value in metrics.items()},
                        })

        final_path = os.path.join(fold_dir, "agem_titan_final.pt")
        save_checkpoint(final_path, model, trainer, fold_id, num_tasks - 1, num_classes, run_args)
        if use_wandb:
            import wandb
            wandb.log({"train/final_buffer_size": len(trainer.buffer)})
            wandb.finish()

    if eval_rows:
        eval_csv = os.path.join(save_dir, "agem_titan_eval.csv")
        write_eval_csv(eval_csv, eval_rows)
        write_eval_summary_csv(
            os.path.join(save_dir, "agem_titan_eval_summary_per_fold.csv"),
            eval_rows,
        )


if __name__ == "__main__":
    main()
