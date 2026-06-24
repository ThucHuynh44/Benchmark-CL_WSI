"""
Evaluate TASK-IL masked metrics for continual baselines with one global head.

For each eval task, the oracle task identity is used by restricting logits to
the task's class range [offset_t, offset_t + num_classes_t). This matches the
TASK-IL setting in scripts/eval_taskil.py, adapted to DER++/A-GEM/ER-ACE
checkpoints that use one global classifier.

Examples:
    python scripts/eval_taskil_masked_baselines.py --method agem \
        --ckpt_dir ./checkpoints/agem_titan_buffer10

    python scripts/eval_taskil_masked_baselines.py --method derpp --final_only
"""

import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import csv
import warnings
from typing import Dict, List, Tuple

import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModel

from configs.loader import load_config
from mergeslide.datasets import Sequential_Generic_MIL_Dataset
from mergeslide.derpp import TitanGlobalClassifier
from mergeslide.utils import seed_torch


METHOD_DEFAULTS = {
    "derpp": {
        "config": "derpp.yaml",
        "config_key": "derpp",
        "save_dir": "./checkpoints/derpp_titan",
        "prefix": "derpp_titan",
    },
    "agem": {
        "config": "agem.yaml",
        "config_key": "agem",
        "save_dir": "./checkpoints/agem_titan",
        "prefix": "agem_titan",
    },
    "er_ace": {
        "config": "er_ace.yaml",
        "config_key": "er_ace",
        "save_dir": "./checkpoints/er_ace_titan",
        "prefix": "er_ace_titan",
    },
}


def _cfg_value(args, cfg: dict, key: str, default):
    value = getattr(args, key)
    return value if value is not None else cfg.get(key, default)


def _class_offsets(num_classes: List[int]) -> List[int]:
    offsets, total = [], 0
    for n_classes in num_classes:
        offsets.append(total)
        total += int(n_classes)
    return offsets


def _torch_load(path: str, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _sample_patches(
    features: torch.Tensor,
    coords: torch.Tensor,
    device: torch.device,
    k: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    if k is not None:
        k = int(k)
        if k > 0 and features.shape[0] > k:
            indices = torch.randperm(features.shape[0])[:k]
            features = features[indices]
            coords = coords[indices]

    features = features.to(device, non_blocking=True)
    coords = coords.long().to(device, non_blocking=True)
    return features, coords


def checkpoint_path(ckpt_dir: str, prefix: str, fold_id: int, after_task: int) -> str:
    return os.path.join(ckpt_dir, f"fold_{fold_id}", f"{prefix}_after_task_{after_task}.pt")


def evaluate_masked_task(
    model: TitanGlobalClassifier,
    test_loader,
    task_id: int,
    label_offset: int,
    task_num_classes: int,
    device: torch.device,
    k: int,
    patch_size: int,
) -> Dict[str, float]:
    model.eval()
    preds_all, targets_all = [], []
    patch_size_tensor = torch.tensor(patch_size, dtype=torch.int32, device=device)

    with torch.no_grad():
        for features, coords, labels in tqdm(test_loader, desc=f"masked task {task_id}", leave=False):
            features, coords = _sample_patches(features, coords, device, k)
            targets = labels.to(device, non_blocking=True).long() + label_offset
            logits = model(features, coords, patch_size_tensor).float()
            task_logits = logits[:, label_offset:label_offset + task_num_classes]
            preds = task_logits.argmax(1) + label_offset

            preds_all.append(preds.detach().cpu().numpy())
            targets_all.append(targets.detach().cpu().numpy())

    preds = np.concatenate(preds_all)
    targets = np.concatenate(targets_all)
    return {
        "masked_acc": float(accuracy_score(targets, preds)),
        "masked_bacc": float(balanced_accuracy_score(targets, preds)),
        "n": float(len(targets)),
    }


def write_csv(path: str, rows: List[dict], fieldnames: List[str]) -> None:
    if not rows:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_fold(rows: List[dict], fold_id: int) -> dict:
    max_task = max(int(row["after_task"]) for row in rows)
    final_rows = [row for row in rows if int(row["after_task"]) == max_task]
    return {
        "fold": fold_id,
        "num_tasks": max_task + 1,
        "final_masked_acc": float(np.mean([float(row["masked_acc"]) for row in final_rows])),
        "final_masked_bacc": float(np.mean([float(row["masked_bacc"]) for row in final_rows])),
    }


def append_mean_std(rows: List[dict], metric_cols: List[str]) -> List[dict]:
    output_rows = list(rows)
    mean_row = {"fold": "mean", "num_tasks": float(np.mean([float(row["num_tasks"]) for row in rows]))}
    std_row = {"fold": "std", "num_tasks": float(np.std([float(row["num_tasks"]) for row in rows], ddof=0))}

    for col in metric_cols:
        values = np.array([float(row[col]) for row in rows], dtype=float)
        mean_row[col] = float(np.nanmean(values))
        std_row[col] = float(np.nanstd(values, ddof=0))

    output_rows.extend([mean_row, std_row])
    return output_rows


def build_summary(rows: List[dict]) -> List[dict]:
    fold_rows = []
    for fold_id in sorted({int(row["fold"]) for row in rows}):
        fold_rows.append(summarize_fold([row for row in rows if int(row["fold"]) == fold_id], fold_id))
    return append_mean_std(fold_rows, ["final_masked_acc", "final_masked_bacc"])


def main() -> None:
    parser = argparse.ArgumentParser(description="TASK-IL masked bACC for DER++/A-GEM/ER-ACE checkpoints")
    parser.add_argument("--method", choices=sorted(METHOD_DEFAULTS), required=True)
    parser.add_argument("--ckpt_dir", type=str, default=None, help="Checkpoint directory. Defaults to method config save_dir.")
    parser.add_argument("--checkpoint_prefix", type=str, default=None, help="Checkpoint filename prefix.")
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--summary_csv", type=str, default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fold_start", type=int, default=0)
    parser.add_argument("--fold_end", type=int, default=None)
    parser.add_argument("--after_task", type=int, default=None, help="Evaluate one checkpoint sequence only.")
    parser.add_argument("--final_only", action="store_true", help="Evaluate only the final after-task checkpoint.")
    parser.add_argument("--allow_missing", action="store_true", help="Skip missing checkpoints instead of raising.")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    defaults = METHOD_DEFAULTS[args.method]
    cfg = load_config(default_filename=defaults["config"])
    method_cfg = cfg.get(defaults["config_key"], {})

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else method_cfg.get("save_dir", defaults["save_dir"])
    prefix = args.checkpoint_prefix if args.checkpoint_prefix is not None else defaults["prefix"]
    num_folds = int(_cfg_value(args, method_cfg, "num_folds", 10))
    num_workers = int(_cfg_value(args, method_cfg, "num_workers", 4))
    k = int(_cfg_value(args, method_cfg, "k", 400))
    patch_size = int(_cfg_value(args, method_cfg, "patch_size", 1024))
    seed = int(_cfg_value(args, method_cfg, "seed", 0))
    use_wandb = (args.use_wandb or method_cfg.get("use_wandb", False)) and not args.disable_wandb
    if use_wandb:
        try:
            import wandb  # noqa: F401
        except ImportError:
            warnings.warn("wandb package not found. Disabling wandb tracking.")
            use_wandb = False

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, seed)

    seq_dataset = Sequential_Generic_MIL_Dataset()
    all_num_classes = seq_dataset.num_classes
    if args.num_tasks is not None:
        num_tasks = int(args.num_tasks)
    else:
        cfg_num_tasks = method_cfg.get("num_tasks")
        num_tasks = int(cfg_num_tasks) if cfg_num_tasks is not None else len(all_num_classes)
    num_classes = all_num_classes[:num_tasks]
    offsets = _class_offsets(num_classes)
    total_classes = sum(num_classes)

    fold_end = int(args.fold_end if args.fold_end is not None else num_folds)
    if args.after_task is not None:
        after_tasks = [int(args.after_task)]
    elif args.final_only:
        after_tasks = [num_tasks - 1]
    else:
        after_tasks = list(range(num_tasks))

    raw_rows: List[dict] = []

    for fold_id in tqdm(range(int(args.fold_start), fold_end), desc="folds"):
        if use_wandb:
            import wandb
            wandb.init(
                project=method_cfg.get("wandb_project", f"MergeSlide-{args.method.upper()}"),
                entity=method_cfg.get("wandb_entity"),
                group=f"{args.method}_eval_taskil_masked",
                job_type="eval_taskil_masked",
                name=f"{args.method}_masked_taskil_fold_{fold_id}",
                config={
                    "fold": fold_id,
                    "method": args.method,
                    "num_tasks": num_tasks,
                    "k": k,
                    "patch_size": patch_size,
                    "after_tasks": after_tasks,
                },
                reinit=True,
            )
        base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)
        model = TitanGlobalClassifier(base_model, total_classes).to(device)

        for after_task in after_tasks:
            ckpt_path = checkpoint_path(ckpt_dir, prefix, fold_id, after_task)
            if not os.path.exists(ckpt_path):
                if args.allow_missing:
                    print(f"Skip missing checkpoint: {ckpt_path}")
                    continue
                raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

            checkpoint = _torch_load(ckpt_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            for eval_task_id in range(after_task + 1):
                _, _, test_loader = seq_dataset.get_data_loaders(fold_id, eval_task_id, num_workers=num_workers)
                metrics = evaluate_masked_task(
                    model=model,
                    test_loader=test_loader,
                    task_id=eval_task_id,
                    label_offset=offsets[eval_task_id],
                    task_num_classes=num_classes[eval_task_id],
                    device=device,
                    k=k,
                    patch_size=patch_size,
                )
                row = {
                    "fold": fold_id,
                    "after_task": after_task,
                    "eval_task": eval_task_id,
                    **metrics,
                }
                raw_rows.append(row)
                print(
                    f"masked eval fold={fold_id} after_task={after_task} task={eval_task_id}: "
                    f"masked_acc={metrics['masked_acc']:.4f} masked_bacc={metrics['masked_bacc']:.4f}"
                )
                if use_wandb:
                    import wandb
                    wandb.log({
                        "eval/after_task": after_task,
                        "eval/task_id": eval_task_id,
                        **{f"eval/{key}": value for key, value in metrics.items()},
                    })

        if use_wandb:
            import wandb
            fold_rows = [row for row in raw_rows if int(row["fold"]) == fold_id]
            if fold_rows:
                final_after_task = max(int(row["after_task"]) for row in fold_rows)
                final_rows = [row for row in fold_rows if int(row["after_task"]) == final_after_task]
                wandb.log({
                    "eval/final_masked_acc": float(np.mean([row["masked_acc"] for row in final_rows])),
                    "eval/final_masked_bacc": float(np.mean([row["masked_bacc"] for row in final_rows])),
                })
            wandb.finish()

    output_csv = args.output_csv or os.path.join(ckpt_dir, f"{prefix}_taskil_masked_eval.csv")
    summary_csv = args.summary_csv or os.path.join(ckpt_dir, f"{prefix}_taskil_masked_summary_per_fold.csv")

    write_csv(
        output_csv,
        raw_rows,
        ["fold", "after_task", "eval_task", "masked_acc", "masked_bacc", "n"],
    )
    summary_rows = build_summary(raw_rows) if raw_rows else []
    write_csv(
        summary_csv,
        summary_rows,
        ["fold", "num_tasks", "final_masked_acc", "final_masked_bacc"],
    )
    print(f"CSV saved: {output_csv}")
    print(f"CSV saved: {summary_csv}")


if __name__ == "__main__":
    main()
