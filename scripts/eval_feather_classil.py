"""
Naive CLASS-IL evaluation for FEATHER.

The merged FEATHER backbone is paired with one global classifier formed by
concatenating the independently trained per-task classifier heads.  Prompt/TCP
routing is intentionally not part of the FEATHER pipeline.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score, f1_score
from tqdm import tqdm

from mergeslide.datasets import Sequential_Generic_MIL_Dataset, get_dict_convert_class
from mergeslide.feather_models import (
    DEFAULT_FEATHER_MODEL_NAME,
    FeatherMILWrapper,
    create_feather_model,
    prepare_hf_token_env,
    sample_patch_bag,
    split_feather_state_dict,
)
from mergeslide.utils import get_eval_metrics, seed_torch


REPO_ROOT = Path(__file__).resolve().parent.parent
FEATHER_CONFIG = REPO_ROOT / "configs" / "feather.yaml"
FEATHER_MERGE_CONFIG = REPO_ROOT / "configs" / "merge_feather.yaml"
FEATHER_EVAL_CONFIG = REPO_ROOT / "configs" / "eval_feather.yaml"


def _load_feather_cfg() -> dict:
    with open(FEATHER_CONFIG, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get("feather", {})


def _load_merge_cfg() -> dict:
    with open(FEATHER_MERGE_CONFIG, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get("feather_merging", {})


def _load_eval_cfg() -> dict:
    with open(FEATHER_EVAL_CONFIG, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get("feather_evaluation", {})


def _torch_load(path: str, map_location="cpu"):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _classifier_state(checkpoint: dict, model, num_classes: int) -> tuple:
    full_state = checkpoint.get("model_state_dict", checkpoint)
    classifier_keys = tuple(checkpoint.get("classifier_state_keys", ()))
    if "head_state_dict" in checkpoint:
        head_state = checkpoint["head_state_dict"]
        if not classifier_keys:
            _, _, classifier_keys = split_feather_state_dict(model, num_classes=num_classes)
    else:
        _, inferred_head_state, inferred_classifier_keys = split_feather_state_dict(
            model,
            num_classes=num_classes,
        )
        classifier_keys = classifier_keys or inferred_classifier_keys
        head_state = {key: full_state[key] for key in inferred_head_state if key in full_state}

    missing = [key for key in classifier_keys if key not in head_state]
    if missing:
        missing = [key for key in classifier_keys if key not in full_state]
        if missing:
            raise KeyError(f"Checkpoint is missing FEATHER classifier keys: {missing}")
        head_state = {key: full_state[key] for key in classifier_keys}

    extra_head_keys = sorted(set(head_state) - set(classifier_keys))
    if extra_head_keys:
        raise NotImplementedError(
            "Naive FEATHER CLASS-IL currently supports one final linear classifier "
            f"per task. Additional head keys found: {extra_head_keys[:5]}"
        )

    return {key: head_state[key] for key in classifier_keys}, classifier_keys


def _build_global_model(
    *,
    model_name: str,
    num_classes_per_task: list,
    checkpoint_paths: list,
    merged_backbone_path: str,
    device: torch.device,
    from_pretrained_arch: bool,
) -> FeatherMILWrapper:
    total_classes = sum(num_classes_per_task)
    base_model = create_feather_model(
        model_name,
        num_classes=total_classes,
        from_pretrained=from_pretrained_arch,
    )
    merged_backbone = _torch_load(merged_backbone_path)
    _, unexpected = base_model.load_state_dict(merged_backbone, strict=False)
    if unexpected:
        raise KeyError(f"Merged FEATHER backbone has unexpected keys: {unexpected[:5]}")

    _, _, global_classifier_keys = split_feather_state_dict(base_model, num_classes=total_classes)
    global_head_state = {}
    task_classifier_states = []
    reference_keys = None
    for task_id, checkpoint_path in enumerate(checkpoint_paths):
        task_model = create_feather_model(
            model_name,
            num_classes=num_classes_per_task[task_id],
            from_pretrained=False,
        )
        checkpoint = _torch_load(checkpoint_path)
        classifier_state, classifier_keys = _classifier_state(
            checkpoint,
            task_model,
            num_classes=num_classes_per_task[task_id],
        )
        if reference_keys is None:
            reference_keys = classifier_keys
        elif classifier_keys != reference_keys:
            raise ValueError(
                "Task checkpoints expose different FEATHER classifier keys; "
                "naive CLASS-IL requires a shared classifier layout."
            )
        task_classifier_states.append(classifier_state)

    if len(global_classifier_keys) != len(reference_keys or ()):
        raise ValueError(
            "Global FEATHER model classifier differs from task classifier layout."
        )

    for task_key, global_key in zip(reference_keys or (), global_classifier_keys):
        tensors = [state[task_key] for state in task_classifier_states]
        if tensors[0].ndim not in (1, 2):
            raise ValueError(f"Unsupported FEATHER classifier tensor: {task_key} {tuple(tensors[0].shape)}")
        combined = torch.cat(tensors, dim=0)
        expected_shape = tuple(base_model.state_dict()[global_key].shape)
        if tuple(combined.shape) != expected_shape:
            raise ValueError(
                f"Cannot concatenate {task_key}: got {tuple(combined.shape)}, "
                f"expected global shape {expected_shape}."
            )
        global_head_state[global_key] = combined

    _, unexpected = base_model.load_state_dict(global_head_state, strict=False)
    if unexpected:
        raise KeyError(f"Global FEATHER head has unexpected keys: {unexpected[:5]}")
    return FeatherMILWrapper(base_model, num_classes=total_classes).to(device)


def evaluate_task(
    test_loader,
    *,
    task_id: int,
    model: nn.Module,
    dict_convert_class: dict,
    device: torch.device,
    k: int,
    patch_size: int,
):
    model.eval()
    preds_all, probs_all, targets_all = [], [], []
    patch_size_tensor = torch.tensor(patch_size, dtype=torch.int32, device=device)
    amp_enabled = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        for features, coords, labels in tqdm(test_loader, leave=False):
            features, coords = sample_patch_bag(features, coords, k)
            features = features.to(device, non_blocking=True)
            coords = coords.long().to(device, non_blocking=True)
            logits = model(features, coords, patch_size_tensor).float()
            probs = nn.functional.softmax(logits, dim=1)
            preds_all.append(logits.argmax(1).cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            targets_all.append(
                np.asarray([dict_convert_class[task_id][int(labels[0])]], dtype=np.int64)
            )

    preds_all = np.concatenate(preds_all)
    probs_all = np.concatenate(probs_all)
    targets_all = np.concatenate(targets_all)
    roc_kwargs = {"multi_class": "ovo", "average": "macro"} if probs_all.shape[1] > 2 else {}
    metrics = get_eval_metrics(targets_all, preds_all, probs_all, roc_kwargs=roc_kwargs)
    return metrics, preds_all, targets_all


def build_summary_frames(task_rows: list, fold_rows: list) -> tuple:
    """Build TITAN-style per-fold and per-task summaries from detailed rows."""
    metric_columns = ("acc", "bacc", "macro_f1", "weighted_f1")
    per_fold = pd.DataFrame(fold_rows, columns=("fold", *metric_columns))
    if per_fold.empty:
        return per_fold, pd.DataFrame(columns=("task", "mean_acc", "std_acc"))

    summary_rows = [*per_fold.to_dict("records")]
    for label, reducer in (("mean", np.mean), ("std", np.std)):
        summary_rows.append({
            "fold": label,
            **{metric: float(reducer(per_fold[metric].to_numpy(dtype=float))) for metric in metric_columns},
        })

    per_task = pd.DataFrame(task_rows)
    task_summary_rows = []
    for task_id in sorted(per_task["task"].unique()):
        values = per_task.loc[per_task["task"] == task_id, "acc"].to_numpy(dtype=float)
        task_summary_rows.append({
            "task": int(task_id),
            "mean_acc": float(np.mean(values)),
            "std_acc": float(np.std(values)),
        })
    return pd.DataFrame(summary_rows), pd.DataFrame(task_summary_rows)


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser(description="Naive CLASS-IL evaluation for FEATHER")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--merge_model_path", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--fold_start", type=int, default=0)
    parser.add_argument("--fold_end", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--from_pretrained_arch", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    feather_cfg = _load_feather_cfg()
    merge_cfg = _load_merge_cfg()
    eval_cfg = _load_eval_cfg()
    model_name = str(args.model_name or feather_cfg.get("model_name", DEFAULT_FEATHER_MODEL_NAME))
    save_dir = str(
        args.save_dir or eval_cfg.get("save_dir", feather_cfg.get("save_dir", "./checkpoints/feather_finetuned"))
    )
    merge_model_path = str(
        args.merge_model_path
        or eval_cfg.get("merge_model_path")
        or merge_cfg.get("des_merged_checkpoints", "./checkpoints/feather_merged")
    )
    num_folds = int(args.num_folds if args.num_folds is not None else eval_cfg.get("num_folds", 10))
    num_workers = int(args.num_workers if args.num_workers is not None else eval_cfg.get("num_workers", 0))
    k = int(args.k if args.k is not None else eval_cfg.get("k", feather_cfg.get("k", 400)))
    patch_size = int(args.patch_size if args.patch_size is not None else eval_cfg.get("patch_size", feather_cfg.get("patch_size", 256)))
    output_csv = args.output_csv or eval_cfg.get("classil_output_csv")
    fold_end = int(args.fold_end if args.fold_end is not None else num_folds)
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
    seq_dataset = Sequential_Generic_MIL_Dataset(config_path=str(FEATHER_CONFIG))
    num_tasks = int(args.num_tasks if args.num_tasks is not None else eval_cfg.get("num_tasks", len(seq_dataset.num_classes)))
    num_classes = seq_dataset.num_classes[:num_tasks]
    dict_convert_class = get_dict_convert_class(num_classes)

    rows = []
    task_rows = []
    fold_summary_rows = []
    for fold_id in range(args.fold_start, fold_end):
        fold = f"fold_{fold_id}"
        if use_wandb:
            import wandb
            wandb.init(
                project=eval_cfg.get("wandb_project", "MergeSlide-FEATHER"),
                entity=eval_cfg.get("wandb_entity"),
                group="feather_eval_classil",
                job_type="eval_classil",
                name=f"feather_classil_{fold}",
                config={
                    "fold": fold_id,
                    "model_name": model_name,
                    "num_tasks": num_tasks,
                    "k": k,
                    "patch_size": patch_size,
                    "mode": "naive",
                },
                reinit=True,
            )
        if os.path.isdir(merge_model_path):
            merged_backbone_path = os.path.join(
                merge_model_path,
                f"_{fold}",
                f"merged_backbone_feather_opcm_{fold}_task_{num_tasks - 1}.pth",
            )
        else:
            merged_backbone_path = merge_model_path
        if not os.path.exists(merged_backbone_path):
            raise FileNotFoundError(f"Missing merged FEATHER backbone: {merged_backbone_path}")

        checkpoint_paths = [
            os.path.join(save_dir, fold, f"feather_task_{task_id}.pt")
            for task_id in range(num_tasks)
        ]
        model = _build_global_model(
            model_name=model_name,
            num_classes_per_task=num_classes,
            checkpoint_paths=checkpoint_paths,
            merged_backbone_path=merged_backbone_path,
            device=device,
            from_pretrained_arch=args.from_pretrained_arch,
        )

        fold_accs, fold_baccs = [], []
        fold_preds, fold_targets = [], []
        for task_id in range(num_tasks):
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id, num_workers=num_workers)
            metrics, preds_all, targets_all = evaluate_task(
                test_loader,
                task_id=task_id,
                model=model,
                dict_convert_class=dict_convert_class,
                device=device,
                k=k,
                patch_size=patch_size,
            )
            acc = float((preds_all == targets_all).mean())
            bacc = balanced_accuracy_score(targets_all, preds_all)
            fold_accs.append(acc)
            fold_baccs.append(bacc)
            task_row = {
                "fold": fold_id,
                "task": task_id,
                "acc": metrics.get("/acc", np.nan),
                "bacc": metrics.get("/bacc", np.nan),
                "kappa": metrics.get("/kappa", np.nan),
                "nw_kappa": metrics.get("/nw_kappa", np.nan),
                "weighted_f1": metrics.get("/weighted_f1", np.nan),
                "loss": metrics.get("/loss", np.nan),
                "auroc": metrics.get("/auroc", np.nan),
            }
            rows.append(task_row)
            task_rows.append(task_row)
            fold_preds.append(preds_all)
            fold_targets.append(targets_all)
            if use_wandb:
                import wandb
                wandb.log({
                    "eval/task_id": task_id,
                    **{f"eval/task_{task_id}{key}": value for key, value in metrics.items()},
                })

        rows.append({
            "fold": fold_id,
            "task": "overall",
            "acc": float(np.mean(fold_accs)),
            "bacc": float(np.mean(fold_baccs)),
        })
        all_fold_preds = np.concatenate(fold_preds)
        all_fold_targets = np.concatenate(fold_targets)
        fold_summary_rows.append({
            "fold": fold_id,
            "acc": float(np.mean(fold_accs)),
            "bacc": float(np.mean(fold_baccs)),
            "macro_f1": float(f1_score(all_fold_targets, all_fold_preds, average="macro", zero_division=0)),
            "weighted_f1": float(f1_score(all_fold_targets, all_fold_preds, average="weighted", zero_division=0)),
        })
        print(
            f"[FEATHER] fold={fold_id} classil_acc={np.mean(fold_accs):.4f} "
            f"classil_bacc={np.mean(fold_baccs):.4f}"
        )
        if use_wandb:
            import wandb
            wandb.log({
                "eval/overall_acc": float(np.mean(fold_accs)),
                "eval/overall_bacc": float(np.mean(fold_baccs)),
            })
            wandb.finish()

    if output_csv:
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"[FEATHER] CSV saved: {output_csv}")
        output_base, output_ext = os.path.splitext(output_csv)
        output_ext = output_ext or ".csv"
        per_fold_csv = f"{output_base}_per_fold{output_ext}"
        per_task_csv = f"{output_base}_per_task{output_ext}"
        per_fold_df, per_task_df = build_summary_frames(task_rows, fold_summary_rows)
        per_fold_df.to_csv(per_fold_csv, index=False)
        per_task_df.to_csv(per_task_csv, index=False)
        print(f"[FEATHER] per-fold summary saved: {per_fold_csv}")
        print(f"[FEATHER] per-task summary saved: {per_task_csv}")


if __name__ == "__main__":
    main()
