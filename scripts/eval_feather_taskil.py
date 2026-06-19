"""
TASK-IL evaluation for FEATHER per-task checkpoints.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm

from mergeslide.datasets import Sequential_Generic_MIL_Dataset
from mergeslide.feather_models import (
    DEFAULT_FEATHER_MODEL_NAME,
    FeatherMILWrapper,
    create_feather_model,
    prepare_hf_token_env,
    sample_patch_bag,
)
from mergeslide.utils import get_eval_metrics, seed_torch


REPO_ROOT = Path(__file__).resolve().parent.parent
FEATHER_CONFIG = REPO_ROOT / "configs" / "feather.yaml"


def _load_feather_cfg() -> dict:
    with open(FEATHER_CONFIG, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get("feather", {})


def _torch_load(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def evaluate(test_loader, model, num_classes: int, device: torch.device, k: int, patch_size: int):
    model.eval()
    preds_all, probs_all, targets_all = [], [], []
    patch_size_tensor = torch.tensor(patch_size, dtype=torch.int32, device=device)
    amp_enabled = device.type == "cuda"

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp_enabled):
        for features, coords, labels in tqdm(test_loader, leave=False):
            features, coords = sample_patch_bag(features, coords, k)
            features = features.to(device, non_blocking=True)
            coords = coords.long().to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features, coords, patch_size_tensor).float()

            preds = logits.argmax(1)
            if num_classes == 2:
                probs = nn.functional.softmax(logits, dim=1)[:, 1]
                roc_kwargs = {}
            else:
                probs = nn.functional.softmax(logits, dim=1)
                roc_kwargs = {"multi_class": "ovo", "average": "macro"}

            preds_all.append(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            targets_all.append(labels.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    probs_all = np.concatenate(probs_all)
    targets_all = np.concatenate(targets_all)
    metrics = get_eval_metrics(targets_all, preds_all, probs_all, roc_kwargs=roc_kwargs)
    return metrics, preds_all, targets_all


def _load_model_from_checkpoint(
    checkpoint_path: str,
    *,
    model_name: str,
    num_classes: int,
    device: torch.device,
    from_pretrained_arch: bool,
) -> FeatherMILWrapper:
    checkpoint = _torch_load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        ckpt_model_name = checkpoint.get("model_name")
        if ckpt_model_name:
            model_name = ckpt_model_name
    else:
        state_dict = checkpoint

    base_model = create_feather_model(
        model_name,
        num_classes=num_classes,
        from_pretrained=from_pretrained_arch,
    )
    missing, unexpected = base_model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        print(
            f"[FEATHER] checkpoint={checkpoint_path} load_state_dict "
            f"missing={len(missing)} unexpected={len(unexpected)}"
        )
    model = FeatherMILWrapper(base_model, num_classes=num_classes).to(device)
    if isinstance(checkpoint, dict) and checkpoint.get("forward_mode"):
        model._forward_mode = checkpoint["forward_mode"]
    return model


def main() -> None:
    torch.multiprocessing.set_sharing_strategy("file_system")
    parser = argparse.ArgumentParser(description="TASK-IL evaluation for FEATHER")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--output_csv", type=str, default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--k", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--fold_start", type=int, default=0)
    parser.add_argument("--fold_end", type=int, default=None)
    parser.add_argument("--only_task", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--from_pretrained_arch", action="store_true",
                        help="Initialize from pretrained FEATHER before loading checkpoint weights.")
    args = parser.parse_args()

    feather_cfg = _load_feather_cfg()
    model_name = str(args.model_name or feather_cfg.get("model_name", DEFAULT_FEATHER_MODEL_NAME))
    save_dir = str(args.save_dir or feather_cfg.get("save_dir", "./checkpoints/feather_finetuned"))
    num_folds = int(args.num_folds if args.num_folds is not None else feather_cfg.get("num_folds", 10))
    num_workers = int(args.num_workers if args.num_workers is not None else feather_cfg.get("num_workers", 0))
    k = int(args.k if args.k is not None else feather_cfg.get("k", 400))
    patch_size = int(args.patch_size if args.patch_size is not None else feather_cfg.get("patch_size", 512))
    output_csv = args.output_csv
    fold_end = int(args.fold_end if args.fold_end is not None else num_folds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, args.seed)
    prepare_hf_token_env()

    seq_dataset = Sequential_Generic_MIL_Dataset(config_path=str(FEATHER_CONFIG))
    num_classes = seq_dataset.num_classes
    task_ids = [args.only_task] if args.only_task is not None else list(range(len(num_classes)))

    fold_summaries = []
    task_rows = []
    for fold_id in tqdm(range(args.fold_start, fold_end)):
        all_baccs, all_accs = [], []
        fold = f"fold_{fold_id}"
        for task_id in task_ids:
            checkpoint_path = os.path.join(save_dir, fold, f"feather_task_{task_id}.pt")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Missing FEATHER checkpoint: {checkpoint_path}")

            _, _, test_loader = seq_dataset.get_data_loaders(
                fold_id,
                task_id,
                num_workers=num_workers,
            )
            model = _load_model_from_checkpoint(
                checkpoint_path,
                model_name=model_name,
                num_classes=num_classes[task_id],
                device=device,
                from_pretrained_arch=args.from_pretrained_arch,
            )
            metrics, preds_all, targets_all = evaluate(
                test_loader,
                model,
                num_classes=num_classes[task_id],
                device=device,
                k=k,
                patch_size=patch_size,
            )
            acc = float((preds_all == targets_all).mean())
            bacc = balanced_accuracy_score(targets_all, preds_all)
            all_accs.append(acc)
            all_baccs.append(bacc)
            print(f"[FEATHER] fold={fold_id} task={task_id} metrics={metrics}")

            row = {"fold": fold_id, "task": task_id}
            row.update({
                "acc": metrics.get("/acc", np.nan),
                "bacc": metrics.get("/bacc", np.nan),
                "kappa": metrics.get("/kappa", np.nan),
                "nw_kappa": metrics.get("/nw_kappa", np.nan),
                "weighted_f1": metrics.get("/weighted_f1", np.nan),
                "loss": metrics.get("/loss", np.nan),
                "auroc": metrics.get("/auroc", np.nan),
            })
            task_rows.append(row)

        fold_summary = {
            "fold": fold_id,
            "task": "overall",
            "acc": float(np.mean(all_accs)),
            "bacc": float(np.mean(all_baccs)),
        }
        fold_summaries.append(fold_summary)
        print(
            f"[FEATHER] fold={fold_id} overall_acc={fold_summary['acc']:.4f} "
            f"overall_bacc={fold_summary['bacc']:.4f}"
        )

    if output_csv:
        rows = task_rows + fold_summaries
        rows.append({
            "fold": "mean",
            "task": "overall",
            "acc": np.mean([row["acc"] for row in fold_summaries]),
            "bacc": np.mean([row["bacc"] for row in fold_summaries]),
        })
        rows.append({
            "fold": "std",
            "task": "overall",
            "acc": np.std([row["acc"] for row in fold_summaries]),
            "bacc": np.std([row["bacc"] for row in fold_summaries]),
        })
        pd.DataFrame(rows).to_csv(output_csv, index=False)
        print(f"[FEATHER] CSV saved: {output_csv}")


if __name__ == "__main__":
    main()
