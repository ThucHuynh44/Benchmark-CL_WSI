"""
scripts/eval_classil_metrics.py
CLASS-IL evaluation for continual learning metrics: mACC, Forgetting (FGT), BWT.

Evaluates intermediate merged checkpoints (one per accumulated task) to compute
metrics that require tracking performance over the continual learning sequence.

Usage:
    python scripts/eval_classil_metrics.py \
        --save_dir /path/to/finetuned/checkpoints \
        --merge_model_path /path/to/merged/checkpoints/
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel

from configs.loader import load_config
from mergeslide.datasets import Sequential_Generic_MIL_Dataset, get_dict_convert_class
from mergeslide.models import CustomSequential
from mergeslide.utils import get_eval_metrics, seed_torch

# Patch sampling budget per forward pass
K = 400


# DICT_CONVERT_CLASS sẽ được tính động sau khi load seq_dataset (dòng 217)
# (tránh hardcode class ranges khi số task thay đổi)


def forgetting(results: list) -> float:
    """Compute average forgetting over all tasks.

    Args:
        results: List of lists. results[i][j] = accuracy on task j after training on task i.

    Returns:
        Mean forgetting across tasks 0..n-2.
    """
    n_tasks = len(results)
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    return float(np.mean([maxx[i] - results[-1][i] for i in range(n_tasks - 1)]))


def backward_transfer(results: list) -> float:
    """Compute Backward Transfer (BWT).

    Args:
        results: Same format as for forgetting().

    Returns:
        Mean BWT across tasks 0..n-2.
    """
    n_tasks = len(results)
    return float(np.mean([results[-1][i] - results[i][i] for i in range(n_tasks - 1)]))


def evaluate(
    test_loader,
    task_id: int,
    model,
    num_classes: list,
    device,
    task_prompts,
    task_model_paths: list,
    dict_convert_class: dict,
    prefix: str = "",
):
    """Run CLASS-IL inference using task-to-class prompt routing.

    Args:
        test_loader: DataLoader for the test set.
        task_id: Ground-truth task id for this test loader.
        model: CustomSequential with merged backbone.
        num_classes: Number of classes per task.
        device: Target device.
        task_prompts: Task prompt embeddings (one per task, stacked).
        task_model_paths: Paths to per-task finetuned checkpoints.
        dict_convert_class: Mapping from task-local class ids to global class ids.
        prefix: Metric key prefix.

    Returns:
        (eval_metrics, preds_all, targets_all)
    """
    preds_all, probs_all, targets_all = [], [], []
    total_num_classes = sum(num_classes)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader, leave=False):
            indices = torch.randperm(features.shape[0])[:K]
            features = features.to(device)[indices]
            coords = coords.long().to(device)[indices]

            slide_embed = model.backbone(features, coords, torch.tensor(1024).int().to(device))
            predicted_task_id = int(torch.argmax(slide_embed @ task_prompts.T))

            raw = torch.load(task_model_paths[predicted_task_id], map_location='cpu')
            mlp_state = {k.split('mlp.')[-1]: raw[k] for k in list(raw.keys())[-2:]}
            mlp = nn.Linear(768, num_classes[predicted_task_id]).to(device)
            mlp.load_state_dict(mlp_state)

            logits = mlp(slide_embed).float()
            probs_local = nn.functional.softmax(logits, dim=1)
            pred_local = int(logits.argmax(1).item())

            true_local = int(label[0])
            true_global = dict_convert_class[task_id][true_local]
            pred_global = dict_convert_class[predicted_task_id][pred_local]

            probs_global = np.zeros((1, total_num_classes), dtype=np.float32)
            for local_idx, global_idx in dict_convert_class[predicted_task_id].items():
                probs_global[0, global_idx] = float(probs_local[0, local_idx].detach().cpu())

            preds_all.append(np.array([pred_global], dtype=np.int64))
            probs_all.append(probs_global)
            targets_all.append(np.array([true_global], dtype=np.int64))

    preds_all = np.concatenate(preds_all)
    probs_all = np.concatenate(probs_all)
    targets_all = np.concatenate(targets_all)

    roc_kwargs = {"multi_class": "ovo", "average": "macro"}
    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, roc_kwargs=roc_kwargs, prefix=prefix)
    return eval_metrics, preds_all, targets_all


def build_merged_mlp(task_model_paths: list, device) -> nn.Linear:
    """Build the naive classifier for the currently accumulated tasks."""
    mlp_task_weights = []
    for path in task_model_paths:
        state = torch.load(path, map_location='cpu')
        mlp_task_weights.append(
            {k.split('mlp.')[-1]: state[k] for k in list(state.keys())[-2:]}
        )
    merged_mlp_data = {
        "weight": torch.cat([w["weight"] for w in mlp_task_weights], dim=0),
        "bias": torch.cat([w["bias"] for w in mlp_task_weights], dim=0),
    }
    merged_mlp = nn.Linear(768, merged_mlp_data["weight"].shape[0]).to(device)
    merged_mlp.load_state_dict(merged_mlp_data)
    merged_mlp.eval()
    return merged_mlp


def evaluate_naive(
    test_loader,
    task_id: int,
    model,
    device,
    merged_mlp,
    prefix: str = "",
):
    """Run CLASS-IL naive inference with a concatenated global classifier."""
    preds_all, probs_all, targets_all = [], [], []

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader, leave=False):
            indices = torch.randperm(features.shape[0])[:K]
            features = features.to(device)[indices]
            coords = coords.long().to(device)[indices]

            slide_embed = model.backbone(features, coords, torch.tensor(1024).int().to(device))
            logits = merged_mlp(slide_embed.float()).float()
            preds = logits.argmax(1)
            probs = nn.functional.softmax(logits, dim=1)
            global_label = DICT_CONVERT_CLASS[task_id][int(label)]

            preds_all.append(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            targets_all.append(np.array([global_label]))

    preds_all = np.concatenate(preds_all)
    probs_all = np.concatenate(probs_all)
    targets_all = np.concatenate(targets_all)

    roc_kwargs = {"multi_class": "ovo", "average": "macro"}
    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, roc_kwargs=roc_kwargs, prefix=prefix)
    return eval_metrics, preds_all, targets_all


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser(description="CLASS-IL: mACC, FGT, BWT metrics")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory with per-task finetuned checkpoints")
    parser.add_argument("--merge_model_path", type=str, default=None,
                        help="Directory with merged model checkpoints")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional path to save results as CSV (e.g. results_metrics.csv)")
    parser.add_argument("--mode", type=str, default=None, choices=["tcp", "naive"],
                        help="CLASS-IL inference mode: tcp routes by task prompts; naive uses one global MLP")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers for evaluation. Use 0 to debug HDF5/IO stalls.")
    args = parser.parse_args()

    cfg = load_config(default_filename="eval.yaml")
    eval_cfg = cfg.get("evaluation", {})

    save_dir = args.save_dir if args.save_dir is not None else eval_cfg.get("save_dir", "./checkpoints/finetuned")
    merge_model_path = args.merge_model_path if args.merge_model_path is not None else eval_cfg.get("merge_model_path", "./checkpoints/merged")
    mode = args.mode if args.mode is not None else eval_cfg.get("mode", "tcp")
    output_csv = args.output_csv if args.output_csv is not None else eval_cfg.get("classil_metrics_output_csv")
    num_folds = int(eval_cfg.get("num_folds", 10))
    num_workers = int(args.num_workers if args.num_workers is not None else eval_cfg.get("num_workers", 0))

    seq_dataset = Sequential_Generic_MIL_Dataset()
    num_classes = seq_dataset.num_classes
    num_tasks = int(eval_cfg.get("num_tasks", len(num_classes)))
    num_classes = num_classes[:num_tasks]
    DICT_CONVERT_CLASS = get_dict_convert_class(num_classes)
    list_num_tasks = list(range(1, num_tasks + 1))
    task_prompts_all = (
        torch.load("./task_prompts.pt", map_location=device).to(device)
        if mode == "tcp" else None
    )

    mACCs_all_folds, fgt_all_folds, bwt_all_folds = [], [], []
    ACC_all_seqs_all_folds = []

    for fold_id in tqdm(range(num_folds)):
        fold = f"fold_{fold_id}"
        task_model_paths = [
            f"{save_dir}/{fold}/ckpts_outputs_finetuning_task_{task_id}.pt"
            for task_id in range(num_tasks)
        ]

        acc_per_task_all_tasks = []
        ACC_all_seqs = []

        for seq_task in tqdm(list_num_tasks, leave=False):
            seed_torch(device, 0)
            acc_per_task = [0.0] * seq_task

            # Load the merged checkpoint after `seq_task` tasks
            resolved_merge_model_path = (
                f"{merge_model_path}/_{fold}"
                f"/merged_weight_opcm_random_sampling_{fold}_task_{seq_task - 1}.pth"
            )

            base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
            base_model = base_model.to(device)
            base_model.vision_encoder.load_state_dict(torch.load(resolved_merge_model_path, map_location=device))
            model = CustomSequential(base_model, nn.Identity())
            model.eval()

            task_prompts = task_prompts_all[:seq_task] if mode == "tcp" else None
            merged_mlp = (
                build_merged_mlp(task_model_paths[:seq_task], device)
                if mode == "naive" else None
            )
            num_correct, num_total = 0.0, 0.0

            for task_id in range(seq_task):
                _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id, num_workers=num_workers)
                if mode == "tcp":
                    results, preds_all, targets_all = evaluate(
                        test_loader, task_id, model,
                        num_classes[:seq_task], device,
                        task_prompts, task_model_paths[:seq_task],
                        DICT_CONVERT_CLASS,
                    )
                else:
                    results, preds_all, targets_all = evaluate_naive(
                        test_loader, task_id, model, device, merged_mlp,
                    )
                num_correct += sum(preds_all == targets_all)
                num_total += len(test_loader)
                acc_per_task[task_id] = sum(preds_all == targets_all) / len(targets_all)

            overall_acc = num_correct / num_total
            ACC_all_seqs.append(float(overall_acc))
            acc_per_task_all_tasks.append(acc_per_task)

        mACC = float(np.mean(ACC_all_seqs))
        ACC_all_seqs_all_folds.append(ACC_all_seqs)
        mACCs_all_folds.append(mACC)
        fgt_all_folds.append(forgetting(acc_per_task_all_tasks))
        bwt_all_folds.append(backward_transfer(acc_per_task_all_tasks))

    print(ACC_all_seqs_all_folds)
    print(f"mACC: {np.mean(mACCs_all_folds):.4f} (std {np.std(mACCs_all_folds):.4f})")
    print(f"BWT:  {np.mean(bwt_all_folds):.4f} (std {np.std(bwt_all_folds):.4f})")
    print(f"FGT:  {np.mean(fgt_all_folds):.4f} (std {np.std(fgt_all_folds):.4f})")

    if output_csv:
        rows = []
        for i in range(num_folds):
            rows.append({
                "fold": i,
                "mACC": mACCs_all_folds[i],
                "BWT": bwt_all_folds[i],
                "FGT": fgt_all_folds[i],
            })
        rows.append({
            "fold": "mean",
            "mACC": np.mean(mACCs_all_folds),
            "BWT": np.mean(bwt_all_folds),
            "FGT": np.mean(fgt_all_folds),
        })
        rows.append({
            "fold": "std",
            "mACC": np.std(mACCs_all_folds),
            "BWT": np.std(bwt_all_folds),
            "FGT": np.std(fgt_all_folds),
        })
        df = pd.DataFrame(rows)
        df.to_csv(output_csv, index=False)
        print(f"\nCSV saved: {output_csv}")
