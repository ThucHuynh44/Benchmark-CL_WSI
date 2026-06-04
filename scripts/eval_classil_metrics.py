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

import argparse

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel

from configs.loader import load_config
from mergeslide.datasets import Sequential_Generic_MIL_Dataset
from mergeslide.models import CustomSequential, pad_numpy_arrays
from mergeslide.utils import get_eval_metrics, seed_torch

# Patch sampling budget per forward pass
K = 400


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
    model,
    num_classes: list,
    device,
    task_prompts,
    task_model_paths: list,
    prefix: str = "",
):
    """Run CLASS-IL inference using task-to-class prompt routing.

    Args:
        test_loader: DataLoader for the test set.
        model: CustomSequential with merged backbone.
        num_classes: Number of classes per task.
        device: Target device.
        task_prompts: Task prompt embeddings (one per task, stacked).
        task_model_paths: Paths to per-task finetuned checkpoints.
        prefix: Metric key prefix.

    Returns:
        (eval_metrics, preds_all, targets_all)
    """
    preds_all, probs_all, targets_all = [], [], []

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
            preds = logits.argmax(1)
            probs = nn.functional.softmax(logits, dim=1)

            preds_all.append(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            targets_all.append(label.numpy())

    preds_all = np.concatenate(preds_all)
    try:
        probs_all = np.concatenate(probs_all)
    except ValueError:
        probs_all = pad_numpy_arrays(probs_all)
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
    args = parser.parse_args()

    cfg = load_config(default_filename="merge.yaml")
    eval_cfg = cfg.get("evaluation", {})

    save_dir = args.save_dir if args.save_dir is not None else eval_cfg.get("save_dir", "./checkpoints/finetuned")
    merge_model_path = args.merge_model_path if args.merge_model_path is not None else eval_cfg.get("merge_model_path", "./checkpoints/merged")

    num_tasks = 6
    num_classes = [2, 3, 2, 2, 2, 2]
    list_num_tasks = list(range(1, num_tasks + 1))
    seq_dataset = Sequential_Generic_MIL_Dataset()
    task_prompts_all = torch.load("./task_prompts.pt")

    mACCs_all_folds, fgt_all_folds, bwt_all_folds = [], [], []
    ACC_all_seqs_all_folds = []

    for fold_id in tqdm(range(10)):
        fold = f"fold_{fold_id}"
        task_model_paths = [
            f"{save_dir}/{fold_id}/ckpts_outputs_finetuning_task_{task_id}.pt"
            for task_id in range(num_tasks)
        ]

        acc_per_task_all_tasks = []
        ACC_all_seqs = []

        for seq_task in tqdm(list_num_tasks, leave=False):
            seed_torch(device, 0)
            acc_per_task = [0.0] * seq_task

            # Load the merged checkpoint after `seq_task` tasks
            resolved_merge_model_path = (
                f"{merge_model_path}_{{fold}}"
                f"/merged_weight_opcm_random_sampling_{fold}_task_{seq_task - 1}.pth"
            ).format(fold=fold)

            base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
            base_model = base_model.to(device)
            base_model.vision_encoder.load_state_dict(torch.load(resolved_merge_model_path))
            model = CustomSequential(base_model, nn.Identity())
            model.eval()

            task_prompts = task_prompts_all[:seq_task]
            num_correct, num_total = 0.0, 0.0

            for task_id in range(seq_task):
                _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
                results, preds_all, targets_all = evaluate(
                    test_loader, model,
                    num_classes[:seq_task], device,
                    task_prompts, task_model_paths[:seq_task],
                )
                num_correct += sum(preds_all == targets_all)
                num_total += len(targets_all)
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
