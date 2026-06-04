"""
scripts/eval_taskil.py
TASK-IL evaluation: each task uses its own task-specific MLP head (oracle task identity).

Usage:
    python scripts/eval_taskil.py \
        --save_dir /path/to/finetuned/checkpoints \
        --merge_model_path /path/to/merged/checkpoints/fold_0.pth
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModel

from configs.loader import load_config
from mergeslide.datasets import Sequential_Generic_MIL_Dataset
from mergeslide.models import CustomSequential
from mergeslide.utils import get_eval_metrics, seed_torch

# Patch sampling budget per forward pass
K = 300


def evaluate(test_loader, model, num_classes: int, device, prefix: str = "", **kwargs):
    """Evaluate a model with a task-specific MLP head (TASK-IL setting).

    Args:
        test_loader: DataLoader for the test set.
        model: CustomSequential with merged backbone + task MLP.
        num_classes: Number of classes for this task.
        device: Target device.
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

            try:
                logits = model(features, coords, torch.tensor(1024).int().to(device), **kwargs)
            except Exception:
                model.cpu()
                logits = model(features, coords, torch.tensor(1024).int().cpu(), **kwargs)
                model.to(device)

            logits = logits.float()
            preds = logits.argmax(1)
            if num_classes == 2:
                probs = nn.functional.softmax(logits, dim=1)[:, 1]
                roc_kwargs = {}
            else:
                probs = nn.functional.softmax(logits, dim=1)
                roc_kwargs = {"multi_class": "ovo", "average": "macro"}

            preds_all.append(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            targets_all.append(label.numpy())

    preds_all = np.concatenate(preds_all)
    probs_all = np.concatenate(probs_all)
    targets_all = np.concatenate(targets_all)

    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, roc_kwargs=roc_kwargs, prefix=prefix)
    return eval_metrics, preds_all, targets_all


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, 0)

    parser = argparse.ArgumentParser(description="TASK-IL evaluation with oracle task identity")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory with per-task finetuned checkpoints")
    parser.add_argument("--merge_model_path", type=str, default=None,
                        help="Path to the merged backbone checkpoint (.pth file)")
    args = parser.parse_args()

    cfg = load_config(default_filename="merge.yaml")
    eval_cfg = cfg.get("evaluation", {})

    save_dir = args.save_dir if args.save_dir is not None else eval_cfg.get("save_dir", "./checkpoints/finetuned")
    merge_model_path = args.merge_model_path if args.merge_model_path is not None else eval_cfg.get("merge_model_path", "./checkpoints/merged")

    # If it is a directory, resolve to the default fold 0 file
    if merge_model_path and os.path.isdir(merge_model_path):
        merge_model_path = os.path.join(merge_model_path, "merged_weight_opcm_random_sampling_fold_0.pth")

    num_tasks = 6
    num_classes = [2, 3, 2, 2, 2, 2]
    seq_dataset = Sequential_Generic_MIL_Dataset()

    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)
    base_model.vision_encoder.load_state_dict(torch.load(merge_model_path))

    overall_accs, all_acc_per_task = [], []

    for fold_id in tqdm(range(1)):  # Note: currently evaluates fold 0 only
        fold = f"fold_{fold_id}"
        task_models = [
            f"{save_dir}/{fold}/ckpts_outputs_finetuning_task_{task_id}.pt"
            for task_id in range(num_tasks)
        ]

        acc_per_task = {}
        all_baccs, all_accs = [], []
        num_correct, num_total = 0.0, 0.0

        for task_id in range(num_tasks):
            print(f"TASK {task_id}")
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            mlp = nn.Linear(768, num_classes[task_id]).to(device)
            mlp.weight.data.normal_(mean=0.0, std=0.01)
            mlp.bias.data.zero_()
            model = CustomSequential(base_model, mlp)

            task_weight = torch.load(task_models[task_id], map_location='cpu')
            model.mlp.load_state_dict(
                {k.split('mlp.')[-1]: task_weight[k] for k in list(task_weight.keys())[-2:]}
            )
            model.eval()

            results, preds_all, targets_all = evaluate(
                test_loader, model, num_classes[task_id], device, prefix=""
            )
            print(results)

            num_correct += sum(preds_all == targets_all)
            num_total += len(test_loader)
            all_baccs.append(balanced_accuracy_score(targets_all, preds_all))
            acc_per_task[task_id] = results['/acc']
            all_accs.append(sum(preds_all == targets_all) / len(test_loader))

        overall_bacc = float(np.mean(all_baccs))
        overall_acc_norm = float(np.mean(all_accs))
        overall_accs.append(overall_bacc)
        all_acc_per_task.append(acc_per_task)
        print(f"Balanced Accuracy: {overall_bacc:.4f}")
        print(f"Accuracy:          {overall_acc_norm:.4f}")

    print(f"\n[Acc per fold]: {[float(a) for a in overall_accs]}")
    print(f"Accuracy: {np.mean(overall_accs):.4f} ({np.std(overall_accs):.4f})")

    accs_by_task = {t: [] for t in range(num_tasks)}
    for fold_accs in all_acc_per_task:
        for t, acc in fold_accs.items():
            accs_by_task[t].append(acc)
    print("\nPer-task accuracy:")
    for t in range(num_tasks):
        print(f"  Task {t}: {np.mean(accs_by_task[t]):.4f} ({np.std(accs_by_task[t]):.4f})")
