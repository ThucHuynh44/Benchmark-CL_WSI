"""
scripts/eval_taskil.py
TASK-IL evaluation: each task uses its own task-specific MLP head (oracle task identity).

Usage:
    python scripts/eval_taskil.py \
        --save_dir /path/to/finetuned/checkpoints \
        --merge_model_path /path/to/merged/checkpoints
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import argparse
import time

import numpy as np
import pandas as pd
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
K = 400


def _torch_load_weights(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def evaluate(test_loader, model, num_classes: int, device, prefix: str = "", debug_io: bool = False, **kwargs):
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

    autocast_enabled = device.type == "cuda"
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
        for batch_idx, (features, coords, label) in enumerate(tqdm(test_loader, leave=False)):
            if debug_io:
                print(
                    f"[EVAL batch] idx={batch_idx}/{len(test_loader)} "
                    f"features={tuple(features.shape)} coords={tuple(coords.shape)} label={label.tolist()}",
                    flush=True,
                )
            indices = torch.randperm(features.shape[0])[:K]
            features = features.to(device)[indices]
            coords = coords.long().to(device)[indices]

            try:
                if debug_io:
                    print(f"[EVAL forward start] idx={batch_idx} sampled={features.shape[0]}", flush=True)
                start = time.time()
                logits = model(features, coords, torch.tensor(1024).int().to(device), **kwargs)
            except Exception:
                if debug_io:
                    print(f"[EVAL forward retry cpu] idx={batch_idx}", flush=True)
                model.cpu()
                logits = model(features.cpu(), coords.cpu(), torch.tensor(1024).int().cpu(), **kwargs)
                model.to(device)
            if debug_io:
                print(f"[EVAL forward done] idx={batch_idx} elapsed={time.time() - start:.3f}s", flush=True)

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
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional path to save results as CSV (e.g. results_taskil.csv)")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers for evaluation. Use 0 to debug HDF5/IO stalls.")
    parser.add_argument("--debug_io", action="store_true",
                        help="Print per-slide IO and forward timing to find stalls.")
    parser.add_argument("--fold_start", type=int, default=0,
                        help="First fold to evaluate, inclusive.")
    parser.add_argument("--fold_end", type=int, default=None,
                        help="Last fold to evaluate, exclusive. Defaults to num_folds.")
    parser.add_argument("--only_task", type=int, default=None,
                        help="Evaluate only one task id, useful for debugging stalls.")
    args = parser.parse_args()

    cfg = load_config(default_filename="eval.yaml")
    eval_cfg = cfg.get("evaluation", {})

    save_dir = args.save_dir if args.save_dir is not None else eval_cfg.get("save_dir", "./checkpoints/finetuned")
    merge_model_path = args.merge_model_path if args.merge_model_path is not None else eval_cfg.get("merge_model_path", "./checkpoints/merged")
    output_csv = args.output_csv if args.output_csv is not None else eval_cfg.get("taskil_output_csv")
    num_folds = int(eval_cfg.get("num_folds", 10))
    num_workers = int(args.num_workers if args.num_workers is not None else eval_cfg.get("num_workers", 0))
    fold_start = int(args.fold_start)
    fold_end = int(args.fold_end if args.fold_end is not None else num_folds)
    if args.debug_io:
        os.environ["MERGESLIDE_DEBUG_IO"] = "1"

    seq_dataset = Sequential_Generic_MIL_Dataset()
    num_classes = seq_dataset.num_classes
    num_tasks = int(eval_cfg.get("num_tasks", len(num_classes)))
    num_classes = num_classes[:num_tasks]
    task_ids_to_eval = [args.only_task] if args.only_task is not None else list(range(num_tasks))

    overall_accs, overall_baccs_folds, overall_acc_norms = [], [], []
    all_acc_per_task, all_results_per_task = [], []

    for fold_id in tqdm(range(fold_start, fold_end)):
        fold = f"fold_{fold_id}"
        task_models = [
            f"{save_dir}/{fold}/ckpts_outputs_finetuning_task_{task_id}.pt"
            for task_id in range(num_tasks)
        ]

        acc_per_task = {}
        results_per_task = {}
        all_baccs, all_accs = [], []
        num_correct, num_total = 0.0, 0.0

        # Load merged backbone for this fold
        if os.path.isdir(merge_model_path):
            fold_ckpt = os.path.join(
                merge_model_path,
                f"_{fold}",
                f"merged_weight_opcm_random_sampling_{fold}_task_{num_tasks - 1}.pth",
            )
        else:
            fold_ckpt = merge_model_path  # single file path (e.g. fold 0 only)
        base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device)
        base_model.vision_encoder.load_state_dict(_torch_load_weights(fold_ckpt, map_location=device))
        base_model.eval()

        for task_id in task_ids_to_eval:
            print(f"TASK {task_id}")
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id, num_workers=num_workers)

            mlp = nn.Linear(768, num_classes[task_id]).to(device)
            mlp.weight.data.normal_(mean=0.0, std=0.01)
            mlp.bias.data.zero_()
            model = CustomSequential(base_model, mlp)

            task_weight = _torch_load_weights(task_models[task_id], map_location='cpu')
            model.mlp.load_state_dict(
                {k.split('mlp.')[-1]: task_weight[k] for k in list(task_weight.keys())[-2:]}
            )
            model.eval()

            results, preds_all, targets_all = evaluate(
                test_loader, model, num_classes[task_id], device, prefix="", debug_io=args.debug_io
            )
            print(results)

            num_correct += sum(preds_all == targets_all)
            num_total += len(test_loader)
            all_baccs.append(balanced_accuracy_score(targets_all, preds_all))
            acc_per_task[task_id] = results['/acc']
            results_per_task[task_id] = results
            all_accs.append(sum(preds_all == targets_all) / len(test_loader))

        overall_bacc = float(np.mean(all_baccs))
        overall_acc_norm = float(np.mean(all_accs))
        overall_accs.append(overall_bacc)
        overall_baccs_folds.append(overall_bacc)
        overall_acc_norms.append(overall_acc_norm)
        all_acc_per_task.append(acc_per_task)
        all_results_per_task.append(results_per_task)
        print(f"Balanced Accuracy: {overall_bacc:.4f}")
        print(f"Accuracy:          {overall_acc_norm:.4f}")

    print(f"\n[Acc per fold]: {[float(a) for a in overall_accs]}")
    print(f"Accuracy: {np.mean(overall_accs):.4f} ({np.std(overall_accs):.4f})")

    accs_by_task = {t: [] for t in task_ids_to_eval}
    for fold_accs in all_acc_per_task:
        for t, acc in fold_accs.items():
            accs_by_task[t].append(acc)
    print("\nPer-task accuracy:")
    for t in task_ids_to_eval:
        print(f"  Task {t}: {np.mean(accs_by_task[t]):.4f} ({np.std(accs_by_task[t]):.4f})")

    if output_csv:
        metric_keys = ['/acc', '/bacc', '/kappa', '/nw_kappa', '/weighted_f1', '/loss', '/auroc']
        col_names   = ['acc', 'bacc', 'kappa', 'nw_kappa', 'weighted_f1', 'loss', 'auroc']

        # --- Per-task rows (one row per fold × task) ---
        detail_rows = []
        for fold_idx, res_fold in enumerate(all_results_per_task):
            fold_id = fold_idx  # matches the tqdm range
            for task_id in task_ids_to_eval:
                res = res_fold[task_id]
                row = {"fold": fold_id, "task": task_id}
                for key, col in zip(metric_keys, col_names):
                    row[col] = res.get(key, float('nan'))
                detail_rows.append(row)

        # Summary rows: mean & std across folds per task
        for task_id in task_ids_to_eval:
            task_vals = {col: [] for col in col_names}
            for res_fold in all_results_per_task:
                for key, col in zip(metric_keys, col_names):
                    task_vals[col].append(res_fold[task_id].get(key, float('nan')))
            mean_row = {"fold": "mean", "task": task_id}
            std_row  = {"fold": "std",  "task": task_id}
            for col in col_names:
                mean_row[col] = np.nanmean(task_vals[col])
                std_row[col]  = np.nanstd(task_vals[col])
            detail_rows.append(mean_row)
            detail_rows.append(std_row)

        # Overall fold-level summary rows
        for fold_idx in range(len(overall_baccs_folds)):
            detail_rows.append({
                "fold": fold_idx, "task": "overall",
                "acc": overall_acc_norms[fold_idx],
                "bacc": overall_baccs_folds[fold_idx],
            })
        detail_rows.append({"fold": "mean", "task": "overall",
                            "acc": np.mean(overall_acc_norms),
                            "bacc": np.mean(overall_baccs_folds)})
        detail_rows.append({"fold": "std",  "task": "overall",
                            "acc": np.std(overall_acc_norms),
                            "bacc": np.std(overall_baccs_folds)})

        df = pd.DataFrame(detail_rows)
        df.to_csv(output_csv, index=False)
        print(f"\nCSV saved: {output_csv}")
