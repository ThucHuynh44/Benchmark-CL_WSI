"""
scripts/eval_classil.py
CLASS-IL evaluation: Accuracy, Balanced Accuracy, Macro/Weighted F1, Precision, Recall, AUC.

Uses the task-to-class prompt alignment strategy (MergeSlide inference).

Usage:
    python scripts/eval_classil.py \
        --save_dir /path/to/finetuned/checkpoints \
        --merge_model_path /path/to/merged/checkpoints/
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
from sklearn.metrics import (
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from tqdm import tqdm
from transformers import AutoModel

from configs.loader import load_config
from mergeslide.datasets import Sequential_Generic_MIL_Dataset, get_dict_convert_class
from mergeslide.models import CustomSequential
from mergeslide.prompts import ALL_TASK_PROMPTS, TEMPLATES
from mergeslide.utils import get_eval_metrics, seed_torch

# Patch sampling budget per forward pass
K = 400

# DICT_CLASSES và DICT_CONVERT_CLASS sẽ được tính động sau khi load seq_dataset
# (tránh hardcode class ranges khi số task thay đổi)


def build_task_prompts(titan_model, num_tasks: int, device: str):
    """Compute task-level prompt embeddings (one per task)."""
    class_prompts = []
    for prompt_fn in ALL_TASK_PROMPTS[:num_tasks]:
        prompts, _ = prompt_fn()
        class_prompts.extend(prompts)
    with torch.autocast('cuda', torch.float16), torch.inference_mode():
        classifier = titan_model.zero_shot_classifier(class_prompts, TEMPLATES, device=device)
    return classifier


def load_mlp_weights(task_model_paths: list) -> list:
    """Load the MLP head state dict for each task."""
    weights = []
    for path in task_model_paths:
        raw = torch.load(path, map_location="cpu")
        mlp_state = {}
        for key, value in raw.items():
            if key.endswith("mlp.weight"):
                mlp_state["weight"] = value
            elif key.endswith("mlp.bias"):
                mlp_state["bias"] = value

        if "weight" not in mlp_state or "bias" not in mlp_state:
            raise KeyError(f"Cannot find mlp.weight/mlp.bias in checkpoint: {path}")

        weights.append(mlp_state)
    return weights


def build_merged_mlp(mlp_task_weights: list, device) -> nn.Linear:
    """Build the naive global classifier by concatenating all task heads."""
    merged_mlp_data = {
        'weight': torch.cat([w['weight'] for w in mlp_task_weights]),
        'bias': torch.cat([w['bias'] for w in mlp_task_weights]),
    }
    merged_mlp = nn.Linear(768, merged_mlp_data['weight'].shape[0]).to(device)
    merged_mlp.load_state_dict(merged_mlp_data)
    merged_mlp.eval()
    return merged_mlp


def _task_column(prefix: str, task_id: int, task_names: list) -> str:
    if task_id < len(task_names):
        return f"{prefix}_{task_id}_{task_names[task_id]}"
    return f"{prefix}_{task_id}"


def routing_confusion_to_df(matrix: np.ndarray, task_names: list, normalize: bool = False) -> pd.DataFrame:
    """Format a true-task x predicted-task routing confusion matrix."""
    values = matrix.astype(float) if normalize else matrix.astype(int)
    if normalize:
        row_sums = values.sum(axis=1, keepdims=True)
        values = np.divide(values, row_sums, out=np.zeros_like(values), where=row_sums != 0)

    rows = []
    for true_task_id in range(values.shape[0]):
        row = {
            "true_task": true_task_id,
            "true_task_name": task_names[true_task_id] if true_task_id < len(task_names) else "",
        }
        for pred_task_id in range(values.shape[1]):
            row[_task_column("pred", pred_task_id, task_names)] = values[true_task_id, pred_task_id]
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_task(
    test_loader,
    task_id: int,
    model,
    num_classes: list,
    device,
    task_prompts,
    task_weights: list,
    dict_convert_class: dict,
    prefix: str = "",
):
    """Run CLASS-IL inference on one task's test set using global labels.

    Returns:
        (eval_metrics, preds_all, targets_all, slide_per_task, slide_per_class,
         probs_all, convert_preds_all, convert_targets_all, predicted_task_ids,
         total_inference_time)
    """
    preds_all, probs_all, targets_all = [], [], []
    convert_preds_all, convert_targets_all = [], []
    predicted_task_ids = []
    slide_per_task, slide_per_class = [], {}
    times = []
    total_num_classes = sum(num_classes)

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader):
            features = features.to(device)
            coords = coords.long().to(device)
            indices = torch.randperm(features.shape[0])[:K]
            features = features[indices]
            coords = coords[indices]

            start = time.time()
            slide_embed = model.backbone(features, coords, torch.tensor(1024).int().to(device))
            predicted_task_id = int(torch.argmax(slide_embed @ task_prompts.T))
            predicted_task_ids.append(predicted_task_id)

            mlp = nn.Linear(768, num_classes[predicted_task_id]).to(device)
            mlp.load_state_dict(task_weights[predicted_task_id])
            mlp.eval()
            logits = mlp(slide_embed).float()
            probs_local = nn.functional.softmax(logits, dim=1)
            pred_local = int(logits.argmax(1).item())
            times.append(time.time() - start)

            true_local = int(label[0])
            true_global = dict_convert_class[task_id][true_local]
            pred_global = dict_convert_class[predicted_task_id][pred_local]

            probs_global = np.zeros((1, total_num_classes), dtype=np.float32)
            for local_idx, global_idx in dict_convert_class[predicted_task_id].items():
                probs_global[0, global_idx] = float(probs_local[0, local_idx].detach().cpu())

            preds_all.append(np.array([pred_global], dtype=np.int64))
            targets_all.append(np.array([true_global], dtype=np.int64))
            probs_all.append(probs_global)

            slide_per_task.append(slide_embed)
            slide_per_class.setdefault(true_global, []).append(slide_embed)

            convert_targets_all.append(np.array([true_global], dtype=np.int64))
            convert_preds_all.append(np.array([pred_global], dtype=np.int64))

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    probs_all = np.concatenate(probs_all)

    convert_preds_all = np.concatenate(convert_preds_all)
    convert_targets_all = np.concatenate(convert_targets_all)

    eval_metrics = {
        f"{prefix}/acc": float(np.mean(preds_all == targets_all))
    }

    return (
        eval_metrics, preds_all, targets_all,
        slide_per_task, slide_per_class,
        probs_all, convert_preds_all, convert_targets_all,
        np.array(predicted_task_ids, dtype=np.int64),
        sum(times),
    )


def evaluate_task_naive(
    test_loader,
    task_id: int,
    model,
    device,
    merged_mlp,
    prefix: str = "",
):
    """Run CLASS-IL naive inference with one concatenated global classifier."""
    preds_all, probs_all, targets_all = [], [], []
    convert_preds_all, convert_targets_all = [], []
    slide_per_task, slide_per_class = [], {}
    times = []

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader):
            features = features.to(device)
            coords = coords.long().to(device)
            indices = torch.randperm(features.shape[0])[:K]
            features = features[indices]
            coords = coords[indices]

            start = time.time()
            slide_embed = model.backbone(features, coords, torch.tensor(1024).int().to(device))
            logits = merged_mlp(slide_embed.float()).float()
            preds = logits.argmax(1)
            times.append(time.time() - start)

            probs = nn.functional.softmax(logits, dim=1)
            global_label = DICT_CONVERT_CLASS[task_id][int(label)]

            preds_all.append(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            targets_all.append(np.array([global_label]))

            slide_per_task.append(slide_embed)
            slide_per_class.setdefault(global_label, []).append(slide_embed)
            convert_targets_all.append(np.array([global_label]))
            convert_preds_all.append(preds.cpu().numpy())

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    probs_all = np.concatenate(probs_all)
    convert_preds_all = np.concatenate(convert_preds_all)
    convert_targets_all = np.concatenate(convert_targets_all)

    roc_kwargs = {"multi_class": "ovo", "average": "macro"}
    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, roc_kwargs=roc_kwargs, prefix=prefix)

    return (
        eval_metrics, preds_all, targets_all,
        slide_per_task, slide_per_class,
        probs_all, convert_preds_all, convert_targets_all,
        sum(times),
    )


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, 0)

    parser = argparse.ArgumentParser(description="CLASS-IL evaluation with task-to-class prompt alignment")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory with per-task finetuned checkpoints")
    parser.add_argument("--merge_model_path", type=str, default=None,
                        help="Directory with merged model checkpoints")
    parser.add_argument("--output_csv", type=str, default=None,
                        help="Optional path to save results as CSV (e.g. results_classil.csv)")
    parser.add_argument("--mode", type=str, default=None, choices=["tcp", "naive"],
                        help="CLASS-IL inference mode: tcp routes by task prompts; naive uses one global MLP")
    parser.add_argument("--num_workers", type=int, default=None,
                        help="DataLoader workers for evaluation. Use 0 to debug HDF5/IO stalls.")
    parser.add_argument("--routing_confusion_csv", type=str, default=None,
                        help="Optional path to save TCP routing confusion counts CSV")
    args = parser.parse_args()

    cfg = load_config(default_filename="eval.yaml")
    eval_cfg = cfg.get("evaluation", {})

    save_dir = args.save_dir if args.save_dir is not None else eval_cfg.get("save_dir", "./checkpoints/finetuned")
    merge_model_path = args.merge_model_path if args.merge_model_path is not None else eval_cfg.get("merge_model_path", "./checkpoints/merged")
    mode = args.mode if args.mode is not None else eval_cfg.get("mode", "tcp")
    output_csv = args.output_csv if args.output_csv is not None else eval_cfg.get("classil_output_csv")
    routing_confusion_csv = (
        args.routing_confusion_csv
        if args.routing_confusion_csv is not None
        else eval_cfg.get("classil_routing_confusion_csv")
    )
    num_folds = int(eval_cfg.get("num_folds", 10))
    num_workers = int(args.num_workers if args.num_workers is not None else eval_cfg.get("num_workers", 0))

    seq_dataset = Sequential_Generic_MIL_Dataset()
    num_classes = seq_dataset.num_classes
    num_tasks = int(eval_cfg.get("num_tasks", len(num_classes)))
    num_classes = num_classes[:num_tasks]
    # Tính động class ranges theo num_classes thực tế
    DICT_CONVERT_CLASS = get_dict_convert_class(num_classes)

    # Load TITAN base model
    titan_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True).to(device)
    task_prompts = torch.load("./task_prompts.pt", map_location=device).to(device) if mode == "tcp" else None

    # Accumulators across folds
    overall_accs, overall_baccs, overall_aucs = [], [], []
    overall_recalls, overall_precisions = [], []
    overall_macro_f1s, overall_weighted_f1s = [], []
    overall_time_all_folds = []
    all_acc_per_task = []
    routing_confusions = []
    task_names = seq_dataset.task_order[:num_tasks]

    for fold_id in tqdm(range(num_folds)):
        fold = f"fold_{fold_id}"
        task_model_paths = [
            f"{save_dir}/{fold}/ckpts_outputs_finetuning_task_{task_id}.pt"
            for task_id in range(num_tasks)
        ]

        # Load all MLP weights. TCP uses per-task heads; naive concatenates them.
        mlp_task_weights = load_mlp_weights(task_model_paths)
        merged_mlp = build_merged_mlp(mlp_task_weights, device) if mode == "naive" else None

        # Build model: merged backbone + identity head (task routing is done inside eval)
        model = CustomSequential(titan_model, nn.Identity())
        # Load the final merged backbone for this fold
        merged_ckpt = f"{merge_model_path}/_{fold}/merged_weight_opcm_random_sampling_{fold}_task_{num_tasks - 1}.pth"
        model.backbone.load_state_dict(torch.load(merged_ckpt, map_location=device))
        model.eval()

        task_prompts_fold = task_prompts[:num_tasks] if mode == "tcp" else None
        acc_per_task = {}
        all_baccs, all_accs, aucs = [], [], []
        all_predictions, all_labels = [], []
        routing_confusion = np.zeros((num_tasks, num_tasks), dtype=np.int64) if mode == "tcp" else None
        overall_time = 0.0

        for task_id in range(num_tasks):
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id, num_workers=num_workers)
            if mode == "tcp":
                (results, preds_all, targets_all, slide_per_task, slide_per_class,
                 probs_all, convert_preds_all, convert_targets_all,
                 predicted_task_ids, sum_time) = evaluate_task(
                    test_loader=test_loader,
                    task_id=task_id,
                    model=model,
                    num_classes=num_classes,
                    device=device,
                    task_prompts=task_prompts_fold,
                    task_weights=mlp_task_weights,
                    dict_convert_class=DICT_CONVERT_CLASS,
                    prefix="",
                )
                routing_confusion[task_id] += np.bincount(
                    predicted_task_ids, minlength=num_tasks
                )[:num_tasks]
            else:
                (results, preds_all, targets_all, slide_per_task, slide_per_class,
                 probs_all, convert_preds_all, convert_targets_all, sum_time) = evaluate_task_naive(
                    test_loader, task_id, model, device, merged_mlp, prefix="",
                )

            acc_per_task[task_id] = results['/acc']
            all_baccs.append(balanced_accuracy_score(targets_all, preds_all))
            all_accs.append(sum(preds_all == targets_all) / len(test_loader))
            all_predictions.append(convert_preds_all)
            all_labels.append(convert_targets_all)
            overall_time += sum_time / len(test_loader)

            if len(probs_all.shape) == 3:
                probs_all = probs_all.squeeze(1)
            if mode == "tcp":
                for global_idx in DICT_CONVERT_CLASS[task_id].values():
                    y_true_binary = (targets_all == global_idx).astype(int)
                    if len(np.unique(y_true_binary)) < 2:
                        continue
                    aucs.append(roc_auc_score(y_true_binary, probs_all[:, global_idx]))
            else:
                for global_idx in DICT_CONVERT_CLASS[task_id].values():
                    y_true_binary = (targets_all == global_idx).astype(int)
                    if len(np.unique(y_true_binary)) < 2:
                        continue
                    aucs.append(roc_auc_score(y_true_binary, probs_all[:, global_idx]))

        all_labels = np.concatenate(all_labels)
        all_predictions = np.concatenate(all_predictions)

        overall_baccs.append(np.mean(all_baccs))
        overall_accs.append(np.mean(all_accs))
        overall_aucs.append(np.array(aucs))
        overall_recalls.append(recall_score(all_labels, all_predictions, average=None))
        overall_precisions.append(precision_score(all_labels, all_predictions, average=None))
        overall_macro_f1s.append(f1_score(all_labels, all_predictions, average="macro"))
        overall_weighted_f1s.append(f1_score(all_labels, all_predictions, average="weighted"))
        overall_time_all_folds.append(overall_time / num_tasks)
        all_acc_per_task.append(acc_per_task)

        print(f"[Fold {fold_id}] ACC: {np.mean(all_accs):.4f}, BACC: {np.mean(all_baccs):.4f}")
        if mode == "tcp":
            routing_confusions.append(routing_confusion)
            route_total = routing_confusion.sum()
            route_acc = np.trace(routing_confusion) / route_total if route_total else 0.0
            print(f"[Fold {fold_id}] Routing ACC: {route_acc:.4f}")

    print(f"\nAccuracy:          {np.mean(overall_accs):.4f} ({np.std(overall_accs):.4f})")
    print(f"Balanced Accuracy: {np.mean(overall_baccs):.4f} ({np.std(overall_baccs):.4f})")
    print(f"Macro F1:          {np.mean(overall_macro_f1s):.4f} ({np.std(overall_macro_f1s):.4f})")
    print(f"Weighted F1:       {np.mean(overall_weighted_f1s):.4f} ({np.std(overall_weighted_f1s):.4f})")

    print("\nRecall per class:")
    for v, s in zip(np.mean(np.stack(overall_recalls), axis=0), np.std(np.stack(overall_recalls), axis=0)):
        print(f"  {v:.4f} ({s:.4f})")

    print("\nPrecision per class:")
    for v, s in zip(np.mean(np.stack(overall_precisions), axis=0), np.std(np.stack(overall_precisions), axis=0)):
        print(f"  {v:.4f} ({s:.4f})")

    print("\nAUC per class:")
    for v, s in zip(np.mean(np.stack(overall_aucs), axis=0), np.std(np.stack(overall_aucs), axis=0)):
        print(f"  {v:.4f} ({s:.4f})")

    print(f"\nAvg inference time per task: {np.mean(overall_time_all_folds):.4f}s ({np.std(overall_time_all_folds):.4f}s)")

    accs_by_task = {t: [] for t in range(num_tasks)}
    for fold_accs in all_acc_per_task:
        for t, acc in fold_accs.items():
            accs_by_task[t].append(acc)
    print("\nPer-task accuracy:")
    for t in range(num_tasks):
        print(f"  Task {t}: {np.mean(accs_by_task[t]):.4f} ({np.std(accs_by_task[t]):.4f})")

    if mode == "tcp" and routing_confusions:
        routing_total = np.sum(np.stack(routing_confusions), axis=0)
        route_total = routing_total.sum()
        route_acc = np.trace(routing_total) / route_total if route_total else 0.0
        print(f"\nTask routing accuracy: {route_acc:.4f}")
        print("\nTask routing confusion counts:")
        print(routing_confusion_to_df(routing_total, task_names).to_string(index=False))

    if output_csv:
        # Per-fold summary
        fold_rows = []
        for i in range(num_folds):
            fold_rows.append({
                "fold": i,
                "acc": overall_accs[i],
                "bacc": overall_baccs[i],
                "macro_f1": overall_macro_f1s[i],
                "weighted_f1": overall_weighted_f1s[i],
                "inference_time_per_task": overall_time_all_folds[i],
            })
        fold_rows.append({
            "fold": "mean",
            "acc": np.mean(overall_accs),
            "bacc": np.mean(overall_baccs),
            "macro_f1": np.mean(overall_macro_f1s),
            "weighted_f1": np.mean(overall_weighted_f1s),
            "inference_time_per_task": np.mean(overall_time_all_folds),
        })
        fold_rows.append({
            "fold": "std",
            "acc": np.std(overall_accs),
            "bacc": np.std(overall_baccs),
            "macro_f1": np.std(overall_macro_f1s),
            "weighted_f1": np.std(overall_weighted_f1s),
            "inference_time_per_task": np.std(overall_time_all_folds),
        })
        df_fold = pd.DataFrame(fold_rows)

        # Per-task accuracy summary
        task_rows = [{
            "task": t,
            "mean_acc": np.mean(accs_by_task[t]),
            "std_acc": np.std(accs_by_task[t]),
        } for t in range(num_tasks)]
        df_task = pd.DataFrame(task_rows)

        output_base, output_ext = os.path.splitext(output_csv)
        output_ext = output_ext or ".csv"
        fold_csv = f"{output_base}_per_fold{output_ext}"
        task_csv = f"{output_base}_per_task{output_ext}"
        df_fold.to_csv(fold_csv, index=False)
        df_task.to_csv(task_csv, index=False)
        print(f"\nCSV saved: {fold_csv}")
        print(f"CSV saved: {task_csv}")

    if mode == "tcp" and routing_confusions and (routing_confusion_csv or output_csv):
        if routing_confusion_csv:
            routing_base, routing_ext = os.path.splitext(routing_confusion_csv)
        else:
            routing_base, routing_ext = os.path.splitext(output_csv)
            routing_base = f"{routing_base}_routing_confusion"
        routing_ext = routing_ext or ".csv"

        routing_total = np.sum(np.stack(routing_confusions), axis=0)
        counts_csv = f"{routing_base}_counts{routing_ext}"
        rates_csv = f"{routing_base}_rates{routing_ext}"

        routing_confusion_to_df(routing_total, task_names).to_csv(counts_csv, index=False)
        routing_confusion_to_df(routing_total, task_names, normalize=True).to_csv(rates_csv, index=False)
        print(f"CSV saved: {counts_csv}")
        print(f"CSV saved: {rates_csv}")
