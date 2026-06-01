"""
scripts/eval_classil.py
CLASS-IL evaluation: Accuracy, Balanced Accuracy, Macro/Weighted F1, Precision, Recall, AUC.

Uses the task-to-class prompt alignment strategy (MergeSlide inference).

Usage:
    python scripts/eval_classil.py \
        --save_dir /path/to/finetuned/checkpoints \
        --merge_model_path /path/to/merged/checkpoints/
"""

import argparse
import time

import numpy as np
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

from mergeslide.datasets import Sequential_Generic_MIL_Dataset
from mergeslide.models import CustomSequential, pad_numpy_arrays
from mergeslide.prompts import ALL_TASK_PROMPTS, TEMPLATES
from mergeslide.utils import get_eval_metrics, seed_torch

# Patch sampling budget per forward pass
K = 300

# Map task_id → global class index range
DICT_CLASSES = {
    0: [0, 1],
    1: [2, 4],
    2: [5, 6],
    3: [7, 8],
    4: [9, 10],
    5: [11, 12],
}

# Map (task_id, local_class_id) → global class id
DICT_CONVERT_CLASS = {
    0: {0: 0,  1: 1},
    1: {0: 2,  1: 3, 2: 4},
    2: {0: 5,  1: 6},
    3: {0: 7,  1: 8},
    4: {0: 9,  1: 10},
    5: {0: 11, 1: 12},
}


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
        raw = torch.load(path, map_location='cpu')
        mlp_state = {k.split('mlp.')[-1]: raw[k] for k in list(raw.keys())[-2:]}
        weights.append(mlp_state)
    return weights


def evaluate_task(
    test_loader,
    task_id: int,
    model,
    dict_class: dict,
    num_classes: list,
    device,
    task_prompts,
    task_weights: list,
    prefix: str = "",
):
    """Run CLASS-IL inference on one task's test set.

    Returns:
        (eval_metrics, preds_all, targets_all, slide_per_task, slide_per_class,
         probs_all, convert_preds_all, convert_targets_all, total_inference_time)
    """
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
            predicted_task_id = int(torch.argmax(slide_embed @ task_prompts.T))

            mlp = nn.Linear(768, num_classes[predicted_task_id]).to(device)
            mlp.load_state_dict(task_weights[predicted_task_id])
            logits = mlp(slide_embed).float()
            preds = logits.argmax(1)
            times.append(time.time() - start)

            probs = nn.functional.softmax(logits, dim=1)
            preds_all.append(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            targets_all.append(label.numpy())

            slide_per_task.append(slide_embed)
            global_label = DICT_CONVERT_CLASS[task_id][int(label)]
            slide_per_class.setdefault(global_label, []).append(slide_embed)

            convert_targets_all.append(torch.Tensor([dict_class[int(label[0])]]))
            try:
                convert_preds_all.append(torch.Tensor([dict_class[int(preds[0])]]))
            except Exception:
                convert_preds_all.append(torch.Tensor([4]))

    preds_all = np.concatenate(preds_all)
    targets_all = np.concatenate(targets_all)
    try:
        probs_all = np.concatenate(probs_all)
    except ValueError:
        probs_all = pad_numpy_arrays(probs_all)

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
    parser.add_argument("--save_dir", type=str, default="./checkpoints/finetuned",
                        help="Directory with per-task finetuned checkpoints")
    parser.add_argument("--merge_model_path", type=str, default="./checkpoints/merged",
                        help="Directory with merged model checkpoints")
    args = parser.parse_args()

    num_tasks = 6
    num_classes = [2, 3, 2, 2, 2, 2]
    seq_dataset = Sequential_Generic_MIL_Dataset()

    # Load TITAN base model
    titan_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True).to(device)
    task_prompts = torch.load("./task_prompts.pt")

    # Accumulators across folds
    overall_accs, overall_baccs, overall_aucs = [], [], []
    overall_recalls, overall_precisions = [], []
    overall_macro_f1s, overall_weighted_f1s = [], []
    overall_time_all_folds = []
    all_acc_per_task = []

    for fold_id in tqdm(range(10)):
        fold = f"fold_{fold_id}"
        task_model_paths = [
            f"{args.save_dir}/{fold_id}/ckpts_outputs_finetuning_task_{task_id}.pt"
            for task_id in range(num_tasks)
        ]

        # Load all MLP weights and build the merged MLP head
        mlp_task_weights = load_mlp_weights(task_model_paths)
        merge_mlp_data = {
            'weight': torch.cat([w['weight'] for w in mlp_task_weights]),
            'bias': torch.cat([w['bias'] for w in mlp_task_weights]),
        }

        # Build model: merged backbone + identity head (task routing is done inside eval)
        model = CustomSequential(titan_model, nn.Identity())
        # Load the final merged backbone for this fold
        merged_ckpt = f"{args.merge_model_path}/fold_{fold_id}/merged_weight_opcm_random_sampling_{fold}_task_{num_tasks - 1}.pth"
        model.backbone.load_state_dict(torch.load(merged_ckpt))
        model.eval()

        task_prompts_fold = task_prompts[:num_tasks]
        acc_per_task = {}
        all_baccs, all_accs, aucs = [], [], []
        all_predictions, all_labels = [], []
        overall_time = 0.0

        for task_id in range(num_tasks):
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
            (results, preds_all, targets_all, slide_per_task, slide_per_class,
             probs_all, convert_preds_all, convert_targets_all, sum_time) = evaluate_task(
                test_loader, task_id, model,
                DICT_CONVERT_CLASS[task_id], num_classes, device,
                task_prompts_fold, mlp_task_weights, prefix="",
            )

            acc_per_task[task_id] = results['/acc']
            all_baccs.append(balanced_accuracy_score(targets_all, preds_all))
            all_accs.append(sum(preds_all == targets_all) / len(test_loader))
            all_predictions.append(convert_preds_all)
            all_labels.append(convert_targets_all)
            overall_time += sum_time / len(test_loader)

            if len(probs_all.shape) == 3:
                probs_all = probs_all.squeeze(1)
            for i in range(len(DICT_CONVERT_CLASS[task_id])):
                y_true_binary = (targets_all == i).astype(int)
                aucs.append(roc_auc_score(y_true_binary, probs_all[:, i]))

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
