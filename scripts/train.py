"""
scripts/train.py
Per-task finetuning of TITAN's slide aggregator on each TCGA task.

Usage:
    python scripts/train.py --save_dir /path/to/finetuned/checkpoints
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import pickle
import time
import warnings

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModel

from configs.loader import load_config
from mergeslide.datasets import Sequential_Generic_MIL_Dataset
from mergeslide.models import CustomSequential, EarlyStopping, cosine_lr, create_mlp
from mergeslide.prompts import ALL_TASK_PROMPTS, TEMPLATES
from mergeslide.utils import get_eval_metrics, seed_torch

# Patch sampling budget per forward pass
K = 400

# Map task_id → column indices in the joint classifier matrix
DICT_CLASSES = {
    0: [0, 1],
    1: [2, 4],
    2: [5, 6],
    3: [7, 8],
    4: [9, 10],
    5: [11, 12],
}


def build_classifier(titan_model, device: str):
    """Build zero-shot classifier from all class-aware prompts."""
    class_prompts = []
    for prompt_fn in ALL_TASK_PROMPTS:
        prompts, _ = prompt_fn()
        class_prompts.extend(prompts)
    with torch.autocast('cuda', torch.float16), torch.inference_mode():
        classifier = titan_model.zero_shot_classifier(class_prompts, TEMPLATES, device=device)
    return classifier


def train(train_loader, val_loader, model, num_epochs, lr, weight_decay, device, use_wandb=False, **kwargs):
    """Train model for one epoch with cosine LR and early stopping.

    Args:
        train_loader: DataLoader for training set.
        val_loader: DataLoader for validation set.
        model: CustomSequential model to train.
        num_epochs: Total number of training epochs.
        lr: Base learning rate.
        weight_decay: Weight decay for AdamW.
        device: Target device string.
        use_wandb: Set to True to enable wandb tracking.

    Returns:
        Trained model.
    """
    named_parameters = list(model.named_parameters())
    exclude = lambda n, p: p.ndim < 2 or any(x in n for x in ('bn', 'ln', 'bias', 'logit_scale'))
    include = lambda n, p: not exclude(n, p)
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]

    optimizer = torch.optim.AdamW(
        [{"params": gain_or_bias_params, "weight_decay": 0.0}, {"params": rest_params, "weight_decay": weight_decay}],
        lr=lr,
    )
    scheduler = cosine_lr(
        optimizer=optimizer,
        base_lr=lr,
        warmup_length=int(len(train_loader) * num_epochs * 0.1),
        steps=len(train_loader) * num_epochs,
    )
    loss_fn = nn.CrossEntropyLoss()
    fp16_scaler = torch.cuda.amp.GradScaler()
    early_stopping = EarlyStopping(patience=2, verbose=True)
    step = 0

    for epoch in tqdm(range(num_epochs)):
        model.train()
        preds_all, targets_all = [], []
        total_train_loss = 0.0

        for features, coords, label in tqdm(train_loader):
            scheduler(step)
            features = features.to(device)
            coords = coords.long().to(device)
            indices = torch.randperm(features.shape[0])[:K]
            features = features[indices]
            coords = coords[indices]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(features, coords, torch.tensor(1024).int().to(device))
                loss = loss_fn(logits, label.to(device))

            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            optimizer.zero_grad()

            preds_all.append(logits.argmax(1).cpu().numpy())
            targets_all.append(label.numpy())
            if use_wandb and step % 10 == 0:
                import wandb
                wandb.log({"train/step_loss": loss.item(), "lr": optimizer.param_groups[0]['lr'], "step": step})
            step += 1
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        bacc = balanced_accuracy_score(np.concatenate(targets_all), np.concatenate(preds_all))

        if epoch > 1:
            model.eval()
            preds_val, targets_val, total_val_loss = [], [], 0.0
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for features, coords, labels in val_loader:
                    indices = torch.randperm(features.shape[0])[:K]
                    features = features.to(device)[indices]
                    coords = coords.long().to(device)[indices]
                    try:
                        logits = model(features, coords, torch.tensor(1024).int().to(device), **kwargs)
                    except Exception:
                        model.cpu()
                        logits = model(features, coords, torch.tensor(1024).int().cpu(), **kwargs)
                        model.to(device)
                    val_loss = loss_fn(logits, labels.to(logits.device))
                    preds_val.append(logits.argmax(1).cpu().numpy())
                    targets_val.append(labels.numpy())
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            bacc_val = balanced_accuracy_score(np.concatenate(targets_val), np.concatenate(preds_val))
            tqdm.write(f"epoch {epoch}, bacc: {bacc:.4f}, bacc_val: {bacc_val:.4f}, loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")
            if use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "train/bacc": bacc,
                    "val/loss": avg_val_loss,
                    "val/bacc": bacc_val,
                })
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            tqdm.write(f"epoch {epoch}, bacc: {bacc:.4f}, loss: {avg_train_loss:.4f}")
            if use_wandb:
                import wandb
                wandb.log({
                    "epoch": epoch,
                    "train/loss": avg_train_loss,
                    "train/bacc": bacc,
                })

    model.eval()
    return model


def evaluate(test_loader, model, num_classes: int, device: str, prefix: str, save_location: str = None, **kwargs):
    """Evaluate model on a test set and optionally save raw outputs.

    Args:
        test_loader: DataLoader for the test set.
        model: Trained CustomSequential model.
        num_classes: Number of output classes for this task.
        device: Target device string.
        prefix: Metric key prefix.
        save_location: Optional path to save predictions as a pickle file.

    Returns:
        eval_metrics dict.
    """
    preds_all, probs_all, targets_all = [], [], []
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader):
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

    if save_location:
        with open(save_location, "wb") as f:
            pickle.dump({"targets": targets_all, "preds": preds_all, "probs": probs_all}, f)

    return eval_metrics


if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, 0)

    parser = argparse.ArgumentParser(description="Per-task finetuning of TITAN on TCGA WSI tasks")
    parser.add_argument("--num_epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory to save per-task finetuned checkpoints")
    parser.add_argument("--use_wandb", action="store_true", help="Enable weights and biases tracking")
    args = parser.parse_args()

    cfg = load_config(default_filename="train.yaml")
    train_cfg = cfg.get("training", {})

    num_epochs = int(args.num_epochs if args.num_epochs is not None else train_cfg.get("num_epochs", 10))
    lr = float(args.lr if args.lr is not None else train_cfg.get("lr", 1e-5))
    weight_decay = float(args.weight_decay if args.weight_decay is not None else train_cfg.get("weight_decay", 1e-4))
    save_dir = args.save_dir if args.save_dir is not None else train_cfg.get("save_dir", "./checkpoints/finetuned")
    use_wandb = args.use_wandb or train_cfg.get("use_wandb", False)

    if use_wandb:
        try:
            import wandb
        except ImportError:
            warnings.warn("wandb package not found. Disabling wandb tracking.")
            use_wandb = False

    device_str = str(device)
    num_tasks = 6
    num_classes = [2, 3, 2, 2, 2, 2]
    seq_dataset = Sequential_Generic_MIL_Dataset()

    # Load TITAN once for prompt encoding
    titan_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    titan_model = titan_model.to(device_str)
    classifier = build_classifier(titan_model, device_str)

    for fold_id in range(10):
        fold_dir = os.path.join(save_dir, f"fold_{fold_id}")
        os.makedirs(fold_dir, exist_ok=True)

        for task_id in range(num_tasks):
            train_loader, val_loader, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)

            if use_wandb:
                import wandb
                wandb.init(
                    project=train_cfg.get("wandb_project", "MergeSlide-Finetuning"),
                    name=f"fold_{fold_id}_task_{task_id}",
                    config={
                        "fold": fold_id,
                        "task": task_id,
                        "num_epochs": num_epochs,
                        "lr": lr,
                        "weight_decay": weight_decay,
                    },
                    reinit=True
                )

            model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True).to(device_str)
            mlp = nn.Linear(768, num_classes[task_id]).to(device_str)
            mlp.bias.data.zero_()

            # Initialize MLP weights from class-aware prompt prototypes
            col_lo, col_hi = DICT_CLASSES[task_id]
            prompt_prototypes = classifier[:, col_lo:col_hi + 1]
            mlp.weight.data = prompt_prototypes.T

            model = CustomSequential(model, mlp)
            # Freeze the classification head — only finetune the backbone
            for param in model.mlp.parameters():
                param.requires_grad = False

            start = time.time()
            model = train(train_loader, val_loader, model, num_epochs, lr, weight_decay, device_str, use_wandb=use_wandb)
            elapsed = time.time() - start
            print(f"Fold {fold_id}, Task {task_id}: training took {elapsed:.1f}s")

            ckpt_path = os.path.join(fold_dir, f"ckpts_outputs_finetuning_task_{task_id}.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"Saved checkpoint: {ckpt_path}")

            if use_wandb:
                import wandb
                wandb.finish()
