import argparse
import os
import pickle
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
torch.manual_seed(42)  # Set seed
import torch.nn as nn
import torch.nn.functional as F
import yaml
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModel

from utils import bootstrap, get_eval_metrics, seed_torch
from datasets import Sequential_Generic_MIL_Dataset

import sys
import time
from prompts_zeroshot import brca_prompts, rcc_prompts, nsclc_prompts, esca_prompts, tgct_prompts, cesc_prompts
device = 'cuda:0'
titan_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
titan_model = titan_model.to(device)

_, TEMPLATES = brca_prompts()
CLASS_PROMPTS = []

print("Getting Prompt Prototypes ...")
for prompts in [brca_prompts, rcc_prompts, nsclc_prompts, esca_prompts, tgct_prompts, cesc_prompts]:
    class_prompts, _ = prompts()
    CLASS_PROMPTS.extend(class_prompts)

with torch.autocast('cuda', torch.float16), torch.inference_mode():
    classifier = titan_model.zero_shot_classifier(CLASS_PROMPTS, TEMPLATES, device=device)

dict_classes = {
    0: [0, 1],
    1: [2, 4],
    2: [5, 6],
    3: [7, 8],
    4: [9, 10],
    5: [11, 12]
}

"""
Script to finetune TITAN on a dummy dataset. Dataset class needs to be adapted to a custom dataset and task.
"""

MAX_NUM_PATCHES = 10000

class CustomSequential(nn.Module):
    def __init__(self, model, mlp):
        super(CustomSequential, self).__init__()
        self.backbone = model.vision_encoder
        self.mlp = mlp

    def forward(self, features, coords, ps):
        x = self.backbone(features, coords, ps)
        x = self.mlp(x)
        return x

def create_mlp(in_dim=None, hid_dims=[], act=nn.ReLU(), dropout=0.0, out_dim=None, end_with_fc=True):
    layers = []
    if len(hid_dims) > 0:
        for hid_dim in hid_dims:
            layers.append(nn.Linear(in_dim, hid_dim))
            layers.append(act)
            layers.append(nn.Dropout(dropout))
            in_dim = hid_dim
    layers.append(nn.Linear(in_dim, out_dim))
    if not end_with_fc:
        layers.append(act)
        layers.append(nn.Dropout(dropout))
    mlp = nn.Sequential(*layers)
    return mlp


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    """Copied from https://github.com/mlfoundations/open_clip/blob/main/src/open_clip_train/scheduler.py
    """
    def _warmup_lr(base_lr, warmup_length, step):
        return base_lr * (step + 1) / warmup_length
    
    def _assign_learning_rate(optimizer, new_lr):
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = new_lr * param_group["lr_scale"]
            else:
                param_group["lr"] = new_lr
    
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = _warmup_lr(base_lr, warmup_length, step)
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        _assign_learning_rate(optimizer, lr)
        return lr

    return _lr_adjuster

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0, verbose=False):
        """
        Args:
            patience (int): How long to wait after the last improvement.
            min_delta (float): Minimum change to qualify as an improvement.
            verbose (bool): If True, prints a message for each improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.best_model_weights = None

    def __call__(self, val_loss, model):
        # Check if the new loss is an improvement
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_weights = model.state_dict()
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0
            self.best_model_weights = model.state_dict()

def count_parameters(model): return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train(train_loader, val_loader, model, num_epochs, lr, weight_decay, device, prompt_prototypes, **kwargs):
    # load trainable parameters
    named_parameters = list(model.named_parameters())
    exclude = (
        lambda n, p: p.ndim < 2
        or "bn" in n
        or "ln" in n
        or "bias" in n
        or "logit_scale" in n
    )
    include = lambda n, p: not exclude(n, p)
    gain_or_bias_params = [p for n, p in named_parameters if exclude(n, p) and p.requires_grad]
    rest_params = [p for n, p in named_parameters if include(n, p) and p.requires_grad]
    
    # set optimizer, scheduler, and loss function
    optimizer = torch.optim.AdamW([{"params": gain_or_bias_params, "weight_decay": 0.0}, {"params": rest_params, "weight_decay": weight_decay}], lr=lr)
    lr_scheduler = cosine_lr(
        optimizer=optimizer,
        base_lr=lr,
        warmup_length=int(len(train_loader) * num_epochs * 0.1),
        steps=(len(train_loader) * num_epochs),
    )
    loss_fn = nn.CrossEntropyLoss()
    
    # training loop
    model.train()
    fp16_scaler = torch.cuda.amp.GradScaler()
    step = 0
    early_stopping = EarlyStopping(patience=2, verbose=True)
    K = 400
    for epoch in tqdm(range(1)):
        model.train()
        preds_all = []
        targets_all = []
        total_train_loss = 0

        for features, coords, label in tqdm(train_loader):
            lr_scheduler(step)

            features = features.to(device)
            coords = coords.long().to(device)

            indices = torch.randperm(features.shape[0])[:K]

            features = features[indices, :]
            coords = coords[indices, :]

            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(features, coords, torch.tensor(1024).int().to(device))
                loss = loss_fn(logits, label.to(device))
            
            fp16_scaler.scale(loss).backward()
            fp16_scaler.step(optimizer)
            fp16_scaler.update()
            optimizer.zero_grad()

            preds_all.append(logits.argmax(1).cpu().numpy())
            targets_all.append(label.numpy())
            step += 1
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        preds_all = np.concatenate(preds_all)
        targets_all = np.concatenate(targets_all)
        bacc = balanced_accuracy_score(targets_all, preds_all)

        # validate the model
        if epoch > 1:
            model.eval()
            preds_val, targets_val = [], []
            total_val_loss = 0
            with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
                for features, coords, labels in val_loader:
                    indices = torch.randperm(features.shape[0])[:K]

                    features = features.to(device)
                    coords = coords.long().to(device)

                    features = features[indices, :]
                    coords = coords[indices, :]

                    try:
                        logits = model(features, coords, torch.tensor(1024).int().to(device), **kwargs)
                    except:
                        model.cpu()
                        logits = model(features, coords, torch.tensor(1024).int().cpu(), **kwargs)
                        model.to(device)
                    val_loss = loss_fn(logits, labels.to(logits.device))
                    preds_val.append(logits.argmax(1).cpu().numpy())
                    targets_val.append(labels.numpy())
                    total_val_loss += val_loss.item()

            avg_val_loss = total_val_loss / len(val_loader)
            
            preds_val = np.concatenate(preds_val)
            targets_val = np.concatenate(targets_val)
            bacc_val = balanced_accuracy_score(targets_val, preds_val)

            tqdm.write(f"epoch {epoch}, bacc: {np.round(bacc, 4):.4f}, bacc_val: {np.round(bacc_val, 4):.4f}, loss: {avg_train_loss:.4f}, val_loss: {avg_val_loss:.4f}")
            early_stopping(avg_val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break
        else:
            tqdm.write(f"epoch {epoch}, bacc: {np.round(bacc, 4):.4f}, loss: {avg_train_loss:.4f}")

    model.eval()
    # model.load_state_dict(early_stopping.best_model_weights)

    return model

def eval(test_loader, model, num_classes, device, prefix, save_location, prompt_prototypes, **kwargs):
    preds_all = []
    probs_all = []
    targets_all = []
    K = 400
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader):

            features = features.to(device)
            coords = coords.long().to(device)

            indices = torch.randperm(features.shape[0])[:K]

            features = features[indices, :]
            coords = coords[indices, :]

            try:
                logits = model(features, coords, torch.tensor(1024).int().to(device), **kwargs)
            except:
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
        # save outputs as pickle objects
        outputs = {
            "targets": targets_all,
            "preds": preds_all,
            "probs": probs_all,
        }
        with open(save_location, "wb") as f:
            pickle.dump(outputs, f)

    return eval_metrics

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_torch(device, 0)

    parser = argparse.ArgumentParser(description="Finetune TITAN")
    
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_CROSSSITE_computational_profile")

    for fold_id in range(0, 10):
        args = parser.parse_args()    
        num_tasks = 6
        num_classes = [2, 3, 2, 2, 2, 2]
        seq_dataset = Sequential_Generic_MIL_Dataset()

        per_task_epoch_times = []

        for task_id in range(3):
            train_loader, val_loader, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
            # load model from huggingface
            model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)    
            model = model.to(device)
            
            # add mlp head for finetuning
            mlp = nn.Linear(768, num_classes[task_id]).to(device)            
            mlp.bias.data.zero_()

            prompt_prototypes = classifier[:, dict_classes[task_id][0]:dict_classes[task_id][1]+1]
            mlp.weight.data = prompt_prototypes.T

            model = CustomSequential(model, mlp)

            # Freeze MLP
            for param in model.mlp.parameters():
                param.requires_grad = False

            # finetune model
            start = time.time()
            model = train(
                train_loader,
                val_loader,
                model,
                args.num_epochs,
                args.lr,
                args.weight_decay,
                device,
                prompt_prototypes
            )
            end = time.time()
            per_task_epoch_times.append(end - start)

            # os.makedirs(f"{args.save_dir}/fold_{fold_id}", exist_ok=True)
            # save_location = f"{args.save_dir}/fold_{fold_id}/outputs_finetuning_task_{task_id}.pkl"
            # results = eval(test_loader, model, num_classes[task_id], device, prefix="", save_location=save_location, prompt_prototypes=prompt_prototypes)
            
            # save_pt_location = f"{args.save_dir}/fold_{fold_id}/ckpts_outputs_finetuning_task_{task_id}.pt"
            # torch.save(model.state_dict(), save_pt_location)

            # # compute bootstrapping results (for tasks with only one fold)
            # bootstrap_kwargs = {"n": 1000, "alpha": 0.95}
            # with open(f"{args.save_dir}/fold_{fold_id}/outputs_finetuning_task_{task_id}.pkl", "rb") as f:
            #     outputs = pickle.load(f)
            # results_mean, results_std = bootstrap(results_dict=outputs, **bootstrap_kwargs)

            # print("=============================")
            # print(f"Final finetuning results")
            # for keys, values in results_mean.items():
            #     print(f"{keys: <15}: {values:.4f} ± {results_std[keys]:.4f}")

            # df_path = f"{args.save_dir}/fold_{fold_id}/results_fintetuning.csv"
            # pd.DataFrame([results_mean, results_std], index=['mean', 'std']).to_csv(df_path)
            # print(f"results saved to {df_path}")

        import pdb; pdb.set_trace()
        per_task_epoch_times