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
from omegaconf import OmegaConf
from sklearn.metrics import balanced_accuracy_score
from tqdm import tqdm
from transformers import AutoModel

from utils import bootstrap, get_eval_metrics, seed_torch
from datasets import Sequential_Generic_MIL_Dataset

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

def eval(test_loader, model, num_classes, device, prefix, save_location, **kwargs):
    preds_all = []
    probs_all = []
    targets_all = []

    K = 300
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

    return eval_metrics, preds_all, targets_all

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
    parser.add_argument("--save_dir", type=str, default="./logs")

    args = parser.parse_args()
    num_classes = 2
    
    fold_id = 0
    num_tasks = 6
    num_classes = [2, 3, 2, 2, 2, 2]
    seq_dataset = Sequential_Generic_MIL_Dataset()

    # load model from huggingface
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)    
    base_model = base_model.to(device)

    overall_accs = []
    all_acc_per_task = []

    for fold_id in tqdm(range(0, 1)):
        num_total = 0.
        num_correct = 0.
        acc_per_task = {}
        all_baccs = []
        fold = "fold_" + str(fold_id)
        # merge_model_path = "/home/bui/continual_learning/WSIModelMerging/fusion_bench/method/opcm_no_rank_new/merged_weight_opcm_random_sampling_" + fold + ".pth"
        merge_model_path = "/home/bui/continual_learning/WSIModelMerging_rebuttal/fusion_bench/method/opcm/soict_task_arithmetic/merged_task_arithmetic_fold_5.pth"
        print(merge_model_path)
        base_model.vision_encoder.load_state_dict(torch.load(merge_model_path))
        # All task-specific models
        task_models = [
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_0.pt",
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_1.pt",
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_2.pt",
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_3.pt",
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_4.pt",
            "/home/bui/continual_learning/WSIModelMerging/finetune/logs_prompt_prototypes_as_freeze_MLP_random_patch_sampling_NEW/" + fold + "/ckpts_outputs_finetuning_task_5.pt"
        ]

        all_accs = []

        for task_id in range(num_tasks):
            print("TASK", task_id)
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
            # add mlp head for finetuning
            mlp = nn.Linear(768, num_classes[task_id]).to(device)
            mlp.weight.data.normal_(mean=0.0, std=0.01)
            mlp.bias.data.zero_()
            model = CustomSequential(base_model, mlp)
            # load MLP
            task_weight = torch.load(task_models[task_id])
            model.mlp.load_state_dict({k.split('mlp.')[-1]:task_weight[k] for k in list(task_weight.keys())[-2:]})
            model.eval()
            results, preds_all, targets_all = eval(test_loader, model, num_classes[task_id], device, prefix="", save_location=None)
            
            num_correct += sum(preds_all == targets_all)
            num_total += len(test_loader)

            bacc = balanced_accuracy_score(targets_all, preds_all)
            all_baccs.append(bacc)

            acc_per_task[task_id] = results['/acc']
            print(results)
            all_accs.append(sum(preds_all == targets_all) / len(test_loader))
        
        acc_new = np.mean(all_accs)
        overall_acc = np.mean(all_baccs)
        overall_accs.append(overall_acc)
        all_acc_per_task.append(acc_per_task)
        print("overall_acc", overall_acc)
        print("overall_normal_acc", acc_new)
    
    print([float(acc) for acc in overall_accs])
    print("Accuracy:", np.mean(overall_accs), "(", np.std(overall_accs), ")")

    accs = {task_id:list() for task_id in range(num_tasks)}
    for i in range(len(all_acc_per_task)):
        for task_id in range(num_tasks):
            accs[task_id].append(all_acc_per_task[i][task_id])
    
    for task_id in range(len(accs)):
        print("Acc ", task_id, np.mean(accs[task_id]), np.std(accs[task_id]))