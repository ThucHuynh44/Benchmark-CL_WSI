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
import torch.nn.functional as F
from prompts_zeroshot import brca_prompts, rcc_prompts, nsclc_prompts, esca_prompts, tgct_prompts, cesc_prompts

def pad_numpy_arrays(arrays, pad_value=0.0):
    """
    Pads a list of NumPy arrays with varying shapes to the same shape and stacks them.

    Args:
        arrays (List[np.ndarray]): List of NumPy arrays with varying shapes.
        pad_value (float): Value to use for padding.

    Returns:
        np.ndarray: A stacked NumPy array of shape (len(arrays), *max_shape)
    """
    # Step 1: Normalize dimensions
    max_dim = max(arr.ndim for arr in arrays)
    arrays = [arr.reshape((1,) * (max_dim - arr.ndim) + arr.shape) for arr in arrays]

    # Step 2: Compute max shape
    max_shape = np.max([arr.shape for arr in arrays], axis=0)

    # Step 3: Pad each array
    padded_arrays = []
    for arr in arrays:
        pad_width = [(0, max_dim_i - arr.shape[i]) for i, max_dim_i in enumerate(max_shape)]
        padded = np.pad(arr, pad_width=pad_width, mode='constant', constant_values=pad_value)
        padded_arrays.append(padded)

    # Step 4: Stack
    return np.stack(padded_arrays)

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

def forgetting(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        results[i] += [0.0] * (n_tasks - len(results[i]))
    np_res = np.array(results)
    maxx = np.max(np_res, axis=0)
    for i in range(n_tasks - 1):
        li.append(maxx[i] - results[-1][i])

    return np.mean(li)

def backward_transfer(results):
    n_tasks = len(results)
    li = list()
    for i in range(n_tasks - 1):
        li.append(results[-1][i] - results[i][i])

    return np.mean(li)

def eval(test_loader, model, num_classes, device, task_prompts, task_model_paths, merge_mlp_data, prefix, save_location, **kwargs):
    preds_all = []
    probs_all = []
    targets_all = []
    K = 400
    dict_convert_class = {0: 0, 1: 1, 2: 0, 3: 1, 4: 2, 5: 0, 6:1, 7:0, 8:1, 9:0, 10:1, 11:0, 12:1}
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        for features, coords, label in tqdm(test_loader):
            features = features.to(device)
            coords = coords.long().to(device)

            indices = torch.randperm(features.shape[0])[:K]

            features = features[indices, :]
            coords = coords[indices, :]

            slide_embed = model.backbone(features, coords, torch.tensor(1024).int().to(device), **kwargs)
            predicted_task_id = torch.argmax(slide_embed @ task_prompts.T)

            mlp = nn.Linear(768, num_classes[predicted_task_id]).to(device)
            task_weight = torch.load(task_model_paths[int(predicted_task_id)])
            mlp.load_state_dict({k.split('mlp.')[-1]:task_weight[k] for k in list(task_weight.keys())[-2:]})
            logits = mlp(slide_embed)
            logits = logits.float()
            preds = logits.argmax(1)
                
            probs = nn.functional.softmax(logits, dim=1)
            roc_kwargs = {"multi_class": "ovo", "average": "macro"}
            
            preds_all.append(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            targets_all.append(label.numpy())

        preds_all = np.concatenate(preds_all)
        try:
            probs_all = np.concatenate(probs_all)
        except:
            probs_all = pad_numpy_arrays(probs_all)
        targets_all = np.concatenate(targets_all)

    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, roc_kwargs=roc_kwargs, prefix=prefix)

    return eval_metrics, preds_all, targets_all

if __name__ == "__main__":
    torch.multiprocessing.set_sharing_strategy("file_system")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    parser = argparse.ArgumentParser(description="Finetune TITAN")
    
    parser.add_argument("--name", default=None, type=str)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./logs")
    parser.add_argument("--merge_model_path", type=str, default="/path/to/merged/checkpoints")
    
    args = parser.parse_args()
    
    num_tasks = 6
    num_classes = [2, 3, 2, 2, 2, 2]
    seq_dataset = Sequential_Generic_MIL_Dataset()
    overall_accs = []
    list_num_tasks = [1, 2, 3, 4, 5, 6]
    mACCs_all_folds = []
    fgt_all_folds = []
    bwt_all_folds = []
    ACC_all_seqs_all_folds = []
    task_prompts = torch.load("./task_prompts.pt")

    for fold_id in tqdm(range(0, 10)):
        fold = "fold_" + str(fold_id)
        task_model_paths = [
            f"{args.save_dir}" + fold_id + "/ckpts_outputs_finetuning_task_" + str(task_id) + ".pt" 
                for task_id in range(0, args.num_tasks)
        ]
        # print("Testing", fold)
        mean_ACCs = []
        acc_per_task_all_tasks = []
        ACC_all_seqs = []
        
        for seq_task in tqdm(list_num_tasks):
            seed_torch(device, 0)
            num_correct = 0.
            num_total = 0.

            acc_per_task = [0 for t in range(0, seq_task)]
            merge_model_path = str(args.merge_model_path) + "_{}".format(fold) + \
                                "/" + "merged_weight_opcm_random_sampling_{}_task_{}".format(fold, seq_task-1) + ".pth"
            
            # print(merge_model_path)
            base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
            base_model = base_model.to(device)
            base_model.vision_encoder.load_state_dict(torch.load(merge_model_path))
            model = CustomSequential(base_model, nn.Identity())
            model.eval()

            # load all MLPs
            mlp_task_weights = [torch.load(task_model_paths[task_id]) for task_id in range(seq_task)]
            for i in range(len(mlp_task_weights)):
                mlp_task_weights[i] = {k.split('mlp.')[-1]:mlp_task_weights[i][k] for k in list(mlp_task_weights[i].keys())[-2:]}
            
            merge_mlp_data = dict()
            merge_mlp_data['weight'] = torch.cat([data['weight'] for data in mlp_task_weights])
            merge_mlp_data['bias'] = torch.cat([data['bias'] for data in mlp_task_weights])

            for task_id in range(seq_task):
                _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
                results, preds_all, targets_all = eval(test_loader, model, num_classes[:seq_task], device, task_prompts[:seq_task], task_model_paths[:seq_task], merge_mlp_data, prefix="", save_location=None)
                
                num_correct += sum(preds_all == targets_all)
                num_total += len(test_loader)
                acc_per_task[task_id] = sum(preds_all == targets_all) / len(targets_all)
            
            overall_acc = num_correct / num_total
            ACC_all_seqs.append(overall_acc)
            acc_per_task_all_tasks.append(acc_per_task)

        # print(overall_acc)
        # print(acc_per_task)

        mACC = np.mean(ACC_all_seqs)
        ACC_all_seqs_all_folds.append(ACC_all_seqs)
        mACCs_all_folds.append(mACC)
        fgt = forgetting(acc_per_task_all_tasks)
        bwt = backward_transfer(acc_per_task_all_tasks)
        fgt_all_folds.append(fgt)
        bwt_all_folds.append(bwt)

    print(ACC_all_seqs_all_folds)
    print("mACC", np.mean(mACCs_all_folds), "std", np.std(mACCs_all_folds))
    print("BWT", np.mean(bwt_all_folds), "std", np.std(bwt_all_folds))
    print("FGT", np.mean(fgt_all_folds), "std", np.std(fgt_all_folds))