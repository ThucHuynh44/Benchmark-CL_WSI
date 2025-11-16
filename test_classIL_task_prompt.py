import argparse

import numpy as np
import torch
torch.manual_seed(42)  # Set seed
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, precision_score, recall_score, f1_score
from tqdm import tqdm
from transformers import AutoModel
from utils import get_eval_metrics, seed_torch
from datasets import Sequential_Generic_MIL_Dataset
import time
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

def eval(test_loader, task_id, model, dict_class, num_classes, device, task_prompts, task_model_paths, merge_mlp_data, prefix, save_location, **kwargs):
    preds_all = []
    probs_all = []
    targets_all = []

    convert_preds_all = []
    convert_targets_all = []

    K = 300
    dict_convert_class = {
        0: {0:0, 1:1},
        1: {0:2, 1:3, 2:4},
        2: {0:5, 1:6},
        3: {0:7, 1:8},
        4: {0:9, 1:10},
        5: {0:11, 1:12}
    }

    times = []
    
    task_weights = [torch.load(model_path) for model_path in task_model_paths]
    task_weights = [{k.split('mlp.')[-1]:task_weight[k] for k in list(task_weight.keys())[-2:]} for task_weight in task_weights]

    with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        slide_per_task = []
        slide_per_class = {}
        for features, coords, label in tqdm(test_loader):
            features = features.to(device)
            coords = coords.long().to(device)

            # 1. Random sampling
            indices = torch.randperm(features.shape[0])[:K]
            features = features[indices, :]
            coords = coords[indices, :]

            start = time.time()
            # 2. Get slide embedding
            slide_embed = model.backbone(features, coords, torch.tensor(1024).int().to(device), **kwargs)
            
            # 3. Get predicted task_id
            predicted_task_id = torch.argmax(slide_embed @ task_prompts.T)

            # if predicted task id usable, use it
            # task_weight = torch.load(task_model_paths[int(predicted_task_id)])
            mlp = nn.Linear(768, num_classes[predicted_task_id]).to(device)
            mlp.load_state_dict(task_weights[int(predicted_task_id)])
            # logits = (slide_embed @ task_weights[int(predicted_task_id)]['weight'].T) + task_weights[int(predicted_task_id)]['bias']
            
            # task_weight
            logits = mlp(slide_embed)
            logits = logits.float()
            preds = logits.argmax(1)
            end = time.time()
            times.append(end - start)
                
            probs = nn.functional.softmax(logits, dim=1)
            roc_kwargs = {"multi_class": "ovo", "average": "macro"}
            
            preds_all.append(preds.cpu().numpy())
            probs_all.append(probs.cpu().numpy())
            targets_all.append(label.numpy())

            # 4. For visualization
            slide_per_task.append(slide_embed)
            if dict_convert_class[task_id][int(label)] not in slide_per_class:
                slide_per_class[dict_convert_class[task_id][int(label)]] = [slide_embed]
            else:
                slide_per_class[dict_convert_class[task_id][int(label)]].append(slide_embed)

            convert_label = torch.Tensor([dict_class[int(label[0])]])

            try:
                convert_pred = torch.Tensor([dict_class[int(preds[0])]])
            except:
                convert_pred = torch.Tensor([4])

            convert_targets_all.append(convert_label)
            convert_preds_all.append(convert_pred)
        
        preds_all = np.concatenate(preds_all)
        try:
            probs_all = np.concatenate(probs_all)
        except:
            # Padding, just in case
            probs_all = pad_numpy_arrays(probs_all)

        targets_all = np.concatenate(targets_all)

        convert_preds_all = np.concatenate(convert_preds_all)
        convert_targets_all = np.concatenate(convert_targets_all)

    eval_metrics = get_eval_metrics(targets_all, preds_all, probs_all, roc_kwargs=roc_kwargs, prefix=prefix)

    return eval_metrics, preds_all, targets_all, slide_per_task, slide_per_class, probs_all, convert_preds_all, convert_targets_all, sum(times)

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
    parser.add_argument("--merge_model_path", type=str, default="/path/to/merged/checkpoints")

    args = parser.parse_args()

    # A little bit hard code here for 6 tasks    
    num_tasks = 6
    num_classes = [2, 3, 2, 2, 2, 2]
    dict_convert_class = {
        0: {0:0, 1:1},
        1: {0:2, 1:3, 2:4},
        2: {0:5, 1:6},
        3: {0:7, 1:8},
        4: {0:9, 1:10},
        5: {0:11, 1:12}
    }
    seq_dataset = Sequential_Generic_MIL_Dataset()

    # load model from huggingface
    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)    
    base_model = base_model.to(device)
    overall_accs = []
    task_prompts = torch.load("./task_prompts.pt")

    overall_accs = []
    overall_baccs = []
    overall_aucs = []
    overall_recalls = []
    overall_precisions = []
    overall_macro_f1s = []
    overall_weighted_f1s = []

    overall_time_all_folds = []
    for fold_id in tqdm(range(0, 10)):
        num_total = 0.
        num_correct = 0.
        fold = "fold_" + str(fold_id)
        merge_model_path = str(args.merge_model_path)
        print(merge_model_path)
        task_model_paths = [
            f"{args.save_dir}" + fold_id + "/ckpts_outputs_finetuning_task_" + str(task_id) + ".pt" 
                for task_id in range(0, args.num_tasks)
        ]

        # Load all MLPs just in case to use
        mlp_task_weights = [torch.load(task_model_paths[task_id]) for task_id in range(num_tasks)]
        for i in range(len(mlp_task_weights)):
            mlp_task_weights[i] = {k.split('mlp.')[-1]:mlp_task_weights[i][k] for k in list(mlp_task_weights[i].keys())[-2:]}
        
        merge_mlp_data = dict()
        merge_mlp_data['weight'] = torch.cat([data['weight'] for data in mlp_task_weights])
        merge_mlp_data['bias'] = torch.cat([data['bias'] for data in mlp_task_weights])
        
        # Load base model            
        mlp = nn.Identity()
        model = CustomSequential(base_model, mlp)
        model.eval()

        # Load test dataset for all tasks
        dict_slide_per_task = {i:[] for i in range(num_tasks)}
        dict_slide_per_class = {i:[] for i in range(0, sum(num_classes))}

        acc_per_task = {}
        
        overall_time = 0.

        all_acc_per_task = []
        all_baccs = []
        all_predictions = []
        all_labels = []
        all_logits = []
        all_probs = []
        all_baccs = []
        all_accs = []
        aucs = []

        throughputs = []
        per_task_epoch_times = []

        for task_id in range(num_tasks):
            # print("TASK", task_id)
            _, _, test_loader = seq_dataset.get_data_loaders(fold_id, task_id)
            start = time.time()
            results, preds_all, targets_all, slide_per_task, slide_per_class, probs_all, convert_preds_all, convert_targets_all, sum_time = eval(test_loader, task_id, model, dict_convert_class[task_id], num_classes[:num_tasks], device, task_prompts[:num_tasks], task_model_paths[:num_tasks], merge_mlp_data, prefix="", save_location=None)
            end = time.time()
            throughputs.append(targets_all.shape[0] / sum_time)
            per_task_epoch_times.append(sum_time)
            
            num_correct += sum(preds_all == targets_all)
            num_total += len(test_loader)
            # print(results)
            acc_per_task[task_id] = results['/acc']

            dict_slide_per_task[task_id].extend(slide_per_task)
            for class_id in slide_per_class:
                dict_slide_per_class[class_id].extend(slide_per_class[class_id])
        
            overall_time_per_task = sum_time / len(test_loader)
            overall_time += overall_time_per_task

            all_predictions.append(convert_preds_all)
            all_labels.append(convert_targets_all)

            bacc = balanced_accuracy_score(targets_all, preds_all)
            all_baccs.append(bacc)
            all_accs.append(sum(preds_all == targets_all) / len(test_loader))

            for i in range(len(dict_convert_class[task_id])):
                # Binarize the true labels for the current class (one-vs-rest)
                y_true_binary = (targets_all == i).astype(int)
                # Get the predicted probabilities for the current class
                if len(probs_all.shape) == 3:
                    probs_all = probs_all.squeeze(1)

                y_score_class_i = probs_all[:, i]

                # Calculate ROC AUC for the current class
                auc_score = roc_auc_score(y_true_binary, y_score_class_i)
                aucs.append(auc_score)
        
        all_labels, all_predictions = np.concatenate(all_labels), np.concatenate(all_predictions)
        bacc = np.mean(all_baccs)
        acc_new = np.mean(all_accs)
        precision_per_class = precision_score(all_labels, all_predictions, average=None)
        recall_per_class = recall_score(all_labels, all_predictions, average=None)
        weighted_f1_score = f1_score(all_labels, all_predictions, average="weighted")
        macro_f1_score = f1_score(all_labels, all_predictions, average="macro")

        overall_baccs.append(bacc)
        overall_aucs.append(np.array(aucs))
        overall_recalls.append(recall_per_class)
        overall_precisions.append(precision_per_class)
        overall_macro_f1s.append(macro_f1_score)
        overall_weighted_f1s.append(weighted_f1_score)

        overall_time = overall_time / num_tasks
        overall_time_all_folds.append(overall_time)
        # overall_acc = num_correct / num_total
        overall_accs.append(acc_new)
        all_acc_per_task.append(acc_per_task)

        print("overall_acc", acc_new)
        print("overall_time", overall_time)
    
    print([float(acc) for acc in overall_accs])
    print("Accuracy:", np.mean(overall_accs), "(", np.std(overall_accs), ")")
    print("Balanced Accuracy:", np.mean(overall_baccs), "(", np.std(overall_baccs), ")")
    print("Macro F1:", np.mean(overall_macro_f1s), "(", np.std(overall_macro_f1s), ")")
    print("Weighted F1:", np.mean(overall_weighted_f1s), "(", np.std(overall_weighted_f1s), ")")

    print("Recall:")
    for value, std in zip(list(np.mean(np.stack(overall_recalls), axis=0)), list(np.std(np.stack(overall_recalls), axis=0))):
        print(value, "(", std, ")")
    
    print("Precision:")
    for value, std in zip(list(np.mean(np.stack(overall_precisions), axis=0)), list(np.std(np.stack(overall_precisions), axis=0))):
        print(value, "(", std, ")")
    
    print("AUC:")
    for value, std in zip(list(np.mean(np.stack(overall_aucs), axis=0)), list(np.std(np.stack(overall_aucs), axis=0))):
        print(value, "(", std, ")")

    print("Over time all folds:", np.mean(overall_time_all_folds), "(", np.std(overall_time_all_folds), ")")
    print("Acc per task:")
    
    accs = {task_id:list() for task_id in range(num_tasks)}
    for i in range(len(all_acc_per_task)):
        for task_id in range(num_tasks):
            accs[task_id].append(all_acc_per_task[i][task_id])
    
    for task_id in range(len(accs)):
        print("Acc ", task_id, np.mean(accs[task_id]), np.std(accs[task_id]))