import os
from typing import TYPE_CHECKING

import numpy as np
import torch
from torch import Tensor, nn
from tqdm.auto import tqdm
from transformers import AutoModel
import argparse
import sys

from utils import get_task_vector_norm, is_leaf_module, svd, get_task_vector_state_dict
import matplotlib.pyplot as plt
import seaborn as sns

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

import pickle
import torch
import torch.nn.functional as F

def merge_linear_weights(
        merged_W: Tensor,
        pretrained_W: Tensor,
        task_W: Tensor,
        param_name: str,
        alpha: float,
        previous_lambda_t: float,
        lambda_t: float,
        accelerator: str = "cpu",
    ):
    original_device = merged_W.device
    merged_W = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W = task_W.to(accelerator)

    previous_merged_tv = merged_W - pretrained_W
    task_tv = task_W - pretrained_W

    u, s, v = svd(previous_merged_tv)

    projected_task_tv = u.T @ task_tv @ v
    projected_task_tv.diag().fill_(0)

    cleaned_task_tv = u @ projected_task_tv @ v.T

    new_merged_W = (
        pretrained_W
        + (previous_lambda_t * previous_merged_tv + cleaned_task_tv) / lambda_t
    )
    return new_merged_W.to(original_device)

def merge_other_parameters(
        merged_W: Tensor,
        pretrained_W: Tensor,
        task_W: Tensor,
        param_name: str,
        previous_lambda_t: float,
        lambda_t: float,
        accelerator: str = "cpu"
    ):
    original_device = merged_W.device
    merged_W = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W = task_W.to(accelerator)

    previous_merged_tv = merged_W - pretrained_W
    task_tv = task_W - pretrained_W

    new_merged_W = (
        pretrained_W + (previous_lambda_t * previous_merged_tv + task_tv) / lambda_t
    )
    return new_merged_W.to(original_device)

def compute_lambda_t(
        previous_merged_tv: Tensor, task_tv: Tensor, previous_lambda_t: float
    ):
    previous_merged_tv = torch.flatten(previous_merged_tv)
    task_tv = torch.flatten(task_tv)

    lambda_t = torch.linalg.vector_norm(
        previous_lambda_t * previous_merged_tv + task_tv
    ) / torch.linalg.vector_norm(previous_merged_tv)
    return lambda_t.item()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Model merging")
    parser.add_argument("--num_tasks", type=int, default=6)
    parser.add_argument("--src_finedtuned_checkpoints", type=str, default="/path/to/finetuned/checkpoints/")
    parser.add_argument("--des_merged_checkpoints", type=str, default="/path/to/merged/checkpoints/")
    args = parser.parse_args()

    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    for fold_id in range(0, 10):
        fold_id = "fold_" + str(fold_id)
        task_models = [
            f"{args.src_finedtuned_checkpoints}" + fold_id + "/ckpts_outputs_finetuning_task_" + str(task_id) + ".pt" 
                for task_id in range(0, args.num_tasks)
        ]
        base_weight = base_model.vision_encoder.state_dict()
        all_task_vectors = []
        all_task_weights = []
        for task_model in task_models:
            task_weight = torch.load(task_model, map_location=torch.device('cpu'))
            non_mlp_task_weight = {k.split('backbone.')[-1]:task_weight[k] for k in list(task_weight.keys())[:-2]}
            task_vector = get_task_vector_state_dict(non_mlp_task_weight, base_weight)
            all_task_vectors.append(task_vector)
            all_task_weights.append(non_mlp_task_weight)

        def filter_key(task_weight): return {k.split('backbone.')[-1]:task_weight[k].detach() for k in list(task_weight.keys())[:-2]}
        
        base_weight = {k:base_weight[k].detach() for k in list(base_weight.keys())}
        # Initialize merged_model with first task
        task_weight = filter_key(torch.load(task_models[0], map_location=torch.device('cpu')))
        merged_weight = task_weight
        
        # Performing C.OPCM
        alpha = 0.5
        previous_lambda_t = 1
        lambda_t = None
        avg_task_vector_norm = get_task_vector_norm(task_weight, base_weight)
        all_task_vector_norm = [avg_task_vector_norm]

        accumulated_path = str(args.des_merged_checkpoints) + fold_id
        os.makedirs(accumulated_path, exist_ok=True)
        
        for model_idx, task_model in enumerate(task_models[1:]):
            task_weight = filter_key(torch.load(task_model, map_location=torch.device('cpu')))
            all_task_vector_norm.append(
                get_task_vector_norm(task_weight, base_weight)
            )
            avg_task_vector_norm = np.mean(all_task_vector_norm)
            lambda_t = 1  # temporary value

            for module_name, module in tqdm(
                list(base_model.vision_encoder.named_modules()),
                desc=f"Processing {model_idx + 2}",
                leave=False,
            ):
                if not is_leaf_module(module):
                    continue

                if isinstance(module, nn.Linear):
                    merged_weight[module_name + '.weight'] = merge_linear_weights(
                        merged_weight[module_name + '.weight'].detach(), # continual merged model
                        base_model.vision_encoder.get_submodule(module_name).weight.detach(), # base model weight
                        task_weight[module_name + '.weight'].detach(), # task model weight
                        param_name=".".join([module_name, "weight"]),
                        alpha=alpha,
                        previous_lambda_t=previous_lambda_t,
                        lambda_t=lambda_t
                    )
                    if module.bias is not None:
                        merged_weight[module_name + '.bias'] = merge_other_parameters(
                            merged_weight[module_name + '.bias'].detach(),
                            base_model.vision_encoder.get_submodule(module_name).bias.detach(),
                            task_weight[module_name + '.bias'].detach(),
                            param_name=".".join([module_name, "bias"]),
                            previous_lambda_t=previous_lambda_t,
                            lambda_t=lambda_t
                        )
                else:
                    for param_name, param in module.named_parameters():
                        merged_weight[module_name + '.' + param_name] = merge_other_parameters(
                            merged_W=merged_weight[module_name + '.' + param_name].detach(),
                            pretrained_W=base_model.vision_encoder.get_submodule(
                                module_name
                            ).get_parameter(param_name).detach(),
                            task_W=task_weight[module_name + '.' + param_name].detach(),
                            param_name=".".join([module_name, param_name]),
                            previous_lambda_t=previous_lambda_t,
                            lambda_t=lambda_t
                        )

            task_vector_norm = get_task_vector_norm(merged_weight, base_weight)
            lambda_t *= task_vector_norm / avg_task_vector_norm
            previous_lambda_t = lambda_t

            for param_name, param in base_model.vision_encoder.named_parameters():
                merged_weight[param_name] = base_model.vision_encoder.get_parameter(param_name).detach() + ( # base model
                    merged_weight[param_name] - base_model.vision_encoder.get_parameter(param_name).detach() # task vector
                ) * (avg_task_vector_norm / task_vector_norm) # lambda

            torch.save(merged_weight, accumulated_path + "/merged_weight_opcm_random_sampling_" + fold_id + "_" + "task_" + str(model_idx + 1) + ".pth")
            
        torch.save(merged_weight, "merged_weight_opcm_random_sampling_" + fold_id + ".pth")
