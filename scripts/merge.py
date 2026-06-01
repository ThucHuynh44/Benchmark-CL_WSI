"""
scripts/merge.py
Continual OPCM (Orthogonal Projection-based Continual Model Merging).

Usage:
    python scripts/merge.py \
        --num_tasks 6 \
        --src_finetuned_checkpoints /path/to/finetuned/checkpoints \
        --des_merged_checkpoints /path/to/merged/checkpoints/
"""

import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm
from transformers import AutoModel

from mergeslide.utils import (
    get_task_vector_norm,
    get_task_vector_state_dict,
    is_leaf_module,
    svd,
    Tensor,
)


def merge_linear_weights(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    previous_lambda_t: float,
    lambda_t: float,
    accelerator: str = "cpu",
) -> Tensor:
    """Merge a linear weight matrix using orthogonal projection.

    Projects the incoming task vector onto the null space of the current
    merged task vector, then accumulates.

    Args:
        merged_W: Current merged weight matrix.
        pretrained_W: Pre-trained base weight matrix.
        task_W: New task's fine-tuned weight matrix.
        previous_lambda_t: Scaling factor for the previous merged task vector.
        lambda_t: Scaling factor for the combined task vector.
        accelerator: Device to run SVD on.

    Returns:
        New merged weight matrix.
    """
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

    new_merged_W = pretrained_W + (previous_lambda_t * previous_merged_tv + cleaned_task_tv) / lambda_t
    return new_merged_W.to(original_device)


def merge_other_parameters(
    merged_W: Tensor,
    pretrained_W: Tensor,
    task_W: Tensor,
    previous_lambda_t: float,
    lambda_t: float,
    accelerator: str = "cpu",
) -> Tensor:
    """Merge non-linear parameters (biases, norms) with simple averaging.

    Args:
        merged_W: Current merged parameter tensor.
        pretrained_W: Pre-trained base parameter tensor.
        task_W: New task's fine-tuned parameter tensor.
        previous_lambda_t: Scaling factor for the previous merged task vector.
        lambda_t: Scaling factor for the combined task vector.
        accelerator: Device to run computation on.

    Returns:
        New merged parameter tensor.
    """
    original_device = merged_W.device
    merged_W = merged_W.to(accelerator)
    pretrained_W = pretrained_W.to(accelerator)
    task_W = task_W.to(accelerator)

    previous_merged_tv = merged_W - pretrained_W
    task_tv = task_W - pretrained_W
    new_merged_W = pretrained_W + (previous_lambda_t * previous_merged_tv + task_tv) / lambda_t
    return new_merged_W.to(original_device)


def compute_lambda_t(
    previous_merged_tv: Tensor, task_tv: Tensor, previous_lambda_t: float
) -> float:
    """Compute the normalisation factor λ_t for the merged task vector."""
    prev_flat = torch.flatten(previous_merged_tv)
    task_flat = torch.flatten(task_tv)
    lambda_t = (
        torch.linalg.vector_norm(previous_lambda_t * prev_flat + task_flat)
        / torch.linalg.vector_norm(prev_flat)
    )
    return lambda_t.item()


def _filter_backbone_weights(raw_state_dict: dict) -> dict:
    """Strip the 'backbone.' prefix and exclude the MLP head (last 2 keys)."""
    keys = list(raw_state_dict.keys())[:-2]
    return {k.split('backbone.')[-1]: raw_state_dict[k].detach() for k in keys}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Continual OPCM model merging")
    parser.add_argument("--num_tasks", type=int, default=6)
    parser.add_argument("--src_finetuned_checkpoints", type=str,
                        default="./checkpoints/finetuned",
                        help="Directory containing per-task finetuned checkpoints")
    parser.add_argument("--des_merged_checkpoints", type=str,
                        default="./checkpoints/merged",
                        help="Output directory for merged checkpoints")
    args = parser.parse_args()

    base_model = AutoModel.from_pretrained("MahmoodLab/TITAN", trust_remote_code=True)
    base_weight = {k: v.detach() for k, v in base_model.vision_encoder.state_dict().items()}

    for fold_id in range(10):
        fold_name = f"fold_{fold_id}"
        task_model_paths = [
            os.path.join(args.src_finetuned_checkpoints, fold_name,
                         f"ckpts_outputs_finetuning_task_{task_id}.pt")
            for task_id in range(args.num_tasks)
        ]

        # Pre-compute task vectors for λ normalization
        all_task_vector_norms = []
        all_task_weights = []
        for path in task_model_paths:
            task_weight = _filter_backbone_weights(torch.load(path, map_location='cpu'))
            all_task_weights.append(task_weight)
            all_task_vector_norms.append(get_task_vector_norm(task_weight, base_weight))

        # Initialize merged model with the first task
        merged_weight = all_task_weights[0]
        previous_lambda_t = 1.0
        avg_task_vector_norm = all_task_vector_norms[0]

        output_dir = os.path.join(args.des_merged_checkpoints, f"_{fold_name}")
        os.makedirs(output_dir, exist_ok=True)

        # Save the task-0-only checkpoint (index 0)
        torch.save(
            merged_weight,
            os.path.join(output_dir, f"merged_weight_opcm_random_sampling_{fold_name}_task_0.pth"),
        )

        for model_idx, task_weight in enumerate(all_task_weights[1:], start=1):
            avg_task_vector_norm = np.mean(all_task_vector_norms[:model_idx + 1])
            lambda_t = 1.0  # temporary placeholder; updated after merging

            for module_name, module in tqdm(
                list(base_model.vision_encoder.named_modules()),
                desc=f"Merging task {model_idx + 1}/{args.num_tasks}",
                leave=False,
            ):
                if not is_leaf_module(module):
                    continue

                if isinstance(module, nn.Linear):
                    w_key = f"{module_name}.weight"
                    merged_weight[w_key] = merge_linear_weights(
                        merged_weight[w_key].detach(),
                        base_model.vision_encoder.get_submodule(module_name).weight.detach(),
                        task_weight[w_key].detach(),
                        previous_lambda_t=previous_lambda_t,
                        lambda_t=lambda_t,
                    )
                    if module.bias is not None:
                        b_key = f"{module_name}.bias"
                        merged_weight[b_key] = merge_other_parameters(
                            merged_weight[b_key].detach(),
                            base_model.vision_encoder.get_submodule(module_name).bias.detach(),
                            task_weight[b_key].detach(),
                            previous_lambda_t=previous_lambda_t,
                            lambda_t=lambda_t,
                        )
                else:
                    for param_name, _ in module.named_parameters():
                        pk = f"{module_name}.{param_name}"
                        merged_weight[pk] = merge_other_parameters(
                            merged_weight[pk].detach(),
                            base_model.vision_encoder.get_submodule(module_name).get_parameter(param_name).detach(),
                            task_weight[pk].detach(),
                            previous_lambda_t=previous_lambda_t,
                            lambda_t=lambda_t,
                        )

            # Update λ after merging
            task_vector_norm = get_task_vector_norm(merged_weight, base_weight)
            lambda_t *= task_vector_norm / avg_task_vector_norm
            previous_lambda_t = lambda_t

            # Renormalize merged weight
            for param_name, param in base_model.vision_encoder.named_parameters():
                base_p = base_model.vision_encoder.get_parameter(param_name).detach()
                task_vector = merged_weight[param_name] - base_p
                merged_weight[param_name] = base_p + task_vector * (avg_task_vector_norm / task_vector_norm)

            ckpt_name = f"merged_weight_opcm_random_sampling_{fold_name}_task_{model_idx}.pth"
            torch.save(merged_weight, os.path.join(output_dir, ckpt_name))
            print(f"Saved: {os.path.join(output_dir, ckpt_name)}")

        # Also save the final merged checkpoint at the top-level output directory
        final_path = os.path.join(args.des_merged_checkpoints, f"merged_weight_opcm_random_sampling_{fold_name}.pth")
        torch.save(merged_weight, final_path)
        print(f"Final checkpoint: {final_path}")
