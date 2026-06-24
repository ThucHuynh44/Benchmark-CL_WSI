"""
OPCM merge for FEATHER per-task backbones.

Task-specific FEATHER classifier heads are intentionally excluded.  The output
is a merged backbone state dict that is paired with task heads at evaluation.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import yaml

from configs.loader import load_config
from mergeslide.datasets import get_num_classes
from mergeslide.feather_models import (
    DEFAULT_FEATHER_MODEL_NAME,
    create_feather_model,
    prepare_hf_token_env,
    split_feather_state_dict,
)


REPO_ROOT = Path(__file__).resolve().parent.parent
FEATHER_CONFIG = REPO_ROOT / "configs" / "feather.yaml"
FEATHER_MERGE_CONFIG = REPO_ROOT / "configs" / "merge_feather.yaml"


def _load_feather_cfg() -> dict:
    with open(FEATHER_CONFIG, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get("feather", {})


def _load_merge_cfg() -> dict:
    with open(FEATHER_MERGE_CONFIG, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get("feather_merging", {})


def _torch_load(path: str):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _float_tensor(tensor: torch.Tensor) -> bool:
    return tensor.is_floating_point() or tensor.is_complex()


def _validate_state_dict(state_dict: dict, base_state: dict, source: str) -> None:
    missing = sorted(set(base_state) - set(state_dict))
    unexpected = sorted(set(state_dict) - set(base_state))
    if missing or unexpected:
        raise KeyError(
            f"Backbone state mismatch for {source}: missing={missing[:5]} "
            f"unexpected={unexpected[:5]}"
        )
    shape_mismatches = [
        key for key in base_state
        if tuple(base_state[key].shape) != tuple(state_dict[key].shape)
    ]
    if shape_mismatches:
        raise ValueError(f"Backbone tensor shape mismatch for {source}: {shape_mismatches[:5]}")


def _task_vector_norm(state_dict: dict, base_state: dict) -> float:
    squared_norm = 0.0
    for key, base_tensor in base_state.items():
        task_tensor = state_dict[key]
        if not _float_tensor(base_tensor):
            continue
        delta = (task_tensor.float() - base_tensor.float()).reshape(-1)
        squared_norm += float(torch.dot(delta, delta))
    return squared_norm ** 0.5


def _merge_matrix(
    merged_tensor: torch.Tensor,
    base_tensor: torch.Tensor,
    task_tensor: torch.Tensor,
    previous_lambda: float,
) -> torch.Tensor:
    """Apply OPCM's orthogonal projection to one 2-D floating tensor."""
    original_dtype = merged_tensor.dtype
    merged = merged_tensor.float()
    base = base_tensor.float()
    task = task_tensor.float()
    previous_tv = merged - base
    task_tv = task - base

    u, _, vh = torch.linalg.svd(previous_tv, full_matrices=True)
    v = vh.T
    projected_task_tv = u.T @ task_tv @ v
    projected_task_tv.diagonal().zero_()
    cleaned_task_tv = u @ projected_task_tv @ v.T
    return (base + previous_lambda * previous_tv + cleaned_task_tv).to(original_dtype)


def _merge_other(
    merged_tensor: torch.Tensor,
    base_tensor: torch.Tensor,
    task_tensor: torch.Tensor,
    previous_lambda: float,
) -> torch.Tensor:
    original_dtype = merged_tensor.dtype
    merged = merged_tensor.float()
    base = base_tensor.float()
    task = task_tensor.float()
    return (base + previous_lambda * (merged - base) + (task - base)).to(original_dtype)


def _extract_backbone_state(checkpoint: dict, model, num_classes: int) -> dict:
    if isinstance(checkpoint, dict) and "backbone_state_dict" in checkpoint:
        return checkpoint["backbone_state_dict"]
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    backbone_state, _, _ = split_feather_state_dict(model, num_classes=num_classes)
    head_keys = set(model.state_dict()) - set(backbone_state)
    return {key: value for key, value in state_dict.items() if key not in head_keys}


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge FEATHER backbones with OPCM")
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--src_finetuned_checkpoints", type=str, default=None)
    parser.add_argument("--des_merged_checkpoints", type=str, default=None)
    parser.add_argument("--num_folds", type=int, default=None)
    parser.add_argument("--num_tasks", type=int, default=None)
    parser.add_argument("--base_num_classes", type=int, default=None)
    parser.add_argument("--no_pretrained_base", action="store_true")
    parser.add_argument("--use_wandb", action="store_true")
    parser.add_argument("--disable_wandb", action="store_true")
    args = parser.parse_args()

    feather_cfg = _load_feather_cfg()
    merge_cfg = _load_merge_cfg()
    prepare_hf_token_env(feather_cfg.get("hf_token"))
    use_wandb = (args.use_wandb or merge_cfg.get("use_wandb", False)) and not args.disable_wandb
    if use_wandb:
        try:
            import wandb  # noqa: F401
        except ImportError:
            warnings.warn("wandb package not found. Disabling wandb tracking.")
            use_wandb = False
    model_name = str(args.model_name or feather_cfg.get("model_name", DEFAULT_FEATHER_MODEL_NAME))
    src_dir = str(
        args.src_finetuned_checkpoints
        or merge_cfg.get("src_finetuned_checkpoints", feather_cfg.get("save_dir", "./checkpoints/feather_finetuned"))
    )
    dst_dir = str(
        args.des_merged_checkpoints
        or merge_cfg.get("des_merged_checkpoints", "./checkpoints/feather_merged")
    )
    num_folds = int(args.num_folds if args.num_folds is not None else merge_cfg.get("num_folds", 10))

    dataset_cfg = load_config(default_filename="feather.yaml")
    num_classes = get_num_classes(dataset_cfg.get("task_order"))
    num_tasks = int(args.num_tasks if args.num_tasks is not None else merge_cfg.get("num_tasks", len(num_classes)))
    num_classes = num_classes[:num_tasks]
    configured_base_num_classes = merge_cfg.get("base_num_classes")
    base_num_classes = int(
        args.base_num_classes
        if args.base_num_classes is not None
        else configured_base_num_classes if configured_base_num_classes is not None else num_classes[0]
    )

    base_model = create_feather_model(
        model_name,
        num_classes=base_num_classes,
        from_pretrained=bool(merge_cfg.get("from_pretrained_base", True)) and not args.no_pretrained_base,
    )
    base_state, _, _ = split_feather_state_dict(base_model, num_classes=base_num_classes)
    base_state = {key: value.detach().cpu() for key, value in base_state.items()}

    for fold_id in range(num_folds):
        fold_name = f"fold_{fold_id}"
        if use_wandb:
            import wandb
            wandb.init(
                project=merge_cfg.get("wandb_project", "MergeSlide-FEATHER"),
                entity=merge_cfg.get("wandb_entity"),
                group="feather_merge",
                job_type="merge",
                name=f"feather_merge_{fold_name}",
                config={
                    "fold": fold_id,
                    "model_name": model_name,
                    "num_tasks": num_tasks,
                    "base_num_classes": base_num_classes,
                },
                reinit=True,
            )
        task_states, task_norms = [], []
        for task_id in range(num_tasks):
            checkpoint_path = os.path.join(src_dir, fold_name, f"feather_task_{task_id}.pt")
            if not os.path.exists(checkpoint_path):
                raise FileNotFoundError(f"Missing FEATHER checkpoint: {checkpoint_path}")
            checkpoint = _torch_load(checkpoint_path)
            task_model = create_feather_model(
                model_name,
                num_classes=num_classes[task_id],
                from_pretrained=False,
            )
            task_state = _extract_backbone_state(checkpoint, task_model, num_classes[task_id])
            task_state = {key: value.detach().cpu() for key, value in task_state.items()}
            _validate_state_dict(task_state, base_state, checkpoint_path)
            task_states.append(task_state)
            task_norms.append(_task_vector_norm(task_state, base_state))

        merged_state = {key: value.clone() for key, value in task_states[0].items()}
        output_dir = os.path.join(dst_dir, f"_{fold_name}")
        os.makedirs(output_dir, exist_ok=True)
        torch.save(
            merged_state,
            os.path.join(output_dir, f"merged_backbone_feather_opcm_{fold_name}_task_0.pth"),
        )
        if use_wandb:
            import wandb
            wandb.log({
                "merge/task_id": 0,
                "merge/task_vector_norm": task_norms[0],
                "merge/average_task_vector_norm": task_norms[0],
                "merge/previous_lambda": 1.0,
            })

        previous_lambda = 1.0
        for task_id, task_state in enumerate(task_states[1:], start=1):
            average_norm = float(np.mean(task_norms[:task_id + 1]))
            for key, base_tensor in base_state.items():
                if not _float_tensor(base_tensor):
                    merged_state[key] = base_tensor.clone()
                elif base_tensor.ndim == 2:
                    merged_state[key] = _merge_matrix(
                        merged_state[key],
                        base_tensor,
                        task_state[key],
                        previous_lambda,
                    )
                else:
                    merged_state[key] = _merge_other(
                        merged_state[key],
                        base_tensor,
                        task_state[key],
                        previous_lambda,
                    )

            merged_norm = _task_vector_norm(merged_state, base_state)
            scale = 1.0
            if average_norm > 0 and merged_norm > 0:
                scale = average_norm / merged_norm
                for key, base_tensor in base_state.items():
                    if _float_tensor(base_tensor):
                        merged_state[key] = base_tensor + (merged_state[key] - base_tensor) * scale
                previous_lambda = merged_norm / average_norm
            else:
                previous_lambda = 1.0

            output_path = os.path.join(
                output_dir,
                f"merged_backbone_feather_opcm_{fold_name}_task_{task_id}.pth",
            )
            torch.save(merged_state, output_path)
            print(f"[FEATHER] saved: {output_path}")
            if use_wandb:
                import wandb
                wandb.log({
                    "merge/task_id": task_id,
                    "merge/task_vector_norm": task_norms[task_id],
                    "merge/merged_task_vector_norm": merged_norm,
                    "merge/average_task_vector_norm": average_norm,
                    "merge/normalization_scale": scale,
                    "merge/previous_lambda": previous_lambda,
                })

        final_path = os.path.join(dst_dir, f"merged_backbone_feather_opcm_{fold_name}.pth")
        torch.save(merged_state, final_path)
        print(f"[FEATHER] final checkpoint: {final_path}")
        if use_wandb:
            import wandb
            wandb.log({"merge/final_task_vector_norm": _task_vector_norm(merged_state, base_state)})
            wandb.finish()


if __name__ == "__main__":
    main()
