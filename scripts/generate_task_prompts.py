"""
scripts/generate_task_prompts.py
Tạo lại task_prompts.pt dùng TITAN text encoder.

task_prompts.pt có shape [num_tasks, 768]:
  - Mỗi hàng là embedding trung bình của tất cả class prompts trong task đó.
  - Dùng để routing task trong CLASS-IL inference (TCP mode).

Usage:
    python scripts/generate_task_prompts.py                  # 10 tasks (mặc định)
    python scripts/generate_task_prompts.py --num_tasks 6    # 6 tasks (backward compat)
    python scripts/generate_task_prompts.py --output /path/to/task_prompts.pt
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import argparse
import torch
from transformers import AutoModel

from mergeslide.prompts import ALL_TASK_PROMPTS, TEMPLATES
from mergeslide.datasets import DEFAULT_TASK_ORDER


def generate_task_prompts(titan_model, num_tasks: int, device: str) -> torch.Tensor:
    """
    Tạo task-level prompt embeddings bằng TITAN text encoder.

    Với mỗi task, lấy trung bình embedding của tất cả class prompts trong task đó.
    Kết quả shape: [num_tasks, 768]

    Args:
        titan_model: TITAN model đã load.
        num_tasks: Số task cần generate prompts.
        device: Device string ('cuda' hoặc 'cpu').

    Returns:
        Tensor shape [num_tasks, 768].
    """
    task_embeddings = []

    for task_id, prompt_fn in enumerate(ALL_TASK_PROMPTS[:num_tasks]):
        task_name = DEFAULT_TASK_ORDER[task_id]
        class_prompts, _ = prompt_fn()
        num_classes = len(class_prompts)
        print(f"  Task {task_id:2d} ({task_name:12s}): {num_classes} classes")

        # Tính class-level embeddings dùng zero_shot_classifier của TITAN
        with torch.autocast('cuda', torch.float16), torch.inference_mode():
            # classifier shape: [embed_dim, total_classes_in_task]
            classifier = titan_model.zero_shot_classifier(class_prompts, TEMPLATES, device=device)

        # Lấy mean theo chiều class → shape [embed_dim]
        task_embed = classifier.mean(dim=1)  # [768]
        task_embed = task_embed / task_embed.norm()  # L2 normalize
        task_embeddings.append(task_embed.cpu())

    task_prompts = torch.stack(task_embeddings, dim=0)  # [num_tasks, 768]
    return task_prompts


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate task_prompts.pt for CLASS-IL routing")
    parser.add_argument("--num_tasks", type=int, default=len(ALL_TASK_PROMPTS),
                        help=f"Number of tasks to generate prompts for (default: {len(ALL_TASK_PROMPTS)})")
    parser.add_argument("--output", type=str, default="./task_prompts.pt",
                        help="Output path for task_prompts.pt (default: ./task_prompts.pt)")
    parser.add_argument("--device", type=str, default=None,
                        help="Device to use: 'cuda' or 'cpu' (default: auto-detect)")
    args = parser.parse_args()

    if args.num_tasks > len(ALL_TASK_PROMPTS):
        raise ValueError(
            f"--num_tasks={args.num_tasks} vượt quá số prompt functions có sẵn ({len(ALL_TASK_PROMPTS)}). "
            f"Task order: {DEFAULT_TASK_ORDER}"
        )

    device_str = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    print(f"[INFO] Generating task_prompts.pt")
    print(f"[INFO] num_tasks = {args.num_tasks}")
    print(f"[INFO] device    = {device_str}")
    print(f"[INFO] output    = {args.output}")
    print(f"[INFO] task order: {DEFAULT_TASK_ORDER[:args.num_tasks]}")
    print()

    print("[INFO] Loading TITAN model...")
    titan_model = AutoModel.from_pretrained('MahmoodLab/TITAN', trust_remote_code=True)
    titan_model = titan_model.to(device_str)
    titan_model.eval()
    print("[INFO] TITAN loaded.\n")

    print(f"[INFO] Encoding {args.num_tasks} tasks...")
    task_prompts = generate_task_prompts(titan_model, args.num_tasks, device_str)

    print(f"\n[INFO] task_prompts shape: {task_prompts.shape}")

    output_path = args.output
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    torch.save(task_prompts, output_path)
    print(f"[INFO] Saved: {output_path}")
