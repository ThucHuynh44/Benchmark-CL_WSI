"""
Probe FEATHER/MIL-Lab loading and forward behavior.

Usage:
    HF_TOKEN=... python scripts/probe_feather.py
    python scripts/probe_feather.py --feature_path /path/to/slide.h5
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import torch
import yaml

from mergeslide.feather_models import (
    DEFAULT_FEATHER_MODEL_NAME,
    DEFAULT_FEATURE_DIM,
    FeatherMILWrapper,
    create_feather_model,
    prepare_hf_token_env,
)
from mergeslide.utils import seed_torch


REPO_ROOT = Path(__file__).resolve().parent.parent
FEATHER_CONFIG = REPO_ROOT / "configs" / "feather.yaml"


def _config_token():
    if not FEATHER_CONFIG.exists():
        return None
    with open(FEATHER_CONFIG, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get("feather", {}).get("hf_token")


def _load_features(path: str, k: int) -> torch.Tensor:
    with h5py.File(path, "r") as handle:
        if "features" not in handle:
            raise KeyError(f"{path} does not contain an HDF5 'features' dataset.")
        features = torch.from_numpy(handle["features"][:])
    if k > 0 and features.shape[0] > k:
        features = features[:k]
    return features


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe FEATHER model loading and forward pass")
    parser.add_argument("--model_name", type=str, default=DEFAULT_FEATHER_MODEL_NAME)
    parser.add_argument("--num_classes", type=int, default=2)
    parser.add_argument("--feature_path", type=str, default=None)
    parser.add_argument("--k", type=int, default=16)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no_pretrained", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    seed_torch(device, 0)
    token = prepare_hf_token_env(_config_token())
    print(f"[FEATHER] HF token available: {bool(token)}")
    print(f"[FEATHER] model_name={args.model_name}")

    model = create_feather_model(
        args.model_name,
        num_classes=args.num_classes,
        from_pretrained=not args.no_pretrained,
    )
    wrapper = FeatherMILWrapper(model, num_classes=args.num_classes).to(device)
    wrapper.eval()

    if args.feature_path:
        features = _load_features(args.feature_path, args.k)
        source = str(Path(args.feature_path))
    else:
        features = torch.randn(max(args.k, 1), DEFAULT_FEATURE_DIM)
        source = "random"

    features = features.to(device)
    with torch.no_grad():
        logits = wrapper(features)

    print(f"[FEATHER] feature_source={source}")
    print(f"[FEATHER] features_shape={tuple(features.shape)}")
    print(f"[FEATHER] logits_shape={tuple(logits.shape)}")
    print(f"[FEATHER] forward_mode={wrapper._forward_mode}")


if __name__ == "__main__":
    main()
