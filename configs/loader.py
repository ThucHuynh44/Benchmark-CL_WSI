"""
configs/loader.py
Load dataset paths from configs/datasets.yaml.

Priority:
  1. Path given explicitly via load_config(path=...)
  2. Environment variable MERGESLIDE_CONFIG
  3. configs/datasets.yaml  (default, git-ignored)
  4. configs/datasets.yaml.example  (fallback with placeholder paths — shows a warning)
"""

import os
import warnings
from pathlib import Path
from typing import Dict, List

import yaml

# Repository root = two levels up from this file (configs/loader.py)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "datasets.yaml"
_EXAMPLE_CONFIG = _REPO_ROOT / "configs" / "datasets.yaml.example"


def _resolve(value: str, data_root: str) -> str:
    """Replace '{data_root}' placeholder with the actual data_root value."""
    return value.replace("{data_root}", data_root)


def load_config(path: str = None, default_filename: str = "train.yaml") -> dict:
    """Load and resolve the YAML dataset config.

    Args:
        path: Optional explicit path to a YAML config file.
        default_filename: Default config filename inside configs/ directory.

    Returns:
        dict with keys: brca_csv, rcc_csv, nsclc_csv,
                        brca_features, rcc_features, nsclc_features,
                        esca_features, tgct_features, cesc_features,
                        split_dirs (list of 6 paths in task order)
    """
    # 1. Load shared dataset paths from datasets.yaml
    shared_path = _REPO_ROOT / "configs" / "datasets.yaml"
    if not shared_path.exists():
        shared_path = _REPO_ROOT / "configs" / "datasets.yaml.example"

    with open(shared_path, "r") as f:
        shared_raw = yaml.safe_load(f) or {}

    # 2. Determine and load stage-specific config file to use
    if path:
        cfg_path = Path(path)
    elif "MERGESLIDE_CONFIG" in os.environ:
        cfg_path = Path(os.environ["MERGESLIDE_CONFIG"])
    else:
        candidate = _REPO_ROOT / "configs" / default_filename
        if candidate.exists():
            cfg_path = candidate
        else:
            cfg_path = None

    stage_raw = {}
    if cfg_path and cfg_path.exists():
        with open(cfg_path, "r") as f:
            stage_raw = yaml.safe_load(f) or {}

    # Merge them (stage_raw overrides shared_raw if overlap)
    raw = {**shared_raw, **stage_raw}

    # Ensure required keys exist
    if "data_root" not in raw:
        raw["data_root"] = "/path/to/dataset"
    if "annotations" not in raw:
        raw["annotations"] = {"brca": "", "rcc": "", "nsclc": ""}
    if "features" not in raw:
        raw["features"] = {"brca": "", "rcc": "", "nsclc": "", "esca": "", "tgct": "", "cesc": ""}
    if "split_dirs" not in raw:
        raw["split_dirs"] = {"brca": "", "rcc": "", "nsclc": "", "esca": "", "tgct": "", "cesc": ""}

    data_root: str = raw["data_root"]
    hf_token = raw.get("hf_token") or os.environ.get("HF_TOKEN")

    if hf_token and hf_token != "YOUR_HF_TOKEN":
        try:
            from huggingface_hub import login
            login(token=hf_token, write_permission=False)
        except Exception as e:
            warnings.warn(f"Failed to log in to Hugging Face Hub using token: {e}")

    def r(v: str) -> str:
        return _resolve(v, data_root)

    return {
        # Settings
        "training": raw.get("training", {}),
        "merging": raw.get("merging", {}),
        "evaluation": raw.get("evaluation", {}),
        # Hugging Face token
        "hf_token": hf_token,
        # Annotation CSV/ZIP paths
        "brca_csv":  r(raw["annotations"]["brca"]),
        "rcc_csv":   r(raw["annotations"]["rcc"]),
        "nsclc_csv": r(raw["annotations"]["nsclc"]),
        # Feature directories
        "brca_features":  r(raw["features"]["brca"]),
        "rcc_features":   r(raw["features"]["rcc"]),
        "nsclc_features": r(raw["features"]["nsclc"]),
        "esca_features":  r(raw["features"]["esca"]),
        "tgct_features":  r(raw["features"]["tgct"]),
        "cesc_features":  r(raw["features"]["cesc"]),
        # Split directories (ordered: BRCA, RCC, NSCLC, ESCA, TGCT, CESC)
        "split_dirs": [
            r(raw["split_dirs"]["brca"]),
            r(raw["split_dirs"]["rcc"]),
            r(raw["split_dirs"]["nsclc"]),
            r(raw["split_dirs"]["esca"]),
            r(raw["split_dirs"]["tgct"]),
            r(raw["split_dirs"]["cesc"]),
        ],
    }
