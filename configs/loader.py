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


def load_config(path: str = None) -> dict:
    """Load and resolve the YAML dataset config.

    Args:
        path: Optional explicit path to a YAML config file.

    Returns:
        dict with keys: brca_csv, rcc_csv, nsclc_csv,
                        brca_features, rcc_features, nsclc_features,
                        esca_features, tgct_features, cesc_features,
                        split_dirs (list of 6 paths in task order)
    """
    # Determine config file to use
    if path:
        cfg_path = Path(path)
    elif "MERGESLIDE_CONFIG" in os.environ:
        cfg_path = Path(os.environ["MERGESLIDE_CONFIG"])
    elif _DEFAULT_CONFIG.exists():
        cfg_path = _DEFAULT_CONFIG
    else:
        warnings.warn(
            f"configs/datasets.yaml not found. "
            f"Using example config with placeholder paths.\n"
            f"Run:  cp configs/datasets.yaml.example configs/datasets.yaml\n"
            f"Then edit configs/datasets.yaml with your local dataset root.",
            stacklevel=3,
        )
        cfg_path = _EXAMPLE_CONFIG

    with open(cfg_path, "r") as f:
        raw = yaml.safe_load(f)

    data_root: str = raw["data_root"]

    def r(v: str) -> str:
        return _resolve(v, data_root)

    return {
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
