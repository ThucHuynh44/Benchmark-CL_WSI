"""
Check whether configured WSI feature files are compatible with FEATHER.

This script validates only stored feature bags.  It does not re-extract patches.
"""

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ.setdefault("HDF5_USE_FILE_LOCKING", "FALSE")

import h5py
import yaml


REPO_ROOT = Path(__file__).resolve().parent.parent
FEATHER_CONFIG = REPO_ROOT / "configs" / "feather.yaml"
DATASETS_CONFIG = REPO_ROOT / "configs" / "datasets.yaml"
DEFAULT_FEATURE_DIM = 768


def _load_feather_cfg() -> dict:
    if not FEATHER_CONFIG.exists():
        return {}
    with open(FEATHER_CONFIG, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    return raw.get("feather", {})


def _load_dataset_paths() -> tuple:
    with open(DATASETS_CONFIG, "r") as handle:
        raw = yaml.safe_load(handle) or {}
    data_root = str(raw.get("data_root", ""))
    task_order = list(raw.get("task_order", []))
    raw_features = raw.get("features", {})
    features = {
        name: str(path).replace("{data_root}", data_root)
        for name, path in raw_features.items()
    }
    return task_order, features


def _iter_h5_files(root: Path, limit: int):
    if root.is_file() and root.suffix == ".h5":
        yield root
        return
    yielded = 0
    for path in root.rglob("*.h5"):
        yield path
        yielded += 1
        if limit > 0 and yielded >= limit:
            return


def _read_attr_int(attrs, names):
    for name in names:
        if name in attrs:
            try:
                return int(attrs[name])
            except Exception:
                return attrs[name]
    return None


def _protocol_warnings(path: Path, attrs, expected_patch_size: int, expected_mag: int):
    warnings = []
    patch_attr = _read_attr_int(attrs, ("patch_size", "patch_size_px", "tile_size", "tile_size_px"))
    mag_attr = _read_attr_int(attrs, ("magnification", "mag", "target_mag"))
    lower_path = str(path).lower()

    if patch_attr is not None and patch_attr != expected_patch_size:
        warnings.append(f"patch attr={patch_attr}, expected {expected_patch_size}")
    elif patch_attr is None and str(expected_patch_size) not in lower_path:
        warnings.append(f"no patch_size attr/path hint for {expected_patch_size}px")

    if mag_attr is not None and mag_attr != expected_mag:
        warnings.append(f"magnification attr={mag_attr}, expected {expected_mag}x")
    elif mag_attr is None and f"{expected_mag}x" not in lower_path:
        warnings.append(f"no magnification attr/path hint for {expected_mag}x")

    return warnings


def main() -> None:
    parser = argparse.ArgumentParser(description="Check FEATHER feature compatibility")
    parser.add_argument("--max_slides", type=int, default=5,
                        help="Maximum H5 slides to inspect per task. Use 0 for all.")
    parser.add_argument("--feature_dim", type=int, default=DEFAULT_FEATURE_DIM)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--magnification", type=int, default=None)
    args = parser.parse_args()

    feather_cfg = _load_feather_cfg()
    expected_patch_size = int(args.patch_size or feather_cfg.get("patch_size", 512))
    expected_mag = int(args.magnification or feather_cfg.get("magnification", 20))

    task_order, features_by_task = _load_dataset_paths()

    total_checked, total_errors, total_warnings = 0, 0, 0
    for task_name in task_order:
        root = Path(features_by_task.get(task_name, ""))
        if not root.exists():
            total_errors += 1
            print(f"[ERROR] task={task_name} missing feature path: {root}")
            continue

        checked_for_task = 0
        for h5_path in _iter_h5_files(root, args.max_slides):
            checked_for_task += 1
            total_checked += 1
            try:
                with h5py.File(h5_path, "r") as handle:
                    if "features" not in handle:
                        total_errors += 1
                        print(f"[ERROR] task={task_name} file={h5_path} missing key 'features'")
                        continue
                    shape = handle["features"].shape
                    attrs = dict(handle.attrs)
                    if len(shape) != 2 or shape[-1] != args.feature_dim:
                        total_errors += 1
                        print(
                            f"[ERROR] task={task_name} file={h5_path} "
                            f"features_shape={shape}, expected [N, {args.feature_dim}]"
                        )
                        continue
                    warnings = _protocol_warnings(
                        h5_path,
                        attrs,
                        expected_patch_size=expected_patch_size,
                        expected_mag=expected_mag,
                    )
                    if warnings:
                        total_warnings += len(warnings)
                        print(
                            f"[WARN] task={task_name} file={h5_path} "
                            f"features_shape={shape}: {'; '.join(warnings)}"
                        )
                    else:
                        print(f"[OK] task={task_name} file={h5_path} features_shape={shape}")
            except Exception as exc:
                total_errors += 1
                print(f"[ERROR] task={task_name} file={h5_path}: {type(exc).__name__}: {exc}")

        if checked_for_task == 0:
            total_errors += 1
            print(f"[ERROR] task={task_name} no H5 files found under {root}")

    print(
        f"\nChecked {total_checked} H5 files. "
        f"errors={total_errors}, warnings={total_warnings}"
    )
    if total_errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
