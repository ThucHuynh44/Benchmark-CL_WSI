"""
Compute BWT and FGT from a continual eval CSV.

Input CSV must contain:
    fold, after_task, eval_task, acc

By default this matches train_feather_continual.py: BWT and FGT are computed
with standard accuracy. Use --metric bacc if you explicitly want balanced
accuracy instead.

Example:
    python scripts/compute_bwt_fgt.py \
        --csv checkpoints/feather_agem_buffer10_fullpatch/feather_agem_eval.csv
"""

import argparse
import os
from typing import Dict, List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"fold", "after_task", "eval_task"}


def _finite_mean(values: List[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.mean(finite_values)) if finite_values else np.nan


def _finite_std(values: List[float]) -> float:
    finite_values = [float(value) for value in values if np.isfinite(float(value))]
    return float(np.std(finite_values, ddof=0)) if finite_values else np.nan


def validate_csv(df: pd.DataFrame, metric: str) -> None:
    missing = (REQUIRED_COLUMNS | {metric}) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Input CSV is empty.")


def summarize_fold(fold_df: pd.DataFrame, fold_id, metric: str) -> Dict[str, float]:
    fold_df = fold_df.copy()
    fold_df["after_task"] = fold_df["after_task"].astype(int)
    fold_df["eval_task"] = fold_df["eval_task"].astype(int)
    fold_df[metric] = fold_df[metric].astype(float)

    final_after_task = int(fold_df["after_task"].max())
    final_rows = fold_df[fold_df["after_task"] == final_after_task]

    final_by_task = {
        int(row["eval_task"]): float(row[metric])
        for _, row in final_rows.iterrows()
    }
    diagonal = {
        int(row["eval_task"]): float(row[metric])
        for _, row in fold_df.iterrows()
        if int(row["after_task"]) == int(row["eval_task"])
    }

    bwt_values = [
        final_by_task[task_id] - diagonal[task_id]
        for task_id in range(final_after_task)
        if task_id in final_by_task and task_id in diagonal
    ]

    fgt_values = []
    for task_id in range(final_after_task):
        trajectory = [
            float(row[metric])
            for _, row in fold_df.iterrows()
            if int(row["eval_task"]) == task_id and int(row["after_task"]) >= task_id
        ]
        if trajectory and task_id in final_by_task:
            fgt_values.append(max(trajectory) - final_by_task[task_id])

    return {
        "fold": fold_id,
        "num_tasks": final_after_task + 1,
        "metric": metric,
        "BWT": _finite_mean(bwt_values),
        "FGT": _finite_mean(fgt_values),
    }


def append_mean_std(rows: List[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    mean_row = {
        "fold": "mean",
        "num_tasks": float(df["num_tasks"].mean()),
        "metric": df["metric"].iloc[0],
        "BWT": _finite_mean(df["BWT"].tolist()),
        "FGT": _finite_mean(df["FGT"].tolist()),
    }
    std_row = {
        "fold": "std",
        "num_tasks": float(df["num_tasks"].std(ddof=0)),
        "metric": df["metric"].iloc[0],
        "BWT": _finite_std(df["BWT"].tolist()),
        "FGT": _finite_std(df["FGT"].tolist()),
    }
    return pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute BWT and FGT from continual eval CSV.")
    parser.add_argument("--csv", required=True, help="Path to eval CSV.")
    parser.add_argument("--metric", default="acc", choices=("acc", "bacc"), help="Metric used for BWT/FGT.")
    parser.add_argument("--output_csv", default=None, help="Optional output CSV. Defaults to '<input>_bwt_fgt.csv'.")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    validate_csv(df, args.metric)

    rows = [
        summarize_fold(fold_df, fold_id, args.metric)
        for fold_id, fold_df in df.groupby("fold", sort=True)
    ]
    summary = append_mean_std(rows)

    output_csv = args.output_csv
    if output_csv is None:
        root, _ = os.path.splitext(args.csv)
        output_csv = f"{root}_bwt_fgt.csv"

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    summary.to_csv(output_csv, index=False)

    mean_row = summary[summary["fold"] == "mean"].iloc[0]
    std_row = summary[summary["fold"] == "std"].iloc[0]
    print(f"Metric: {args.metric}")
    print(f"BWT: {mean_row['BWT']:.6f} ({std_row['BWT']:.6f})")
    print(f"FGT: {mean_row['FGT']:.6f} ({std_row['FGT']:.6f})")
    print(f"CSV saved: {output_csv}")


if __name__ == "__main__":
    main()
