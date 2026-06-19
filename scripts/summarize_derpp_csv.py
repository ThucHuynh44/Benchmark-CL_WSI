"""
Summarize DER++ continual metrics from train_derpp.py's eval CSV.

The input CSV must contain:
    fold, after_task, eval_task, acc, bacc, n

This script computes the metrics that can be recovered from aggregated CSV
rows, matching MergeSlide's continual-metric convention where FGT/BWT use
standard accuracy rather than balanced accuracy.
"""

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


REQUIRED_COLUMNS = {"fold", "after_task", "eval_task", "acc", "bacc", "n"}


def forgetting(results: List[List[float]]) -> float:
    """Compute average forgetting over tasks 0..T-2."""
    n_tasks = len(results)
    padded = [row + [0.0] * (n_tasks - len(row)) for row in results]
    np_res = np.array(padded, dtype=float)
    max_per_task = np.max(np_res, axis=0)
    return float(np.mean([max_per_task[i] - padded[-1][i] for i in range(n_tasks - 1)]))


def backward_transfer(results: List[List[float]]) -> float:
    """Compute average BWT over tasks 0..T-2."""
    n_tasks = len(results)
    return float(np.mean([results[-1][i] - results[i][i] for i in range(n_tasks - 1)]))


def validate_csv(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if df.empty:
        raise ValueError("Input CSV is empty.")


def weighted_seen_accuracy(seq_df: pd.DataFrame, metric_col: str = "acc") -> float:
    """Compute seen-task micro average from per-task metric and sample counts."""
    total_n = float(seq_df["n"].sum())
    if total_n <= 0:
        return float("nan")
    return float((seq_df[metric_col] * seq_df["n"]).sum() / total_n)


def summarize_fold(fold_df: pd.DataFrame, fold_id) -> dict:
    fold_df = fold_df.copy()
    fold_df["after_task"] = fold_df["after_task"].astype(int)
    fold_df["eval_task"] = fold_df["eval_task"].astype(int)

    max_task = int(fold_df["after_task"].max())
    results_by_seq: List[List[float]] = []
    acc_all_seqs = []
    bacc_all_seqs = []

    for seq_task in range(max_task + 1):
        seq_df = fold_df[fold_df["after_task"] == seq_task].sort_values("eval_task")
        expected_tasks = list(range(seq_task + 1))
        present_tasks = seq_df["eval_task"].tolist()
        if present_tasks != expected_tasks:
            raise ValueError(
                f"Fold {fold_id}, after_task {seq_task}: expected eval_task "
                f"{expected_tasks}, got {present_tasks}"
            )

        acc_per_task = seq_df["acc"].astype(float).tolist()
        results_by_seq.append(acc_per_task)
        acc_all_seqs.append(weighted_seen_accuracy(seq_df, metric_col="acc"))
        bacc_all_seqs.append(float(seq_df["bacc"].mean()))

    final_df = fold_df[fold_df["after_task"] == max_task].sort_values("eval_task")
    return {
        "fold": fold_id,
        "num_tasks": max_task + 1,
        "final_acc": float(final_df["acc"].mean()),
        "final_bacc": float(final_df["bacc"].mean()),
        "final_weighted_acc": weighted_seen_accuracy(final_df, metric_col="acc"),
        "mACC": float(np.mean(acc_all_seqs)),
        "mBACC": float(np.mean(bacc_all_seqs)),
        "BWT": backward_transfer(results_by_seq),
        "FGT": forgetting(results_by_seq),
    }


def append_mean_std(rows: List[dict], metric_cols: List[str]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    mean_row = {"fold": "mean"}
    std_row = {"fold": "std"}
    for col in metric_cols:
        mean_row[col] = float(df[col].mean())
        std_row[col] = float(df[col].std(ddof=0))
    if "num_tasks" in df.columns:
        mean_row["num_tasks"] = float(df["num_tasks"].mean())
        std_row["num_tasks"] = float(df["num_tasks"].std(ddof=0))
    return pd.concat([df, pd.DataFrame([mean_row, std_row])], ignore_index=True)


def build_per_task_summary(df: pd.DataFrame) -> pd.DataFrame:
    final_rows = []
    for fold_id, fold_df in df.groupby("fold", sort=True):
        max_task = int(fold_df["after_task"].max())
        final_rows.append(fold_df[fold_df["after_task"] == max_task])
    final_df = pd.concat(final_rows, ignore_index=True)

    rows = []
    for task_id, task_df in final_df.groupby("eval_task", sort=True):
        rows.append(
            {
                "task": int(task_id),
                "mean_acc": float(task_df["acc"].mean()),
                "std_acc": float(task_df["acc"].std(ddof=0)),
                "mean_bacc": float(task_df["bacc"].mean()),
                "std_bacc": float(task_df["bacc"].std(ddof=0)),
                "mean_n": float(task_df["n"].mean()),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize DER++ metrics from derpp_titan_eval.csv")
    parser.add_argument(
        "--csv",
        default="./checkpoints/derpp_titan/derpp_titan_eval.csv",
        help="Path to derpp_titan_eval.csv.",
    )
    parser.add_argument(
        "--output_prefix",
        default=None,
        help="Prefix for output CSV files. Defaults to '<input>_summary'.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    validate_csv(df)

    for col in ["after_task", "eval_task"]:
        df[col] = df[col].astype(int)
    for col in ["acc", "bacc", "n"]:
        df[col] = df[col].astype(float)

    fold_rows = [summarize_fold(fold_df, fold_id) for fold_id, fold_df in df.groupby("fold", sort=True)]
    metric_cols = ["final_bacc", "mACC", "BWT", "FGT"]
    fold_summary = append_mean_std(fold_rows, metric_cols)
    fold_summary = fold_summary[["fold", "num_tasks", "final_bacc", "mACC", "BWT", "FGT"]]
    task_summary = build_per_task_summary(df)

    output_prefix = args.output_prefix
    if output_prefix is None:
        root, _ = os.path.splitext(args.csv)
        output_prefix = f"{root}_summary"

    fold_csv = f"{output_prefix}_per_fold.csv"
    task_csv = f"{output_prefix}_per_task.csv"
    os.makedirs(os.path.dirname(fold_csv) or ".", exist_ok=True)
    fold_summary.to_csv(fold_csv, index=False)
    task_summary.to_csv(task_csv, index=False)

    mean_row = fold_summary[fold_summary["fold"] == "mean"].iloc[0]
    std_row = fold_summary[fold_summary["fold"] == "std"].iloc[0]
    print(
        "Final BACC: "
        f"{mean_row['final_bacc']:.4f} ({std_row['final_bacc']:.4f})\n"
        "mACC:       "
        f"{mean_row['mACC']:.4f} ({std_row['mACC']:.4f})\n"
        "BWT:        "
        f"{mean_row['BWT']:.4f} ({std_row['BWT']:.4f})\n"
        "FGT:        "
        f"{mean_row['FGT']:.4f} ({std_row['FGT']:.4f})"
    )
    print(f"CSV saved: {fold_csv}")
    print(f"CSV saved: {task_csv}")


if __name__ == "__main__":
    main()
