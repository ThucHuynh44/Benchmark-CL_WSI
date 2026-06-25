# FEATHER Backend

This backend runs alongside the existing TITAN code.  It uses MIL-Lab's
FEATHER model `abmil.base.conch_v15.pc108-24k` on CONCH v1.5 feature bags.

## Setup

Install MIL-Lab in the environment used for MergeSlide.  If MIL-Lab exposes
`create_model` from a module name not covered by `mergeslide/feather_models.py`,
set:

```bash
export FEATHER_CREATE_MODEL="package.module:create_model"
```

For gated Hugging Face access, set `feather.hf_token` in `configs/feather.yaml`
or pass a token at runtime:

```bash
export HF_TOKEN="..."
```

`HF_TOKEN` takes precedence over the config value when both are set.

## Smoke Tests

Check feature bags:

```bash
python scripts/check_feather_features.py --max_slides 5
```

Probe FEATHER loading and forward behavior:

```bash
python scripts/probe_feather.py
```

## Training, Merge, And Evaluation

Train a quick 1-fold, 1-epoch run:

```bash
python scripts/train_feather.py --num_epochs 1 --num_folds 1 --k 64 --disable_wandb
```

Merge parameters are in `configs/merge_feather.yaml`, and evaluation parameters
are in `configs/eval_feather.yaml`. The task-specific classifier heads are
retained separately and are not merged:

```bash
python scripts/merge_feather.py --num_folds 1
```

Evaluate TASK-IL with the merged backbone and the oracle task head:

```bash
python scripts/eval_feather_taskil.py --num_folds 1
```

Evaluate naive CLASS-IL with one global classifier assembled by concatenating
the per-task classifier heads:

```bash
python scripts/eval_feather_classil.py --num_folds 1
```

FEATHER intentionally has no TCP task routing because TCP depends on TITAN's
text-prompt embeddings.

## Continual Baselines

FEATHER versions of DER++, A-GEM, ER-ACE, online EWC, and LwF reuse the
continual-learning loop while replacing the TITAN global classifier with
FEATHER ABMIL. They share `configs/feather.yaml` and use per-method settings
from `configs/feather_continual.yaml`.

```bash
python scripts/train_feather_continual.py --method derpp
python scripts/train_feather_continual.py --method agem
python scripts/train_feather_continual.py --method er_ace
python scripts/train_feather_continual.py --method ewc_on
python scripts/train_feather_continual.py --method lwf
```

For a short smoke run, add `--num_folds 1 --num_tasks 1 --num_epochs 1 --k 64`.

Evaluate TASK-IL for continual checkpoints with oracle task masking:

```bash
python scripts/eval_feather_taskil_masked_baselines.py --method derpp --final_only
python scripts/eval_feather_taskil_masked_baselines.py --method agem --final_only
python scripts/eval_feather_taskil_masked_baselines.py --method er_ace --final_only
python scripts/eval_feather_taskil_masked_baselines.py --method ewc_on --final_only
python scripts/eval_feather_taskil_masked_baselines.py --method lwf --final_only
```

Without `--final_only`, the evaluator writes the full continual matrix for
every available `after_task`.

The FEATHER LwF implementation follows Mammoth's classifier warm-up: from
task 2 onward, it runs an SGD-only warm-up of the new classifier rows for
`num_epochs` before the main LwF training loop.

Each continual run writes benchmark artifacts inside its method `save_dir`:
`run_manifest.json`, `eval_matrix.csv`, `per_slide_predictions.csv`,
`per_fold_summary.csv`, `per_task_summary.csv`, and one CSV confusion matrix
for each `(fold, after_task, eval_task)` result.

The FEATHER model must expose a final `nn.Linear` classifier with
`out_features=num_classes`. MergeSlide keeps that layer per task, merges every
other floating-point weight as the backbone, and concatenates the final-layer
weights for naive CLASS-IL.
