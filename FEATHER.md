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

For gated Hugging Face access, pass the token at runtime:

```bash
export HF_TOKEN="..."
```

Do not store HF tokens in config files.

## Smoke Tests

Check feature bags:

```bash
python scripts/check_feather_features.py --max_slides 5
```

Probe FEATHER loading and forward behavior:

```bash
python scripts/probe_feather.py
```

## Training And Evaluation

Train a quick 1-fold, 1-epoch run:

```bash
python scripts/train_feather.py --num_epochs 1 --num_folds 1 --k 64 --disable_wandb
```

Evaluate FEATHER TASK-IL checkpoints:

```bash
python scripts/eval_feather_taskil.py --num_folds 1
```

FEATHER CLASS-IL prompt routing and OPCM merge are intentionally not wired in
phase 1 because those paths depend on TITAN-specific text and vision encoder
interfaces.
