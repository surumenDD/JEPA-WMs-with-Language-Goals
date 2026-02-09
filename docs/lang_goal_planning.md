# Language-Conditioned Goal Planning for JEPA-WM

This document describes the implementation for language-conditioned latent planning
using JEPA-WM on MetaWorld tasks.

## Overview

This implementation enables goal-conditioned planning in latent space using
**language instructions** instead of goal images. The key components are:

1. **GoalHead**: Predicts goal latent `z_goal` from initial image `I0` and text instruction `T`
2. **Language Goal Agent**: Uses GoalHead predictions for MPC planning
3. **Evaluation Framework**: Compares language goals vs. image goals (baseline)

## Architecture

```
Language-Conditioned Planning Pipeline:

                    ┌─────────────────┐
    Initial Image   │  Visual Encoder │
        I0  ───────►│    (DINOv2)     │───────┐
                    │   [Frozen]      │       │
                    └─────────────────┘       │
                                              ▼
                    ┌─────────────────┐   ┌──────────┐
    Text Instruction│  Text Encoder   │   │ GoalHead │──► z_goal_pred
        T  ────────►│ (SentenceT)     │──►│  (MLP)   │
                    │   [Frozen]      │   │[Trained] │
                    └─────────────────┘   └──────────┘
                                              │
                                              ▼
                    ┌─────────────────────────────────┐
    Current Obs ───►│     CEM Planner (MPC)           │──► Action
                    │  Objective: ||z_t+H - z_goal||  │
                    │  World Model: JEPA-WM [Frozen]  │
                    └─────────────────────────────────┘
```

## Directory Structure

```
jepa-wms/
├── app/lang_goal/
│   ├── __init__.py
│   ├── data_index.py        # Data indexing and text generation
│   ├── env_utils.py         # Environment variable utilities
│   ├── goal_head.py         # GoalHead model definition
│   ├── goalhead_dataset.py  # Dataset for GoalHead training
│   ├── text_encoder.py      # Text encoder wrappers
│   └── train_goalhead.py    # GoalHead training script
├── evals/lang_goal/
│   ├── __init__.py
│   ├── eval.py              # Main evaluation script
│   └── lang_goal_agent.py   # Language-conditioned agent
├── configs/lang_goal/
│   ├── data_index.yaml      # Data indexing config
│   ├── goalhead_train.yaml  # GoalHead training config
│   └── mw_lang_goal_eval.yaml  # Evaluation config
├── assets/lang_goal/
│   └── task_to_text.yaml    # Task name to instruction mapping
└── scripts/
    └── download_weights.sh  # Pretrained weights download script
```

## Setup

### 1. Environment Variables

Set the following environment variables:

```bash
export JEPAWM_HOME=$HOME/workspace/jepa_lang_goal
export JEPAWM_DSET=$HOME/datasets/jepa_lang_goal
export JEPAWM_LOGS=$HOME/logs/jepa_lang_goal
export JEPAWM_CKPT=$HOME/checkpoints/jepa_lang_goal
export JEPAWM_OSSCKPT=$HOME/oss_checkpoints/jepa_lang_goal
```

### 2. Download Pretrained Weights

```bash
cd $JEPAWM_HOME/jepa-wms
chmod +x scripts/download_weights.sh
./scripts/download_weights.sh
```

Or manually:

```bash
mkdir -p "$JEPAWM_CKPT/pretrained"
cd "$JEPAWM_CKPT/pretrained"
curl -L -O https://dl.fbaipublicfiles.com/jepa-wms/mw_jepa-wm.pth.tar
```

### 3. Prepare Data

Ensure MetaWorld data is available at:
```
$JEPAWM_DSET/Metaworld/data/*.parquet
```

## Running Experiments

### Step 1: Create Data Index

Generate the train/val/test splits and language instructions:

```bash
cd $JEPAWM_HOME/jepa-wms
conda activate jepa-wms

python -m app.lang_goal.data_index \
    --config configs/lang_goal/data_index.yaml
```

This creates:
- `$JEPAWM_LOGS/lang_goal/index/train_index.jsonl`
- `$JEPAWM_LOGS/lang_goal/index/val_index.jsonl`
- `$JEPAWM_LOGS/lang_goal/index/test_index.jsonl`
- `$JEPAWM_LOGS/lang_goal/index/full_index.csv`
- `$JEPAWM_LOGS/lang_goal/index/metadata.json`

### Step 2: Train GoalHead

Train the GoalHead to predict goal latents from language:

```bash
python -m app.lang_goal.train_goalhead \
    --config configs/lang_goal/goalhead_train.yaml
```

Training outputs:
- `$JEPAWM_CKPT/lang_goal/goalhead/<run_id>/best_model.pth`
- `$JEPAWM_CKPT/lang_goal/goalhead/<run_id>/training_log.csv`
- `$JEPAWM_CKPT/lang_goal/goalhead/<run_id>/config.yaml`

Monitor with wandb (if enabled).

### Step 3: Run Evaluation

Evaluate language-conditioned planning:

```bash
# Update goalhead_checkpoint in config first, then:
python -m evals.main \
    --fname configs/lang_goal/mw_lang_goal_eval.yaml \
    --debug
```

For full distributed evaluation:
```bash
python -m evals.main \
    --fname configs/lang_goal/mw_lang_goal_eval.yaml
```

## Evaluation Outputs

### Result Files

- `results.json`: Full results with all metrics
- `lang_goal_episodes.csv`: Per-episode results for language goal
- `baseline_episodes.csv`: Per-episode results for baseline
- `task_summary.csv`: Per-task summary comparing both methods

### Metrics

1. **Offline Metrics** (GoalHead quality):
   - L2 distance: `||z_goal_pred - z_goal_gt||_2`
   - Cosine similarity: `cos(z_goal_pred, z_goal_gt)`

2. **Online Metrics** (Planning performance):
   - Success rate (per-task and overall)
   - Steps to success (for successful episodes)
   - Total reward

3. **Degradation Metrics** (Comparison):
   - Success rate difference: `success(lang_goal) - success(baseline)`
   - Steps difference: `steps(lang_goal) - steps(baseline)`

## Configuration Reference

### GoalHead Training (`goalhead_train.yaml`)

Key parameters:
- `fusion_mode`: "concat" | "cross_attn" | "film"
- `num_layers`: Number of MLP layers (default: 3)
- `hidden_dim`: Hidden dimension (default: 512)
- `l2_weight`: Weight for L2 loss (default: 1.0)
- `cosine_weight`: Weight for cosine loss (default: 0.1)

### Evaluation (`mw_lang_goal_eval.yaml`)

Key parameters:
- `goalhead_checkpoint`: Path to trained GoalHead
- `run_baseline`: Whether to run baseline comparison
- `planner.horizon`: Planning horizon H (default: 6)
- `planner.num_samples`: CEM samples N (default: 300)
- `planner.num_elites`: CEM elites K (default: 10)
- `planner.iterations`: CEM iterations (default: 15)

## Task-to-Text Mapping

The file `assets/lang_goal/task_to_text.yaml` maps MetaWorld task names
to natural language instructions. Examples:

```yaml
reach: "Reach to the target position"
drawer-open: "Open the drawer by pulling it"
button-press: "Press the button"
```

For unknown tasks, the system generates instructions from task names
using pattern matching.

## Extending the Implementation

### Adding New Text Encoders

1. Create a wrapper class in `app/lang_goal/text_encoder.py`
2. Implement `encode()` and `forward()` methods
3. Register in `build_text_encoder()` function

### Adding New Fusion Methods

1. Add fusion layer in `app/lang_goal/goal_head.py`
2. Update `GoalHead.__init__()` to handle new fusion mode
3. Update `GoalHead.forward()` with fusion logic

### Adding New Tasks

1. Add task-to-text mapping in `assets/lang_goal/task_to_text.yaml`
2. Update environment config in evaluation YAML

## Troubleshooting

### Common Issues

1. **"Environment variable not set"**:
   Set all required environment variables (see Setup section)

2. **"Pretrained checkpoint not found"**:
   Run `scripts/download_weights.sh` to download weights

3. **"Index file not found"**:
   Run data indexing step first

4. **Out of memory**:
   Reduce batch size or use `--debug` mode

### Debug Mode

Add `--debug` flag for single-GPU execution with reduced data:

```bash
python -m app.lang_goal.train_goalhead --config ... --debug
python -m evals.main --fname ... --debug
```

## References

- JEPA-WM Paper: [arXiv:2512.24497](https://arxiv.org/abs/2512.24497)
- MetaWorld Benchmark: [Meta-World Paper](https://zhanpenghe.github.io/publications/meta_world.pdf)
- JEPA-WMs Repository: [GitHub](https://github.com/facebookresearch/jepa-wms)
