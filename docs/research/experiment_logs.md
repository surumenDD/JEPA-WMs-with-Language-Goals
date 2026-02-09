# Experiment Logs: Language-Conditioned Goal Planning

## Overview
This document records experimental results for language-conditioned goal planning using JEPA-WM.

---

## Experiment 1: Oracle Baseline (mw-reach)

### Date
2026-02-03

### Objective
Reproduce the Oracle baseline from JEPA-WM paper to verify the evaluation pipeline is working correctly.

### Configuration
- **Config file**: `configs/online_plan_evals/mw/oracle_reach_eval.yaml`
- **Task**: mw-reach (MetaWorld Reach)
- **Goal source**: expert (Oracle - goal image from expert rollout)
- **Planner**: CEM (Cross-Entropy Method)
  - Horizon: 6
  - Iterations: 15
  - Samples: 300
  - Elites: 10
  - num_act_stepped: 3
- **Episodes**: 20
- **Frameskip**: 5
- **Context window**: 2

### Model
- **Checkpoint**: `/home/ryo52/checkpoints/jepa_lang_goal/pretrained/mw_jepa-wm.pth.tar`
- **Encoder**: DinoV2 ViT-S/14 (384 dim, frozen)
- **Predictor**: VisionTransformerAdaLN (6 blocks, 384 dim)

### Environment Variables
```bash
export JEPAWM_DSET=/home/ryo52/datasets/jepa_lang_goal
export JEPAWM_CKPT=/home/ryo52/checkpoints/jepa_lang_goal
export MUJOCO_GL=egl
```

### Command
```bash
python -m evals.main \
  --fname configs/online_plan_evals/mw/oracle_reach_eval.yaml \
  --debug
```

### Results
| Metric | Value |
|--------|-------|
| **Success Rate** | **50.0%** |
| Episode Reward | 745.25 |
| Expert Success | 100% |
| Success Distance | 0.9 |
| End Distance | 0.0279 |
| Total Time | 54,697s (~15.2 hours) |
| Avg Time/Episode | ~2,735s (~45.6 min) |

### Episode Breakdown (from visualizations)
- **Success**: ep_0, ep_1, ep_2, ep_3, ep_5, ep_12, ep_13, ep_15, ep_18, ep_19 (10/20)
- **Fail**: ep_4, ep_6, ep_7, ep_8, ep_9, ep_10, ep_11, ep_14, ep_16, ep_17 (10/20)

### Comparison with Paper
| Method | Paper Result | Our Result | Difference |
|--------|-------------|------------|------------|
| JEPA-WM (Oracle, Reach) | 58.2% | 50.0% | -8.2% |

### Notes
- Result is slightly lower than paper (50% vs 58.2%)
- Possible reasons:
  1. Different random seeds
  2. Fewer episodes (20 vs paper's likely 48)
  3. Hardware differences affecting numerical precision
- The evaluation pipeline is confirmed working correctly
- CEM planning is very slow on CPU (~45 min/episode)

### Output Location
```
/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/online_eval/simu_env_planning/oracle_reach_eval/
├── eval.csv                    # Summary results
├── args_eval.yaml              # Full configuration used
└── mw-reach/
    └── ep_*/visualisation/     # Per-episode videos (mp4, gif, pdf)
```

---

## Experiment 2: Lang-Goal Evaluation (mw-reach) - Planned

### Objective
Evaluate language-conditioned goal planning where z_goal is predicted from GoalHead(initial_image, text_instruction) instead of being encoded from the expert goal image.

### Key Differences from Oracle
- **Oracle**: z_goal = Encoder(goal_image)
- **Lang-Goal**: z_goal = GoalHead(Encoder(initial_image), TextEncoder(instruction))

### GoalHead Model
- **Checkpoint**: `/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/goalhead/20260201_234248/best_model.pth`
- **Architecture**: 
  - Visual dim: 384
  - Text dim: 384
  - Hidden dim: 512
  - Layers: 3
  - Fusion: concat

### Plan
1. Start with small number of episodes (5-10) to verify implementation
2. Analyze results and potential improvements
3. If reasonable, run full evaluation (20+ episodes)

### Expected Challenges
- GoalHead prediction quality may be lower than direct goal encoding
- Text instruction mapping may not capture goal precisely
- Single z_goal from GoalHead vs Oracle's exact goal representation

---

## Notes on Reproducibility

### Hardware
- GPU: NVIDIA GeForce RTX 5070 Ti (17GB VRAM)
- CUDA available: Yes

### Software Versions
- Python: 3.10
- PyTorch: 2.10.0+cu128
- NumPy: 2.2.6 (Warning: incompatible with Gym, but works)

### Known Issues
1. **Gym/NumPy compatibility warning**: Gym not maintained since 2022, NumPy 2.x not officially supported
2. **libOpenGL.so.0**: Missing library warning (using EGL backend instead)
3. **Slow CEM**: ~45 min per episode on CPU-bound operations

### Data
- Dataset: `/home/ryo52/datasets/jepa_lang_goal/Metaworld/data/`
- Total rollouts: 12,600
