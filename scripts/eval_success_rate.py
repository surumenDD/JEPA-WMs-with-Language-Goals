#!/usr/bin/env python3
"""
Online Success Rate Evaluation: Oracle vs Language-Conditioned Goal

This script measures actual task success rates in MetaWorld by running:
1. Oracle planning: z_goal from goal image (standard baseline)
2. Lang-Goal planning: z_goal from GoalHead(I0, text)

Metrics:
- Task Success Rate: Percentage of episodes where task is completed
- Steps to Success: Average steps when successful
- Success Rate Degradation: Oracle vs Lang-Goal comparison
"""

import argparse
import json
import logging
import os
import sys

# Set MuJoCo to use EGL for headless rendering
os.environ['MUJOCO_GL'] = 'egl'

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.lang_goal.goal_head import GoalHead
from app.lang_goal.text_encoder import SentenceTransformerEncoder
from app.lang_goal.train_goalhead import load_pretrained_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class Config:
    """Configuration for online evaluation."""
    # Model checkpoints
    jepa_checkpoint: str = "/home/ryo52/checkpoints/jepa_lang_goal/pretrained/mw_jepa-wm.pth.tar"
    goalhead_checkpoint: str = "/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/goalhead/20260201_234248/best_model.pth"
    
    # GoalHead architecture
    visual_dim: int = 384
    text_dim: int = 384
    output_dim: int = 384
    hidden_dim: int = 512
    num_layers: int = 3
    fusion_mode: str = "concat"
    
    # Environment settings
    img_size: int = 224
    max_episode_steps: int = 100
    frameskip: int = 5
    action_skip: int = 1
    
    # Planner settings
    horizon: int = 6
    iterations: int = 15
    num_samples: int = 300
    num_elites: int = 10
    
    # Evaluation settings
    episodes_per_task: int = 20
    seed: int = 42
    device: str = "cuda:0"
    
    # Output
    output_dir: str = "/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/online_evaluation"


# Task instructions
TASK_INSTRUCTIONS = {
    "mw-reach": "Reach the target position with your gripper",
    "mw-push": "Push the puck to the target position",
    "mw-door-open": "Open the door",
    "mw-door-close": "Close the door",
    "mw-drawer-open": "Open the drawer",
    "mw-drawer-close": "Close the drawer",
    "mw-button-press": "Press the button on the box",
    "mw-window-open": "Open the window",
    "mw-window-close": "Close the window",
    "mw-faucet-open": "Open the faucet",
    "mw-faucet-close": "Close the faucet",
    "mw-handle-press": "Press the handle downward",
    "mw-handle-pull": "Pull the handle towards you",
    "mw-lever-pull": "Pull the lever towards you",
    "mw-coffee-button": "Press the coffee machine button",
    "mw-push-back": "Push the puck backwards",
    "mw-reach-wall": "Reach the target position over the wall",
    "mw-pick-place": "Pick up the puck and place it at the target",
    "mw-assembly": "Assemble the ring onto the peg",
    "mw-bin-picking": "Pick up the puck and place it in the bin",
}


# ============================================================================
# Model Loading
# ============================================================================

def get_model_kwargs(img_size: int = 224, frameskip: int = 5, action_skip: int = 1):
    """Get model kwargs for JEPA-WM."""
    return {
        "module_name": "app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds",
        "data": {
            "img_size": img_size,
            "custom": {
                "frameskip": frameskip,
                "action_skip": action_skip,
                "state_skip": 1,
            }
        },
        "data_aug": {
            "normalize": [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]],
        },
        "wrapper_kwargs": {
            "ctxt_window": 2,
        },
        "pretrain_kwargs": {
            "grid_size": 16,
            "tubelet_size_enc": 1,
            "use_activation_checkpointing": False,
            "action_conditioning": "token",
            "proprio_encoding": "feature",
            "num_frames_pred": 4,
            "visual_encoder": {
                "enc_type": "dino",
                "enc_version": "dinov2_vits14",
                "pretrain_enc_path": None,
                "pretrain_enc_ckpt_key": "target_encoder",
                "embed_dim": 384,
                "enc_use_rope": None,
                "enc_name": None,
                "use_sdpa_enc": True,
                "num_frames_enc": 16,
                "uniform_power": True,
            },
            "action_encoder": {
                "action_tokens": 1,
                "action_emb_dim": 0,
                "act_mlp": False,
                "action_encoder_inpred": True,
            },
            "proprio_encoder": {
                "proprio_tokens": 0,
                "proprio_emb_dim": 16,
                "prop_mlp": False,
                "proprio_encoder_inpred": False,
            },
            "predictor": {
                "tubelet_size": 1,
                "pred_num_heads": 16,
                "pred_depth": 6,
                "pred_embed_dim": 384,
                "pred_use_extrinsics": False,
                "pred_type": "AdaLN",
                "act_pred_projector": False,
                "use_SiLU": False,
                "use_rope": True,
            },
            "wm_encoding": {
                "batchify_video": True,
                "dup_image": False,
                "normalize_reps": False,
            },
            "attn": {
                "local_window_time": 3,
                "local_window_h": -1,
                "local_window_w": -1,
            },
            "heads_cfg": {
                "architectures": {},
                "pretrain_dec_path": {},
            },
        },
    }


def load_models(config: Config, device: torch.device):
    """Load all required models."""
    logger.info("=" * 60)
    logger.info("Loading models...")
    logger.info("=" * 60)
    
    model_kwargs = get_model_kwargs(config.img_size, config.frameskip, config.action_skip)
    
    # Load JEPA-WM
    logger.info(f"📦 Loading JEPA-WM from {config.jepa_checkpoint}")
    encoder = load_pretrained_model(
        checkpoint_path=config.jepa_checkpoint,
        device=device,
        model_kwargs=model_kwargs,
    )
    logger.info("✅ JEPA-WM loaded")
    
    # Load GoalHead
    logger.info(f"📦 Loading GoalHead from {config.goalhead_checkpoint}")
    goalhead = GoalHead(
        visual_dim=config.visual_dim,
        text_dim=config.text_dim,
        output_dim=config.output_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        fusion_mode=config.fusion_mode,
    )
    checkpoint = torch.load(config.goalhead_checkpoint, map_location="cpu")
    state_dict_key = "goal_head_state_dict" if "goal_head_state_dict" in checkpoint else "model_state_dict"
    goalhead.load_state_dict(checkpoint[state_dict_key])
    goalhead.to(device)
    goalhead.eval()
    logger.info(f"✅ GoalHead loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    
    # Load Text Encoder
    logger.info("📦 Loading Text Encoder")
    text_encoder = SentenceTransformerEncoder(
        model_name="all-MiniLM-L6-v2",
        device=str(device),
    )
    logger.info("✅ Text Encoder loaded")
    
    return encoder, goalhead, text_encoder


# ============================================================================
# Environment Setup
# ============================================================================

def make_metaworld_env(task_name: str, config: Config, seed: int = None):
    """Create MetaWorld environment."""
    from metaworld.env_dict import ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE
    from evals.simu_env_planning.envs.wrappers.time_limit import TimeLimit
    
    # Convert task name to env_id
    env_id = task_name.split("-", 1)[-1] + "-v3" + "-goal-observable"
    
    if env_id not in ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE:
        raise ValueError(f"Unknown task: {task_name}, env_id: {env_id}")
    
    env = ALL_V3_ENVIRONMENTS_GOAL_OBSERVABLE[env_id](seed=seed or config.seed)
    env.seeded_rand_vec = False
    
    # Create minimal cfg for wrapper
    cfg = OmegaConf.create({
        "task_specification": {
            "img_size": config.img_size,
            "max_episode_steps": config.max_episode_steps,
            "env": {
                "freeze_rand_vec": False,
            }
        },
        "logging": {
            "tqdm_silent": True,
        }
    })
    
    from evals.simu_env_planning.envs.metaworld import MetaWorldWrapper
    env = MetaWorldWrapper(env, cfg)
    env = TimeLimit(env, max_episode_steps=config.max_episode_steps)
    
    return env


def get_expert_policy(task_name: str):
    """Get expert policy for a task."""
    from metaworld import policies
    
    special_cases = {
        "mw-peg-insert-side": "SawyerPegInsertionSideV3Policy",
    }
    
    if task_name in special_cases:
        policy_name = special_cases[task_name]
    else:
        parts = task_name.split("-")
        policy_name = "Sawyer" + "".join([word.capitalize() for word in parts[1:]]) + "V3Policy"
    
    return getattr(policies, policy_name)()


# ============================================================================
# Image Processing
# ============================================================================

def preprocess_image(img: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    """Preprocess image for encoder."""
    import torchvision.transforms as T
    
    transform = T.Compose([
        T.ToPILImage(),
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # img is [H, W, 3], uint8
    return transform(img).unsqueeze(0).to(device)


# ============================================================================
# Evaluation Episodes
# ============================================================================

@torch.no_grad()
def run_episode_oracle(
    env,
    encoder,
    expert_policy,
    planner,  # Added: CEM planner
    config: Config,
    device: torch.device,
) -> Tuple[bool, int, float, torch.Tensor]:
    """
    Run episode with oracle goal from expert demonstration using CEM planning.
    
    This gets the goal state by running expert policy, then uses CEM planner
    (NOT expert policy) for actual action selection.
    
    Returns:
        (success, steps, reward, z_goal_oracle)
    """
    # Reset environment
    obs, info = env.reset()
    
    # Capture initial image
    init_img = env.render()
    init_tensor = preprocess_image(init_img, config.img_size, device)
    
    # Run expert policy to get goal image (NOT for action selection!)
    temp_obs, temp_info = env.reset()  # Reset again
    expert_done = False
    expert_steps = 0
    
    while not expert_done and expert_steps < config.max_episode_steps:
        action = expert_policy.get_action(temp_info["state"])
        temp_obs, reward, trunc, done, temp_info = env.step(action)
        expert_steps += 1
        expert_done = done or trunc
        if temp_info.get("success", False):
            break
    
    # Capture goal image after expert execution
    goal_img = env.render()
    goal_tensor = preprocess_image(goal_img, config.img_size, device)
    
    # Encode goal image to get z_goal_oracle
    goal_obs_dict = {"visual": goal_tensor.unsqueeze(1), "proprio": None}
    z_goal_dict = encoder.encode_obs(goal_obs_dict)
    z_goal = z_goal_dict["visual"]
    if z_goal.dim() == 6:
        z_goal_pooled = z_goal.mean(dim=(1, 2, 3, 4))
    else:
        z_goal_pooled = z_goal.mean(dim=1) if z_goal.dim() > 2 else z_goal
    
    # Now reset environment and run CEM planning (NOT expert!)
    obs, info = env.reset()
    init_img = env.render()
    
    # Set goal for planner
    planner.set_goal_from_embedding(z_goal)
    
    # Run CEM planning loop
    done = False
    success = False
    total_reward = 0.0
    steps = 0
    current_img = init_img
    
    while not done and steps < config.max_episode_steps:
        # Preprocess current observation
        current_tensor = preprocess_image(current_img, config.img_size, device)
        current_obs_dict = {"visual": current_tensor.unsqueeze(1), "proprio": None}
        z_current = encoder.encode_obs(current_obs_dict)["visual"]
        
        # Plan action using CEM
        steps_left = max((config.max_episode_steps - steps) // config.frameskip, 1)
        action = planner.act(z_current, steps_left=steps_left)
        
        # Execute action
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if action.ndim > 1:
            action = action.squeeze(0)
        
        obs, reward, trunc, terminated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or trunc
        
        # Update current image
        current_img = env.render()
        
        if info.get("success", False):
            success = True
            break
    
    return success, steps, total_reward, z_goal_pooled


@torch.no_grad()
def run_episode_langgoal(
    env,
    encoder,
    goalhead: GoalHead,
    text_encoder,
    instruction: str,
    planner,  # Changed: CEM planner instead of expert policy
    config: Config,
    device: torch.device,
) -> Tuple[bool, int, float, torch.Tensor]:
    """
    Run episode with language-conditioned goal using CEM planning.
    
    This is the correct implementation that uses CEM planner for action selection,
    NOT expert policy. The goal latent z_goal is predicted by GoalHead from the
    initial image and language instruction.
    
    Returns:
        (success, steps, reward, z_goal_pred)
    """
    # Reset environment
    obs, info = env.reset()
    
    # Get initial image
    init_img = env.render()
    init_tensor = preprocess_image(init_img, config.img_size, device)
    
    # Encode initial image
    init_obs_dict = {"visual": init_tensor.unsqueeze(1), "proprio": None}
    z_init_dict = encoder.encode_obs(init_obs_dict)
    z_init = z_init_dict["visual"]
    
    # Encode instruction
    text_features = text_encoder.encode([instruction])
    
    # Predict goal with GoalHead
    z_goal_pred = goalhead(z_init, text_features)
    
    # Set goal for planner (expand z_goal_pred to match expected shape)
    # GoalHead outputs [B, D], need to expand to [B, T, V, H, W, D] or similar
    z_goal_expanded = expand_goal_embedding(z_goal_pred, z_init)
    planner.set_goal_from_embedding(z_goal_expanded)
    
    # Run CEM planning loop (NOT expert policy!)
    done = False
    success = False
    total_reward = 0.0
    steps = 0
    
    # Current observation for planning
    current_img = init_img
    
    while not done and steps < config.max_episode_steps:
        # Preprocess current observation
        current_tensor = preprocess_image(current_img, config.img_size, device)
        current_obs_dict = {"visual": current_tensor.unsqueeze(1), "proprio": None}
        z_current = encoder.encode_obs(current_obs_dict)["visual"]
        
        # Plan action using CEM
        steps_left = max((config.max_episode_steps - steps) // config.frameskip, 1)
        action = planner.act(z_current, steps_left=steps_left)
        
        # Execute action in environment
        if isinstance(action, torch.Tensor):
            action = action.cpu().numpy()
        if action.ndim > 1:
            action = action.squeeze(0)
        
        obs, reward, trunc, terminated, info = env.step(action)
        total_reward += reward
        steps += 1
        done = terminated or trunc
        
        # Update current image
        current_img = env.render()
        
        if info.get("success", False):
            success = True
            break
    
    return success, steps, total_reward, z_goal_pred


def compute_goal_similarity(z_pred: torch.Tensor, z_oracle: torch.Tensor) -> float:
    """Compute cosine similarity between predicted and oracle goals."""
    z_pred_flat = z_pred.flatten()
    z_oracle_flat = z_oracle.flatten()
    cos_sim = F.cosine_similarity(z_pred_flat.unsqueeze(0), z_oracle_flat.unsqueeze(0))
    return cos_sim.item()


# ============================================================================
# Main Evaluation
# ============================================================================

def run_evaluation(config: Config, tasks: List[str]):
    """Run full online evaluation."""
    device = torch.device(config.device if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Starting online success rate evaluation on {device}")
    
    # Load models
    encoder, goalhead, text_encoder = load_models(config, device)
    
    results = {
        "tasks": {},
        "aggregate": {},
        "config": {
            "episodes_per_task": config.episodes_per_task,
            "max_episode_steps": config.max_episode_steps,
            "seed": config.seed,
        }
    }
    
    all_oracle_successes = []
    all_langgoal_successes = []
    all_oracle_steps = []
    all_langgoal_steps = []
    all_cosine_sims = []
    
    for task_name in tqdm(tasks, desc="Evaluating tasks"):
        logger.info(f"\n{'='*60}")
        logger.info(f"📋 Evaluating: {task_name}")
        logger.info(f"{'='*60}")
        
        instruction = TASK_INSTRUCTIONS.get(task_name, f"Complete the {task_name} task")
        
        oracle_successes = []
        langgoal_successes = []
        oracle_steps = []
        langgoal_steps = []
        cosine_sims = []
        
        try:
            expert_policy = get_expert_policy(task_name)
        except Exception as e:
            logger.warning(f"⚠️ Failed to get expert policy for {task_name}: {e}")
            continue
        
        for ep in tqdm(range(config.episodes_per_task), desc=f"  {task_name}", leave=False):
            seed = config.seed + ep * 100
            
            try:
                # Oracle evaluation
                env_oracle = make_metaworld_env(task_name, config, seed)
                oracle_success, oracle_step, _, z_oracle = run_episode_oracle(
                    env_oracle, encoder, expert_policy, config, device
                )
                oracle_successes.append(int(oracle_success))
                if oracle_success:
                    oracle_steps.append(oracle_step)
                env_oracle.close()
                
                # Lang-Goal evaluation (same seed for fair comparison)
                env_lang = make_metaworld_env(task_name, config, seed)
                lang_success, lang_step, _, z_pred = run_episode_langgoal(
                    env_lang, encoder, goalhead, text_encoder,
                    instruction, expert_policy, config, device
                )
                langgoal_successes.append(int(lang_success))
                if lang_success:
                    langgoal_steps.append(lang_step)
                env_lang.close()
                
                # Compute goal similarity
                cos_sim = compute_goal_similarity(z_pred, z_oracle)
                cosine_sims.append(cos_sim)
                
            except Exception as e:
                logger.warning(f"⚠️ Episode {ep} failed: {e}")
                continue
        
        # Task results
        if oracle_successes:
            oracle_sr = np.mean(oracle_successes)
            langgoal_sr = np.mean(langgoal_successes)
            
            results["tasks"][task_name] = {
                "oracle_success_rate": float(oracle_sr),
                "langgoal_success_rate": float(langgoal_sr),
                "success_rate_degradation": float(oracle_sr - langgoal_sr),
                "oracle_avg_steps": float(np.mean(oracle_steps)) if oracle_steps else None,
                "langgoal_avg_steps": float(np.mean(langgoal_steps)) if langgoal_steps else None,
                "goal_cosine_similarity": float(np.mean(cosine_sims)) if cosine_sims else None,
                "n_episodes": len(oracle_successes),
            }
            
            logger.info(f"  Oracle SR: {oracle_sr:.1%} ({sum(oracle_successes)}/{len(oracle_successes)})")
            logger.info(f"  Lang-Goal SR: {langgoal_sr:.1%} ({sum(langgoal_successes)}/{len(langgoal_successes)})")
            logger.info(f"  Degradation: {oracle_sr - langgoal_sr:.1%}")
            logger.info(f"  Goal Cosine Sim: {np.mean(cosine_sims):.4f}")
            
            all_oracle_successes.extend(oracle_successes)
            all_langgoal_successes.extend(langgoal_successes)
            all_oracle_steps.extend(oracle_steps)
            all_langgoal_steps.extend(langgoal_steps)
            all_cosine_sims.extend(cosine_sims)
    
    # Aggregate results
    if all_oracle_successes:
        agg_oracle_sr = np.mean(all_oracle_successes)
        agg_langgoal_sr = np.mean(all_langgoal_successes)
        
        results["aggregate"] = {
            "oracle_success_rate": float(agg_oracle_sr),
            "langgoal_success_rate": float(agg_langgoal_sr),
            "success_rate_degradation": float(agg_oracle_sr - agg_langgoal_sr),
            "success_rate_degradation_percent": float((agg_oracle_sr - agg_langgoal_sr) / agg_oracle_sr * 100) if agg_oracle_sr > 0 else 0,
            "oracle_avg_steps": float(np.mean(all_oracle_steps)) if all_oracle_steps else None,
            "langgoal_avg_steps": float(np.mean(all_langgoal_steps)) if all_langgoal_steps else None,
            "avg_goal_cosine_similarity": float(np.mean(all_cosine_sims)) if all_cosine_sims else None,
            "total_episodes": len(all_oracle_successes),
            "total_tasks": len(results["tasks"]),
        }
    
    return results


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Online Success Rate Evaluation")
    parser.add_argument("--tasks", type=str, nargs="+", 
                       default=["mw-reach", "mw-push", "mw-door-open", "mw-drawer-close"],
                       help="Tasks to evaluate")
    parser.add_argument("--episodes", type=int, default=20, help="Episodes per task")
    parser.add_argument("--output-dir", type=str,
                       default="/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/online_evaluation",
                       help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    config = Config(
        episodes_per_task=args.episodes,
        output_dir=args.output_dir,
        device=args.device,
        seed=args.seed,
    )
    
    # Run evaluation
    results = run_evaluation(config, args.tasks)
    
    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_path = os.path.join(config.output_dir, f"success_rate_results_{timestamp}.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 ONLINE SUCCESS RATE EVALUATION SUMMARY")
    logger.info("=" * 70)
    
    if "aggregate" in results and results["aggregate"]:
        agg = results["aggregate"]
        logger.info(f"\n🎯 AGGREGATE RESULTS ({agg.get('total_tasks', 0)} tasks, {agg.get('total_episodes', 0)} episodes)")
        logger.info(f"   Oracle Success Rate:      {agg['oracle_success_rate']:.1%}")
        logger.info(f"   Lang-Goal Success Rate:   {agg['langgoal_success_rate']:.1%}")
        logger.info(f"   Success Rate Degradation: {agg['success_rate_degradation']:.1%} ({agg.get('success_rate_degradation_percent', 0):.1f}% relative)")
        if agg.get('oracle_avg_steps'):
            logger.info(f"   Oracle Avg Steps:         {agg['oracle_avg_steps']:.1f}")
        if agg.get('langgoal_avg_steps'):
            logger.info(f"   Lang-Goal Avg Steps:      {agg['langgoal_avg_steps']:.1f}")
        if agg.get('avg_goal_cosine_similarity'):
            logger.info(f"   Avg Goal Cosine Sim:      {agg['avg_goal_cosine_similarity']:.4f}")
    
    logger.info(f"\n📁 Results saved to: {output_path}")
    
    return results


if __name__ == "__main__":
    main()
