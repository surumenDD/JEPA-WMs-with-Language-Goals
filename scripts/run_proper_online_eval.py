#!/usr/bin/env python3
"""
Proper Online Evaluation for Language-Conditioned Goal Planning

This script runs the CORRECT evaluation using:
1. Oracle: CEM planner with z_goal from goal image (NOT expert policy for actions!)
2. Lang-Goal: CEM planner with z_goal from GoalHead prediction

Reference paper results for mw-reach / mw-reach-wall:
- JEPA-WM:  Reach 58.2%, Reach-Wall 41.6%
- DINO-WM:  Reach 44.8%, Reach-Wall 35.1%

Usage:
    # Run oracle baseline evaluation (standard JEPA-WM paper setup)
    python scripts/run_proper_online_eval.py --task mw-reach --mode oracle --episodes 48
    
    # Run lang-goal evaluation with GoalHead
    python scripts/run_proper_online_eval.py --task mw-reach --mode langgoal --episodes 48
    
    # Compare both
    python scripts/run_proper_online_eval.py --task mw-reach --mode both --episodes 48
"""

import argparse
import importlib
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import torch
import torch.distributed as dist
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.gc_agent import GC_Agent
from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
from evals.simu_env_planning.planning.utils import set_seed, make_td
from evals.utils import make_datasets, prepare_obs
from src.utils.distributed import init_distributed
from src.utils.yaml_utils import expand_env_vars

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def init_module(folder, checkpoint, module_name, model_kwargs, device, cfgs_data=None,
                wrapper_kwargs=None, action_dim=None, proprio_dim=None, preprocessor=None):
    """Build model and load checkpoint."""
    model = importlib.import_module(f"{module_name}").init_module(
        folder=folder,
        checkpoint=checkpoint,
        model_kwargs=model_kwargs,
        device=device,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        preprocessor=preprocessor,
        cfgs_data=cfgs_data,
        wrapper_kwargs=wrapper_kwargs,
    )
    return model


def load_config(config_path: str) -> dict:
    """Load and expand config file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg = expand_env_vars(cfg)
    return cfg


def run_oracle_evaluation(cfg: OmegaConf, model, agent: GC_Agent, dset, preprocessor, device):
    """
    Run Oracle evaluation using CEM planner with goal image.
    
    This is the CORRECT baseline: expert policy is used ONLY to get goal image,
    then CEM planner is used for action selection.
    """
    logger.info("=" * 60)
    logger.info("🎯 Running ORACLE evaluation (CEM planning with goal image)")
    logger.info("=" * 60)
    
    evaluator = PlanEvaluator(cfg, agent)
    env = make_env(cfg)
    
    results = {
        "successes": [],
        "rewards": [],
        "steps": [],
        "end_distances": [],
    }
    
    for ep in tqdm(range(cfg.meta.eval_episodes), desc="Oracle episodes"):
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        
        # Reset environment
        init_obs, info = env.reset(seed=ep_seed, task_idx=0)
        
        # Set episode (this runs expert to get goal image, but NOT for action selection)
        init_obs, goal_obs, expert_obses, expert_success = evaluator.set_episode(
            cfg, agent, env, ep_seed, task_idx=0
        )
        
        # Set goal from goal image (standard oracle approach)
        agent.set_goal(prepare_obs(cfg.task_specification.obs, goal_obs))
        
        # Run agent with CEM planning (NOT expert policy!)
        episode_obses, ep_reward, planned_actions, infos, success, state_dist = evaluator.unroll_agent(
            env, init_obs, info, 
            lambda obs, steps_left: agent.act(prepare_obs(cfg.task_specification.obs, obs), steps_left=steps_left),
            preprocessor=agent.preprocessor
        )
        
        results["successes"].append(int(success))
        results["rewards"].append(ep_reward)
        if success:
            results["steps"].append(len(planned_actions))
        results["end_distances"].append(state_dist)
    
    success_rate = np.mean(results["successes"])
    logger.info(f"✅ Oracle Success Rate: {success_rate:.1%} ({sum(results['successes'])}/{len(results['successes'])})")
    
    return {
        "success_rate": float(success_rate),
        "avg_reward": float(np.mean(results["rewards"])),
        "avg_steps_on_success": float(np.mean(results["steps"])) if results["steps"] else None,
        "avg_end_distance": float(np.mean(results["end_distances"])),
        "n_episodes": len(results["successes"]),
        "raw_results": results,
    }


def run_langgoal_evaluation(cfg: OmegaConf, model, agent: GC_Agent, goalhead, text_encoder, 
                             dset, preprocessor, device, task_instruction: str):
    """
    Run Lang-Goal evaluation using CEM planner with GoalHead prediction.
    
    This predicts z_goal from GoalHead(I0, text), then uses CEM planner for action selection.
    """
    logger.info("=" * 60)
    logger.info("🗣️ Running LANG-GOAL evaluation (CEM planning with GoalHead)")
    logger.info(f"   Instruction: '{task_instruction}'")
    logger.info("=" * 60)
    
    evaluator = PlanEvaluator(cfg, agent)
    env = make_env(cfg)
    
    results = {
        "successes": [],
        "rewards": [],
        "steps": [],
        "end_distances": [],
        "goal_cosine_sims": [],
    }
    
    for ep in tqdm(range(cfg.meta.eval_episodes), desc="Lang-Goal episodes"):
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        
        # Reset environment
        init_obs, info = env.reset(seed=ep_seed, task_idx=0)
        init_obs = make_td(init_obs, info)
        
        # Get oracle goal for comparison (expert trajectory to get goal image)
        _, goal_obs, _, _ = evaluator.set_episode(cfg, agent, env, ep_seed, task_idx=0)
        
        # Encode oracle goal for comparison
        z_goal_oracle = model.encode(
            prepare_obs(cfg.task_specification.obs, goal_obs).unsqueeze(0).to(device), 
            act=False
        ).detach()
        
        # Reset again for lang-goal evaluation
        init_obs, info = env.reset(seed=ep_seed, task_idx=0)
        init_obs = make_td(init_obs, info)
        
        # Encode initial observation
        init_visual = prepare_obs(cfg.task_specification.obs, init_obs).unsqueeze(0).to(device)
        z_init = model.encode(init_visual, act=False).detach()
        
        # Get text features
        text_features = text_encoder.encode([task_instruction])
        
        # Predict z_goal with GoalHead
        with torch.no_grad():
            # Pool z_init to match GoalHead input
            if z_init.dim() == 6:  # [B, T, V, H, W, D]
                z_init_pooled = z_init.mean(dim=(1, 2, 3, 4))
            elif z_init.dim() == 5:  # [B, T, N, H, D] 
                z_init_pooled = z_init.mean(dim=(1, 2, 3))
            else:
                z_init_pooled = z_init.mean(dim=1) if z_init.dim() > 2 else z_init
            
            z_goal_pred = goalhead(z_init_pooled, text_features)
        
        # Compute goal similarity for analysis
        z_oracle_pooled = z_goal_oracle.mean(dim=(1, 2, 3, 4)) if z_goal_oracle.dim() == 6 else z_goal_oracle.flatten(1)
        cos_sim = torch.nn.functional.cosine_similarity(
            z_goal_pred.flatten(1), z_oracle_pooled.flatten(1)
        ).item()
        results["goal_cosine_sims"].append(cos_sim)
        
        # Set goal from GoalHead prediction (using embedding directly)
        # Expand z_goal_pred to match expected shape for planning objective
        z_goal_expanded = expand_goal_for_planning(z_goal_pred, z_goal_oracle.shape)
        agent.set_goal_from_embedding(z_goal_expanded)
        
        # Run agent with CEM planning
        episode_obses, ep_reward, planned_actions, infos, success, state_dist = evaluator.unroll_agent(
            env, init_obs, info,
            lambda obs, steps_left: agent.act(prepare_obs(cfg.task_specification.obs, obs), steps_left=steps_left),
            preprocessor=agent.preprocessor
        )
        
        results["successes"].append(int(success))
        results["rewards"].append(ep_reward)
        if success:
            results["steps"].append(len(planned_actions))
        results["end_distances"].append(state_dist)
    
    success_rate = np.mean(results["successes"])
    logger.info(f"✅ Lang-Goal Success Rate: {success_rate:.1%} ({sum(results['successes'])}/{len(results['successes'])})")
    logger.info(f"   Avg Goal Cosine Sim: {np.mean(results['goal_cosine_sims']):.4f}")
    
    return {
        "success_rate": float(success_rate),
        "avg_reward": float(np.mean(results["rewards"])),
        "avg_steps_on_success": float(np.mean(results["steps"])) if results["steps"] else None,
        "avg_end_distance": float(np.mean(results["end_distances"])),
        "avg_goal_cosine_sim": float(np.mean(results["goal_cosine_sims"])),
        "n_episodes": len(results["successes"]),
        "raw_results": results,
    }


def expand_goal_for_planning(z_goal_pred: torch.Tensor, target_shape: tuple) -> torch.Tensor:
    """
    Expand GoalHead output to match planning objective expected shape.
    
    GoalHead outputs [B, D], but planning expects [B, T, V, H, W, D] or similar.
    """
    # z_goal_pred: [B, D]
    # target_shape: e.g., [B, T, V, H, W, D]
    
    if len(target_shape) == 6:
        # [B, T, V, H, W, D]
        B, T, V, H, W, D = target_shape
        z_expanded = z_goal_pred.view(B, 1, 1, 1, 1, -1)
        z_expanded = z_expanded.expand(B, T, V, H, W, -1)
    elif len(target_shape) == 5:
        # [B, T, N, H, D]
        B, T, N, H, D = target_shape
        z_expanded = z_goal_pred.view(B, 1, 1, 1, -1)
        z_expanded = z_expanded.expand(B, T, N, H, -1)
    elif len(target_shape) == 4:
        # [B, T, N, D]
        B, T, N, D = target_shape
        z_expanded = z_goal_pred.view(B, 1, 1, -1)
        z_expanded = z_expanded.expand(B, T, N, -1)
    else:
        z_expanded = z_goal_pred
    
    return z_expanded


def load_goalhead_and_text_encoder(config, device):
    """Load GoalHead and text encoder."""
    from app.lang_goal.goal_head import GoalHead
    from app.lang_goal.text_encoder import SentenceTransformerEncoder
    
    # GoalHead
    goalhead_ckpt = config.get("goalhead_checkpoint", 
        "/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/goalhead/20260201_234248/best_model.pth")
    
    goalhead = GoalHead(
        visual_dim=384,
        text_dim=384,
        output_dim=384,
        hidden_dim=512,
        num_layers=3,
        fusion_mode="concat",
    )
    
    checkpoint = torch.load(goalhead_ckpt, map_location="cpu")
    state_dict_key = "goal_head_state_dict" if "goal_head_state_dict" in checkpoint else "model_state_dict"
    goalhead.load_state_dict(checkpoint[state_dict_key])
    goalhead.to(device)
    goalhead.eval()
    logger.info(f"✅ Loaded GoalHead from {goalhead_ckpt}")
    
    # Text encoder
    text_encoder = SentenceTransformerEncoder(
        model_name="all-MiniLM-L6-v2",
        device=str(device),
    )
    logger.info("✅ Loaded text encoder")
    
    return goalhead, text_encoder


# Task instructions
TASK_INSTRUCTIONS = {
    "mw-reach": "Reach the target position with your gripper",
    "mw-reach-wall": "Reach the target position over the wall",
    "mw-push": "Push the puck to the target position",
    "mw-door-open": "Open the door",
    "mw-drawer-close": "Close the drawer",
}


def main():
    parser = argparse.ArgumentParser(description="Proper Online Evaluation")
    parser.add_argument("--config", type=str, 
                       default="configs/evals/simu_env_planning/mw/jepa-wm/reach_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml",
                       help="Config file path")
    parser.add_argument("--task", type=str, default="mw-reach",
                       choices=["mw-reach", "mw-reach-wall"],
                       help="Task to evaluate")
    parser.add_argument("--mode", type=str, default="both",
                       choices=["oracle", "langgoal", "both"],
                       help="Evaluation mode")
    parser.add_argument("--episodes", type=int, default=48, help="Number of episodes")
    parser.add_argument("--output-dir", type=str, 
                       default="/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/proper_eval",
                       help="Output directory")
    parser.add_argument("--goalhead-ckpt", type=str,
                       default="/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/goalhead/20260201_234248/best_model.pth",
                       help="GoalHead checkpoint path")
    
    args = parser.parse_args()
    
    # Setup
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load config
    config_path = Path(__file__).parent.parent / args.config
    if not config_path.exists():
        # Try direct path
        config_path = Path(args.config)
    
    cfg_dict = load_config(str(config_path))
    cfg_dict["meta"]["eval_episodes"] = args.episodes
    cfg_dict["task_specification"]["task"] = args.task
    cfg_dict["goalhead_checkpoint"] = args.goalhead_ckpt
    
    # Add work_dir if not present
    if "work_dir" not in cfg_dict:
        cfg_dict["work_dir"] = args.output_dir
    
    # Add device as string (OmegaConf doesn't support torch.device)
    cfg_dict["device"] = str(device)
    cfg_dict["local_seed"] = cfg_dict["meta"]["seed"]
    
    cfg = OmegaConf.create(cfg_dict)
    cfg = parse_cfg(cfg)
    set_seed(cfg.meta.seed)
    
    logger.info(f"🚀 Proper Online Evaluation")
    logger.info(f"   Task: {args.task}")
    logger.info(f"   Mode: {args.mode}")
    logger.info(f"   Episodes: {args.episodes}")
    logger.info(f"   Reference: JEPA-WM Reach=58.2%, Reach-Wall=41.6%")
    
    # Initialize model
    model_kwargs = cfg_dict["model_kwargs"]
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})
    
    # Get checkpoint path
    checkpoint_folder = cfg_dict.get("checkpoint_folder", cfg_dict.get("folder", ""))
    checkpoint = model_kwargs.get("checkpoint", "jepa-latest.pth.tar")
    
    # Make datasets for normalization
    dset, preprocessor = make_datasets(cfgs_data, cfgs_data_aug, 1, 0)
    
    # Initialize model
    model = init_module(
        folder=checkpoint_folder,
        checkpoint=checkpoint,
        module_name=model_kwargs["module_name"],
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=device,
        action_dim=dset.action_dim,
        proprio_dim=dset.proprio_dim,
        preprocessor=preprocessor,
    )
    logger.info("✅ Loaded JEPA-WM model")
    
    # Create agent
    agent = GC_Agent(cfg, model, dset=dset, preprocessor=preprocessor)
    
    results = {
        "task": args.task,
        "mode": args.mode,
        "episodes": args.episodes,
        "timestamp": timestamp,
        "reference": {
            "jepa_wm_reach": 0.582,
            "jepa_wm_reach_wall": 0.416,
            "dino_wm_reach": 0.448,
            "dino_wm_reach_wall": 0.351,
        }
    }
    
    # Run evaluations
    if args.mode in ["oracle", "both"]:
        oracle_results = run_oracle_evaluation(cfg, model, agent, dset, preprocessor, device)
        results["oracle"] = oracle_results
    
    if args.mode in ["langgoal", "both"]:
        goalhead, text_encoder = load_goalhead_and_text_encoder(cfg_dict, device)
        instruction = TASK_INSTRUCTIONS.get(args.task, f"Complete {args.task}")
        langgoal_results = run_langgoal_evaluation(
            cfg, model, agent, goalhead, text_encoder, dset, preprocessor, device, instruction
        )
        results["langgoal"] = langgoal_results
    
    # Compute degradation
    if args.mode == "both":
        oracle_sr = results["oracle"]["success_rate"]
        langgoal_sr = results["langgoal"]["success_rate"]
        results["comparison"] = {
            "success_rate_degradation": oracle_sr - langgoal_sr,
            "success_rate_degradation_percent": (oracle_sr - langgoal_sr) / oracle_sr * 100 if oracle_sr > 0 else 0,
        }
    
    # Save results
    output_path = os.path.join(args.output_dir, f"proper_eval_{args.task}_{args.mode}_{timestamp}.json")
    with open(output_path, "w") as f:
        # Remove raw_results for cleaner output
        clean_results = {k: {kk: vv for kk, vv in v.items() if kk != "raw_results"} 
                        if isinstance(v, dict) else v for k, v in results.items()}
        json.dump(clean_results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("📊 EVALUATION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Task: {args.task}")
    
    if "oracle" in results:
        logger.info(f"\n🎯 Oracle (CEM with goal image):")
        logger.info(f"   Success Rate: {results['oracle']['success_rate']:.1%}")
        logger.info(f"   (Reference JEPA-WM: ~58.2% for reach, ~41.6% for reach-wall)")
    
    if "langgoal" in results:
        logger.info(f"\n🗣️ Lang-Goal (CEM with GoalHead):")
        logger.info(f"   Success Rate: {results['langgoal']['success_rate']:.1%}")
        logger.info(f"   Goal Cosine Sim: {results['langgoal']['avg_goal_cosine_sim']:.4f}")
    
    if "comparison" in results:
        logger.info(f"\n📉 Degradation:")
        logger.info(f"   Absolute: {results['comparison']['success_rate_degradation']:.1%}")
        logger.info(f"   Relative: {results['comparison']['success_rate_degradation_percent']:.1f}%")
    
    logger.info(f"\n📁 Results saved to: {output_path}")


if __name__ == "__main__":
    main()
