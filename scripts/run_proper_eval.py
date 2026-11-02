#!/usr/bin/env python3
"""
Proper evaluation script for language-conditioned goal planning.

This script uses the ORIGINAL JEPA-WMs evaluation pipeline to ensure
correct comparison with the paper's baseline.

Reference paper values:
- JEPA-WMs: Reach 58.2%, Reach-Wall 41.6%
- DINO-WM: Reach 44.8%, Reach-Wall 35.1%

Usage:
    # Baseline evaluation (goal_source: expert)
    python scripts/run_proper_eval.py --task mw-reach --eval_type baseline --episodes 48
    
    # Language-goal evaluation (goal_source: lang)
    python scripts/run_proper_eval.py --task mw-reach --eval_type lang_goal --episodes 48
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import torch
import yaml
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.lang_goal.goal_head import GoalHead
from app.lang_goal.text_encoder import SentenceTransformerEncoder
from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.gc_logger import Logger
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.gc_agent import GC_Agent
from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
from evals.simu_env_planning.planning.utils import aggregate_results, compute_task_distribution, set_seed
from evals.utils import make_datasets, prepare_obs
from evals.simu_env_planning.planning.utils import make_td
from src.utils.logging import get_logger
from src.utils.yaml_utils import expand_env_vars

logging.basicConfig(level=logging.INFO)
logger = get_logger(__name__)


# Task to text mapping for language goals
TASK_TO_TEXT = {
    "mw-reach": "Reach the target position with your gripper",
    "mw-reach-wall": "Reach the target position over the wall",
    "mw-button-press": "Press the button",
    "mw-door-open": "Open the door",
    "mw-drawer-close": "Close the drawer",
    "mw-drawer-open": "Open the drawer",
}


def init_module(folder, checkpoint, module_name, model_kwargs, device, cfgs_data=None,
                wrapper_kwargs=None, action_dim=None, proprio_dim=None, preprocessor=None):
    """Build (frozen) model and initialize from pretrained checkpoint."""
    import importlib
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


def load_base_config(task: str) -> dict:
    """Load base config for the task from existing JEPA-WMs configs."""
    script_dir = Path(__file__).parent.parent
    
    # Use existing reach/reach-wall configs
    if task == "mw-reach":
        config_path = script_dir / "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml"
    elif task == "mw-reach-wall":
        config_path = script_dir / "configs/evals/simu_env_planning/mw/jepa-wm/reach-wall_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml"
    else:
        raise ValueError(f"Unknown task: {task}. Supported: mw-reach, mw-reach-wall")
    
    if not config_path.exists():
        # Try alternate paths
        alt_paths = [
            script_dir / "configs/evals/simu_env_planning/mw/jepa-wm/reach_L2_cem_sourcexp_H6_nas3_ctxt2_r256_alpha0.1_ep48_decode.yaml",
            script_dir / "configs/lang_goal/mw_lang_goal_eval.yaml",
        ]
        for alt_path in alt_paths:
            if alt_path.exists():
                config_path = alt_path
                break
    
    if not config_path.exists():
        logger.warning(f"Config not found at {config_path}, using mw_lang_goal_eval.yaml")
        config_path = script_dir / "configs/lang_goal/mw_lang_goal_eval.yaml"
    
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # Override task
    if "task_specification" in config:
        config["task_specification"]["task"] = task
    
    return config


def load_goalhead_and_text_encoder(
    goalhead_checkpoint: str,
    text_encoder_name: str = "all-MiniLM-L6-v2",
    device: torch.device = torch.device("cuda:0"),
):
    """Load GoalHead and text encoder for language-conditioned evaluation."""
    logger.info(f"📦 Loading GoalHead from {goalhead_checkpoint}")
    
    goalhead = GoalHead(
        visual_dim=384,
        text_dim=384,
        output_dim=384,
        hidden_dim=512,
        num_layers=3,
        fusion_mode="concat",
    )
    
    checkpoint = torch.load(goalhead_checkpoint, map_location="cpu")
    state_dict_key = "goal_head_state_dict" if "goal_head_state_dict" in checkpoint else "model_state_dict"
    goalhead.load_state_dict(checkpoint[state_dict_key])
    goalhead.to(device)
    goalhead.eval()
    
    logger.info(f"✅ GoalHead loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    logger.info(f"📦 Loading TextEncoder: {text_encoder_name}")
    text_encoder = SentenceTransformerEncoder(
        model_name=text_encoder_name,
        device=str(device),
    )
    
    return goalhead, text_encoder


class LangGoalPlanEvaluator(PlanEvaluator):
    """Modified PlanEvaluator that supports language-conditioned goals."""
    
    def __init__(self, cfg, agent, goalhead=None, text_encoder=None, task_to_text=None):
        super().__init__(cfg, agent)
        self.goalhead = goalhead
        self.text_encoder = text_encoder
        self.task_to_text = task_to_text or TASK_TO_TEXT
    
    def set_episode_lang_goal(self, cfg, agent, env, ep_seed, task_idx=-1):
        """
        Set up episode using language-conditioned goal instead of expert trajectory.
        
        Key difference from set_episode():
        - Uses GoalHead(I_0, text) instead of Encoder(I_goal) for z_goal
        """
        # Get initial observation (same as baseline)
        init_obs, info = env.reset_warmup(seed=ep_seed)
        init_obs = make_td(init_obs, info)
        
        # Get task name and instruction
        task_name = cfg.tasks[task_idx]
        instruction = self.task_to_text.get(task_name, f"Perform {task_name} task")
        logger.info(f"🗣️  Language instruction: '{instruction}'")
        
        # Get goal from expert trajectory (we still need this for ground truth comparison)
        # but we won't use it for planning - only for success evaluation
        expert_obses, ep_reward, expert_actions, expert_infos = self.get_goal_state_mw(
            env, task_idx=task_idx,
        )
        self.expert_actions = torch.stack(expert_actions).permute(1, 0, 2)
        goal_obs = expert_obses[-1]
        expert_success = expert_infos[-1]["success"]
        self.state_g = expert_infos[-1]["state"]
        
        # --- Language Goal: Predict z_goal using GoalHead ---
        # Encode initial observation
        initial_visual = prepare_obs(cfg.task_specification.obs, init_obs)
        initial_visual = initial_visual.unsqueeze(0).to(agent.device)
        
        # Get visual features
        with torch.no_grad():
            z_initial = agent.model.encode(initial_visual, act=False)
            z_initial_visual = z_initial["visual"]  # [1, T, V, H, W, D] or [1, T, N, D]
        
        # Get text features
        text_features = self.text_encoder.encode([instruction])
        text_features = text_features.to(agent.device)
        
        # Predict z_goal using GoalHead
        with torch.no_grad():
            # GoalHead expects pooled visual features
            if z_initial_visual.dim() > 2:
                # Pool to [B, D]
                z_visual_pooled = z_initial_visual.mean(dim=tuple(range(1, z_initial_visual.dim() - 1)))
            else:
                z_visual_pooled = z_initial_visual
            
            z_goal_pred = self.goalhead(z_visual_pooled, text_features)  # [1, D]
        
        logger.info(f"🎯 Predicted z_goal shape: {z_goal_pred.shape}")
        
        # Expand z_goal_pred to match the shape expected by planner
        # We need to convert [1, D] -> [1, T, V, H, W, D] or similar
        # Based on how the original set_goal works, we need:
        # goal_state_enc["visual"] shape to match what unroll predicts
        
        # Use the original z_initial_visual shape as template
        template_shape = z_initial_visual.shape  # e.g., [1, T, V, H, W, D]
        if len(template_shape) == 6:
            # [1, T, V, H, W, D]
            z_goal_expanded = z_goal_pred.view(1, 1, 1, 1, 1, -1).expand(
                1, template_shape[1], template_shape[2], template_shape[3], template_shape[4], -1
            )
        elif len(template_shape) == 4:
            # [1, T, N, D]
            z_goal_expanded = z_goal_pred.view(1, 1, 1, -1).expand(
                1, template_shape[1], template_shape[2], -1
            )
        else:
            # Fallback: just add dimensions
            z_goal_expanded = z_goal_pred.unsqueeze(1)
        
        # Set goal using the embedding method
        agent.set_goal_from_embedding(z_goal_expanded)
        
        return init_obs, goal_obs, expert_obses, expert_success
    
    def eval_lang_goal(self, cfg, agent, env, task_idx=-1, ep=0):
        """
        Evaluation using language-conditioned goal prediction.
        """
        work_dir = cfg.work_dir / cfg.tasks[task_idx] / f"ep_{ep}"
        dist_work_dir = work_dir / "distances"
        os.makedirs(dist_work_dir, exist_ok=True)
        vis_work_dir = work_dir / "visualisation"
        os.makedirs(vis_work_dir, exist_ok=True)
        if cfg.planner.decode_each_iteration:
            self.ep_plan_vis_dir = work_dir / "plan_vis"
            os.makedirs(self.ep_plan_vis_dir, exist_ok=True)
        
        ep_seed = (cfg.local_seed * cfg.local_seed + ep * cfg.local_seed) % (2**32 - 2)
        self.expert_actions = None
        
        init_obs, info = env.reset(seed=ep_seed, task_idx=task_idx)
        unwrapped = env.proprio_env.unwrapped
        unwrapped._freeze_rand_vec = False
        unwrapped.seeded_rand_vec = True
        env.seed(ep_seed)

        # Use language goal instead of image goal
        init_obs, goal_obs, expert_obses, expert_success = self.set_episode_lang_goal(
            cfg, agent, env, ep_seed, task_idx=task_idx,
        )

        # Rest is same as original eval
        self.prev_losses = []
        self.prev_elite_losses_mean = []
        self.prev_elite_losses_std = []
        self.prev_pred_frames_over_iterations = []
        self.predicted_best_encs_over_iterations = []

        def agent_actor(obs, steps_left, plan_vis_path=None):
            act = agent.act(
                prepare_obs(agent.cfg.task_specification.obs, obs),
                steps_left=steps_left,
            )
            self.prev_losses.append(agent._prev_losses)
            self.prev_elite_losses_mean.append(agent._prev_elite_losses_mean)
            self.prev_elite_losses_std.append(agent._prev_elite_losses_std)
            self.prev_pred_frames_over_iterations.append(agent._prev_pred_frames_over_iterations)
            self.predicted_best_encs_over_iterations.append(agent._predicted_best_encs_over_iterations)
            return act

        episode_obses, ep_reward, planned_actions, infos, success, state_dist = self.unroll_agent(
            env, init_obs, info, agent_actor, preprocessor=agent.preprocessor,
        )

        # Compute distances (same as original)
        if cfg.logging.optional_plots and cfg.task_specification.num_frames > agent.model.tubelet_size_enc:
            for x in episode_obses:
                x["visual"] = x["visual"][-agent.model.tubelet_size_enc:]

        coord_diffs = [-1.0]  # placeholder if no analysis
        if cfg.logging.optional_plots:
            from evals.simu_env_planning.planning.episode_plot_utils import analyze_distances
            coord_diffs, _repr_diffs = analyze_distances(
                agent, episode_obses, goal_obs, str(dist_work_dir / "agent"),
                objective=agent.objective,
            )
        
        success_dist = float(coord_diffs[-1] < 0.05) if coord_diffs[-1] != -1.0 else 0.0
        end_distance = coord_diffs[-1]
        end_distance_xyz, end_distance_orientation, end_distance_closure = -1.0, -1.0, -1.0
        total_lpips, total_emb_l2 = -1.0, -1.0

        return (
            expert_success,
            success,
            ep_reward,
            success_dist,
            end_distance,
            end_distance_xyz,
            end_distance_orientation,
            end_distance_closure,
            state_dist,
            total_lpips,
            total_emb_l2,
        )


def run_baseline_eval(cfg, model, dset, preprocessor, device):
    """Run baseline evaluation (goal_source: expert) - same as original JEPA-WMs."""
    logger.info("=" * 60)
    logger.info("Running BASELINE evaluation (goal from expert trajectory)")
    logger.info("=" * 60)
    
    agent = GC_Agent(cfg, model, dset=dset, preprocessor=preprocessor)
    env = make_env(cfg)
    evaluator = PlanEvaluator(cfg, agent)
    
    results = run_evaluation_loop(cfg, agent, env, evaluator, eval_type="baseline")
    return results


def run_lang_goal_eval(cfg, model, dset, preprocessor, device, goalhead, text_encoder):
    """Run language-conditioned goal evaluation."""
    logger.info("=" * 60)
    logger.info("Running LANGUAGE GOAL evaluation (goal from GoalHead)")
    logger.info("=" * 60)
    
    agent = GC_Agent(cfg, model, dset=dset, preprocessor=preprocessor)
    env = make_env(cfg)
    evaluator = LangGoalPlanEvaluator(cfg, agent, goalhead=goalhead, text_encoder=text_encoder)
    
    results = run_evaluation_loop(cfg, agent, env, evaluator, eval_type="lang_goal")
    return results


def run_evaluation_loop(cfg, agent, env, evaluator, eval_type="baseline"):
    """Common evaluation loop for both baseline and lang_goal."""
    results = {
        "eval_type": eval_type,
        "task": cfg.tasks[0] if len(cfg.tasks) == 1 else "multi",
        "episodes": [],
        "successes": [],
        "rewards": [],
        "steps": [],
    }
    
    cfg.task_indices, cfg.episodes_per_task = compute_task_distribution(cfg)
    logger.info(f"Task indices: {cfg.task_indices}")
    logger.info(f"Episodes per task: {cfg.episodes_per_task}")
    
    for task_pos, (task_idx, episodes) in enumerate(zip(cfg.task_indices, cfg.episodes_per_task)):
        task_name = cfg.tasks[task_idx]
        logger.info(f"\n🎯 Evaluating task: {task_name}")
        
        task_successes = []
        task_rewards = []
        
        for i, ep in enumerate(episodes):
            logger.info(f"  Episode {i + 1}/{len(episodes)}")
            
            episode_start_time = time()
            
            if eval_type == "lang_goal":
                # Use language goal evaluation
                result = evaluator.eval_lang_goal(cfg, agent, env, task_idx=task_idx, ep=ep)
            else:
                # Use baseline evaluation
                result = evaluator.eval(cfg, agent, env, task_idx=task_idx, ep=ep)
            
            (expert_success, success, ep_reward, success_dist, end_distance,
             end_distance_xyz, end_distance_orientation, end_distance_closure,
             state_dist, total_lpips, total_emb_l2) = result
            
            episode_time = time() - episode_start_time
            
            results["episodes"].append({
                "task": task_name,
                "episode": ep,
                "success": success,
                "reward": ep_reward,
                "end_distance": end_distance,
                "time": episode_time,
            })
            results["successes"].append(success)
            results["rewards"].append(ep_reward)
            task_successes.append(success)
            task_rewards.append(ep_reward)
            
            logger.info(f"    Success: {success}, Reward: {ep_reward:.2f}, "
                       f"Distance: {end_distance:.4f}, Time: {episode_time:.1f}s")
        
        task_success_rate = np.mean(task_successes) * 100
        logger.info(f"  📊 {task_name} success rate: {task_success_rate:.1f}%")
    
    # Aggregate results
    results["aggregate"] = {
        "success_rate": np.mean(results["successes"]) * 100,
        "mean_reward": np.mean(results["rewards"]),
        "num_episodes": len(results["episodes"]),
    }
    
    logger.info(f"\n📊 Overall success rate: {results['aggregate']['success_rate']:.1f}%")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Proper evaluation for JEPA-WMs Lang-Goal")
    parser.add_argument("--task", type=str, default="mw-reach",
                       choices=["mw-reach", "mw-reach-wall"],
                       help="Task to evaluate")
    parser.add_argument("--eval_type", type=str, default="both",
                       choices=["baseline", "lang_goal", "both"],
                       help="Type of evaluation to run")
    parser.add_argument("--episodes", type=int, default=48,
                       help="Number of episodes per task")
    parser.add_argument("--seed", type=int, default=1,
                       help="Random seed")
    parser.add_argument("--goalhead_checkpoint", type=str,
                       default="/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/goalhead/20260201_234248/best_model.pth",
                       help="Path to GoalHead checkpoint")
    parser.add_argument("--output_dir", type=str,
                       default="/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/proper_eval",
                       help="Output directory for results")
    parser.add_argument("--quick_debug", action="store_true",
                       help="Quick debug mode with fewer episodes/samples")
    args = parser.parse_args()
    
    # Load config
    config = load_base_config(args.task)
    config = expand_env_vars(config)
    
    # Override settings
    config["meta"]["eval_episodes"] = args.episodes
    config["meta"]["seed"] = args.seed
    if args.quick_debug:
        config["meta"]["quick_debug"] = True
        config["meta"]["eval_episodes"] = 2
        config["planner"]["iterations"] = 2
        config["planner"]["num_samples"] = 10
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Setup config
    folder = config.get("folder", output_dir)
    config["work_dir"] = output_dir / f"{args.task}_{args.eval_type}_{timestamp}"
    config["work_dir"].mkdir(parents=True, exist_ok=True)
    
    cfg = OmegaConf.create(config)
    cfg = parse_cfg(cfg)
    set_seed(cfg.meta.seed)
    cfg.rank = 0
    cfg.world_size = 1
    cfg.device = device
    cfg.num_active_gpus = 1
    cfg.active_ranks = [0]
    cfg.local_seed = cfg.meta.seed
    cfg.task_specification.goal_source = cfg.task_specification.get("goal_source", "expert")
    
    # Load model
    logger.info("Loading JEPA-WM model...")
    model_kwargs = config["model_kwargs"]
    checkpoint_folder = config.get("checkpoint_folder", config.get("folder"))
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})
    
    dset, preprocessor = make_datasets(cfgs_data, cfgs_data_aug, 1, 0)
    cfg.frameskip = cfgs_data["custom"]["frameskip"]
    
    model = init_module(
        folder=checkpoint_folder,
        checkpoint=model_kwargs.get("checkpoint"),
        module_name=model_kwargs.get("module_name"),
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=device,
        action_dim=dset.action_dim,
        proprio_dim=dset.proprio_dim,
        preprocessor=preprocessor,
    )
    logger.info("✅ JEPA-WM model loaded")
    
    # Create environment for action_ratio calculation
    env = make_env(cfg)
    if cfg.planner.repeat_actskip:
        cfg.action_ratio = 1
    else:
        cfg.action_ratio = cfg.frameskip // model.action_skip
    
    results = {"task": args.task, "timestamp": timestamp}
    
    # Run baseline evaluation
    if args.eval_type in ["baseline", "both"]:
        baseline_results = run_baseline_eval(cfg, model, dset, preprocessor, device)
        results["baseline"] = baseline_results
    
    # Run language goal evaluation
    if args.eval_type in ["lang_goal", "both"]:
        goalhead, text_encoder = load_goalhead_and_text_encoder(
            args.goalhead_checkpoint,
            device=device,
        )
        lang_goal_results = run_lang_goal_eval(
            cfg, model, dset, preprocessor, device,
            goalhead, text_encoder,
        )
        results["lang_goal"] = lang_goal_results
    
    # Save results
    results_path = output_dir / f"results_{args.task}_{args.eval_type}_{timestamp}.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info(f"💾 Results saved to {results_path}")
    
    # Print comparison
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Task: {args.task}")
    logger.info(f"Episodes: {args.episodes}")
    
    if "baseline" in results:
        logger.info(f"\n📊 Baseline (Image Goal) Success Rate: {results['baseline']['aggregate']['success_rate']:.1f}%")
        logger.info(f"   Reference (JEPA-WMs paper):")
        logger.info(f"     - Reach: 58.2%")
        logger.info(f"     - Reach-Wall: 41.6%")
    
    if "lang_goal" in results:
        logger.info(f"\n📊 Language Goal Success Rate: {results['lang_goal']['aggregate']['success_rate']:.1f}%")
    
    if "baseline" in results and "lang_goal" in results:
        degradation = results["baseline"]["aggregate"]["success_rate"] - results["lang_goal"]["aggregate"]["success_rate"]
        logger.info(f"\n📉 Degradation (Baseline - LangGoal): {degradation:.1f}%")


if __name__ == "__main__":
    main()
