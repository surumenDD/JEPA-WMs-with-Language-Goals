# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Evaluation script for language-conditioned goal planning.

This implements the main evaluation loop for comparing:
1. Language-conditioned planning (GoalHead predicted z_goal)
2. Baseline image-conditioned planning (encoded I_goal)
"""

import importlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from time import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

from app.lang_goal.data_index import TaskToTextMapper, load_index
from app.lang_goal.env_utils import get_git_info, get_paths, validate_all_env_vars
from evals.lang_goal.lang_goal_agent import LangGoalAgent, LangGoalAgentFactory
from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.common.gc_logger import Logger
from evals.simu_env_planning.planning.common.parser import parse_cfg
from evals.simu_env_planning.planning.gc_agent import GC_Agent
from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
from evals.simu_env_planning.planning.utils import (
    aggregate_results,
    compute_task_distribution,
    set_seed,
)
from evals.utils import make_datasets
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger
from src.utils.yaml_utils import expand_env_vars

logger = get_logger(__name__)

# Global seed
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


def init_module(
    folder,
    checkpoint,
    module_name,
    model_kwargs,
    device,
    cfgs_data=None,
    wrapper_kwargs=None,
    action_dim=None,
    proprio_dim=None,
    preprocessor=None,
):
    """Build (frozen) model and initialize from pretrained checkpoint."""
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


def evaluate_episode(
    agent,
    env,
    cfg,
    task_idx: int,
    text_instruction: str,
    use_language_goal: bool = True,
    goal_image: Optional[torch.Tensor] = None,
) -> Dict:
    """
    Evaluate a single episode.
    
    Args:
        agent: LangGoalAgent or GC_Agent
        env: Environment instance
        cfg: Configuration
        task_idx: Task index
        text_instruction: Language instruction for the task
        use_language_goal: If True, use language goal; else use image goal
        goal_image: Goal image for baseline (required if use_language_goal=False)
        
    Returns:
        Dictionary with episode metrics
    """
    # Reset environment
    obs = env.reset()
    
    # Handle observation format
    if isinstance(obs, dict):
        rgb_obs = obs.get("rgb", obs.get("image", obs.get("pixels")))
    else:
        rgb_obs = obs
    
    # Convert to tensor
    if isinstance(rgb_obs, np.ndarray):
        rgb_obs = torch.from_numpy(rgb_obs).float()
    
    # Normalize if needed
    if rgb_obs.max() > 1.0:
        rgb_obs = rgb_obs / 255.0
    
    # Rearrange to [C, H, W] if needed
    if rgb_obs.dim() == 3 and rgb_obs.shape[-1] in [1, 3]:
        rgb_obs = rgb_obs.permute(2, 0, 1)
    
    # Set goal
    if use_language_goal and isinstance(agent, LangGoalAgent):
        agent.set_goal_from_language(rgb_obs, text_instruction)
    else:
        if goal_image is None:
            raise ValueError("goal_image required for baseline evaluation")
        agent.set_goal(goal_image)
    
    # Run episode
    total_reward = 0.0
    success = False
    steps = 0
    max_steps = cfg.task_specification.max_episode_steps
    
    for step in range(max_steps):
        steps += 1
        
        # Get action from agent
        steps_left = max_steps - step
        action = agent.act(rgb_obs, steps_left=steps_left)
        
        # Step environment
        obs, reward, done, info = env.step(action.numpy())
        
        # Process new observation
        if isinstance(obs, dict):
            rgb_obs = obs.get("rgb", obs.get("image", obs.get("pixels")))
        else:
            rgb_obs = obs
        
        if isinstance(rgb_obs, np.ndarray):
            rgb_obs = torch.from_numpy(rgb_obs).float()
        if rgb_obs.max() > 1.0:
            rgb_obs = rgb_obs / 255.0
        if rgb_obs.dim() == 3 and rgb_obs.shape[-1] in [1, 3]:
            rgb_obs = rgb_obs.permute(2, 0, 1)
        
        total_reward += reward
        
        # Check success
        if info.get("success", False):
            success = True
            if cfg.task_specification.done_at_succ:
                break
        
        if done:
            break
    
    return {
        "success": int(success),
        "total_reward": total_reward,
        "steps": steps,
        "reached_goal": success,
    }


def run_evaluation(
    cfg,
    model,
    lang_agent_factory: Optional[LangGoalAgentFactory],
    dset,
    preprocessor,
    text_mapper: TaskToTextMapper,
    run_baseline: bool = True,
) -> Tuple[Dict, Dict]:
    """
    Run full evaluation comparing language goal vs image goal.
    
    Args:
        cfg: Configuration
        model: World model
        lang_agent_factory: Factory for creating language goal agents
        dset: Dataset
        preprocessor: Image preprocessor
        text_mapper: Task to text mapper
        run_baseline: Whether to also run baseline evaluation
        
    Returns:
        Tuple of (lang_goal_results, baseline_results)
    """
    device = cfg.device
    
    # Create environment
    env = make_env(cfg)
    
    # Create evaluator for standard metrics
    evaluator = PlanEvaluator(cfg, None)  # Will set agent per episode
    
    # Results storage
    lang_goal_results = {
        "per_task": {},
        "per_episode": [],
    }
    baseline_results = {
        "per_task": {},
        "per_episode": [],
    }
    
    # Get task distribution
    cfg.task_indices, cfg.episodes_per_task = compute_task_distribution(cfg)
    
    # Iterate over tasks
    for task_pos, (task_idx, episodes) in enumerate(zip(cfg.task_indices, cfg.episodes_per_task)):
        task_name = cfg.tasks[task_idx]
        text_instruction = text_mapper.get_text(task_name)
        
        logger.info(f"🎯 Evaluating task: {task_name}")
        logger.info(f"   Text instruction: '{text_instruction}'")
        
        lang_task_metrics = {
            "successes": [],
            "rewards": [],
            "steps": [],
        }
        baseline_task_metrics = {
            "successes": [],
            "rewards": [],
            "steps": [],
        }
        
        for ep_idx, ep in enumerate(episodes):
            logger.info(f"   Episode {ep_idx + 1}/{len(episodes)}")
            
            # Create fresh agent for language goal
            if lang_agent_factory is not None:
                lang_agent = lang_agent_factory.create_agent(
                    cfg, model, dset=dset, preprocessor=preprocessor
                )
                
                # Run language goal evaluation
                lang_metrics = evaluate_episode(
                    agent=lang_agent,
                    env=env,
                    cfg=cfg,
                    task_idx=task_idx,
                    text_instruction=text_instruction,
                    use_language_goal=True,
                )
                
                lang_task_metrics["successes"].append(lang_metrics["success"])
                lang_task_metrics["rewards"].append(lang_metrics["total_reward"])
                lang_task_metrics["steps"].append(lang_metrics["steps"])
                
                lang_goal_results["per_episode"].append({
                    "task": task_name,
                    "episode": ep_idx,
                    **lang_metrics,
                })
            
            # Run baseline evaluation (goal image)
            if run_baseline:
                baseline_agent = GC_Agent(cfg, model, dset=dset, preprocessor=preprocessor)
                
                # Get goal image from dataset or environment
                # This would need to be adapted based on how goal images are stored
                goal_obs = env.get_goal_observation() if hasattr(env, 'get_goal_observation') else None
                
                if goal_obs is not None:
                    baseline_metrics = evaluate_episode(
                        agent=baseline_agent,
                        env=env,
                        cfg=cfg,
                        task_idx=task_idx,
                        text_instruction=text_instruction,
                        use_language_goal=False,
                        goal_image=goal_obs,
                    )
                    
                    baseline_task_metrics["successes"].append(baseline_metrics["success"])
                    baseline_task_metrics["rewards"].append(baseline_metrics["total_reward"])
                    baseline_task_metrics["steps"].append(baseline_metrics["steps"])
                    
                    baseline_results["per_episode"].append({
                        "task": task_name,
                        "episode": ep_idx,
                        **baseline_metrics,
                    })
        
        # Aggregate task metrics
        if lang_task_metrics["successes"]:
            lang_goal_results["per_task"][task_name] = {
                "success_rate": np.mean(lang_task_metrics["successes"]),
                "mean_reward": np.mean(lang_task_metrics["rewards"]),
                "mean_steps": np.mean([s for s, succ in zip(lang_task_metrics["steps"], lang_task_metrics["successes"]) if succ]),
                "num_episodes": len(lang_task_metrics["successes"]),
            }
        
        if baseline_task_metrics["successes"]:
            baseline_results["per_task"][task_name] = {
                "success_rate": np.mean(baseline_task_metrics["successes"]),
                "mean_reward": np.mean(baseline_task_metrics["rewards"]),
                "mean_steps": np.mean([s for s, succ in zip(baseline_task_metrics["steps"], baseline_task_metrics["successes"]) if succ]),
                "num_episodes": len(baseline_task_metrics["successes"]),
            }
    
    # Compute overall metrics
    if lang_goal_results["per_episode"]:
        all_successes = [ep["success"] for ep in lang_goal_results["per_episode"]]
        lang_goal_results["overall"] = {
            "success_rate": np.mean(all_successes),
            "num_episodes": len(all_successes),
        }
    
    if baseline_results["per_episode"]:
        all_successes = [ep["success"] for ep in baseline_results["per_episode"]]
        baseline_results["overall"] = {
            "success_rate": np.mean(all_successes),
            "num_episodes": len(all_successes),
        }
    
    return lang_goal_results, baseline_results


def compute_degradation(
    lang_results: Dict,
    baseline_results: Dict,
) -> Dict:
    """
    Compute degradation metrics comparing language goal to baseline.
    
    Args:
        lang_results: Results from language goal planning
        baseline_results: Results from baseline (image goal) planning
        
    Returns:
        Dictionary with degradation metrics
    """
    degradation = {
        "per_task": {},
        "overall": {},
    }
    
    # Per-task degradation
    for task_name in lang_results.get("per_task", {}):
        lang_task = lang_results["per_task"].get(task_name, {})
        baseline_task = baseline_results.get("per_task", {}).get(task_name, {})
        
        if lang_task and baseline_task:
            degradation["per_task"][task_name] = {
                "success_rate_diff": lang_task["success_rate"] - baseline_task["success_rate"],
                "steps_diff": lang_task.get("mean_steps", 0) - baseline_task.get("mean_steps", 0),
            }
    
    # Overall degradation
    if "overall" in lang_results and "overall" in baseline_results:
        degradation["overall"] = {
            "success_rate_diff": (
                lang_results["overall"]["success_rate"] - 
                baseline_results["overall"]["success_rate"]
            ),
        }
    
    return degradation


def save_results(
    output_dir: Path,
    lang_results: Dict,
    baseline_results: Dict,
    degradation: Dict,
    config: Dict,
):
    """Save evaluation results to files."""
    
    # Save as JSON
    results = {
        "lang_goal": lang_results,
        "baseline": baseline_results,
        "degradation": degradation,
        "config": config,
        "timestamp": datetime.now().isoformat(),
        "git_info": get_git_info(),
    }
    
    json_path = output_dir / "results.json"
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save per-episode results as CSV
    if lang_results.get("per_episode"):
        df_lang = pd.DataFrame(lang_results["per_episode"])
        df_lang.to_csv(output_dir / "lang_goal_episodes.csv", index=False)
    
    if baseline_results.get("per_episode"):
        df_baseline = pd.DataFrame(baseline_results["per_episode"])
        df_baseline.to_csv(output_dir / "baseline_episodes.csv", index=False)
    
    # Save per-task summary
    task_summary = []
    for task_name in set(
        list(lang_results.get("per_task", {}).keys()) + 
        list(baseline_results.get("per_task", {}).keys())
    ):
        row = {
            "task": task_name,
            "lang_success_rate": lang_results.get("per_task", {}).get(task_name, {}).get("success_rate"),
            "baseline_success_rate": baseline_results.get("per_task", {}).get(task_name, {}).get("success_rate"),
            "success_rate_diff": degradation.get("per_task", {}).get(task_name, {}).get("success_rate_diff"),
        }
        task_summary.append(row)
    
    if task_summary:
        df_summary = pd.DataFrame(task_summary)
        df_summary.to_csv(output_dir / "task_summary.csv", index=False)
    
    logger.info(f"💾 Results saved to {output_dir}")


def main(args_eval: Dict = None, resume_preempt: bool = False):
    """Main evaluation function."""
    
    # Expand environment variables
    if args_eval is not None:
        args_eval = expand_env_vars(args_eval)
    
    # Get paths
    env_paths = get_paths()
    
    # Setup output directory
    eval_tag = args_eval.get("tag", "lang_goal_eval")
    pretrain_folder = args_eval.get("folder", str(env_paths["lang_goal_logs"]))
    output_dir = Path(pretrain_folder) / "lang_goal_planning" / eval_tag
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(args_eval, f)
    
    # Setup distributed
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    world_size, rank = init_distributed()
    logger.info(f"🚀 Initialized (rank/world-size) {rank}/{world_size}")
    
    # Parse config - set work_dir BEFORE parse_cfg since parser expects it
    cfg = OmegaConf.create(args_eval)
    cfg.work_dir = str(output_dir)  # Must be set before parse_cfg
    cfg = parse_cfg(cfg)
    set_seed(cfg.meta.seed)
    cfg.rank = rank
    cfg.world_size = world_size
    cfg.device = str(device)  # Store as string for OmegaConf compatibility
    
    # Set local_seed for agent initialization
    cfg.local_seed = cfg.meta.seed
    seed_shift_cfg = args_eval.get("distributed", {}).get("seed_shift", "horizon_1000")
    if seed_shift_cfg == "horizon_1000":
        seed_shift = 1000
    else:
        seed_shift = int(seed_shift_cfg) if seed_shift_cfg else 0
    cfg.local_seed += rank * seed_shift
    
    logger.info(f"📂 Work dir: {output_dir}")
    
    # Load model
    model_kwargs = args_eval.get("model_kwargs", {})
    module_name = model_kwargs.get("module_name")
    checkpoint = model_kwargs.get("checkpoint")
    
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})
    
    dset, preprocessor = make_datasets(cfgs_data, cfgs_data_aug, world_size, rank)
    
    model = init_module(
        folder=args_eval.get("checkpoint_folder", pretrain_folder),
        checkpoint=checkpoint,
        module_name=module_name,
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=device,
        action_dim=dset.action_dim,
        proprio_dim=dset.proprio_dim,
        preprocessor=preprocessor,
    )
    logger.info("✅ Loaded world model")
    
    # Load GoalHead
    goalhead_checkpoint = args_eval.get("goalhead_checkpoint")
    lang_agent_factory = None
    
    if goalhead_checkpoint and Path(goalhead_checkpoint).exists():
        text_encoder_config = args_eval.get("text_encoder", {
            "type": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2",
        })
        
        lang_agent_factory = LangGoalAgentFactory(
            goal_head_checkpoint=goalhead_checkpoint,
            text_encoder_config=text_encoder_config,
            device=str(device),
        )
        logger.info("✅ Loaded GoalHead for language-conditioned planning")
    else:
        logger.warning(f"⚠️ GoalHead checkpoint not found: {goalhead_checkpoint}")
        logger.warning("   Running baseline evaluation only")
    
    # Load task-to-text mapper
    text_mapper = TaskToTextMapper()
    
    # Run evaluation
    run_baseline = args_eval.get("run_baseline", True)
    
    lang_results, baseline_results = run_evaluation(
        cfg=cfg,
        model=model,
        lang_agent_factory=lang_agent_factory,
        dset=dset,
        preprocessor=preprocessor,
        text_mapper=text_mapper,
        run_baseline=run_baseline,
    )
    
    # Compute degradation
    degradation = compute_degradation(lang_results, baseline_results)
    
    # Save results
    save_results(
        output_dir=output_dir,
        lang_results=lang_results,
        baseline_results=baseline_results,
        degradation=degradation,
        config=args_eval,
    )
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Evaluation Summary")
    logger.info("=" * 60)
    
    if "overall" in lang_results:
        logger.info(f"Language Goal Success Rate: {lang_results['overall']['success_rate']:.4f}")
    
    if "overall" in baseline_results:
        logger.info(f"Baseline (Image Goal) Success Rate: {baseline_results['overall']['success_rate']:.4f}")
    
    if "overall" in degradation:
        logger.info(f"Degradation (Lang - Baseline): {degradation['overall']['success_rate_diff']:.4f}")
    
    logger.info("=" * 60)
    logger.info(f"✅ Evaluation complete! Results saved to {output_dir}")
    
    return lang_results, baseline_results, degradation


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Language Goal Planning Evaluation")
    parser.add_argument("--fname", type=str, required=True, help="Config file path")
    parser.add_argument("--debug", action="store_true", help="Debug mode (single GPU)")
    
    args = parser.parse_args()
    
    # Load config
    with open(args.fname, "r") as f:
        config = yaml.safe_load(f)
    
    if args.debug:
        config["meta"] = config.get("meta", {})
        config["meta"]["quick_debug"] = True
    
    main(args_eval=config)
