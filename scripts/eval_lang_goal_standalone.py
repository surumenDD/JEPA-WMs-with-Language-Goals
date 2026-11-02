#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Standalone evaluation script for language-conditioned goal planning.

This script implements Approach D: a simple standalone evaluation that:
1. Computes offline errors (L2/cosine) between z_pred and z_oracle
2. Runs online evaluation comparing Lang-Goal vs Oracle planning
3. Generates task-wise breakdown of success rates and steps

Research validation metrics:
- 推定誤差: L2距離 and コサイン距離 (z_pred vs z_oracle)
- 成功率の劣化: Lang-Goal vs Oracle comparison
- 到達ステップの劣化: Average steps comparison
- 成立範囲と限界: Task-wise and condition-wise analysis
"""

import argparse
import csv
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.lang_goal.goal_head import GoalHead, ImprovedGoalHead
from app.lang_goal.text_encoder import SentenceTransformerEncoder, build_text_encoder
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
class ModelConfig:
    """Configuration for model paths and architecture."""
    # Checkpoints
    jepa_wm_checkpoint: str = "/home/ryo52/checkpoints/jepa_lang_goal/pretrained/mw_jepa-wm.pth.tar"
    goalhead_checkpoint: str = "/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/goalhead/20260207_200729/best_model.pth"
    
    # GoalHead architecture (must match training config from checkpoint)
    visual_dim: int = 384
    text_dim: int = 1024  # Must match text encoder output (SigLIP2 = 1024)
    output_dim: int = 384
    hidden_dim: int = 512
    num_layers: int = 2  # Must match training config
    fusion_mode: str = "qformer"  # v2 uses qformer fusion
    num_queries: int = 16  # Q-Former query count
    use_gated_attn: bool = True  # Gated attention for better fusion
    use_improved: bool = True  # Use ImprovedGoalHead (v2)
    
    # JEPA-WM architecture  
    patch_size: int = 16
    tubelet_size_enc: int = 1  # Matches training config
    img_size: int = 224  # Matches training config
    grid_size: int = 16  # Matches training config
    
    # Text encoder
    text_encoder_type: str = "siglip2"  # siglip2, sentence_transformer, clip
    text_encoder_name: str = "google/siglip2-large-patch16-512"


@dataclass
class PlannerConfig:
    """Configuration for CEM planner."""
    planner_name: str = "cem"
    horizon: int = 12
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    elite_ratio: float = 0.1
    action_dim: int = 4  # MetaWorld action dimension
    action_min: float = -1.0
    action_max: float = 1.0


@dataclass
class EvalConfig:
    """Configuration for evaluation."""
    # Tasks to evaluate
    tasks: List[str] = field(default_factory=lambda: [
        "assembly-v2",
        "bin-picking-v2",
        "button-press-v2",
        "door-close-v2",
        "door-open-v2",
        "drawer-close-v2",
        "drawer-open-v2",
        "hammer-v2",
    ])
    
    # Evaluation settings
    episodes_per_task: int = 10
    max_episode_steps: int = 100
    seed: int = 42
    
    # Output directory
    output_dir: str = "/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/evaluation"
    
    # Device
    device: str = "cuda:0"


# ============================================================================
# Helper Functions
# ============================================================================

def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two tensors."""
    # Ensure at least 2D
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    a_norm = F.normalize(a.flatten(start_dim=1), dim=-1)
    b_norm = F.normalize(b.flatten(start_dim=1), dim=-1)
    return (a_norm * b_norm).sum(dim=-1)


def l2_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute L2 distance between two tensors."""
    # Ensure at least 2D
    if a.dim() == 1:
        a = a.unsqueeze(0)
    if b.dim() == 1:
        b = b.unsqueeze(0)
    return torch.norm(a.flatten(start_dim=1) - b.flatten(start_dim=1), dim=-1)


# ============================================================================
# Model Loading
# ============================================================================

def load_jepa_wm_model(
    checkpoint_path: str,
    device: torch.device,
    img_size: int = 224,
) -> nn.Module:
    """
    Load pretrained JEPA-WM model using the same approach as train_goalhead.py.
    
    Args:
        checkpoint_path: Path to JEPA-WM checkpoint
        device: Device to load on
        img_size: Image size for preprocessing
        
    Returns:
        Frozen EncPredWM model with encode() method
    """
    # Model kwargs matching the GoalHead training config exactly
    model_kwargs = {
        "module_name": "app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds",
        "data": {
            "img_size": img_size,
            "custom": {
                "frameskip": 5,
                "action_skip": 1,
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
    
    model = load_pretrained_model(
        checkpoint_path=checkpoint_path,
        device=device,
        model_kwargs=model_kwargs,
    )
    
    return model


def load_goalhead(
    checkpoint_path: str,
    model_config: ModelConfig,
    device: torch.device,
) -> nn.Module:
    """Load trained GoalHead model (supports both GoalHead and ImprovedGoalHead)."""
    logger.info(f"📦 Loading GoalHead from {checkpoint_path}")
    
    # Choose model class based on config
    if model_config.use_improved:
        logger.info(f"   Using ImprovedGoalHead (v2) with fusion_mode={model_config.fusion_mode}")
        model = ImprovedGoalHead(
            visual_dim=model_config.visual_dim,
            text_dim=model_config.text_dim,
            output_dim=model_config.output_dim,
            hidden_dim=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            fusion_mode=model_config.fusion_mode,
            num_queries=model_config.num_queries,
            use_gated_attn=model_config.use_gated_attn,
        )
    else:
        logger.info(f"   Using GoalHead (v1) with fusion_mode={model_config.fusion_mode}")
        model = GoalHead(
            visual_dim=model_config.visual_dim,
            text_dim=model_config.text_dim,
            output_dim=model_config.output_dim,
            hidden_dim=model_config.hidden_dim,
            num_layers=model_config.num_layers,
            fusion_mode=model_config.fusion_mode,
        )
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    # Support both key names for compatibility
    state_dict_key = "goal_head_state_dict" if "goal_head_state_dict" in checkpoint else "model_state_dict"
    model.load_state_dict(checkpoint[state_dict_key])
    model.to(device)
    model.eval()
    
    logger.info(f"✅ GoalHead loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    return model


def load_text_encoder(
    model_config: ModelConfig,
    device: torch.device,
) -> nn.Module:
    """Load text encoder using build_text_encoder for consistency with training."""
    logger.info(f"📦 Loading TextEncoder: {model_config.text_encoder_name} (type={model_config.text_encoder_type})")
    
    text_encoder_config = {
        "type": model_config.text_encoder_type,
        "model_name": model_config.text_encoder_name,
        "device": str(device),
    }
    
    text_encoder = build_text_encoder(text_encoder_config)
    
    return text_encoder


# ============================================================================
# Offline Error Measurement
# ============================================================================

def compute_offline_errors(
    encoder: nn.Module,
    goalhead: nn.Module,
    text_encoder: nn.Module,
    test_data: List[Dict[str, Any]],
    device: torch.device,
) -> Dict[str, float]:
    """
    Compute offline errors between predicted and oracle goal latents.
    
    Args:
        encoder: JEPA-WM encoder for extracting latents
        goalhead: Trained GoalHead model
        text_encoder: Text encoder for instructions
        test_data: List of test samples with {initial_image, goal_image, instruction}
        device: Computation device
        
    Returns:
        Dictionary with L2 distance and cosine similarity statistics
    """
    logger.info("📊 Computing offline errors...")
    
    l2_distances = []
    cosine_sims = []
    
    goalhead.eval()
    
    with torch.no_grad():
        for sample in tqdm(test_data, desc="Computing errors"):
            initial_image = sample["initial_image"].to(device)
            goal_image = sample["goal_image"].to(device)
            instruction = sample["instruction"]
            
            # Ensure images have correct shape [B, T, C, H, W]
            if initial_image.dim() == 4:
                initial_image = initial_image.unsqueeze(1)  # Add time dimension
            if goal_image.dim() == 4:
                goal_image = goal_image.unsqueeze(1)  # Add time dimension
            
            # Use encode_obs() to match training (not encode())
            # encode_obs() returns TensorDict with 'visual' key
            goal_obs_dict = {"visual": goal_image, "proprio": None}
            z_oracle_dict = encoder.encode_obs(goal_obs_dict)
            z_oracle_visual = z_oracle_dict["visual"]
            
            # Get visual features from initial image
            initial_obs_dict = {"visual": initial_image, "proprio": None}
            z_initial_dict = encoder.encode_obs(initial_obs_dict)
            z_initial_visual = z_initial_dict["visual"]
            
            # Get text features - encode returns tensor directly
            if isinstance(instruction, str):
                instruction = [instruction]
            text_features = text_encoder.encode(instruction)  # [1, text_dim]
            
            # Predict z_goal from GoalHead
            z_pred = goalhead(z_initial_visual, text_features)  # [1, output_dim]
            
            # Pool z_oracle_visual to match training:
            # Training uses z_goal_gt.mean(dim=(1, 2, 3, 4)) for [B, T, V, H, W, D] -> [B, D]
            if z_oracle_visual.dim() == 6:
                # [B, T, V, H, W, D] -> [B, D]
                z_oracle_pooled = z_oracle_visual.mean(dim=(1, 2, 3, 4))
            elif z_oracle_visual.dim() == 5:
                # [B, T, N, H, D] or similar -> [B, D]
                z_oracle_pooled = z_oracle_visual.mean(dim=(1, 2, 3))
            elif z_oracle_visual.dim() == 4:
                # [B, T, N, D] -> [B, D]
                z_oracle_pooled = z_oracle_visual.mean(dim=(1, 2))
            elif z_oracle_visual.dim() == 3:
                # [B, N, D] -> [B, D]
                z_oracle_pooled = z_oracle_visual.mean(dim=1)
            else:
                z_oracle_pooled = z_oracle_visual
            
            # Compute metrics
            l2_dist = l2_distance(z_pred, z_oracle_pooled).item()
            cos_sim = cosine_similarity(z_pred, z_oracle_pooled).item()
            
            l2_distances.append(l2_dist)
            cosine_sims.append(cos_sim)
    
    results = {
        "l2_distance_mean": float(np.mean(l2_distances)),
        "l2_distance_std": float(np.std(l2_distances)),
        "l2_distance_min": float(np.min(l2_distances)),
        "l2_distance_max": float(np.max(l2_distances)),
        "cosine_sim_mean": float(np.mean(cosine_sims)),
        "cosine_sim_std": float(np.std(cosine_sims)),
        "cosine_sim_min": float(np.min(cosine_sims)),
        "cosine_sim_max": float(np.max(cosine_sims)),
        "num_samples": len(test_data),
    }
    
    logger.info(f"📈 Offline errors computed:")
    logger.info(f"   L2 distance: {results['l2_distance_mean']:.4f} ± {results['l2_distance_std']:.4f}")
    logger.info(f"   Cosine sim:  {results['cosine_sim_mean']:.4f} ± {results['cosine_sim_std']:.4f}")
    
    return results


# ============================================================================
# CEM Planner (Simplified)
# ============================================================================

class SimpleCEMPlanner:
    """Simplified CEM planner for standalone evaluation."""
    
    def __init__(
        self,
        config: PlannerConfig,
        unroll_fn,
        device: torch.device,
    ):
        self.config = config
        self.unroll_fn = unroll_fn
        self.device = device
        
        self.horizon = config.horizon
        self.iterations = config.iterations
        self.num_samples = config.num_samples
        self.num_elites = config.num_elites
        self.action_dim = config.action_dim
        self.action_min = config.action_min
        self.action_max = config.action_max
        
        # Initialize action distribution
        self.mean = torch.zeros(self.horizon, self.action_dim, device=device)
        self.std = torch.ones(self.horizon, self.action_dim, device=device)
        
        self.target_z = None
    
    def set_goal(self, target_z: torch.Tensor):
        """Set the goal latent for planning."""
        self.target_z = target_z.to(self.device)
    
    def reset(self):
        """Reset the planner for a new episode."""
        self.mean = torch.zeros(self.horizon, self.action_dim, device=self.device)
        self.std = torch.ones(self.horizon, self.action_dim, device=self.device)
    
    def plan(self, z_current: torch.Tensor) -> torch.Tensor:
        """
        Plan optimal action sequence using CEM.
        
        Args:
            z_current: Current latent state
            
        Returns:
            First action in the optimal sequence
        """
        assert self.target_z is not None, "Goal must be set before planning"
        
        z_current = z_current.to(self.device)
        
        # CEM optimization loop
        for iteration in range(self.iterations):
            # Sample action sequences from current distribution
            # [num_samples, horizon, action_dim]
            noise = torch.randn(
                self.num_samples, self.horizon, self.action_dim,
                device=self.device
            )
            actions = self.mean.unsqueeze(0) + self.std.unsqueeze(0) * noise
            actions = actions.clamp(self.action_min, self.action_max)
            
            # Evaluate each action sequence
            costs = self._evaluate_sequences(z_current, actions)
            
            # Select elites (lowest cost)
            elite_indices = torch.argsort(costs)[:self.num_elites]
            elite_actions = actions[elite_indices]
            
            # Update distribution from elites
            self.mean = elite_actions.mean(dim=0)
            self.std = elite_actions.std(dim=0).clamp(min=0.01)
        
        # Return first action from mean sequence
        return self.mean[0].unsqueeze(0)
    
    def _evaluate_sequences(
        self,
        z_current: torch.Tensor,
        action_sequences: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate action sequences by unrolling world model.
        
        Args:
            z_current: Current latent [1, N, D]
            action_sequences: [num_samples, horizon, action_dim]
            
        Returns:
            costs: [num_samples] costs for each sequence
        """
        num_samples = action_sequences.shape[0]
        
        # Expand z_current for all samples
        z_expanded = z_current.expand(num_samples, -1, -1)  # [num_samples, N, D]
        
        # Unroll world model with actions
        # actions: [horizon, num_samples, action_dim]
        actions = action_sequences.permute(1, 0, 2)
        
        # Predict future states
        z_pred_seq = self.unroll_fn(z_expanded, actions)  # [horizon, num_samples, N, D]
        
        # Compute cost as distance to goal at final timestep
        z_final = z_pred_seq[-1]  # [num_samples, N, D]
        
        # Pool spatial dimensions
        z_final_pooled = z_final.mean(dim=1)  # [num_samples, D]
        target_pooled = self.target_z.mean(dim=1) if self.target_z.dim() > 2 else self.target_z
        target_pooled = target_pooled.expand(num_samples, -1)  # [num_samples, D]
        
        # Cost: negative cosine similarity (minimize cost = maximize similarity)
        costs = -cosine_similarity(z_final_pooled, target_pooled)
        
        return costs


# ============================================================================
# Online Evaluation
# ============================================================================

def create_metaworld_env(task_name: str, seed: int = 42):
    """Create MetaWorld environment for a specific task."""
    try:
        import metaworld
        
        ml1 = metaworld.ML1(task_name)
        env = ml1.train_classes[task_name]()
        task = ml1.train_tasks[0]
        env.set_task(task)
        env.seed(seed)
        
        return env
    except ImportError:
        logger.warning("MetaWorld not available. Online evaluation will be skipped.")
        return None


def run_online_evaluation(
    encoder: nn.Module,
    goalhead: nn.Module,
    text_encoder: nn.Module,
    config: EvalConfig,
    planner_config: PlannerConfig,
    task_instructions: Dict[str, str],
    device: torch.device,
) -> Dict[str, Any]:
    """
    Run online evaluation comparing Lang-Goal vs Oracle planning.
    
    Args:
        encoder: JEPA-WM encoder
        goalhead: Trained GoalHead model
        text_encoder: Text encoder
        config: Evaluation configuration
        planner_config: Planner configuration
        task_instructions: Dict mapping task names to instructions
        device: Computation device
        
    Returns:
        Dictionary with evaluation results
    """
    logger.info("🎮 Starting online evaluation...")
    
    results = {
        "tasks": {},
        "aggregate": {},
    }
    
    # Check if MetaWorld is available
    test_env = create_metaworld_env(config.tasks[0])
    if test_env is None:
        logger.warning("⚠️ MetaWorld not available. Skipping online evaluation.")
        return results
    test_env.close()
    
    # Placeholder unroll function (simplified for demonstration)
    def dummy_unroll(z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Placeholder unroll function - in real usage, use the JEPA-WM predictor."""
        # This is a simplified version - real implementation would use predictor
        horizon = actions.shape[0]
        num_samples = z.shape[0]
        
        # For demonstration: just return slightly modified z
        z_seq = []
        z_current = z
        for t in range(horizon):
            # Simple placeholder: z doesn't change much
            z_current = z_current + 0.01 * torch.randn_like(z_current)
            z_seq.append(z_current)
        
        return torch.stack(z_seq, dim=0)
    
    all_oracle_successes = []
    all_lang_successes = []
    all_oracle_steps = []
    all_lang_steps = []
    
    for task_name in tqdm(config.tasks, desc="Evaluating tasks"):
        task_results = {
            "oracle_successes": [],
            "lang_successes": [],
            "oracle_steps": [],
            "lang_steps": [],
        }
        
        instruction = task_instructions.get(task_name, f"Perform {task_name} task")
        
        for episode in range(config.episodes_per_task):
            seed = config.seed + episode
            
            # Create environment
            env = create_metaworld_env(task_name, seed)
            if env is None:
                continue
            
            # --- Oracle evaluation ---
            oracle_success, oracle_steps = evaluate_episode_oracle(
                env=env,
                encoder=encoder,
                planner_config=planner_config,
                unroll_fn=dummy_unroll,
                max_steps=config.max_episode_steps,
                device=device,
            )
            task_results["oracle_successes"].append(oracle_success)
            task_results["oracle_steps"].append(oracle_steps)
            
            # Reset environment for lang-goal evaluation
            env.close()
            env = create_metaworld_env(task_name, seed)
            
            # --- Lang-Goal evaluation ---
            lang_success, lang_steps = evaluate_episode_lang_goal(
                env=env,
                encoder=encoder,
                goalhead=goalhead,
                text_encoder=text_encoder,
                instruction=instruction,
                planner_config=planner_config,
                unroll_fn=dummy_unroll,
                max_steps=config.max_episode_steps,
                device=device,
            )
            task_results["lang_successes"].append(lang_success)
            task_results["lang_steps"].append(lang_steps)
            
            env.close()
        
        # Compute task statistics
        results["tasks"][task_name] = {
            "oracle_success_rate": np.mean(task_results["oracle_successes"]),
            "lang_success_rate": np.mean(task_results["lang_successes"]),
            "oracle_avg_steps": np.mean([s for s in task_results["oracle_steps"] if s is not None]),
            "lang_avg_steps": np.mean([s for s in task_results["lang_steps"] if s is not None]),
            "success_rate_delta": np.mean(task_results["lang_successes"]) - np.mean(task_results["oracle_successes"]),
        }
        
        all_oracle_successes.extend(task_results["oracle_successes"])
        all_lang_successes.extend(task_results["lang_successes"])
        all_oracle_steps.extend([s for s in task_results["oracle_steps"] if s is not None])
        all_lang_steps.extend([s for s in task_results["lang_steps"] if s is not None])
        
        logger.info(f"  {task_name}: Oracle={results['tasks'][task_name]['oracle_success_rate']:.2%}, "
                   f"Lang-Goal={results['tasks'][task_name]['lang_success_rate']:.2%}")
    
    # Aggregate results
    results["aggregate"] = {
        "oracle_success_rate": np.mean(all_oracle_successes) if all_oracle_successes else 0.0,
        "lang_success_rate": np.mean(all_lang_successes) if all_lang_successes else 0.0,
        "oracle_avg_steps": np.mean(all_oracle_steps) if all_oracle_steps else 0.0,
        "lang_avg_steps": np.mean(all_lang_steps) if all_lang_steps else 0.0,
        "success_rate_degradation": (
            (np.mean(all_oracle_successes) - np.mean(all_lang_successes)) / np.mean(all_oracle_successes) * 100
            if all_oracle_successes and np.mean(all_oracle_successes) > 0 else 0.0
        ),
        "steps_degradation": (
            (np.mean(all_lang_steps) - np.mean(all_oracle_steps)) / np.mean(all_oracle_steps) * 100
            if all_oracle_steps and np.mean(all_oracle_steps) > 0 else 0.0
        ),
    }
    
    logger.info("📊 Aggregate Results:")
    logger.info(f"   Oracle success rate:     {results['aggregate']['oracle_success_rate']:.2%}")
    logger.info(f"   Lang-Goal success rate:  {results['aggregate']['lang_success_rate']:.2%}")
    logger.info(f"   Success rate degradation: {results['aggregate']['success_rate_degradation']:.1f}%")
    
    return results


def evaluate_episode_oracle(
    env,
    encoder: nn.Module,
    planner_config: PlannerConfig,
    unroll_fn,
    max_steps: int,
    device: torch.device,
) -> Tuple[bool, Optional[int]]:
    """
    Evaluate one episode using oracle goal (from goal image).
    
    Returns:
        (success, steps_to_success)
    """
    obs = env.reset()
    
    # Get goal image from environment (if available)
    # For MetaWorld, we need to render the goal state
    # This is a simplified version - in practice, use expert demos
    goal_obs = obs.copy()  # Placeholder
    
    # Convert observation to image
    img = env.render(offscreen=True)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Encode initial state
    z_current = encoder.encode(img_tensor)
    
    # For oracle, z_goal would come from goal image
    # Here we use the initial state as placeholder
    z_goal = z_current.clone()  # Placeholder
    
    # Create planner
    planner = SimpleCEMPlanner(planner_config, unroll_fn, device)
    planner.set_goal(z_goal)
    
    success = False
    steps = None
    
    for step in range(max_steps):
        # Plan action
        action = planner.plan(z_current)
        action_np = action.squeeze().cpu().numpy()
        
        # Execute action
        obs, reward, done, info = env.step(action_np)
        
        # Check success
        if info.get('success', False):
            success = True
            steps = step + 1
            break
        
        # Update current state
        img = env.render(offscreen=True)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        z_current = encoder.encode(img_tensor)
    
    return success, steps


def evaluate_episode_lang_goal(
    env,
    encoder: nn.Module,
    goalhead: nn.Module,
    text_encoder: nn.Module,
    instruction: str,
    planner_config: PlannerConfig,
    unroll_fn,
    max_steps: int,
    device: torch.device,
) -> Tuple[bool, Optional[int]]:
    """
    Evaluate one episode using language-conditioned goal prediction.
    
    Returns:
        (success, steps_to_success)
    """
    obs = env.reset()
    
    # Render initial image
    img = env.render(offscreen=True)
    img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Encode initial state
    z_initial = encoder.encode(img_tensor)
    
    # Encode instruction
    text_features = text_encoder.encode(instruction)
    
    # Predict goal latent using GoalHead
    with torch.no_grad():
        z_goal = goalhead(z_initial, text_features)
    
    # Create planner
    planner = SimpleCEMPlanner(planner_config, unroll_fn, device)
    planner.set_goal(z_goal)
    
    z_current = z_initial
    success = False
    steps = None
    
    for step in range(max_steps):
        # Plan action
        action = planner.plan(z_current)
        action_np = action.squeeze().cpu().numpy()
        
        # Execute action
        obs, reward, done, info = env.step(action_np)
        
        # Check success
        if info.get('success', False):
            success = True
            steps = step + 1
            break
        
        # Update current state
        img = env.render(offscreen=True)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(device)
        z_current = encoder.encode(img_tensor)
    
    return success, steps


# ============================================================================
# Results Saving
# ============================================================================

def save_results(
    results: Dict[str, Any],
    output_dir: str,
    timestamp: str,
):
    """Save evaluation results to files."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save JSON
    json_path = os.path.join(output_dir, f"results_{timestamp}.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"💾 Results saved to {json_path}")
    
    # Save CSV summary
    csv_path = os.path.join(output_dir, f"summary_{timestamp}.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        
        # Write offline errors
        writer.writerow(["=== Offline Errors ==="])
        if "offline_errors" in results:
            for key, value in results["offline_errors"].items():
                writer.writerow([key, value])
        writer.writerow([])
        
        # Write aggregate results
        writer.writerow(["=== Aggregate Results ==="])
        if "online_eval" in results and "aggregate" in results["online_eval"]:
            for key, value in results["online_eval"]["aggregate"].items():
                writer.writerow([key, value])
        writer.writerow([])
        
        # Write task-wise results
        writer.writerow(["=== Task-wise Results ==="])
        writer.writerow(["Task", "Oracle SR", "Lang-Goal SR", "Delta", "Oracle Steps", "Lang Steps"])
        if "online_eval" in results and "tasks" in results["online_eval"]:
            for task, task_results in results["online_eval"]["tasks"].items():
                writer.writerow([
                    task,
                    f"{task_results.get('oracle_success_rate', 0):.4f}",
                    f"{task_results.get('lang_success_rate', 0):.4f}",
                    f"{task_results.get('success_rate_delta', 0):.4f}",
                    f"{task_results.get('oracle_avg_steps', 0):.1f}",
                    f"{task_results.get('lang_avg_steps', 0):.1f}",
                ])
    
    logger.info(f"📊 Summary saved to {csv_path}")


# ============================================================================
# Main
# ============================================================================

def get_task_instructions() -> Dict[str, str]:
    """Get language instructions for each MetaWorld task."""
    return {
        "assembly-v2": "Assemble the ring onto the peg",
        "bin-picking-v2": "Pick up the puck and place it in the bin",
        "button-press-v2": "Press the button on the box",
        "button-press-topdown-v2": "Press the button from above",
        "button-press-wall-v2": "Press the button on the wall",
        "coffee-button-v2": "Press the coffee machine button",
        "coffee-pull-v2": "Pull the coffee mug towards you",
        "coffee-push-v2": "Push the coffee mug away",
        "dial-turn-v2": "Turn the dial to the target position",
        "disassemble-v2": "Remove the ring from the peg",
        "door-close-v2": "Close the door",
        "door-lock-v2": "Lock the door",
        "door-open-v2": "Open the door",
        "door-unlock-v2": "Unlock the door",
        "drawer-close-v2": "Close the drawer",
        "drawer-open-v2": "Open the drawer",
        "faucet-close-v2": "Close the faucet",
        "faucet-open-v2": "Open the faucet",
        "hammer-v2": "Hammer the nail into the wall",
        "hand-insert-v2": "Insert your hand into the opening",
        "handle-press-side-v2": "Press the handle from the side",
        "handle-press-v2": "Press the handle downward",
        "handle-pull-side-v2": "Pull the handle from the side",
        "handle-pull-v2": "Pull the handle towards you",
        "lever-pull-v2": "Pull the lever towards you",
        "peg-insert-side-v2": "Insert the peg from the side",
        "peg-unplug-side-v2": "Unplug the peg from the side",
        "pick-out-of-hole-v2": "Pick the puck out of the hole",
        "pick-place-v2": "Pick up the puck and place it at the target",
        "pick-place-wall-v2": "Pick up the puck and place it over the wall",
        "plate-slide-back-side-v2": "Slide the plate back from the side",
        "plate-slide-back-v2": "Slide the plate backwards",
        "plate-slide-side-v2": "Slide the plate to the side",
        "plate-slide-v2": "Slide the plate forward",
        "push-back-v2": "Push the puck backwards",
        "push-v2": "Push the puck to the target position",
        "push-wall-v2": "Push the puck past the wall to the target",
        "reach-v2": "Reach the target position with your gripper",
        "reach-wall-v2": "Reach the target position over the wall",
        "shelf-place-v2": "Place the block on the shelf",
        "soccer-v2": "Push the soccer ball into the goal",
        "stick-pull-v2": "Use the stick to pull the box towards you",
        "stick-push-v2": "Use the stick to push the box away",
        "sweep-into-v2": "Sweep the cube into the hole",
        "sweep-v2": "Sweep the cube off the table",
        "window-close-v2": "Close the window",
        "window-open-v2": "Open the window",
    }


def create_dummy_test_data(
    num_samples: int,
    device: torch.device,
) -> List[Dict[str, Any]]:
    """Create dummy test data for offline error computation (placeholder)."""
    # In real usage, load actual test data from GoalHead dataset
    logger.warning("⚠️ Using dummy test data. Replace with actual GoalHead test dataset.")
    
    test_data = []
    for i in range(num_samples):
        test_data.append({
            "initial_image": torch.randn(1, 3, 224, 224),
            "goal_image": torch.randn(1, 3, 224, 224),
            "instruction": f"Perform task {i % 8}",
        })
    
    return test_data


def load_test_data_from_goalhead_dataset(
    index_dir: str,
    data_path: str,
    max_samples: Optional[int] = None,
    img_size: int = 224,
    split: str = "test",
) -> Tuple[List[Dict[str, Any]], Dict[str, List[Dict[str, Any]]]]:
    """
    Load test data from GoalHead dataset.
    
    Args:
        index_dir: Directory containing test_index.jsonl
        data_path: Path to MetaWorld data
        max_samples: Maximum number of samples to load (None = all)
        img_size: Target image size
        split: Which data split to use ("train", "val", "test")
        
    Returns:
        Tuple of (all_samples, task_wise_samples)
    """
    from pathlib import Path
    import torchvision.transforms as T
    
    index_path = Path(index_dir) / f"{split}_index.jsonl"
    
    if not index_path.exists():
        logger.warning(f"⚠️ Index not found at {index_path}")
        return [], {}
    
    # Create simple image transform (not video transform)
    transform = T.Compose([
        T.Resize((img_size, img_size)),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset without transform (we'll apply manually)
    from app.lang_goal.goalhead_dataset import GoalHeadDataset
    
    dataset = GoalHeadDataset(
        index_path=str(index_path),
        data_path=data_path,
        transform=None,  # No transform - we apply manually
        goal_frame="last",
        include_proprio=False,
    )
    
    # Convert to list format expected by compute_offline_errors
    all_samples = []
    task_wise_samples = {}
    
    num_samples = len(dataset) if max_samples is None else min(max_samples, len(dataset))
    
    logger.info(f"📂 Loading {num_samples} test samples from GoalHead dataset...")
    
    for i in tqdm(range(num_samples), desc="Loading test data"):
        sample = dataset[i]
        
        # Get raw images and apply transform
        I0 = sample["I0"]  # [C, H, W] (already normalized to 0-1)
        I_goal = sample["I_goal"]  # [C, H, W]
        
        # Apply resize and normalization
        I0 = transform(I0)
        I_goal = transform(I_goal)
        
        converted = {
            "initial_image": I0.unsqueeze(0),  # [1, C, H, W]
            "goal_image": I_goal.unsqueeze(0),  # [1, C, H, W]
            "instruction": sample["text"],
            "task_name": sample["task_name"],
            "episode_id": sample["episode_id"],
        }
        
        all_samples.append(converted)
        
        # Organize by task
        task_name = sample["task_name"]
        if task_name not in task_wise_samples:
            task_wise_samples[task_name] = []
        task_wise_samples[task_name].append(converted)
    
    logger.info(f"✅ Loaded {len(all_samples)} samples across {len(task_wise_samples)} tasks")
    
    return all_samples, task_wise_samples


def compute_task_wise_offline_errors(
    encoder: nn.Module,
    goalhead: nn.Module,
    text_encoder: nn.Module,
    task_wise_samples: Dict[str, List[Dict[str, Any]]],
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Compute offline errors per task for detailed analysis.
    
    Args:
        encoder: JEPA-WM encoder
        goalhead: Trained GoalHead model
        text_encoder: Text encoder
        task_wise_samples: Dict mapping task names to samples
        device: Computation device
        
    Returns:
        Dictionary with per-task error statistics
    """
    logger.info("📊 Computing task-wise offline errors...")
    
    task_results = {}
    
    goalhead.eval()
    
    for task_name, samples in tqdm(task_wise_samples.items(), desc="Tasks"):
        l2_distances = []
        cosine_sims = []
        
        with torch.no_grad():
            for sample in samples:
                initial_image = sample["initial_image"].to(device)
                goal_image = sample["goal_image"].to(device)
                instruction = sample["instruction"]
                
                # Ensure images have correct shape [B, T, C, H, W]
                if initial_image.dim() == 4:
                    initial_image = initial_image.unsqueeze(1)
                if goal_image.dim() == 4:
                    goal_image = goal_image.unsqueeze(1)
                
                # Use encode_obs() to match training (not encode())
                goal_obs_dict = {"visual": goal_image, "proprio": None}
                z_oracle_dict = encoder.encode_obs(goal_obs_dict)
                z_oracle_visual = z_oracle_dict["visual"]
                
                # Get visual features from initial image
                initial_obs_dict = {"visual": initial_image, "proprio": None}
                z_initial_dict = encoder.encode_obs(initial_obs_dict)
                z_initial_visual = z_initial_dict["visual"]
                
                # Get text features
                if isinstance(instruction, str):
                    instruction = [instruction]
                text_features = text_encoder.encode(instruction)
                
                # Predict z_goal from GoalHead
                z_pred = goalhead(z_initial_visual, text_features)
                
                # Pool z_oracle_visual to match training:
                # Training uses z_goal_gt.mean(dim=(1, 2, 3, 4)) for [B, T, V, H, W, D] -> [B, D]
                if z_oracle_visual.dim() == 6:
                    # [B, T, V, H, W, D] -> [B, D]
                    z_oracle_pooled = z_oracle_visual.mean(dim=(1, 2, 3, 4))
                elif z_oracle_visual.dim() == 5:
                    # [B, T, N, H, D] or similar -> [B, D]
                    z_oracle_pooled = z_oracle_visual.mean(dim=(1, 2, 3))
                elif z_oracle_visual.dim() == 4:
                    # [B, T, N, D] -> [B, D]
                    z_oracle_pooled = z_oracle_visual.mean(dim=(1, 2))
                elif z_oracle_visual.dim() == 3:
                    # [B, N, D] -> [B, D]
                    z_oracle_pooled = z_oracle_visual.mean(dim=1)
                else:
                    z_oracle_pooled = z_oracle_visual
                
                # Compute metrics
                l2_dist = l2_distance(z_pred, z_oracle_pooled).item()
                cos_sim = cosine_similarity(z_pred, z_oracle_pooled).item()
                
                l2_distances.append(l2_dist)
                cosine_sims.append(cos_sim)
        
        task_results[task_name] = {
            "l2_distance_mean": float(np.mean(l2_distances)),
            "l2_distance_std": float(np.std(l2_distances)),
            "cosine_sim_mean": float(np.mean(cosine_sims)),
            "cosine_sim_std": float(np.std(cosine_sims)),
            "num_samples": len(samples),
        }
        
        logger.info(f"  {task_name}: L2={task_results[task_name]['l2_distance_mean']:.4f}, "
                   f"cos={task_results[task_name]['cosine_sim_mean']:.4f} (n={len(samples)})")
    
    return task_results


def main():
    parser = argparse.ArgumentParser(description="Language-conditioned Goal Planning Evaluation")
    parser.add_argument("--jepa-checkpoint", type=str, 
                       default="/home/ryo52/checkpoints/jepa_lang_goal/pretrained/mw_jepa-wm.pth.tar",
                       help="Path to JEPA-WM checkpoint")
    parser.add_argument("--goalhead-checkpoint", type=str,
                       default="/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/goalhead/20260201_234248/best_model.pth",
                       help="Path to GoalHead checkpoint")
    parser.add_argument("--output-dir", type=str,
                       default="/home/ryo52/checkpoints/jepa_lang_goal/lang_goal/evaluation",
                       help="Output directory for results")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to use")
    parser.add_argument("--episodes", type=int, default=10, help="Episodes per task")
    parser.add_argument("--offline-only", action="store_true", help="Only run offline evaluation")
    parser.add_argument("--online-only", action="store_true", help="Only run online evaluation")
    parser.add_argument("--tasks", type=str, nargs="+", default=None, 
                       help="Specific tasks to evaluate")
    # Data paths for real test data
    parser.add_argument("--index-dir", type=str,
                       default="/home/ryo52/logs/jepa_lang_goal/lang_goal/index",
                       help="Directory containing test_index.jsonl")
    parser.add_argument("--data-path", type=str,
                       default="/home/ryo52/datasets/jepa_lang_goal/Metaworld/data",
                       help="Path to MetaWorld data")
    parser.add_argument("--use-real-data", action="store_true",
                       help="Use real test dataset instead of dummy data")
    parser.add_argument("--max-offline-samples", type=int, default=None,
                       help="Maximum number of offline samples to evaluate")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                       help="Which data split to use for evaluation")
    
    args = parser.parse_args()
    
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"🚀 Starting evaluation on {device}")
    
    # Create configs
    model_config = ModelConfig(
        jepa_wm_checkpoint=args.jepa_checkpoint,
        goalhead_checkpoint=args.goalhead_checkpoint,
    )
    
    eval_config = EvalConfig(
        output_dir=args.output_dir,
        episodes_per_task=args.episodes,
        device=str(device),
    )
    
    if args.tasks:
        eval_config.tasks = args.tasks
    
    planner_config = PlannerConfig()
    
    results = {
        "timestamp": timestamp,
        "config": {
            "model": model_config.__dict__,
            "eval": eval_config.__dict__,
            "planner": planner_config.__dict__,
        },
    }
    
    # Load models
    logger.info("=" * 60)
    logger.info("Loading models...")
    logger.info("=" * 60)
    
    try:
        encoder = load_jepa_wm_model(
            checkpoint_path=model_config.jepa_wm_checkpoint,
            device=device,
            img_size=model_config.img_size,
        )
    except Exception as e:
        logger.error(f"❌ Failed to load JEPA-WM encoder: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Continuing with placeholder encoder for demonstration...")
        encoder = None
        encoder = None
    
    try:
        goalhead = load_goalhead(
            checkpoint_path=model_config.goalhead_checkpoint,
            model_config=model_config,
            device=device,
        )
    except Exception as e:
        logger.error(f"❌ Failed to load GoalHead: {e}")
        logger.info("Continuing with placeholder GoalHead for demonstration...")
        goalhead = None
    
    try:
        text_encoder = load_text_encoder(
            model_config=model_config,
            device=device,
        )
    except Exception as e:
        logger.error(f"❌ Failed to load text encoder: {e}")
        import traceback
        traceback.print_exc()
        logger.info("Continuing without text encoder...")
        text_encoder = None
    
    # Run offline evaluation
    if not args.online_only:
        logger.info("=" * 60)
        logger.info("Running offline error measurement...")
        logger.info("=" * 60)
        
        if encoder is not None and goalhead is not None and text_encoder is not None:
            # Load test data
            if args.use_real_data:
                logger.info(f"📂 Loading real {args.split} data from GoalHead dataset...")
                all_samples, task_wise_samples = load_test_data_from_goalhead_dataset(
                    index_dir=args.index_dir,
                    data_path=args.data_path,
                    max_samples=args.max_offline_samples,
                    split=args.split,
                )
                
                if not all_samples:
                    logger.error("❌ Failed to load test data, falling back to dummy data")
                    test_data = create_dummy_test_data(100, device)
                    task_wise_samples = None
                else:
                    # Use all_samples directly (list of sample dicts)
                    test_data = all_samples
                    logger.info(f"✅ Loaded test data: {len(test_data)} samples")
            else:
                logger.info("📂 Using dummy test data for demonstration...")
                test_data = create_dummy_test_data(100, device)
                task_wise_samples = None
            
            # Compute offline errors
            offline_results = compute_offline_errors(
                encoder=encoder,
                goalhead=goalhead,
                text_encoder=text_encoder,
                test_data=test_data,
                device=device,
            )
            results["offline_errors"] = offline_results
            
            # Compute task-wise errors if using real data
            if args.use_real_data and task_wise_samples:
                logger.info("📊 Computing task-wise error statistics...")
                task_wise_results = compute_task_wise_offline_errors(
                    encoder=encoder,
                    goalhead=goalhead,
                    text_encoder=text_encoder,
                    task_wise_samples=task_wise_samples,
                    device=device,
                )
                results["task_wise_errors"] = task_wise_results
        else:
            logger.warning("⚠️ Skipping offline evaluation due to missing models")
    
    # Run online evaluation
    if not args.offline_only:
        logger.info("=" * 60)
        logger.info("Running online evaluation...")
        logger.info("=" * 60)
        
        if encoder is not None and goalhead is not None and text_encoder is not None:
            task_instructions = get_task_instructions()
            online_results = run_online_evaluation(
                encoder=encoder,
                goalhead=goalhead,
                text_encoder=text_encoder,
                config=eval_config,
                planner_config=planner_config,
                task_instructions=task_instructions,
                device=device,
            )
            results["online_eval"] = online_results
        else:
            logger.warning("⚠️ Skipping online evaluation due to missing models")
    
    # Save results
    logger.info("=" * 60)
    logger.info("Saving results...")
    logger.info("=" * 60)
    
    save_results(results, args.output_dir, timestamp)
    
    # Print summary
    logger.info("=" * 60)
    logger.info("Evaluation Complete!")
    logger.info("=" * 60)
    
    if "offline_errors" in results:
        logger.info("\n📊 Offline Evaluation Results:")
        logger.info(f"   L2 distance:   {results['offline_errors']['l2_distance_mean']:.4f} ± {results['offline_errors']['l2_distance_std']:.4f}")
        logger.info(f"   Cosine sim:    {results['offline_errors']['cosine_sim_mean']:.4f} ± {results['offline_errors']['cosine_sim_std']:.4f}")
        logger.info(f"   Samples:       {results['offline_errors']['num_samples']}")
    
    if "task_wise_errors" in results:
        logger.info("\n📊 Task-wise Error Breakdown:")
        for task_name, task_stats in results["task_wise_errors"].items():
            logger.info(f"   {task_name}:")
            logger.info(f"      L2: {task_stats['l2_distance_mean']:.4f} ± {task_stats['l2_distance_std']:.4f}")
            logger.info(f"      Cos: {task_stats['cosine_sim_mean']:.4f} ± {task_stats['cosine_sim_std']:.4f}")
            logger.info(f"      N: {task_stats['num_samples']}")
    
    if "online_eval" in results and "aggregate" in results["online_eval"]:
        agg = results["online_eval"]["aggregate"]
        logger.info("\n🎯 Online Evaluation Results:")
        logger.info(f"   Oracle success rate:    {agg.get('oracle_success_rate', 0):.2%}")
        logger.info(f"   Lang-Goal success rate: {agg.get('lang_success_rate', 0):.2%}")
        logger.info(f"   Success degradation:    {agg.get('success_rate_degradation', 0):.1f}%")
        logger.info(f"   Avg steps (Oracle):     {agg.get('oracle_avg_steps', 0):.1f}")
        logger.info(f"   Avg steps (Lang-Goal):  {agg.get('lang_avg_steps', 0):.1f}")
    
    logger.info(f"\n📁 Results saved to: {args.output_dir}/{timestamp}_results.json")


if __name__ == "__main__":
    main()
