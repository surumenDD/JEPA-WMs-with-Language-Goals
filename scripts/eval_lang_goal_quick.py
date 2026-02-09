#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Quick Language-conditioned Goal Planning Evaluation.

Simplified evaluation script for fast iteration on GoalHead improvements.
Uses the official JEPA-WM evaluation pipeline components.
"""

import os
import sys
import argparse
import yaml
import csv
from time import time
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf, open_dict

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from evals.utils import make_datasets, prepare_obs
from evals.simu_env_planning.envs.init import make_env
from evals.simu_env_planning.planning.gc_agent import GC_Agent
from evals.simu_env_planning.planning.plan_evaluator import PlanEvaluator
from evals.simu_env_planning.planning.planning.objectives import ReprTargetDistMPCObjective
from evals.simu_env_planning.planning.utils import make_td
from app.lang_goal.goal_head import build_goal_head
from app.lang_goal.text_encoder import build_text_encoder
from app.lang_goal.distillation import PromptTuningModule
from app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds import init_module
from src.utils.logging import get_logger

logger = get_logger(__name__)


def extract_proprio_from_obs_state(obs_state, proprio_dim: int = 4) -> np.ndarray:
    """
    Extract proprio (proprioceptive) features from environment observation state.
    
    For MetaWorld environments, the observation is typically a 39-dim array where:
    - First 4 dims: gripper position (x, y, z) + gripper opening
    - Or similar proprio information depending on the task
    
    Args:
        obs_state: Environment observation (numpy array or None)
        proprio_dim: Expected dimension of proprio features (default: 4)
        
    Returns:
        proprio: numpy array of shape [proprio_dim]
    """
    if obs_state is None:
        return np.zeros(proprio_dim, dtype=np.float32)
    
    # Convert to numpy if needed
    if isinstance(obs_state, torch.Tensor):
        obs_state = obs_state.cpu().numpy()
    elif not isinstance(obs_state, np.ndarray):
        try:
            obs_state = np.array(obs_state, dtype=np.float32)
        except (TypeError, ValueError):
            logger.warning(f"Cannot convert obs_state to numpy: {type(obs_state)}")
            return np.zeros(proprio_dim, dtype=np.float32)
    
    # Flatten if needed
    obs_state = obs_state.flatten().astype(np.float32)
    
    # Extract first proprio_dim elements
    if len(obs_state) >= proprio_dim:
        return obs_state[:proprio_dim]
    else:
        # Pad with zeros if obs_state is too short
        proprio = np.zeros(proprio_dim, dtype=np.float32)
        proprio[:len(obs_state)] = obs_state
        return proprio


def get_env_obs_state(env) -> np.ndarray:
    """
    Safely get the current observation state from the environment.
    
    Handles various wrapper structures and exception types.
    
    Args:
        env: Gymnasium/Gym environment (possibly wrapped)
        
    Returns:
        obs_state: numpy array or None if not available
    """
    obs_state = None
    
    # Try multiple ways to get the state
    try:
        # Method 1: get_wrapper_attr (gymnasium style)
        if hasattr(env, 'get_wrapper_attr'):
            obs_state = env.get_wrapper_attr('state')
    except (AttributeError, KeyError, RuntimeError) as e:
        logger.debug(f"get_wrapper_attr('state') failed: {e}")
    
    if obs_state is None:
        try:
            # Method 2: Direct attribute access
            if hasattr(env, 'state'):
                obs_state = env.state
        except (AttributeError, KeyError, RuntimeError) as e:
            logger.debug(f"env.state access failed: {e}")
    
    if obs_state is None:
        try:
            # Method 3: Unwrap and try again
            if hasattr(env, 'unwrapped') and hasattr(env.unwrapped, 'state'):
                obs_state = env.unwrapped.state
        except (AttributeError, KeyError, RuntimeError) as e:
            logger.debug(f"env.unwrapped.state access failed: {e}")
    
    if obs_state is None:
        try:
            # Method 4: Get observation from _get_obs() if available
            if hasattr(env, '_get_obs'):
                obs_state = env._get_obs()
            elif hasattr(env, 'unwrapped') and hasattr(env.unwrapped, '_get_obs'):
                obs_state = env.unwrapped._get_obs()
        except (AttributeError, KeyError, RuntimeError) as e:
            logger.debug(f"_get_obs() failed: {e}")
    
    return obs_state


# Task to text mapping
TASK_TEXT_MAP = {
    "mw-reach": "reach the red target",
    "mw-reach-wall": "reach the red target while avoiding the wall",
    "mw-push": "push the object to the target location",
    "mw-pick-place": "pick up the object and place it at the target",
}

# Text ensemble for each task (multiple paraphrases for robust embedding)
TASK_TEXT_ENSEMBLE = {
    "mw-reach": [
        "Move the robot gripper to the red target marker and touch it.",
        "Touch the red target marker on the table with the gripper tip.",
        "Reach the red target marker and make contact with the end-effector.",
        "The gripper tip is touching the red target marker on the tabletop.",
        "The robot end-effector reaches the red target marker and touches it.",
    ],
    "mw-reach-wall": [
        "Move the robot gripper to the red target marker and touch it without touching the wall.",
        "Touch the red target marker with the gripper tip while avoiding contact with the wall.",
        "Reach the red target marker and make contact, and do not collide with the wall obstacle.",
        "The gripper tip is touching the red target marker and the wall is not touched.",
        "The robot end-effector reaches and touches the red target marker with no wall contact.",
    ],
    "mw-push": [
        "Push the object to the target location with the gripper.",
        "Use the robot gripper to push the object towards the goal position.",
        "Move the object by pushing it to the designated target spot.",
        "The gripper pushes the object until it reaches the target area.",
        "Push the block to the red target marker on the table.",
    ],
    "mw-pick-place": [
        "Pick up the object and place it at the target location.",
        "Grasp the object with the gripper and move it to the goal position.",
        "Lift the object and set it down at the designated target spot.",
        "The gripper picks up the block and places it on the target.",
        "Pick and place the object to the red target marker.",
    ],
}


def load_goalhead(checkpoint_path: str, device: torch.device):
    """Load trained GoalHead model and optional PromptTuningModule."""
    logger.info(f"📥 Loading GoalHead from {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # Get config from checkpoint
    config = checkpoint.get("config", {})
    goalhead_config = config.get("goal_head", {})
    
    # Build and load GoalHead
    goal_head = build_goal_head(goalhead_config)
    goal_head.load_state_dict(checkpoint["goal_head_state_dict"])
    goal_head = goal_head.to(device)
    goal_head.eval()
    
    # Load PromptTuningModule if saved in checkpoint
    prompt_tuning = None
    if "prompt_tuning_state_dict" in checkpoint:
        pt_config = config.get("prompt_tuning", {})
        text_encoder_config = config.get("text_encoder", {})
        # Determine embed_dim from text encoder config
        text_type = text_encoder_config.get("type", "sentence_transformer")
        if text_type == "siglip2":
            embed_dim = goalhead_config.get("text_dim", 1024)
        elif text_type == "clip":
            embed_dim = goalhead_config.get("text_dim", 512)
        else:
            embed_dim = goalhead_config.get("text_dim", 384)
        
        num_context_tokens = pt_config.get("num_context_tokens", 16)
        # Match training code key resolution:
        #   train_goalhead.py uses: prompt_cfg.get("context_position", "prefix")
        #   But training config may use 'position' key instead of 'context_position'
        # Priority: context_position > position > default "prefix"
        position = pt_config.get("context_position", 
                                  pt_config.get("position", "prefix"))
        # Map legacy values to PromptTuningModule's valid values
        # PromptTuningModule accepts: "prefix", "suffix", "both"
        position_map = {"prepend": "prefix", "append": "suffix"}
        position = position_map.get(position, position)
        
        prompt_tuning = PromptTuningModule(
            embed_dim=embed_dim,
            num_context_tokens=num_context_tokens,
            context_position=position,
        )
        prompt_tuning.load_state_dict(checkpoint["prompt_tuning_state_dict"])
        prompt_tuning = prompt_tuning.to(device)
        prompt_tuning.eval()
        logger.info(f"✅ PromptTuningModule loaded ({num_context_tokens} tokens, position={position})")
    
    # Get training info
    epoch = checkpoint.get("epoch", "unknown")
    best_loss = checkpoint.get("best_val_loss", "unknown")
    
    is_patch_level = goalhead_config.get("patch_level", False)
    is_improved = goalhead_config.get("version", "v1") == "v2" or goalhead_config.get("use_improved", False)
    
    if isinstance(best_loss, (int, float)):
        logger.info(f"✅ GoalHead loaded (epoch={epoch}, val_loss={best_loss:.4f}, patch_level={is_patch_level}, improved={is_improved})")
    else:
        logger.info(f"✅ GoalHead loaded (epoch={epoch}, val_loss={best_loss}, patch_level={is_patch_level}, improved={is_improved})")
    
    return goal_head, goalhead_config, prompt_tuning


def load_text_encoder(device: torch.device, config: dict = None):
    """Load text encoder based on config or use default."""
    if config is None:
        config = {
            "type": "sentence_transformer",
            "model_name": "all-MiniLM-L6-v2",
            "device": str(device),
        }
    else:
        config["device"] = str(device)
    
    text_encoder = build_text_encoder(config)
    logger.info("✅ Text encoder loaded")
    return text_encoder


@torch.no_grad()
def compute_goal_latent_from_language(
    goal_head,
    text_encoder,
    world_model,  # Changed from encoder to world_model for consistency with training
    initial_obs_visual: torch.Tensor,
    text_instruction: str,
    device: torch.device,
    task: str = None,
    use_ensemble: bool = False,
    prompt_tuning=None,
    proprio: Optional[torch.Tensor] = None,
):
    """
    Compute goal latent from initial observation and text instruction.
    
    IMPORTANT: Uses encode_obs() to match training path exactly.
    This ensures the same representation space is used for both training and evaluation.
    
    Args:
        world_model: JEPA-WM model with encode_obs() method
        initial_obs_visual: [B, T, C, H, W] normalized visual observation
        text_instruction: Language instruction (used if use_ensemble=False)
        task: Task name for ensemble lookup
        use_ensemble: Whether to use text ensemble
        prompt_tuning: Optional PromptTuningModule to apply to text features
        proprio: Optional proprio tensor [B, proprio_dim] if GoalHead uses proprio
        
    Returns:
        z_goal: [B, D] or [B, N, D] goal latent (depending on GoalHead type)
        z_init: [B, N, D] encoded initial observation (last timestep, flattened patches)
    """
    from einops import rearrange
    
    # Encode initial observation using encode_obs() to match training path
    # encode_obs() expects [B, T, C, H, W] and returns visual: [B, T, V, H, W, D]
    obs_dict = {"visual": initial_obs_visual, "proprio": None}
    visual_features = world_model.encode_obs(obs_dict)["visual"]  # [B, T, V, H, W, D]
    
    # Flatten to [B, N, D] where N = T * V * H * W (for last timestep: N = V * H * W = 1 * 16 * 16 = 256)
    z_init_full = rearrange(visual_features, "b t v h w d -> b (t v h w) d")
    
    # Encode text
    if use_ensemble and task and task in TASK_TEXT_ENSEMBLE:
        # Use ensemble of paraphrases
        if hasattr(text_encoder, 'encode_ensemble'):
            text_features = text_encoder.encode_ensemble(TASK_TEXT_ENSEMBLE[task])
        else:
            # Fallback: manual ensemble
            texts = TASK_TEXT_ENSEMBLE[task]
            embeddings = text_encoder.encode(texts)  # [N, dim]
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            text_features = embeddings.mean(dim=0, keepdim=True)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = text_encoder.encode([text_instruction])
    text_features = text_features.to(device)  # [1, text_dim]
    
    # Apply prompt tuning if available (must match training path)
    if prompt_tuning is not None:
        text_features = prompt_tuning(text_features)
    
    # Prepare input for GoalHead
    # Check if this is a PatchLevelGoalHead by checking for patch_pos_embed attribute
    is_patch_level = hasattr(goal_head, 'patch_pos_embed')
    # Check if this is an ImprovedGoalHead (v2) which accepts [B, N, D] but outputs [B, D]
    is_improved = hasattr(goal_head, 'query_pool') or hasattr(goal_head, 'visual_pool')
    
    # Get the last timestep's patches for GoalHead input
    # z_init_full is [B, T*V*H*W, D], we need last timestep: [B, V*H*W, D] = [B, 256, D]
    b, t, v = initial_obs_visual.shape[0], initial_obs_visual.shape[1], 1  # V=1 view
    h = w = 16  # 16x16 patches for 224x224 image with patch_size=14
    num_patches_per_frame = v * h * w  # 256
    z_init = z_init_full[:, -num_patches_per_frame:, :]  # [B, 256, D] - last timestep
    
    if is_patch_level:
        # PatchLevelGoalHead expects [B, N, D]
        z_goal = goal_head(z_init, text_features, proprio)  # [B, N, D]
    elif is_improved:
        # ImprovedGoalHead: match training path - pool to [B, D] then forward
        # Training code uses: visual_input = visual_features.mean(dim=(1, 2, 3, 4))
        z_init_pooled = z_init.mean(dim=1)  # [B, D]
        z_goal = goal_head(z_init_pooled, text_features, proprio)  # [B, D]
    else:
        # Original GoalHead expects pooled features
        z_init_pooled = z_init.mean(dim=1)  # [B, D]
        z_goal = goal_head(z_init_pooled, text_features, proprio)  # [B, D]
    
    return z_goal, z_init


@torch.no_grad()
def compute_goal_latent_from_image(world_model, goal_obs_visual: torch.Tensor):
    """
    Compute goal latent from goal image (Oracle).
    
    IMPORTANT: Uses encode_obs() to match training path exactly.
    
    Args:
        world_model: JEPA-WM model with encode_obs() method
        goal_obs_visual: [B, T, C, H, W] normalized goal visual observation
        
    Returns:
        z_goal: [B, num_patches, D] goal latent (from last frame)
    """
    from einops import rearrange
    
    # Use encode_obs() to match training path
    obs_dict = {"visual": goal_obs_visual, "proprio": None}
    visual_features = world_model.encode_obs(obs_dict)["visual"]  # [B, T, V, H, W, D]
    
    # Flatten to [B, N, D] for last timestep
    z_goal_full = rearrange(visual_features, "b t v h w d -> b (t v h w) d")
    
    # Get last timestep patches
    b, t = goal_obs_visual.shape[0], goal_obs_visual.shape[1]
    h = w = 16  # 16x16 patches
    num_patches_per_frame = h * w  # 256
    z_goal = z_goal_full[:, -num_patches_per_frame:, :]  # [B, 256, D]
    
    return z_goal


def run_evaluation(
    cfg,
    model,
    agent: GC_Agent,
    env,
    evaluator: PlanEvaluator,
    goal_head,
    text_encoder,
    preprocessor,
    text_instruction: str,
    num_episodes: int,
    device: torch.device,
    output_dir: Path,
    use_oracle: bool = False,
    prompt_tuning=None,
):
    """Run full evaluation.
    
    Args:
        use_oracle: If True, use Oracle z_goal (from goal image) instead of GoalHead prediction.
        prompt_tuning: Optional PromptTuningModule to apply to text features.
    """
    
    results = []
    
    for ep_idx in range(num_episodes):
        ep_seed = cfg.meta.seed + ep_idx
        
        logger.info(f"\n{'='*50}")
        logger.info(f"📍 Episode {ep_idx + 1}/{num_episodes} (seed={ep_seed})")
        logger.info(f"{'='*50}")
        
        ep_start_time = time()
        
        # Step 1: Get expert goal (for comparison and Oracle baseline)
        try:
            # Update env config seed to match current episode seed so that Expert env is created correctly
            with open_dict(env.cfg):
                env.cfg.meta.seed = int(ep_seed)

            init_obs, goal_obs, expert_obses, expert_success = evaluator.set_episode(
                ep_seed=ep_seed,
                cfg=cfg,
                env=env,
                agent=agent,
                task_idx=0,  # Use index 0 for single-task evaluation
            )
            logger.info(f"✅ Expert rollout complete (success={expert_success})")
        except Exception as e:
            logger.error(f"❌ Failed to get expert goal: {e}")
            continue
        
        # Step 2: Compute goal latents
        # Prepare observations
        init_visual = init_obs["visual"].float() / 255.0  # [T, C, H, W]
        init_visual = preprocessor.transform(init_visual.unsqueeze(0)).to(device)  # [1, T, C, H, W]
        
        goal_visual = goal_obs["visual"].float() / 255.0
        goal_visual = preprocessor.transform(goal_visual.unsqueeze(0)).to(device)
        
        # Oracle: z_goal from goal image (always compute for comparison)
        # Use world_model.encode_obs() to match training path
        z_goal_oracle = compute_goal_latent_from_image(
            world_model=model,
            goal_obs_visual=goal_visual,
        )
        # z_goal_oracle shape: [B, num_patches, D] = [1, 256, 384]
        
        # Determine which z_goal to use for planning
        if use_oracle:
            # Oracle mode: use z_goal directly from goal image (patch-wise)
            num_patches = z_goal_oracle.shape[1]  # 256 = 16 * 16
            grid_size = int(num_patches ** 0.5)  # 16
            embed_dim = z_goal_oracle.shape[-1]  # 384
            
            # Reshape to world model format: [B, T, V, H, W, D]
            z_goal_expanded = z_goal_oracle.view(1, 1, 1, grid_size, grid_size, embed_dim)
            
            cos_sim = 1.0  # Oracle is identical to itself
            l2_dist = 0.0
            
            logger.info(f"🔮 Using Oracle z_goal (from goal image)")
            logger.info(f"   z_goal_expanded shape: {z_goal_expanded.shape}")
        else:
            # Lang-Goal mode: z_goal from GoalHead
            # Check if goal_head is patch-level
            is_patch_level = hasattr(goal_head, 'patch_pos_embed')
            task = cfg.task_specification.task
            
            # Prepare proprio if GoalHead uses it (match training path)
            goalhead_proprio = None
            if hasattr(goal_head, 'use_proprio') and goal_head.use_proprio:
                init_proprio = init_obs.get('proprio', None)
                if init_proprio is not None:
                    if isinstance(init_proprio, torch.Tensor):
                        goalhead_proprio = init_proprio.float().unsqueeze(0).to(device)  # [1, proprio_dim]
                    else:
                        goalhead_proprio = torch.tensor(
                            init_proprio, dtype=torch.float32
                        ).unsqueeze(0).to(device)
                    logger.info(f"   Using proprio for GoalHead: shape={goalhead_proprio.shape}")
            
            z_goal_lang, z_init = compute_goal_latent_from_language(
                goal_head=goal_head,
                text_encoder=text_encoder,
                world_model=model,  # Use world_model for encode_obs() to match training
                initial_obs_visual=init_visual,
                text_instruction=text_instruction,
                device=device,
                task=task,
                use_ensemble=True,  # Use text ensemble for robust embedding
                prompt_tuning=prompt_tuning,
                proprio=goalhead_proprio,
            )
            
            # Compute similarity between Lang-Goal and Oracle z_goals
            z_goal_oracle_pooled = z_goal_oracle.mean(dim=1)  # [B, D]
            if is_patch_level:
                # z_goal_lang is [B, N, D], pool for comparison
                z_goal_lang_pooled = z_goal_lang.mean(dim=1)  # [B, D]
            else:
                z_goal_lang_pooled = z_goal_lang
            
            cos_sim = torch.nn.functional.cosine_similarity(
                z_goal_lang_pooled, z_goal_oracle_pooled, dim=-1
            ).item()
            l2_dist = torch.norm(z_goal_lang_pooled - z_goal_oracle_pooled, dim=-1).item()
            
            logger.info(f"🎯 Goal latent comparison (patch_level={is_patch_level}):")
            logger.info(f"   Cosine similarity: {cos_sim:.4f}")
            logger.info(f"   L2 distance: {l2_dist:.4f}")
            
            # z_init is already [B, N, D] from compute_goal_latent_from_language
            num_patches = z_init.shape[1]  # 256 = 16 * 16
            grid_size = int(num_patches ** 0.5)  # 16
            embed_dim = z_init.shape[-1]  # 384
            
            if is_patch_level:
                # PatchLevelGoalHead: z_goal_lang is already [B, N, D]
                z_goal_expanded = z_goal_lang.view(1, 1, 1, grid_size, grid_size, embed_dim)
                logger.info(f"   Goal expansion: patch-level GoalHead output (direct)")
            else:
                # Original GoalHead: use delta-shift method
                # Compute delta between GoalHead prediction and initial observation's pooled embedding
                z_init_pooled = z_init.mean(dim=1)  # [1, D]
                delta = z_goal_lang - z_init_pooled  # [1, D] - direction from init to goal
                
                # Apply delta to each patch of initial observation
                z_goal_patches = z_init + delta.unsqueeze(1)  # [1, N, D]
                
                # Reshape to world model format: [B, T, V, H, W, D]
                z_goal_expanded = z_goal_patches.view(1, 1, 1, grid_size, grid_size, embed_dim)
                
                logger.info(f"   Goal expansion: init patches shifted by delta (norm={delta.norm().item():.4f})")
        
        # Set up agent with Lang-Goal z_goal embedding (this also sets up the objective)
        agent.set_goal_from_embedding(z_goal_visual=z_goal_expanded, z_goal_proprio=None)
        
        # NOTE: Do NOT reset environment here - evaluator.set_episode() already reset the env
        # and set up the correct initial state. Resetting again would create state mismatch
        # between the goal latent (computed from init_obs) and the actual environment state.
        
        # Run episode
        episode_reward = 0.0
        episode_success = 0
        step = 0
        max_steps = cfg.task_specification.max_episode_steps
        
        # Track near_object (distance to target) for analysis
        near_object_values = []
        min_near_object = float('inf')
        
        # Get initial observation from the already-reset environment (via render)
        render_img = env.render()
        obs_visual_init = torch.from_numpy(render_img.copy()).permute(2, 0, 1).unsqueeze(0)
        
        # Get initial proprio from environment state
        # Priority: obs_state from env > init_obs['proprio'] > zeros
        obs_state = get_env_obs_state(env)
        
        # Determine proprio_dim from init_obs if available
        init_proprio = init_obs.get('proprio', None)
        if init_proprio is not None:
            if isinstance(init_proprio, torch.Tensor):
                proprio_dim = init_proprio.numel()
            else:
                proprio_dim = len(np.array(init_proprio).flatten())
        else:
            proprio_dim = 4  # Default for MetaWorld
        
        # Extract proprio from obs_state, fallback to init_obs
        if obs_state is not None:
            current_proprio = extract_proprio_from_obs_state(obs_state, proprio_dim)
        elif init_proprio is not None:
            if isinstance(init_proprio, torch.Tensor):
                current_proprio = init_proprio.cpu().numpy().flatten()[:proprio_dim]
            else:
                current_proprio = np.array(init_proprio, dtype=np.float32).flatten()[:proprio_dim]
        else:
            current_proprio = np.zeros(proprio_dim, dtype=np.float32)
        
        info = {'proprio': current_proprio}
        
        pbar = tqdm(total=max_steps, desc=f"Ep {ep_idx}", leave=False)
        
        planning_times = []
        steps_left = max_steps
        
        while step < max_steps:
            # Get visual observation by rendering
            # Note: encode() expects raw pixel values (0-255) and applies transform internally
            render_img = env.render()  # H, W, 3, uint8 (0-255)
            # Convert to [C, H, W] tensor (keep uint8 scale, encode() will handle /255 and transform)
            obs_visual = torch.from_numpy(render_img.copy()).permute(2, 0, 1)  # [C, H, W] uint8
            # Add time dimension: [T, C, H, W] where T=1
            obs_visual = obs_visual.unsqueeze(0)  # [1, C, H, W]
            
            # Get proprio from info
            proprio_data = info.get("proprio", np.zeros(4, dtype=np.float32))
            if isinstance(proprio_data, torch.Tensor):
                proprio_tensor = proprio_data.float()
            else:
                proprio_tensor = torch.from_numpy(np.array(proprio_data)).float()
            
            # Ensure proprio is [T, P] format where T=1
            # info["proprio"] might already have extra dimensions
            if proprio_tensor.dim() == 1:
                # [P] -> [1, P] (add time dim)
                proprio_tensor = proprio_tensor.unsqueeze(0)
            elif proprio_tensor.dim() == 2:
                # Could be [T, P] or [1, 4], keep as is (T=1)
                pass
            elif proprio_tensor.dim() > 2:
                # Flatten extra dims: [1, 1, 4] -> [1, 4]
                proprio_tensor = proprio_tensor.squeeze()
                if proprio_tensor.dim() == 1:
                    proprio_tensor = proprio_tensor.unsqueeze(0)
            
            # Create TensorDict for agent.act()
            # Expected shapes after make_td: visual [T, C, H, W], proprio [T, P]
            # agent.act() will unsqueeze(0) to add batch dim: visual [B, T, C, H, W], proprio [B, T, P]
            obs_td = make_td(obs_visual, {"proprio": proprio_tensor})
            
            # Plan using agent.act()
            plan_start = time()
            action = agent.act(
                prepare_obs(cfg.task_specification.obs, obs_td),
                steps_left=steps_left
            )
            plan_time = time() - plan_start
            planning_times.append(plan_time)
            
            # Rearrange action: [T, F*D] -> [(T*F), D] where D=env action dim (4)
            # action shape from planner: [num_act_stepped, 20] -> need to split into individual env actions
            from einops import rearrange as einops_rearrange
            env_action_dim = 4  # Metaworld uses 4D actions
            action = einops_rearrange(action.cpu(), "t (f d) -> (t f) d", d=env_action_dim)
            # Now action is [(num_act * frameskip), 4] - multiple environment actions
            
            # Denormalize the flattened actions
            action = preprocessor.denormalize_actions(action)
            
            # Execute each action in the flattened sequence
            num_actions_to_execute = action.shape[0]
            for act_idx in range(num_actions_to_execute):
                if step >= max_steps:
                    break
                
                # Keep as tensor for TensorWrapper, which will call .numpy()
                act = action[act_idx]
                
                obs_state, reward, done, truncated, info = env.step(act)
                
                # Update proprio from obs_state (more reliable than info)
                step_proprio = extract_proprio_from_obs_state(obs_state, proprio_dim)
                info['proprio'] = step_proprio
                
                # Track near_object (distance to target)
                near_obj = info.get("near_object", info.get("distance", None))
                if near_obj is not None:
                    near_object_values.append(float(near_obj))
                    min_near_object = min(min_near_object, float(near_obj))
                
                episode_reward += reward
                step += 1
                steps_left = max_steps - step
                
                if info.get("success", False):
                    episode_success = 1
                
                pbar.update(1)
                pbar.set_postfix({
                    "succ": episode_success,
                    "r": f"{episode_reward:.0f}",
                    "near": f"{near_obj:.3f}" if near_obj is not None else "N/A",
                })
                
                if done or truncated or episode_success:
                    break
            
            if done or truncated or episode_success:
                break
        
        pbar.close()
        
        ep_time = time() - ep_start_time
        avg_plan_time = np.mean(planning_times) if planning_times else 0
        
        # Compute near_object statistics
        if near_object_values:
            final_near_object = near_object_values[-1]
            avg_near_object = np.mean(near_object_values)
            min_near_object_final = min_near_object
        else:
            final_near_object = None
            avg_near_object = None
            min_near_object_final = None
        
        # Record results
        ep_result = {
            "episode": ep_idx,
            "seed": ep_seed,
            "success": episode_success,
            "reward": episode_reward,
            "steps": step,
            "goal_cos_sim": cos_sim,
            "goal_l2_dist": l2_dist,
            "episode_time": ep_time,
            "avg_plan_time": avg_plan_time,
            "expert_success": expert_success,
            "near_object_min": min_near_object_final,
            "near_object_final": final_near_object,
            "near_object_avg": avg_near_object,
        }
        results.append(ep_result)
        
        logger.info(f"📊 Episode {ep_idx} results:")
        logger.info(f"   Success: {episode_success}, Reward: {episode_reward:.1f}, Steps: {step}")
        if min_near_object_final is not None:
            logger.info(f"   📏 Near object: min={min_near_object_final:.4f}, final={final_near_object:.4f}, avg={avg_near_object:.4f}")
            if min_near_object_final < 0.05:
                logger.info(f"   ✅ Very close to target! (threshold ~0.05)")
            elif min_near_object_final < 0.1:
                logger.info(f"   ⚠️  Close but not quite (threshold ~0.05)")
            else:
                logger.info(f"   ❌ Far from target - model may need improvement")
        logger.info(f"   Time: {ep_time:.1f}s, Avg plan time: {avg_plan_time:.1f}s")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Quick Lang-Goal Evaluation")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--goalhead_ckpt", type=str, required=False, default=None)
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--text", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use-oracle", action="store_true", 
                        help="Use Oracle z_goal (from goal image) instead of GoalHead prediction")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        cfg = OmegaConf.create(yaml.safe_load(f))
    
    # Expand environment variables
    def expand_env_vars_recursive(obj):
        if isinstance(obj, str):
            return os.path.expandvars(obj)
        elif isinstance(obj, dict):
            return {k: expand_env_vars_recursive(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [expand_env_vars_recursive(v) for v in obj]
        return obj
    
    cfg = OmegaConf.create(expand_env_vars_recursive(OmegaConf.to_container(cfg)))
    
    # Override with args
    cfg.meta.eval_episodes = args.episodes
    if args.task:
        cfg.task_specification.task = args.task
    
    task = cfg.task_specification.task
    text_instruction = args.text or TASK_TEXT_MAP.get(task, f"complete the {task} task")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Device: {device}")
    
    # Output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(f"outputs/lang_goal_eval/{task}_{args.episodes}ep_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"📁 Output: {output_dir}")
    logger.info(f"📝 Task: {task}")
    
    # Check if text ensemble will be used
    if task in TASK_TEXT_ENSEMBLE:
        ensemble_texts = TASK_TEXT_ENSEMBLE[task]
        logger.info(f"💬 Text Ensemble: {len(ensemble_texts)} paraphrases (averaged)")
        for i, t in enumerate(ensemble_texts):
            logger.info(f"   [{i+1}] '{t}'")
    else:
        logger.info(f"💬 Text: '{text_instruction}'")
    logger.info(f"📊 Episodes: {args.episodes}")
    
    # Initialize models
    logger.info("\n🔧 Initializing models...")
    
    # Get model kwargs
    model_kwargs = cfg.model_kwargs
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})
    
    # Dataset and preprocessor
    dset, preprocessor = make_datasets(cfgs_data, cfgs_data_aug)
    
    # Add frameskip to cfg for plan_evaluator
    cfg.frameskip = cfgs_data["custom"]["frameskip"]
    cfg.work_dir = str(output_dir)
    
    # World model
    checkpoint = model_kwargs.get("checkpoint")
    checkpoint_folder = str(Path(checkpoint).parent) if checkpoint else None
    
    model = init_module(
        folder=checkpoint_folder,
        checkpoint=checkpoint,
        model_kwargs=pretrain_kwargs,
        wrapper_kwargs=wrapper_kwargs,
        cfgs_data=cfgs_data,
        device=device,
        action_dim=dset.action_dim,
        proprio_dim=dset.proprio_dim,
        preprocessor=preprocessor,
    )
    model.eval()
    logger.info("✅ World model loaded")
    
    # Resolve goalhead checkpoint: CLI arg > config field
    goalhead_ckpt_path = args.goalhead_ckpt or cfg.get("goalhead_checkpoint", None)
    
    # GoalHead (only needed if not using Oracle)
    if args.use_oracle:
        goal_head = None
        text_encoder = None
        goalhead_config = None
        prompt_tuning = None
        logger.info("🔮 Using Oracle mode (z_goal from goal image)")
    else:
        if not goalhead_ckpt_path:
            raise ValueError("--goalhead_ckpt is required (or set goalhead_checkpoint in config)")
        goal_head, goalhead_config, prompt_tuning = load_goalhead(goalhead_ckpt_path, device)
        
        # Load text encoder from GoalHead checkpoint config
        checkpoint = torch.load(goalhead_ckpt_path, map_location="cpu")
        text_encoder_config = checkpoint.get("config", {}).get("text_encoder", None)
        text_encoder = load_text_encoder(device, text_encoder_config)
    
    # Set local_seed for agent initialization (required by GC_Agent)
    cfg.local_seed = cfg.meta.seed
    
    # Ensure task_specification has all required fields
    with open_dict(cfg):
        if not cfg.task_specification.get("multitask", None):
            cfg.task_specification.multitask = False
        # tasks list is required for expert goal source (used by get_goal_state_mw)
        if not cfg.get("tasks", None):
            cfg.tasks = [cfg.task_specification.task]  # e.g., ["mw-reach"]
    
    # Agent
    agent = GC_Agent(cfg, model, dset, preprocessor)
    
    # Environment
    env = make_env(cfg)
    
    # Evaluator (takes only cfg and agent)
    evaluator = PlanEvaluator(cfg, agent)
    
    # Run evaluation
    logger.info("\n" + "=" * 60)
    if args.use_oracle:
        logger.info("🚀 Starting ORACLE Evaluation (diagnostic)")
    else:
        logger.info("🚀 Starting LANG-GOAL Evaluation")
    logger.info("=" * 60)
    
    eval_start = time()
    
    results = run_evaluation(
        cfg=cfg,
        model=model,
        agent=agent,
        env=env,
        evaluator=evaluator,
        goal_head=goal_head,
        text_encoder=text_encoder,
        preprocessor=preprocessor,
        text_instruction=text_instruction,
        num_episodes=args.episodes,
        device=device,
        output_dir=output_dir,
        use_oracle=args.use_oracle,
        prompt_tuning=prompt_tuning,
    )
    
    eval_time = time() - eval_start
    
    # Compute summary
    if results:
        n = len(results)
        success_rate = sum(r["success"] for r in results) / n
        avg_reward = sum(r["reward"] for r in results) / n
        avg_cos_sim = sum(r["goal_cos_sim"] for r in results) / n
        avg_l2_dist = sum(r["goal_l2_dist"] for r in results) / n
        
        logger.info("\n" + "=" * 60)
        logger.info("📊 FINAL RESULTS")
        logger.info("=" * 60)
        logger.info(f"Task: {task}")
        if task in TASK_TEXT_ENSEMBLE:
            logger.info(f"Text: Ensemble of {len(TASK_TEXT_ENSEMBLE[task])} paraphrases")
        else:
            logger.info(f"Text: '{text_instruction}'")
        logger.info(f"Episodes: {n}")
        logger.info(f"")
        logger.info(f"SUCCESS RATE: {success_rate * 100:.1f}%")
        logger.info(f"Avg Reward: {avg_reward:.1f}")
        logger.info(f"Avg Goal Cos Sim: {avg_cos_sim:.4f}")
        logger.info(f"Avg Goal L2 Dist: {avg_l2_dist:.4f}")
        logger.info(f"")
        logger.info(f"Total Time: {eval_time:.1f}s ({eval_time/60:.1f} min)")
        logger.info(f"Avg Time/Episode: {eval_time/n:.1f}s")
        logger.info("=" * 60)
        
        # Save results
        summary = {
            "task": task,
            "text_instruction": text_instruction,
            "episodes": n,
            "success_rate": float(success_rate),
            "avg_reward": float(avg_reward),
            "avg_goal_cos_sim": float(avg_cos_sim),
            "avg_goal_l2_dist": float(avg_l2_dist),
            "total_time": float(eval_time),
            "goalhead_checkpoint": goalhead_ckpt_path,
            "config": args.config,
        }
        
        with open(output_dir / "summary.yaml", "w") as f:
            yaml.dump(summary, f)
        
        # Save per-episode results
        with open(output_dir / "episodes.csv", "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)
        
        logger.info(f"✅ Results saved to {output_dir}")
    
    return results


if __name__ == "__main__":
    main()
