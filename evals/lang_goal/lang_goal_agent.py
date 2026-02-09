# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Language-conditioned Goal Agent for planning.

Modified GC_Agent that uses GoalHead predictions instead of goal images.
"""

import torch
import torch.nn as nn

from evals.simu_env_planning.planning.gc_agent import GC_Agent
from evals.simu_env_planning.planning.planning.objectives import (
    ReprTargetDistMPCObjective,
    ReprTargetDistL1MPCObjective,
    ReprTargetCosMPCObjective,
)
from src.utils.logging import get_logger

logger = get_logger(__name__)


class LangGoalAgent(GC_Agent):
    """
    Language-conditioned Goal Agent.
    
    Extends GC_Agent to use GoalHead predictions for goal latents
    instead of encoding goal images directly.
    """
    
    def __init__(
        self,
        cfg,
        model,
        goal_head: nn.Module,
        text_encoder: nn.Module,
        dset=None,
        preprocessor=None,
    ):
        """
        Initialize LangGoalAgent.
        
        Args:
            cfg: Configuration object
            model: World model (VideoWM)
            goal_head: Trained GoalHead model
            text_encoder: Text encoder for language instructions
            dset: Dataset (optional)
            preprocessor: Image preprocessor (optional)
        """
        super().__init__(cfg, model, dset, preprocessor)
        
        self.goal_head = goal_head
        self.text_encoder = text_encoder
        
        # Freeze GoalHead for inference
        self.goal_head.eval()
        for param in self.goal_head.parameters():
            param.requires_grad = False
        
        self.text_instruction = None
        self._initial_obs_features = None
        
        logger.info("🧠 LangGoalAgent initialized with GoalHead")
    
    @torch.no_grad()
    def set_goal_from_language(
        self,
        initial_obs: torch.Tensor,
        text_instruction: str,
        proprio: torch.Tensor = None,
    ):
        """
        Set goal latent from language instruction.
        
        Args:
            initial_obs: Initial observation [C, H, W] or [T, C, H, W]
            text_instruction: Language instruction string
            proprio: Optional proprioception state
        """
        self.text_instruction = text_instruction
        
        # Ensure proper dimensions
        if initial_obs.dim() == 3:
            initial_obs = initial_obs.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
        elif initial_obs.dim() == 4:
            initial_obs = initial_obs.unsqueeze(0)  # [1, T, C, H, W]
        
        initial_obs = initial_obs.to(self.device)
        
        # Encode initial observation
        obs_dict = {"visual": initial_obs, "proprio": proprio}
        visual_features = self.model.encode_obs(obs_dict)["visual"]
        self._initial_obs_features = visual_features
        
        # Encode text instruction
        text_features = self.text_encoder.encode([text_instruction])
        text_features = text_features.to(self.device)
        
        # Predict goal latent using GoalHead
        z_goal_pred = self.goal_head(visual_features, text_features, proprio)
        
        # If z_goal_pred has extra dimensions, pool them
        if z_goal_pred.dim() > 2:
            z_goal_pred = z_goal_pred.mean(dim=tuple(range(1, z_goal_pred.dim() - 1)))
        
        # Expand to match expected format for objective
        # Goal encoding format: [B, T, V, H, W, D] or similar
        self.goal_state_enc = z_goal_pred.unsqueeze(1)  # [1, 1, D]
        
        # Set up objective based on config
        if self.cfg.planner.planning_objective.objective_type == "L2":
            self.objective = ReprTargetDistMPCObjective(
                self.cfg, target_enc=self.goal_state_enc, 
                **self.cfg.planner.planning_objective
            )
        elif self.cfg.planner.planning_objective.objective_type == "L1":
            self.objective = ReprTargetDistL1MPCObjective(
                self.cfg, target_enc=self.goal_state_enc,
                **self.cfg.planner.planning_objective
            )
        elif self.cfg.planner.planning_objective.objective_type == "repr_sim":
            self.objective = ReprTargetCosMPCObjective(
                self.cfg, target_enc=self.goal_state_enc,
                **self.cfg.planner.planning_objective
            )
        else:
            raise ValueError(
                f"Unknown objective type: {self.cfg.planner.planning_objective.objective_type}"
            )
        
        if self.planner is not None:
            self.planner.set_objective(self.objective)
        
        logger.info(f"🎯 Goal set from language: '{text_instruction}'")
        logger.info(f"   Goal latent shape: {z_goal_pred.shape}")
    
    @torch.no_grad()
    def set_goal(self, goal_state):
        """
        Override parent set_goal to also support image-based goals.
        
        This allows the same agent to be used for baseline comparisons
        with goal images.
        """
        # Call parent implementation for image-based goal
        super().set_goal(goal_state)
    
    def get_goal_info(self):
        """Return information about the current goal setting."""
        return {
            "text_instruction": self.text_instruction,
            "goal_latent_shape": self.goal_state_enc.shape if self.goal_state_enc is not None else None,
            "has_goal": self.goal_state_enc is not None,
        }


class LangGoalAgentFactory:
    """Factory for creating LangGoalAgent instances."""
    
    def __init__(
        self,
        goal_head_checkpoint: str,
        text_encoder_config: dict,
        device: str = "cuda",
    ):
        """
        Initialize factory with GoalHead checkpoint.
        
        Args:
            goal_head_checkpoint: Path to trained GoalHead checkpoint
            text_encoder_config: Configuration for text encoder
            device: Device to load models on
        """
        self.device = torch.device(device)
        
        # Load GoalHead
        checkpoint = torch.load(goal_head_checkpoint, map_location="cpu")
        goalhead_config = checkpoint.get("config", {}).get("goal_head", {})
        
        from app.lang_goal.goal_head import build_goal_head
        self.goal_head = build_goal_head(goalhead_config)
        self.goal_head.load_state_dict(checkpoint["goal_head_state_dict"])
        self.goal_head = self.goal_head.to(self.device)
        self.goal_head.eval()
        
        # Build text encoder
        from app.lang_goal.text_encoder import build_text_encoder
        text_encoder_config["device"] = device
        self.text_encoder = build_text_encoder(text_encoder_config)
        
        logger.info(f"✅ LangGoalAgentFactory initialized")
        logger.info(f"   GoalHead checkpoint: {goal_head_checkpoint}")
    
    def create_agent(self, cfg, model, dset=None, preprocessor=None) -> LangGoalAgent:
        """Create a LangGoalAgent instance."""
        return LangGoalAgent(
            cfg=cfg,
            model=model,
            goal_head=self.goal_head,
            text_encoder=self.text_encoder,
            dset=dset,
            preprocessor=preprocessor,
        )
