# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Distillation and advanced training modules for language-conditioned goal learning.

Implements:
- FR-1/FR-2: Teacher-Student distillation (latent distillation)
- FR-3: Behavior distillation (action sequence matching)
- FR-4: Paraphrase invariance loss
- FR-5: Prompt tuning (learnable context tokens)
- FR-6: Contrastive loss with hard negatives
"""

import logging
import sys
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# FR-5: Prompt Tuning (Learnable Context Tokens)
# ==============================================================================

class PromptTuningModule(nn.Module):
    """
    Learnable prompt tokens for adapting frozen text encoders.
    
    Based on CoOp (Context Optimization) approach:
    - Adds learnable context tokens before/after the text input
    - Keeps the main text encoder frozen
    - Only trains the context embeddings and projection layers
    
    Reference: https://arxiv.org/abs/2109.01134
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_context_tokens: int = 16,
        context_position: str = "prefix",  # "prefix", "suffix", or "both"
        dropout: float = 0.1,
    ):
        """
        Initialize PromptTuningModule.
        
        Args:
            embed_dim: Dimension of text embeddings (e.g., 512 for CLIP)
            num_context_tokens: Number of learnable context tokens
            context_position: Where to insert context tokens
            dropout: Dropout probability
        """
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_context_tokens = num_context_tokens
        self.context_position = context_position
        
        # Learnable context embeddings
        if context_position == "both":
            self.prefix_tokens = nn.Parameter(torch.zeros(1, num_context_tokens // 2, embed_dim))
            self.suffix_tokens = nn.Parameter(torch.zeros(1, num_context_tokens // 2, embed_dim))
            nn.init.normal_(self.prefix_tokens, std=0.02)
            nn.init.normal_(self.suffix_tokens, std=0.02)
        else:
            self.context_tokens = nn.Parameter(torch.zeros(1, num_context_tokens, embed_dim))
            nn.init.normal_(self.context_tokens, std=0.02)
        
        # Projection layer for adapting frozen text features
        self.adapt_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.Dropout(dropout),
        )
        
        logger.info(f"🎯 PromptTuning: {num_context_tokens} tokens, position={context_position}")
    
    def forward(self, text_features: torch.Tensor) -> torch.Tensor:
        """
        Add context tokens and adapt text features.
        
        Args:
            text_features: Text embeddings from frozen encoder [B, D] or [B, L, D]
            
        Returns:
            Adapted text features [B, D] or [B, L+context, D]
        """
        batch_size = text_features.shape[0]
        
        if text_features.dim() == 2:
            # [B, D] - pooled features
            # Add context by concatenation and projection
            if self.context_position == "both":
                ctx = torch.cat([
                    self.prefix_tokens.expand(batch_size, -1, -1).mean(dim=1),
                    self.suffix_tokens.expand(batch_size, -1, -1).mean(dim=1),
                ], dim=-1)
                ctx = ctx[..., :self.embed_dim]  # Project back to embed_dim
            else:
                ctx = self.context_tokens.expand(batch_size, -1, -1).mean(dim=1)
            
            # Combine with text features
            combined = text_features + ctx
            return self.adapt_proj(combined)
        
        else:
            # [B, L, D] - sequence features
            if self.context_position == "prefix":
                ctx = self.context_tokens.expand(batch_size, -1, -1)
                combined = torch.cat([ctx, text_features], dim=1)
            elif self.context_position == "suffix":
                ctx = self.context_tokens.expand(batch_size, -1, -1)
                combined = torch.cat([text_features, ctx], dim=1)
            else:  # both
                prefix = self.prefix_tokens.expand(batch_size, -1, -1)
                suffix = self.suffix_tokens.expand(batch_size, -1, -1)
                combined = torch.cat([prefix, text_features, suffix], dim=1)
            
            # Pool and project
            pooled = combined.mean(dim=1)
            return self.adapt_proj(pooled)


# ==============================================================================
# FR-2: Latent Distillation Loss
# ==============================================================================

def compute_distillation_loss(
    z_student: torch.Tensor,
    z_teacher: torch.Tensor,
    temperature: float = 1.0,
    loss_type: str = "mse_cosine",
) -> Dict[str, torch.Tensor]:
    """
    Compute latent distillation loss between student (lang) and teacher (oracle) predictions.
    
    Supports both pooled [B, D] and patch-level [B, N, D] representations.
    
    Args:
        z_student: Student prediction (from language) [B, D] or [B, N, D]
        z_teacher: Teacher prediction (from oracle goal image) [B, D] or [B, N, D]
        temperature: Temperature for softening distributions
        loss_type: Type of loss ("mse", "cosine", "mse_cosine", "kl")
        
    Returns:
        Dictionary with loss terms
    """
    losses = {}
    
    is_patch_level = z_student.dim() == 3
    
    if loss_type == "mse" or loss_type == "mse_cosine":
        # MSE loss (L2)
        mse_loss = F.mse_loss(z_student, z_teacher)
        losses["mse_loss"] = mse_loss
    
    if loss_type == "cosine" or loss_type == "mse_cosine":
        # Cosine similarity loss
        if is_patch_level:
            # Per-patch cosine similarity
            cos_sim = F.cosine_similarity(z_student, z_teacher, dim=-1)  # [B, N]
            # Average over patches
            patch_cos_loss = 1.0 - cos_sim.mean()
            losses["patch_cosine_loss"] = patch_cos_loss
            
            # Also compute pooled cosine
            z_student_pooled = z_student.mean(dim=1)
            z_teacher_pooled = z_teacher.mean(dim=1)
            pooled_cos_sim = F.cosine_similarity(z_student_pooled, z_teacher_pooled, dim=-1)
            pooled_cos_loss = 1.0 - pooled_cos_sim.mean()
            losses["pooled_cosine_loss"] = pooled_cos_loss
            
            # Combined cosine loss
            cosine_loss = 0.5 * patch_cos_loss + 0.5 * pooled_cos_loss
        else:
            cos_sim = F.cosine_similarity(z_student, z_teacher, dim=-1)
            cosine_loss = 1.0 - cos_sim.mean()
        
        losses["cosine_loss"] = cosine_loss
    
    if loss_type == "kl":
        # KL divergence on softmax distributions
        z_s_soft = F.softmax(z_student / temperature, dim=-1)
        z_t_soft = F.softmax(z_teacher / temperature, dim=-1)
        kl_loss = F.kl_div(
            z_s_soft.log(),
            z_t_soft,
            reduction="batchmean"
        ) * (temperature ** 2)
        losses["kl_loss"] = kl_loss
    
    # Total distillation loss
    if loss_type == "mse_cosine":
        total = losses["mse_loss"] + 0.1 * losses["cosine_loss"]
    elif loss_type == "mse":
        total = losses["mse_loss"]
    elif loss_type == "cosine":
        total = losses["cosine_loss"]
    elif loss_type == "kl":
        total = losses["kl_loss"]
    else:
        total = sum(losses.values())
    
    losses["distillation_loss"] = total
    
    return losses


# ==============================================================================
# FR-4: Paraphrase Invariance Loss
# ==============================================================================

def compute_paraphrase_invariance_loss(
    z_goals: List[torch.Tensor],
    invariance_type: str = "pairwise",
) -> Dict[str, torch.Tensor]:
    """
    Compute paraphrase invariance loss to ensure different text expressions
    of the same goal produce similar goal latents.
    
    Args:
        z_goals: List of goal predictions from different paraphrases
                 Each is [B, D] or [B, N, D]
        invariance_type: "pairwise" (all pairs), "anchor" (first vs rest),
                        or "centroid" (distance from mean)
        
    Returns:
        Dictionary with invariance loss
    """
    if len(z_goals) < 2:
        return {"invariance_loss": torch.tensor(0.0, device=z_goals[0].device)}
    
    losses = {}
    
    if invariance_type == "pairwise":
        # Average pairwise MSE between all paraphrases
        total_loss = 0.0
        count = 0
        for i in range(len(z_goals)):
            for j in range(i + 1, len(z_goals)):
                total_loss = total_loss + F.mse_loss(z_goals[i], z_goals[j])
                count += 1
        invariance_loss = total_loss / count
        
    elif invariance_type == "anchor":
        # First paraphrase as anchor, minimize distance to others
        anchor = z_goals[0]
        total_loss = 0.0
        for z in z_goals[1:]:
            total_loss = total_loss + F.mse_loss(z, anchor)
        invariance_loss = total_loss / (len(z_goals) - 1)
        
    elif invariance_type == "centroid":
        # Minimize distance from centroid
        centroid = torch.stack(z_goals, dim=0).mean(dim=0)
        total_loss = 0.0
        for z in z_goals:
            total_loss = total_loss + F.mse_loss(z, centroid)
        invariance_loss = total_loss / len(z_goals)
    
    else:
        raise ValueError(f"Unknown invariance_type: {invariance_type}")
    
    losses["invariance_loss"] = invariance_loss
    
    return losses


# ==============================================================================
# FR-6: Contrastive Loss with Hard Negatives
# ==============================================================================

def compute_contrastive_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    z_negatives: torch.Tensor,
    temperature: float = 0.07,
    margin: float = 0.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute contrastive loss with hard negatives.
    
    Uses InfoNCE-style contrastive loss to push anchor toward positive
    and away from negatives.
    
    Args:
        z_anchor: Anchor embeddings (e.g., predicted goal from lang) [B, D]
        z_positive: Positive embeddings (e.g., oracle goal) [B, D]
        z_negatives: Negative embeddings [B, K, D] where K is num negatives
        temperature: Temperature for softmax
        margin: Optional margin for triplet-style loss
        
    Returns:
        Dictionary with contrastive loss
    """
    losses = {}
    batch_size = z_anchor.shape[0]
    
    # Normalize embeddings
    z_anchor = F.normalize(z_anchor, dim=-1)
    z_positive = F.normalize(z_positive, dim=-1)
    z_negatives = F.normalize(z_negatives, dim=-1)
    
    # Positive similarity
    pos_sim = torch.sum(z_anchor * z_positive, dim=-1) / temperature  # [B]
    
    # Negative similarities
    neg_sim = torch.bmm(z_negatives, z_anchor.unsqueeze(-1)).squeeze(-1) / temperature  # [B, K]
    
    # InfoNCE loss
    logits = torch.cat([pos_sim.unsqueeze(-1), neg_sim], dim=-1)  # [B, 1+K]
    labels = torch.zeros(batch_size, dtype=torch.long, device=z_anchor.device)  # Positive is index 0
    
    contrastive_loss = F.cross_entropy(logits, labels)
    losses["contrastive_loss"] = contrastive_loss
    
    # Also compute accuracy for monitoring
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        accuracy = (pred == 0).float().mean()
        losses["contrastive_accuracy"] = accuracy
    
    return losses


def compute_triplet_loss(
    z_anchor: torch.Tensor,
    z_positive: torch.Tensor,
    z_negative: torch.Tensor,
    margin: float = 0.2,
) -> Dict[str, torch.Tensor]:
    """
    Compute triplet margin loss.
    
    Args:
        z_anchor: Anchor embeddings [B, D]
        z_positive: Positive embeddings [B, D]
        z_negative: Negative embeddings [B, D]
        margin: Margin for triplet loss
        
    Returns:
        Dictionary with triplet loss
    """
    losses = {}
    
    # Distances
    pos_dist = F.pairwise_distance(z_anchor, z_positive)
    neg_dist = F.pairwise_distance(z_anchor, z_negative)
    
    # Triplet loss with margin
    triplet_loss = F.relu(pos_dist - neg_dist + margin).mean()
    losses["triplet_loss"] = triplet_loss
    
    return losses


# ==============================================================================
# FR-3: Behavior Distillation (Action Sequence Matching)
# ==============================================================================

def compute_behavior_distillation_loss(
    student_actions: torch.Tensor,
    teacher_actions: torch.Tensor,
    student_values: Optional[torch.Tensor] = None,
    teacher_values: Optional[torch.Tensor] = None,
    action_weight: float = 1.0,
    value_weight: float = 0.1,
) -> Dict[str, torch.Tensor]:
    """
    Compute behavior distillation loss to match planner outputs.
    
    This ensures that the student's goal latent produces similar
    planning behavior to the teacher's oracle goal latent.
    
    Args:
        student_actions: Action sequences from planner with student goal [B, T, A]
        teacher_actions: Action sequences from planner with teacher goal [B, T, A]
        student_values: Optional value estimates with student goal [B]
        teacher_values: Optional value estimates with teacher goal [B]
        action_weight: Weight for action matching loss
        value_weight: Weight for value matching loss
        
    Returns:
        Dictionary with behavior losses
    """
    losses = {}
    
    # Action sequence matching
    action_loss = F.mse_loss(student_actions, teacher_actions)
    losses["action_loss"] = action_loss
    
    # Value matching (if provided)
    if student_values is not None and teacher_values is not None:
        value_loss = F.mse_loss(student_values, teacher_values)
        losses["value_loss"] = value_loss
    else:
        value_loss = 0.0
    
    # Total behavior loss
    total = action_weight * action_loss + value_weight * value_loss
    losses["behavior_loss"] = total
    
    return losses


def compute_action_ranking_loss(
    student_scores: torch.Tensor,
    teacher_rankings: torch.Tensor,
    temperature: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Compute ranking loss to match CEM action rankings.
    
    The student should assign similar rankings (preferences) to action
    sequences as the teacher does.
    
    Args:
        student_scores: Scores assigned by student's objective [B, K]
        teacher_rankings: Rankings from teacher's objective [B, K] (lower is better)
        temperature: Temperature for softmax
        
    Returns:
        Dictionary with ranking loss
    """
    losses = {}
    
    # Convert rankings to soft targets
    teacher_probs = F.softmax(-teacher_rankings.float() / temperature, dim=-1)
    student_log_probs = F.log_softmax(student_scores / temperature, dim=-1)
    
    # KL divergence between ranking distributions
    ranking_loss = F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")
    losses["ranking_loss"] = ranking_loss
    
    return losses


# ==============================================================================
# Combined Distillation Loss
# ==============================================================================

class DistillationLoss(nn.Module):
    """
    Combined distillation loss module for teacher-student training.
    
    Combines:
    - Latent distillation (FR-2)
    - Paraphrase invariance (FR-4)
    - Contrastive loss with hard negatives (FR-6)
    - Optionally: Behavior distillation (FR-3)
    """
    
    def __init__(
        self,
        # FR-2: Latent distillation
        distillation_weight: float = 1.0,
        distillation_type: str = "mse_cosine",
        distillation_temperature: float = 1.0,
        # FR-4: Paraphrase invariance
        invariance_weight: float = 0.5,
        invariance_type: str = "centroid",
        # FR-6: Contrastive loss
        contrastive_weight: float = 0.1,
        contrastive_temperature: float = 0.07,
        # Original losses
        l2_weight: float = 1.0,
        cosine_weight: float = 0.1,
    ):
        super().__init__()
        
        self.distillation_weight = distillation_weight
        self.distillation_type = distillation_type
        self.distillation_temperature = distillation_temperature
        
        self.invariance_weight = invariance_weight
        self.invariance_type = invariance_type
        
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        
        self.l2_weight = l2_weight
        self.cosine_weight = cosine_weight
        
        logger.info(f"📊 DistillationLoss initialized:")
        logger.info(f"   - distillation_weight={distillation_weight}")
        logger.info(f"   - invariance_weight={invariance_weight}")
        logger.info(f"   - contrastive_weight={contrastive_weight}")
    
    def forward(
        self,
        z_student: torch.Tensor,
        z_teacher: torch.Tensor,
        z_paraphrases: Optional[List[torch.Tensor]] = None,
        z_paraphrases_per_sample: Optional[List[List[torch.Tensor]]] = None,
        z_negatives: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute combined distillation loss.
        
        Args:
            z_student: Student prediction [B, D] or [B, N, D]
            z_teacher: Teacher (oracle) prediction [B, D] or [B, N, D]
            z_paraphrases: Optional list of predictions from paraphrased texts (flat list)
            z_paraphrases_per_sample: Optional list of lists, each inner list contains
                                       paraphrase predictions for one sample
            z_negatives: Optional negative samples [B, K, D]
            
        Returns:
            Dictionary with all loss terms and total
        """
        losses = {}
        
        # FR-2: Latent distillation loss
        distill_losses = compute_distillation_loss(
            z_student, z_teacher,
            temperature=self.distillation_temperature,
            loss_type=self.distillation_type,
        )
        losses.update({f"distill_{k}": v for k, v in distill_losses.items()})
        
        # FR-4: Paraphrase invariance loss
        invariance_loss = torch.tensor(0.0, device=z_student.device)
        
        if z_paraphrases_per_sample is not None and len(z_paraphrases_per_sample) > 0:
            # New format: list of lists, compute invariance per sample and average
            sample_invariance_losses = []
            for sample_paras in z_paraphrases_per_sample:
                if len(sample_paras) >= 2:
                    # Pool if patch-level for invariance computation
                    pooled_paras = []
                    for z in sample_paras:
                        if z.dim() == 3:
                            pooled_paras.append(z.mean(dim=1))  # [1, D]
                        else:
                            pooled_paras.append(z)
                    
                    inv_loss_dict = compute_paraphrase_invariance_loss(
                        pooled_paras,
                        invariance_type=self.invariance_type,
                    )
                    sample_invariance_losses.append(inv_loss_dict["invariance_loss"])
            
            if sample_invariance_losses:
                invariance_loss = torch.stack(sample_invariance_losses).mean()
        
        elif z_paraphrases is not None and len(z_paraphrases) > 0:
            # Legacy format: flat list
            all_z = [z_student] + z_paraphrases
            inv_losses = compute_paraphrase_invariance_loss(
                all_z,
                invariance_type=self.invariance_type,
            )
            invariance_loss = inv_losses["invariance_loss"]
        
        losses["invariance_loss"] = invariance_loss
        
        # FR-6: Contrastive loss (if negatives provided)
        if z_negatives is not None:
            # Pool if patch-level
            if z_student.dim() == 3:
                z_s_pooled = z_student.mean(dim=1)
                z_t_pooled = z_teacher.mean(dim=1)
                z_n_pooled = z_negatives.mean(dim=2) if z_negatives.dim() == 4 else z_negatives
            else:
                z_s_pooled = z_student
                z_t_pooled = z_teacher
                z_n_pooled = z_negatives
            
            contrast_losses = compute_contrastive_loss(
                z_s_pooled, z_t_pooled, z_n_pooled,
                temperature=self.contrastive_temperature,
            )
            losses.update(contrast_losses)
        else:
            losses["contrastive_loss"] = torch.tensor(0.0, device=z_student.device)
        
        # Total loss
        total_loss = (
            self.distillation_weight * losses["distill_distillation_loss"] +
            self.invariance_weight * losses["invariance_loss"] +
            self.contrastive_weight * losses["contrastive_loss"]
        )
        losses["total_loss"] = total_loss
        
        return losses


# ==============================================================================
# Hard Negative Mining
# ==============================================================================

class HardNegativeMiner:
    """
    Mines hard negative samples for contrastive learning.
    
    Hard negatives are:
    1. Similar but incorrect goals (near-miss states)
    2. Goals from other episodes in the same task
    3. Failed states from the same episode
    """
    
    def __init__(
        self,
        num_negatives: int = 5,
        similarity_threshold: float = 0.8,
    ):
        self.num_negatives = num_negatives
        self.similarity_threshold = similarity_threshold
    
    def mine_from_batch(
        self,
        z_goals: torch.Tensor,
        z_current: torch.Tensor,
        episode_ids: List[Union[int, str]],
    ) -> torch.Tensor:
        """
        Mine hard negatives from a batch.
        
        Args:
            z_goals: Goal latents [B, D]
            z_current: Current state latents [B, D]
            episode_ids: Episode IDs for each sample (list of strings or ints)
            
        Returns:
            z_negatives: Hard negative latents [B, K, D]
        """
        batch_size = z_goals.shape[0]
        device = z_goals.device
        
        # Compute pairwise similarities
        z_goals_norm = F.normalize(z_goals, dim=-1)
        similarities = torch.mm(z_goals_norm, z_goals_norm.t())  # [B, B]
        
        # Mask out same-episode samples (they shouldn't be negatives for each other)
        # Handle both string and numeric episode IDs
        same_episode = torch.zeros(batch_size, batch_size, dtype=torch.bool, device=device)
        for i in range(batch_size):
            for j in range(batch_size):
                same_episode[i, j] = episode_ids[i] == episode_ids[j]
        
        similarities = similarities.masked_fill(same_episode, -1.0)
        
        # Also mask out self
        similarities.fill_diagonal_(-1.0)
        
        # Select top-K hardest negatives (most similar but different)
        _, top_indices = similarities.topk(min(self.num_negatives, batch_size - 1), dim=1)
        
        # Gather negatives
        z_negatives = z_goals[top_indices]  # [B, K, D]
        
        return z_negatives
    
    def mine_temporal_negatives(
        self,
        z_trajectory: torch.Tensor,
        goal_idx: int,
        num_negatives: int = 5,
    ) -> torch.Tensor:
        """
        Mine hard negatives from temporal neighbors in a trajectory.
        
        Args:
            z_trajectory: Latents for all frames in trajectory [T, D]
            goal_idx: Index of the goal frame
            num_negatives: Number of negatives to sample
            
        Returns:
            z_negatives: Negative latents [K, D]
        """
        T = z_trajectory.shape[0]
        
        # Exclude goal frame and nearby frames
        exclude_range = 5
        valid_indices = [
            i for i in range(T)
            if abs(i - goal_idx) > exclude_range
        ]
        
        if len(valid_indices) < num_negatives:
            # Sample with replacement
            indices = torch.tensor(valid_indices, device=z_trajectory.device)
            sample_indices = indices[torch.randint(len(indices), (num_negatives,))]
        else:
            # Sample without replacement
            perm = torch.randperm(len(valid_indices))[:num_negatives]
            sample_indices = torch.tensor([valid_indices[i] for i in perm], device=z_trajectory.device)
        
        return z_trajectory[sample_indices]
