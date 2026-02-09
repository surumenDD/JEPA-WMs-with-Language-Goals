# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
GoalHead training script.

Trains GoalHead to predict goal latents from initial images and language instructions.
"""

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

# Setup path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.lang_goal.env_utils import get_git_info, get_paths, validate_all_env_vars
from app.lang_goal.goal_head import (
    GoalHead,
    build_goal_head,
    compute_goal_loss,
    compute_goal_loss_v2,
)
from app.lang_goal.goalhead_dataset import build_dataloaders
from app.lang_goal.text_encoder import build_text_encoder
from app.lang_goal.distillation import (
    DistillationLoss,
    PromptTuningModule,
    HardNegativeMiner,
    compute_paraphrase_invariance_loss,
)
from src.utils.logging import AverageMeter, CSVLogger, get_logger
from src.utils.yaml_utils import expand_env_vars

logger = get_logger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"🎲 Set random seed: {seed}")


def load_pretrained_model(
    checkpoint_path: str,
    device: torch.device,
    model_kwargs: Dict,
) -> nn.Module:
    """
    Load pretrained JEPA-WM model using hardcoded stats (no dataset needed).
    
    This follows the same pattern as hubconf.py to load models without
    requiring access to the actual datasets.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load on
        model_kwargs: Model configuration dict containing data, data_aug, wrapper_kwargs, pretrain_kwargs
        
    Returns:
        Frozen VideoWM model
    """
    from evals.simu_env_planning.eval import init_module
    from app.plan_common.datasets import get_data_stats
    from app.plan_common.datasets.preprocessor import Preprocessor
    from app.plan_common.datasets.transforms import make_inverse_transforms, make_transforms
    
    # Extract config sections
    cfgs_data = model_kwargs.get("data", {})
    cfgs_data_aug = model_kwargs.get("data_aug", {})
    wrapper_kwargs = model_kwargs.get("wrapper_kwargs", {})
    pretrain_kwargs = model_kwargs.get("pretrain_kwargs", {})
    
    # Get hardcoded stats for MetaWorld (no dataset download needed)
    data_stats = get_data_stats("metaworld")
    action_dim = data_stats["action_dim"]
    proprio_dim = data_stats["proprio_dim"]
    logger.info(f"Using hardcoded dimensions: action_dim={action_dim}, proprio_dim={proprio_dim}")
    
    # Build transforms
    img_size = cfgs_data.get("img_size", 224)
    
    transform = make_transforms(
        img_size=img_size,
        normalize=cfgs_data_aug.get("normalize", [[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]]),
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(1.0, 1.0),
        random_resize_scale=(1.0, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
    )
    inverse_transform = make_inverse_transforms(
        img_size=img_size,
        **cfgs_data_aug,
    )
    
    # Build preprocessor with hardcoded normalization stats
    preprocessor = Preprocessor(
        action_mean=torch.tensor(data_stats["action_mean"]),
        action_std=torch.tensor(data_stats["action_std"]),
        state_mean=torch.tensor(data_stats["state_mean"]),
        state_std=torch.tensor(data_stats["state_std"]),
        proprio_mean=torch.tensor(data_stats["proprio_mean"]),
        proprio_std=torch.tensor(data_stats["proprio_std"]),
        transform=transform,
        inverse_transform=inverse_transform,
    )
    
    # Get module name from config
    module_name = model_kwargs.get("module_name", 
        "app.vjepa_wm.modelcustom.simu_env_planning.vit_enc_preds")
    
    # Load model
    model = init_module(
        folder=str(Path(checkpoint_path).parent),
        checkpoint=Path(checkpoint_path).name,
        module_name=module_name,
        model_kwargs=pretrain_kwargs,
        device=device,
        action_dim=action_dim,
        proprio_dim=proprio_dim,
        preprocessor=preprocessor,
        cfgs_data=cfgs_data,
        wrapper_kwargs=wrapper_kwargs,
    )
    
    # Freeze model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    logger.info(f"✅ Loaded pretrained model from {checkpoint_path}")
    logger.info("🔒 Model frozen for GoalHead training")
    
    return model


def save_checkpoint(
    goal_head: GoalHead,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    save_path: str,
    config: Dict,
    best_val_loss: Optional[float] = None,
    prompt_tuning: Optional[nn.Module] = None,
):
    """Save GoalHead checkpoint."""
    checkpoint = {
        "epoch": epoch,
        "goal_head_state_dict": goal_head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
        "best_val_loss": best_val_loss if best_val_loss is not None else loss,
        "config": config,
    }
    
    # Also save prompt tuning if present
    if prompt_tuning is not None:
        checkpoint["prompt_tuning_state_dict"] = prompt_tuning.state_dict()
    
    torch.save(checkpoint, save_path)
    logger.info(f"💾 Saved checkpoint: {save_path}")


def load_checkpoint(
    goal_head: GoalHead,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
) -> int:
    """Load GoalHead checkpoint and return epoch."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    goal_head.load_state_dict(checkpoint["goal_head_state_dict"])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    logger.info(f"✅ Loaded checkpoint from epoch {epoch}: {checkpoint_path}")
    return epoch


def train_epoch(
    goal_head: GoalHead,
    visual_encoder: nn.Module,
    text_encoder: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    config: Dict,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    wandb_run = None,
    prompt_tuning: Optional[nn.Module] = None,
    distillation_loss_fn: Optional[DistillationLoss] = None,
    hard_negative_miner: Optional[HardNegativeMiner] = None,
) -> Dict[str, float]:
    """
    Train for one epoch with optional distillation.
    
    Supports:
    - FR-2: Latent distillation (teacher=oracle, student=lang)
    - FR-4: Paraphrase invariance during training
    - FR-5: Prompt tuning for text encoder
    - FR-6: Contrastive loss with hard negatives
    """
    goal_head.train()
    if prompt_tuning is not None:
        prompt_tuning.train()
    
    loss_meter = AverageMeter()
    l2_meter = AverageMeter()
    cosine_loss_meter = AverageMeter()
    cosine_sim_meter = AverageMeter()
    distill_meter = AverageMeter()
    invariance_meter = AverageMeter()
    contrastive_meter = AverageMeter()
    
    l2_weight = config.get("l2_weight", 1.0)
    cosine_weight = config.get("cosine_weight", 0.0)
    mixed_precision = config.get("mixed_precision", True) and device.type == "cuda"
    clip_grad = config.get("clip_grad", 1.0)
    
    # Loss version: "v1" (original) or "v2" (improved with cosine + contrastive + delta reg)
    loss_version = config.get("loss_version", "v1")
    contrastive_weight = config.get("contrastive_weight", 0.1)
    delta_reg_weight = config.get("delta_reg_weight", 0.01)
    temperature = config.get("temperature", 0.07)
    
    # Distillation settings
    use_distillation = distillation_loss_fn is not None
    
    # FR-4: Paraphrase invariance settings
    invariance_cfg = config.get("paraphrase_invariance", {})
    use_invariance = invariance_cfg.get("enabled", False)
    num_paraphrases = invariance_cfg.get("num_paraphrases", 3)
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [train]")
    
    for batch_idx, batch in enumerate(pbar):
        # Move to device
        I0 = batch["I0"].to(device)
        I_goal = batch["I_goal"].to(device)
        texts = batch["text"]
        proprio = batch.get("proprio")
        if proprio is not None:
            proprio = proprio.to(device)
        
        # Check if using patch-level GoalHead
        is_patch_level = config.get("goal_head", {}).get("patch_level", False)
        
        # Get text ensemble config
        text_ensemble_cfg = config.get("text_ensemble", {})
        use_text_ensemble = text_ensemble_cfg.get("enabled", False)
        
        # Encode visual features (frozen)
        with torch.no_grad():
            # Add batch and time dims if needed
            if I0.dim() == 4:  # [B, C, H, W]
                I0_enc = I0.unsqueeze(1)  # [B, 1, C, H, W]
            else:
                I0_enc = I0
            
            # Create obs dict for visual encoder
            obs_dict = {"visual": I0_enc, "proprio": proprio}
            visual_features = visual_encoder.encode_obs(obs_dict)["visual"]
            # visual_features shape: [B, T, V, H, W, D]
            
            # Encode goal image to get z_goal_gt
            if I_goal.dim() == 3:
                I_goal_enc = I_goal.unsqueeze(0).unsqueeze(0)  # [1, 1, C, H, W]
            elif I_goal.dim() == 4:
                I_goal_enc = I_goal.unsqueeze(1)  # [B, 1, C, H, W]
            else:
                I_goal_enc = I_goal
            
            goal_obs_dict = {"visual": I_goal_enc, "proprio": None}
            z_goal_gt_full = visual_encoder.encode_obs(goal_obs_dict)["visual"]
            # z_goal_gt_full shape: [B, T, V, H, W, D]
            
            if is_patch_level:
                # For patch-level GoalHead, keep patch structure: [B, N, D]
                from einops import rearrange
                z_goal_gt = rearrange(z_goal_gt_full, "b t v h w d -> b (t v h w) d")
            else:
                # For pooled GoalHead, pool to [B, D]
                z_goal_gt = z_goal_gt_full.mean(dim=(1, 2, 3, 4))
        
        # Get task names from batch (these are available!)
        task_names = batch.get("task_name", [])
        
        # Encode text (frozen)
        with torch.no_grad():
            if use_text_ensemble and hasattr(text_encoder, 'encode_ensemble'):
                # Use ensemble encoding: average multiple paraphrases
                task_texts_map = text_ensemble_cfg.get("task_texts", {})
                text_features_list = []
                
                for i, (text, task_name) in enumerate(zip(texts, task_names)):
                    # Use task_name directly to look up ensemble texts
                    # Map task names like "mw-reach" to the config key format
                    task_key = task_name if task_name in task_texts_map else task_name.replace("_", "-")
                    
                    if task_key in task_texts_map:
                        # Found ensemble texts for this task
                        task_texts = task_texts_map[task_key]
                        text_features_list.append(text_encoder.encode_ensemble(task_texts))
                    else:
                        # Fallback to single text
                        text_features_list.append(text_encoder.encode([text]))
                        
                text_features = torch.cat(text_features_list, dim=0)
            else:
                text_features = text_encoder.encode(texts)
        
        # FR-5: Apply prompt tuning if available
        if prompt_tuning is not None:
            text_features = prompt_tuning(text_features)
        
        # Prepare visual features for GoalHead
        if is_patch_level:
            # Reshape to [B, N, D] for PatchLevelGoalHead
            from einops import rearrange
            visual_input = rearrange(visual_features, "b t v h w d -> b (t v h w) d")
            # Keep z_init for delta regularization (v2 loss)
            z_init = visual_input
        else:
            # For pooled GoalHead, also pool the input
            visual_input = visual_features.mean(dim=(1, 2, 3, 4))  # [B, D]
            z_init = visual_input
        
        # Forward through GoalHead
        optimizer.zero_grad()
        
        if mixed_precision and scaler is not None:
            with torch.cuda.amp.autocast():
                z_goal_pred = goal_head(visual_input, text_features, proprio)
                
                # Base losses (L2 + cosine) - select version based on config
                if loss_version == "v2":
                    losses = compute_goal_loss_v2(
                        z_goal_pred, z_goal_gt,
                        z_init=z_init,
                        negative_samples=None,  # Will be populated by distillation
                        l2_weight=l2_weight,
                        cosine_weight=cosine_weight,
                        contrastive_weight=contrastive_weight,
                        delta_reg_weight=delta_reg_weight,
                        temperature=temperature,
                    )
                else:
                    losses = compute_goal_loss(
                        z_goal_pred, z_goal_gt,
                        l2_weight=l2_weight,
                        cosine_weight=cosine_weight,
                    )
                total_loss = losses["total_loss"]
                
                # FR-2/4/6: Distillation losses if enabled
                if use_distillation:
                    # FR-4: Generate paraphrase predictions for invariance loss
                    # Structure: For each sample in batch, generate predictions from multiple paraphrases
                    z_paraphrases_per_sample = None
                    if use_invariance and hasattr(text_encoder, 'encode'):
                        task_texts_map = text_ensemble_cfg.get("task_texts", {})
                        batch_size = len(task_names)
                        z_paraphrases_per_sample = []  # List of lists: [[z_para1, z_para2, ...], ...] per sample
                        
                        for i, task_name in enumerate(task_names):
                            task_key = task_name if task_name in task_texts_map else task_name.replace("_", "-")
                            sample_paraphrases = [z_goal_pred[i:i+1]]  # Start with original prediction
                            
                            if task_key in task_texts_map:
                                paraphrase_texts = task_texts_map[task_key][:num_paraphrases]
                                for pt in paraphrase_texts[1:]:  # Skip first (already used)
                                    pt_feat = text_encoder.encode([pt]).to(device)
                                    if prompt_tuning is not None:
                                        pt_feat = prompt_tuning(pt_feat)
                                    z_para = goal_head(visual_input[i:i+1], pt_feat, None)
                                    sample_paraphrases.append(z_para)
                            
                            z_paraphrases_per_sample.append(sample_paraphrases)
                    
                    # FR-6: Mine hard negatives from batch
                    z_negatives = None
                    if hard_negative_miner is not None:
                        episode_ids = batch.get("episode_id", list(range(len(texts))))
                        z_goal_gt_pooled = z_goal_gt.mean(dim=1) if z_goal_gt.dim() == 3 else z_goal_gt
                        z_negatives = hard_negative_miner.mine_from_batch(
                            z_goal_gt_pooled, z_goal_pred.mean(dim=1) if z_goal_pred.dim() == 3 else z_goal_pred,
                            episode_ids
                        )
                    
                    # FR-6: Use dataset hard negatives if available
                    if "I_negatives" in batch and batch["I_negatives"] is not None:
                        I_negatives = batch["I_negatives"].to(device)  # [B, K, C, H, W]
                        neg_mask = batch.get("neg_mask", None)
                        if neg_mask is not None:
                            neg_mask = neg_mask.to(device)
                        
                        with torch.no_grad():
                            B, K = I_negatives.shape[:2]
                            # Reshape for encoding: [B*K, 1, C, H, W]
                            I_neg_enc = I_negatives.view(B * K, 1, *I_negatives.shape[2:])
                            neg_obs_dict = {"visual": I_neg_enc, "proprio": None}
                            z_neg_full = visual_encoder.encode_obs(neg_obs_dict)["visual"]
                            # z_neg_full: [B*K, T, V, H, W, D]
                            z_neg_pooled = z_neg_full.mean(dim=(1, 2, 3, 4))  # [B*K, D]
                            z_neg_pooled = z_neg_pooled.view(B, K, -1)  # [B, K, D]
                            
                            # Apply mask if available
                            if neg_mask is not None:
                                z_neg_pooled = z_neg_pooled * neg_mask.unsqueeze(-1)
                        
                        # Combine with batch-mined negatives
                        if z_negatives is not None:
                            z_negatives = torch.cat([z_negatives, z_neg_pooled], dim=1)
                        else:
                            z_negatives = z_neg_pooled
                    
                    # Compute distillation loss
                    distill_losses = distillation_loss_fn(
                        z_student=z_goal_pred,
                        z_teacher=z_goal_gt,
                        z_paraphrases_per_sample=z_paraphrases_per_sample,
                        z_negatives=z_negatives,
                    )
                    total_loss = total_loss + distill_losses["total_loss"]
                    losses.update(distill_losses)
                
                loss = total_loss
            
            scaler.scale(loss).backward()
            if clip_grad > 0:
                scaler.unscale_(optimizer)
                params_to_clip = list(goal_head.parameters())
                if prompt_tuning is not None:
                    params_to_clip += list(prompt_tuning.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, clip_grad)
            scaler.step(optimizer)
            scaler.update()
        else:
            z_goal_pred = goal_head(visual_input, text_features, proprio)
            
            # Base losses - select version based on config
            if loss_version == "v2":
                losses = compute_goal_loss_v2(
                    z_goal_pred, z_goal_gt,
                    z_init=z_init,
                    negative_samples=None,  # Will be populated by distillation
                    l2_weight=l2_weight,
                    cosine_weight=cosine_weight,
                    contrastive_weight=contrastive_weight,
                    delta_reg_weight=delta_reg_weight,
                    temperature=temperature,
                )
            else:
                losses = compute_goal_loss(
                    z_goal_pred, z_goal_gt,
                    l2_weight=l2_weight,
                    cosine_weight=cosine_weight,
                )
            total_loss = losses["total_loss"]
            
            # FR-2/4/6: Distillation losses if enabled
            if use_distillation:
                z_paraphrases_per_sample = None
                z_negatives = None
                
                # FR-4: Paraphrase invariance
                if use_invariance and hasattr(text_encoder, 'encode'):
                    task_texts_map = text_ensemble_cfg.get("task_texts", {})
                    z_paraphrases_per_sample = []  # List of lists per sample
                    
                    for i, task_name in enumerate(task_names):
                        task_key = task_name if task_name in task_texts_map else task_name.replace("_", "-")
                        sample_paraphrases = [z_goal_pred[i:i+1]]  # Start with original
                        
                        if task_key in task_texts_map:
                            paraphrase_texts = task_texts_map[task_key][:num_paraphrases]
                            for pt in paraphrase_texts[1:]:
                                pt_feat = text_encoder.encode([pt]).to(device)
                                if prompt_tuning is not None:
                                    pt_feat = prompt_tuning(pt_feat)
                                z_para = goal_head(visual_input[i:i+1], pt_feat, None)
                                sample_paraphrases.append(z_para)
                        
                        z_paraphrases_per_sample.append(sample_paraphrases)
                
                # FR-6: Hard negatives
                if hard_negative_miner is not None:
                    episode_ids = batch.get("episode_id", list(range(len(texts))))
                    z_goal_gt_pooled = z_goal_gt.mean(dim=1) if z_goal_gt.dim() == 3 else z_goal_gt
                    z_negatives = hard_negative_miner.mine_from_batch(
                        z_goal_gt_pooled, z_goal_pred.mean(dim=1) if z_goal_pred.dim() == 3 else z_goal_pred,
                        episode_ids
                    )
                
                # FR-6: Use dataset hard negatives if available
                if "I_negatives" in batch and batch["I_negatives"] is not None:
                    I_negatives = batch["I_negatives"].to(device)  # [B, K, C, H, W]
                    neg_mask = batch.get("neg_mask", None)
                    if neg_mask is not None:
                        neg_mask = neg_mask.to(device)
                    
                    with torch.no_grad():
                        B, K = I_negatives.shape[:2]
                        # Reshape for encoding: [B*K, 1, C, H, W]
                        I_neg_enc = I_negatives.view(B * K, 1, *I_negatives.shape[2:])
                        neg_obs_dict = {"visual": I_neg_enc, "proprio": None}
                        z_neg_full = visual_encoder.encode_obs(neg_obs_dict)["visual"]
                        # z_neg_full: [B*K, T, V, H, W, D]
                        z_neg_pooled = z_neg_full.mean(dim=(1, 2, 3, 4))  # [B*K, D]
                        z_neg_pooled = z_neg_pooled.view(B, K, -1)  # [B, K, D]
                        
                        # Apply mask if available
                        if neg_mask is not None:
                            z_neg_pooled = z_neg_pooled * neg_mask.unsqueeze(-1)
                    
                    # Combine with batch-mined negatives
                    if z_negatives is not None:
                        z_negatives = torch.cat([z_negatives, z_neg_pooled], dim=1)
                    else:
                        z_negatives = z_neg_pooled
                
                distill_losses = distillation_loss_fn(
                    z_student=z_goal_pred,
                    z_teacher=z_goal_gt,
                    z_paraphrases_per_sample=z_paraphrases_per_sample,
                    z_negatives=z_negatives,
                )
                total_loss = total_loss + distill_losses["total_loss"]
                losses.update(distill_losses)
            
            loss = total_loss
            
            loss.backward()
            if clip_grad > 0:
                params_to_clip = list(goal_head.parameters())
                if prompt_tuning is not None:
                    params_to_clip += list(prompt_tuning.parameters())
                torch.nn.utils.clip_grad_norm_(params_to_clip, clip_grad)
            optimizer.step()
        
        # Update meters
        loss_meter.update(loss.item())
        l2_meter.update(losses["l2_loss"].item())
        cosine_loss_meter.update(losses.get("cosine_loss", torch.tensor(0.0)).item())
        
        # Always compute cosine_sim for monitoring (even if cosine_loss weight=0)
        with torch.no_grad():
            if "cosine_sim" in losses and losses["cosine_sim"] is not None:
                _cos_sim = losses["cosine_sim"].item() if isinstance(losses["cosine_sim"], torch.Tensor) else losses["cosine_sim"]
            else:
                # Compute cosine_sim manually
                _pred_flat = z_goal_pred.detach().view(z_goal_pred.shape[0], -1)
                _gt_flat = z_goal_gt.view(z_goal_gt.shape[0], -1)
                _cos_sim = torch.nn.functional.cosine_similarity(_pred_flat, _gt_flat, dim=-1).mean().item()
            cosine_sim_meter.update(_cos_sim)
        
        if use_distillation:
            distill_meter.update(losses.get("distill_distillation_loss", torch.tensor(0.0)).item())
            invariance_meter.update(losses.get("invariance_loss", torch.tensor(0.0)).item())
            contrastive_meter.update(losses.get("contrastive_loss", torch.tensor(0.0)).item())
        
        # Update progress bar (統一: loss, l2, cos, distill)
        postfix = {
            "loss": f"{loss_meter.avg:.4f}",
            "l2": f"{l2_meter.avg:.4f}",
            "cos": f"{cosine_sim_meter.avg:.4f}",
        }
        if use_distillation:
            postfix["distill"] = f"{distill_meter.avg:.4f}"
        pbar.set_postfix(postfix)
        
        # Log to wandb
        if wandb_run is not None and batch_idx % 10 == 0:
            log_dict = {
                "train/total_loss": loss.item(),
                "train/l2_loss": losses["l2_loss"].item(),
                "train/cosine_loss": losses.get("cosine_loss", torch.tensor(0.0)).item(),
                "train/cosine_sim": _cos_sim,
                "epoch": epoch,
                "batch": batch_idx,
            }
            if use_distillation:
                log_dict.update({
                    "train/distill_loss": losses.get("distill_distillation_loss", torch.tensor(0.0)).item(),
                    "train/invariance_loss": losses.get("invariance_loss", torch.tensor(0.0)).item(),
                    "train/contrastive_loss": losses.get("contrastive_loss", torch.tensor(0.0)).item(),
                })
            wandb_run.log(log_dict)
    
    result = {
        "total_loss": loss_meter.avg,
        "l2_loss": l2_meter.avg,
        "cosine_loss": cosine_loss_meter.avg,
        "cosine_sim": cosine_sim_meter.avg,
        "distill_loss": distill_meter.avg,
        "invariance_loss": invariance_meter.avg,
        "contrastive_loss": contrastive_meter.avg,
    }
    return result


@torch.no_grad()
def evaluate(
    goal_head: GoalHead,
    visual_encoder: nn.Module,
    text_encoder: nn.Module,
    val_loader,
    device: torch.device,
    config: Dict,
    prompt_tuning: Optional[nn.Module] = None,
    distillation_loss_fn: Optional[DistillationLoss] = None,
) -> Dict[str, float]:
    """Evaluate GoalHead on validation set.
    
    Now computes the same losses as training for fair comparison:
    - L2 loss (base prediction loss)
    - Cosine similarity loss
    - Distillation loss (if enabled)
    """
    goal_head.eval()
    if prompt_tuning is not None:
        prompt_tuning.eval()
    
    loss_meter = AverageMeter()
    l2_meter = AverageMeter()
    cosine_loss_meter = AverageMeter()
    cosine_sim_meter = AverageMeter()
    distill_meter = AverageMeter()
    total_loss_meter = AverageMeter()  # Total loss matching training
    
    l2_weight = config.get("l2_weight", 1.0)
    cosine_weight = config.get("cosine_weight", 0.0)
    loss_version = config.get("loss_version", "v1")
    
    # Check if using patch-level GoalHead or improved GoalHead
    is_patch_level = config.get("goal_head", {}).get("patch_level", False)
    is_improved = config.get("goal_head", {}).get("version", "v1") == "v2"
    
    use_distillation = distillation_loss_fn is not None
    
    # Per-task metrics
    task_metrics = {}
    
    for batch in tqdm(val_loader, desc="Evaluating "):
        I0 = batch["I0"].to(device)
        I_goal = batch["I_goal"].to(device)
        texts = batch["text"]
        task_names = batch["task_name"]
        proprio = batch.get("proprio")
        if proprio is not None:
            proprio = proprio.to(device)
        
        # Encode visual features
        if I0.dim() == 4:
            I0_enc = I0.unsqueeze(1)
        else:
            I0_enc = I0
        
        obs_dict = {"visual": I0_enc, "proprio": proprio}
        visual_features = visual_encoder.encode_obs(obs_dict)["visual"]
        
        # Encode goal
        if I_goal.dim() == 3:
            I_goal_enc = I_goal.unsqueeze(0).unsqueeze(0)
        elif I_goal.dim() == 4:
            I_goal_enc = I_goal.unsqueeze(1)
        else:
            I_goal_enc = I_goal
        
        goal_obs_dict = {"visual": I_goal_enc, "proprio": None}
        z_goal_gt_full = visual_encoder.encode_obs(goal_obs_dict)["visual"]
        
        if is_patch_level:
            from einops import rearrange
            z_goal_gt = rearrange(z_goal_gt_full, "b t v h w d -> b (t v h w) d")
            visual_input = rearrange(visual_features, "b t v h w d -> b (t v h w) d")
        else:
            z_goal_gt = z_goal_gt_full.mean(dim=(1, 2, 3, 4))
            # For pooled GoalHead, also pool the input (fix for shape mismatch)
            visual_input = visual_features.mean(dim=(1, 2, 3, 4))  # [B, D]
        
        # Encode text (with prompt tuning if enabled)
        if prompt_tuning is not None:
            text_features = text_encoder.encode(texts)
            text_features = prompt_tuning(text_features)
        else:
            text_features = text_encoder.encode(texts)
        
        # Forward
        z_goal_pred = goal_head(visual_input, text_features, proprio)
        
        # Compute base loss (same as training)
        if loss_version == "v2":
            losses = compute_goal_loss_v2(
                z_goal_pred, z_goal_gt,
                l2_weight=l2_weight,
                cosine_weight=cosine_weight,
                contrastive_weight=0.0,  # No contrastive in eval (need negatives)
                delta_reg_weight=0.0,  # No delta reg in eval
            )
        else:
            losses = compute_goal_loss(
                z_goal_pred, z_goal_gt,
                l2_weight=l2_weight,
                cosine_weight=cosine_weight,
            )
        
        total_loss = losses["total_loss"]
        
        # Compute distillation loss (same as training)
        distill_loss_val = 0.0
        if use_distillation:
            # For distillation, we need oracle goal (z_goal_gt)
            # DistillationLoss.forward(z_student, z_teacher) returns dict with "total_loss"
            distill_losses = distillation_loss_fn(
                z_student=z_goal_pred,
                z_teacher=z_goal_gt,  # Oracle is the GT goal latent
            )
            total_loss = total_loss + distill_losses["total_loss"]
            distill_loss_val = distill_losses.get("distill_distillation_loss", torch.tensor(0.0)).item()
        
        # Compute cosine similarity (for patch-level, average over patches)
        if is_patch_level:
            cosine_sim = torch.nn.functional.cosine_similarity(
                z_goal_pred, z_goal_gt, dim=-1
            ).mean(dim=-1)  # Average over patches
        else:
            cosine_sim = torch.nn.functional.cosine_similarity(
                z_goal_pred, z_goal_gt, dim=-1
            )
        
        # Update meters
        loss_meter.update(losses["l2_loss"].item(), n=len(texts))  # L2 only for comparison
        l2_meter.update(losses["l2_loss"].item(), n=len(texts))
        cosine_loss_meter.update(losses.get("cosine_loss", torch.tensor(0.0)).item(), n=len(texts))
        cosine_sim_meter.update(cosine_sim.mean().item(), n=len(texts))
        distill_meter.update(distill_loss_val, n=len(texts))
        total_loss_meter.update(total_loss.item(), n=len(texts))  # Total including distillation
        
        # Per-task metrics
        for i, task in enumerate(task_names):
            if task not in task_metrics:
                task_metrics[task] = {"l2": [], "cosine_sim": []}
            
            if is_patch_level:
                l2_dist = (z_goal_pred[i] - z_goal_gt[i]).pow(2).mean().sqrt().item()
            else:
                l2_dist = (z_goal_pred[i] - z_goal_gt[i]).pow(2).sum().sqrt().item()
            task_metrics[task]["l2"].append(l2_dist)
            task_metrics[task]["cosine_sim"].append(cosine_sim[i].item())
    
    # Aggregate per-task metrics
    task_summary = {}
    for task, metrics in task_metrics.items():
        task_summary[f"l2_{task}"] = np.mean(metrics["l2"])
        task_summary[f"cosine_sim_{task}"] = np.mean(metrics["cosine_sim"])
    
    return {
        "loss": loss_meter.avg,  # L2 only (backward compatible)
        "total_loss": total_loss_meter.avg,  # Total loss matching training
        "l2_loss": l2_meter.avg,
        "cosine_loss": cosine_loss_meter.avg,
        "cosine_sim": cosine_sim_meter.avg,
        "distill_loss": distill_meter.avg,
        **task_summary,
    }


def main(config: Dict = None, args: argparse.Namespace = None):
    """Main training function."""
    
    # Parse arguments if not provided
    if args is None:
        parser = argparse.ArgumentParser(description="Train GoalHead")
        parser.add_argument("--config", type=str, required=True, help="Config file path")
        parser.add_argument("--debug", action="store_true", help="Debug mode")
        parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
        parser.add_argument("--cpu", action="store_true", help="Force CPU execution")
        args = parser.parse_args()
    
    # Load config if not provided
    if config is None:
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Expand environment variables in config
        config = expand_env_vars(config)
        logger.info(f"📂 Loaded config from {args.config}")
    
    # Validate environment
    env_paths = get_paths()
    
    # Setup
    force_cpu = getattr(args, 'cpu', False) or config.get('force_cpu', False)
    if force_cpu:
        device = torch.device("cpu")
        logger.info("🖥️ Forcing CPU execution")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("seed", 42)
    set_seed(seed)
    
    # Create output directory
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = env_paths["lang_goal_ckpt"] / "goalhead" / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    
    # Save git info
    git_info = get_git_info()
    git_path = output_dir / "git_info.json"
    with open(git_path, "w") as f:
        json.dump(git_info, f, indent=2)
    
    logger.info("=" * 60)
    logger.info("GoalHead Training")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Device: {device}")
    logger.info(f"Git commit: {git_info['commit_hash']}")
    logger.info("=" * 60)
    
    # Initialize wandb
    wandb_run = None
    if config.get("use_wandb", True):
        try:
            import wandb
            wandb_run = wandb.init(
                project=config.get("wandb_project", "jepa-lang-goal"),
                name=f"goalhead_{run_id}",
                config=config,
            )
            logger.info("✅ Initialized wandb")
        except Exception as e:
            logger.warning(f"⚠️ Failed to initialize wandb: {e}")
    
    # Load pretrained model
    checkpoint_path = config.get("pretrained_checkpoint")
    if checkpoint_path is None:
        checkpoint_path = str(env_paths["pretrained"] / "mw_jepa-wm.pth.tar")
    
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(
            f"Pretrained checkpoint not found: {checkpoint_path}\n"
            f"Please download using:\n"
            f"  mkdir -p {env_paths['pretrained']}\n"
            f"  curl -L -O https://dl.fbaipublicfiles.com/jepa-wms/mw_jepa-wm.pth.tar\n"
            f"  mv mw_jepa-wm.pth.tar {env_paths['pretrained']}/"
        )
    
    # Load visual encoder from pretrained model
    model_kwargs = config.get("model_kwargs", {})
    
    visual_encoder = load_pretrained_model(
        checkpoint_path=checkpoint_path,
        device=device,
        model_kwargs=model_kwargs,
    )
    
    # Build text encoder
    text_config = config.get("text_encoder", {})
    text_config["device"] = str(device)
    text_encoder = build_text_encoder(text_config)
    
    # Build GoalHead
    goalhead_config = config.get("goal_head", {})
    goalhead_config["visual_dim"] = 384  # DINOv2 ViT-S
    goalhead_config["text_dim"] = text_encoder.embed_dim
    goalhead_config["output_dim"] = 384  # Match visual encoder
    
    goal_head = build_goal_head(goalhead_config)
    goal_head = goal_head.to(device)
    
    # FR-5: Build Prompt Tuning module (if enabled)
    prompt_tuning = None
    prompt_cfg = config.get("prompt_tuning", {})
    if prompt_cfg.get("enabled", False):
        prompt_tuning = PromptTuningModule(
            embed_dim=text_encoder.embed_dim,
            num_context_tokens=prompt_cfg.get("num_context_tokens", 16),
            context_position=prompt_cfg.get("context_position", "prefix"),
            dropout=prompt_cfg.get("dropout", 0.1),
        )
        prompt_tuning = prompt_tuning.to(device)
        logger.info("🎯 Prompt Tuning enabled")
    
    # FR-2/4/6: Build Distillation Loss (if enabled)
    distillation_loss_fn = None
    distill_cfg = config.get("distillation", {})
    if distill_cfg.get("enabled", False):
        distillation_loss_fn = DistillationLoss(
            distillation_weight=distill_cfg.get("distillation_weight", 1.0),
            distillation_type=distill_cfg.get("distillation_type", "mse_cosine"),
            distillation_temperature=distill_cfg.get("temperature", 1.0),
            invariance_weight=distill_cfg.get("invariance_weight", 0.5),
            invariance_type=distill_cfg.get("invariance_type", "centroid"),
            contrastive_weight=distill_cfg.get("contrastive_weight", 0.1),
            contrastive_temperature=distill_cfg.get("contrastive_temperature", 0.07),
            l2_weight=config.get("l2_weight", 1.0),
            cosine_weight=config.get("cosine_weight", 0.1),
        )
        logger.info("📊 Distillation Loss enabled")
    
    # FR-6: Build Hard Negative Miner (if enabled)
    hard_negative_miner = None
    contrastive_cfg = config.get("contrastive", {})
    if contrastive_cfg.get("enabled", False):
        hard_negative_miner = HardNegativeMiner(
            num_negatives=contrastive_cfg.get("num_negatives", 5),
            similarity_threshold=contrastive_cfg.get("similarity_threshold", 0.8),
        )
        logger.info("⚡ Hard Negative Mining enabled")
    
    # Build optimizer - include prompt_tuning parameters if enabled
    params_to_optimize = list(goal_head.parameters())
    if prompt_tuning is not None:
        params_to_optimize += list(prompt_tuning.parameters())
    
    optimizer = torch.optim.AdamW(
        params_to_optimize,
        lr=config.get("learning_rate", 1e-4),
        weight_decay=config.get("weight_decay", 1e-5),
    )
    
    # Mixed precision - only enable on CUDA devices
    scaler = None
    use_mixed_precision = config.get("mixed_precision", True) and device.type == "cuda"
    if use_mixed_precision:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("🔧 Mixed precision enabled (CUDA)")
    else:
        logger.info(f"🔧 Mixed precision disabled (device={device.type})")
    
    # Build dataloaders
    data_path = config.get("data_path", str(env_paths["metaworld_data"] / "data"))
    index_dir = config.get("index_dir", str(env_paths["lang_goal_index"]))
    
    # FR-6: Check if hard negative sampling is needed
    contrastive_cfg = config.get("contrastive", {})
    include_hard_negatives = contrastive_cfg.get("dataset_hard_negatives", False)
    num_hard_negatives = contrastive_cfg.get("num_negatives", 5)
    
    train_loader, val_loader, test_loader = build_dataloaders(
        index_dir=index_dir,
        data_path=data_path,
        batch_size=config.get("batch_size", 32),
        num_workers=config.get("num_workers", 4),
        include_proprio=config.get("use_proprio", False),
        use_augmentation=config.get("use_augmentation", False),  # Disabled by default
        include_hard_negatives=include_hard_negatives,
        num_hard_negatives=num_hard_negatives,
    )
    
    # Setup CSV logger (train/val 同一指標)
    csv_logger = CSVLogger(
        str(output_dir / "training_log.csv"),
        ("%d", "epoch"),
        ("%.5f", "train_total_loss"),
        ("%.5f", "train_l2"),
        ("%.5f", "train_cosine_sim"),
        ("%.5f", "train_distill"),
        ("%.5f", "val_total_loss"),
        ("%.5f", "val_l2"),
        ("%.5f", "val_cosine_sim"),
        ("%.5f", "val_distill"),
    )
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        start_epoch = load_checkpoint(goal_head, optimizer, args.resume)
    
    # Training loop
    num_epochs = config.get("num_epochs", 100)
    best_val_loss = float("inf")
    
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_metrics = train_epoch(
            goal_head=goal_head,
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
            train_loader=train_loader,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            config=config,
            scaler=scaler,
            wandb_run=wandb_run,
            prompt_tuning=prompt_tuning,
            distillation_loss_fn=distillation_loss_fn,
            hard_negative_miner=hard_negative_miner,
        )
        
        # Validate
        val_metrics = evaluate(
            goal_head=goal_head,
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
            val_loader=val_loader,
            device=device,
            config=config,
            prompt_tuning=prompt_tuning,
            distillation_loss_fn=distillation_loss_fn,
        )
        
        # Log (train/val 同一指標で整列表示)
        train_total = train_metrics['total_loss']
        val_total = val_metrics.get('total_loss', val_metrics['loss'])
        logger.info(
            f"Epoch {epoch}:\n"
            f"  [train] loss={train_total:.4f}, l2={train_metrics['l2_loss']:.4f}, "
            f"cos_sim={train_metrics['cosine_sim']:.4f}, "
            f"distill={train_metrics.get('distill_loss', 0):.4f}, "
            f"contrastive={train_metrics.get('contrastive_loss', 0):.4f}\n"
            f"  [val]   loss={val_total:.4f}, l2={val_metrics['l2_loss']:.4f}, "
            f"cos_sim={val_metrics['cosine_sim']:.4f}, "
            f"distill={val_metrics.get('distill_loss', 0):.4f}"
        )
        
        # Per-task metrics
        task_keys = [k for k in val_metrics if k.startswith("cosine_sim_")]
        if task_keys:
            task_lines = []
            for k in sorted(task_keys):
                task_name = k.replace("cosine_sim_", "")
                l2_k = f"l2_{task_name}"
                task_lines.append(
                    f"    {task_name}: cos_sim={val_metrics[k]:.4f}, l2={val_metrics.get(l2_k, 0):.4f}"
                )
            logger.info("  [val per-task]\n" + "\n".join(task_lines))
        
        csv_logger.log(
            epoch,
            train_total,
            train_metrics["l2_loss"],
            train_metrics["cosine_sim"],
            train_metrics.get("distill_loss", 0),
            val_total,
            val_metrics["l2_loss"],
            val_metrics["cosine_sim"],
            val_metrics.get("distill_loss", 0),
        )
        
        if wandb_run is not None:
            epoch_log = {
                "epoch": epoch,
                "train/total_loss": train_total,
                "train/l2_loss": train_metrics["l2_loss"],
                "train/cosine_sim": train_metrics["cosine_sim"],
                "train/cosine_loss": train_metrics.get("cosine_loss", 0),
                "train/distill_loss": train_metrics.get("distill_loss", 0),
                "train/invariance_loss": train_metrics.get("invariance_loss", 0),
                "train/contrastive_loss": train_metrics.get("contrastive_loss", 0),
                "val/total_loss": val_total,
                "val/l2_loss": val_metrics["l2_loss"],
                "val/cosine_sim": val_metrics["cosine_sim"],
                "val/distill_loss": val_metrics.get("distill_loss", 0),
            }
            # Per-task wandb
            for k in task_keys:
                task_name = k.replace("cosine_sim_", "")
                epoch_log[f"val/cosine_sim_{task_name}"] = val_metrics[k]
                l2_k = f"l2_{task_name}"
                if l2_k in val_metrics:
                    epoch_log[f"val/l2_{task_name}"] = val_metrics[l2_k]
            wandb_run.log(epoch_log)
        
        # Save checkpoints (use total_loss for best model selection)
        val_total_loss = val_metrics.get("total_loss", val_metrics["loss"])
        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            save_checkpoint(
                goal_head, optimizer, epoch, val_total_loss,
                str(output_dir / "best_model.pth"),
                config,
                best_val_loss=best_val_loss,
                prompt_tuning=prompt_tuning,
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % config.get("save_every", 10) == 0:
            save_checkpoint(
                goal_head, optimizer, epoch, val_total_loss,
                str(output_dir / f"checkpoint_epoch{epoch}.pth"),
                config,
                best_val_loss=best_val_loss,
                prompt_tuning=prompt_tuning,
            )
    
    # Final evaluation on test set
    if test_loader is not None:
        test_metrics = evaluate(
            goal_head=goal_head,
            visual_encoder=visual_encoder,
            text_encoder=text_encoder,
            val_loader=test_loader,
            device=device,
            config=config,
            prompt_tuning=prompt_tuning,
            distillation_loss_fn=distillation_loss_fn,
        )
        
        # Save test results
        test_results_path = output_dir / "test_results.json"
        with open(test_results_path, "w") as f:
            json.dump(test_metrics, f, indent=2)
        
        test_total = test_metrics.get('total_loss', test_metrics['loss'])
        logger.info(f"✅ Test results saved to {test_results_path}")
        logger.info(f"Test total_loss: {test_total:.4f} (l2={test_metrics['l2_loss']:.4f}, distill={test_metrics.get('distill_loss', 0):.4f})")
        logger.info(f"Test cosine sim: {test_metrics['cosine_sim']:.4f}")
    
    if wandb_run is not None:
        wandb_run.finish()
    
    logger.info("=" * 60)
    logger.info("✅ Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("=" * 60)
    
    return str(output_dir)


if __name__ == "__main__":
    main()
