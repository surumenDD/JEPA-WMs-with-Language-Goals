# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Dataset for GoalHead training.

Provides paired (I0, T, z_goal_gt) samples for supervised learning of GoalHead.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger(__name__)


class GoalHeadDataset(Dataset):
    """
    Dataset for training GoalHead.
    
    Each sample consists of:
    - I0: Initial frame image
    - T: Language instruction (text string)
    - I_goal: Goal frame image (for computing z_goal_gt)
    - proprio: Optional proprioception state
    - I_negatives: Optional hard negative frames for contrastive learning
    """
    
    def __init__(
        self,
        index_path: str,
        data_path: str,
        transform: Optional[Callable] = None,
        goal_frame: str = "last",  # "last" or "success" or specific frame index
        context_frames: int = 1,  # Number of initial frames to use
        include_proprio: bool = True,
        include_hard_negatives: bool = False,  # FR-6: Hard negative sampling
        num_hard_negatives: int = 3,  # Number of hard negatives per sample
    ):
        """
        Initialize GoalHead dataset.
        
        Args:
            index_path: Path to index JSONL file
            data_path: Path to MetaWorld data directory
            transform: Optional image transform
            goal_frame: How to select goal frame ("last", "success", or int)
            context_frames: Number of initial context frames
            include_proprio: Whether to include proprioception
            include_hard_negatives: Whether to sample hard negative frames
            num_hard_negatives: Number of hard negatives to sample per episode
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.goal_frame = goal_frame
        self.context_frames = context_frames
        self.include_proprio = include_proprio
        self.include_hard_negatives = include_hard_negatives
        self.num_hard_negatives = num_hard_negatives
        
        # Load index
        self.episodes = self._load_index(index_path)
        logger.info(f"✅ Loaded {len(self.episodes)} episodes from {index_path}")
        
        # Load underlying dataset
        self._init_dataset()
    
    def _load_index(self, index_path: str) -> List[Dict]:
        """Load episode index from JSONL file."""
        episodes = []
        with open(index_path, "r") as f:
            for line in f:
                if line.strip():
                    episodes.append(json.loads(line))
        return episodes
    
    def _init_dataset(self):
        """Initialize the underlying HuggingFace dataset."""
        from datasets import load_dataset
        
        logger.info(f"📂 Loading dataset from {self.data_path}...")
        self.dataset = load_dataset(
            "parquet",
            data_dir=str(self.data_path),
            split="train"
        )
        logger.info(f"✅ Dataset loaded with {len(self.dataset)} total episodes")
    
    def _decode_video(self, video_data) -> np.ndarray:
        """Decode video frames from dataset."""
        import io
        
        if isinstance(video_data, dict) and "bytes" in video_data:
            # Standard HF format with bytes
            import imageio
            reader = imageio.get_reader(io.BytesIO(video_data["bytes"]), format="mp4")
            frames = [frame for frame in reader]
            reader.close()
            return np.stack(frames)
        else:
            # torchcodec auto-decoded VideoDecoder object
            frames = []
            for i in range(len(video_data)):
                frame = video_data[i]
                frame_np = frame.data.permute(1, 2, 0).numpy()
                frames.append(frame_np)
            return np.stack(frames)
    
    def _get_goal_frame_idx(self, episode_info: Dict, num_frames: int) -> int:
        """Determine which frame to use as goal."""
        if self.goal_frame == "last":
            return num_frames - 1
        elif self.goal_frame == "success":
            # Could use reward info to find success frame
            # For now, default to last
            return num_frames - 1
        elif isinstance(self.goal_frame, int):
            return min(self.goal_frame, num_frames - 1)
        else:
            return num_frames - 1
    
    def __len__(self) -> int:
        return len(self.episodes)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            Dictionary with:
            - I0: Initial frame(s) [context_frames, C, H, W]
            - I_goal: Goal frame [C, H, W]
            - text: Language instruction string
            - proprio: Proprioception state [proprio_dim] (if include_proprio)
        """
        episode_info = self.episodes[idx]
        dataset_idx = episode_info["dataset_idx"]
        
        # Get raw data from dataset
        row = self.dataset[dataset_idx]
        
        # Decode video frames
        video = row["video"]
        frames = self._decode_video(video)  # [T, H, W, C]
        num_frames = len(frames)
        
        # Get initial frame(s)
        I0_frames = frames[:self.context_frames]  # [context_frames, H, W, C]
        I0 = torch.tensor(I0_frames, dtype=torch.float32)
        I0 = I0.permute(0, 3, 1, 2) / 255.0  # [context_frames, C, H, W]
        
        # Get goal frame
        goal_idx = self._get_goal_frame_idx(episode_info, num_frames)
        I_goal = torch.tensor(frames[goal_idx], dtype=torch.float32)
        I_goal = I_goal.permute(2, 0, 1) / 255.0  # [C, H, W]
        
        # Apply transforms
        if self.transform is not None:
            I0 = self.transform(I0)
            I_goal = self.transform(I_goal)
        
        # Get text instruction
        text = episode_info["text_instruction"]
        
        # Prepare output
        sample = {
            "I0": I0.squeeze(0) if self.context_frames == 1 else I0,
            "I_goal": I_goal,
            "text": text,
            "task_name": episode_info["task_name"],
            "episode_id": episode_info["episode_id"],
            "goal_frame_idx": goal_idx,
            "num_frames": num_frames,
        }
        
        # FR-6: Sample hard negative frames (near-miss states)
        if self.include_hard_negatives and num_frames > 10:
            # Sample frames that are close to goal but not the goal itself
            neg_candidates = []
            # Near the goal (temporal proximity)
            near_goal_range = range(max(0, goal_idx - 10), min(num_frames, goal_idx + 10))
            for neg_idx in near_goal_range:
                if abs(neg_idx - goal_idx) > 3:  # Not too close
                    neg_candidates.append(neg_idx)
            
            # Also add early/mid trajectory frames as easier negatives
            if goal_idx > 20:
                neg_candidates.extend([goal_idx // 4, goal_idx // 2])
            
            # Sample up to num_hard_negatives
            if neg_candidates:
                sampled_neg_indices = np.random.choice(
                    neg_candidates,
                    size=min(self.num_hard_negatives, len(neg_candidates)),
                    replace=False
                )
                neg_frames = []
                for neg_idx in sampled_neg_indices:
                    I_neg = torch.tensor(frames[neg_idx], dtype=torch.float32)
                    I_neg = I_neg.permute(2, 0, 1) / 255.0
                    if self.transform is not None:
                        I_neg = self.transform(I_neg)
                    neg_frames.append(I_neg)
                
                if neg_frames:
                    sample["I_negatives"] = torch.stack(neg_frames)  # [K, C, H, W]
        
        # Add proprioception if available
        if self.include_proprio and "states" in row and row["states"] is not None:
            states = np.array(row["states"])
            proprio = torch.tensor(states[0, :4], dtype=torch.float32)  # First 4 dims
            sample["proprio"] = proprio
        
        return sample


class GoalHeadDataLoader:
    """
    DataLoader wrapper for GoalHead training.
    
    Handles batching and collation of text strings.
    """
    
    def __init__(
        self,
        dataset: GoalHeadDataset,
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        from torch.utils.data import DataLoader
        
        self.dataset = dataset
        self.dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            collate_fn=self._collate_fn,
        )
    
    def _collate_fn(self, batch: List[Dict]) -> Dict:
        """Custom collate function to handle text strings."""
        collated = {
            "I0": torch.stack([s["I0"] for s in batch]),
            "I_goal": torch.stack([s["I_goal"] for s in batch]),
            "text": [s["text"] for s in batch],
            "task_name": [s["task_name"] for s in batch],
            "episode_id": [s["episode_id"] for s in batch],
            "goal_frame_idx": torch.tensor([s.get("goal_frame_idx", -1) for s in batch]),
            "num_frames": torch.tensor([s.get("num_frames", -1) for s in batch]),
        }
        
        if "proprio" in batch[0]:
            collated["proprio"] = torch.stack([s["proprio"] for s in batch])
        
        # FR-6: Collate hard negatives (variable number per sample)
        if "I_negatives" in batch[0]:
            # Each sample may have different number of negatives
            # Pad to max and create a mask
            neg_counts = [s["I_negatives"].shape[0] if "I_negatives" in s else 0 for s in batch]
            max_negs = max(neg_counts) if neg_counts else 0
            
            if max_negs > 0:
                neg_list = []
                neg_mask = []
                for s in batch:
                    if "I_negatives" in s:
                        negs = s["I_negatives"]
                        n = negs.shape[0]
                        if n < max_negs:
                            # Pad with zeros
                            pad = torch.zeros(max_negs - n, *negs.shape[1:])
                            negs = torch.cat([negs, pad], dim=0)
                        neg_list.append(negs)
                        neg_mask.append([1.0] * n + [0.0] * (max_negs - n))
                    else:
                        neg_list.append(torch.zeros(max_negs, *batch[0]["I_goal"].shape))
                        neg_mask.append([0.0] * max_negs)
                
                collated["I_negatives"] = torch.stack(neg_list)  # [B, K, C, H, W]
                collated["neg_mask"] = torch.tensor(neg_mask)  # [B, K]
        
        return collated
    
    def __iter__(self):
        return iter(self.dataloader)
    
    def __len__(self):
        return len(self.dataloader)


def build_dataloaders(
    index_dir: str,
    data_path: str,
    batch_size: int = 32,
    num_workers: int = 4,
    transform: Optional[Callable] = None,
    train_transform: Optional[Callable] = None,
    val_transform: Optional[Callable] = None,
    goal_frame: str = "last",
    include_proprio: bool = True,
    use_augmentation: bool = True,
    include_hard_negatives: bool = False,  # FR-6: Enable hard negative sampling
    num_hard_negatives: int = 3,  # FR-6: Number of hard negatives per sample
) -> Tuple[GoalHeadDataLoader, GoalHeadDataLoader, GoalHeadDataLoader]:
    """
    Build train/val/test dataloaders for GoalHead training.
    
    Args:
        index_dir: Directory containing train/val/test index files
        data_path: Path to MetaWorld data
        batch_size: Batch size
        num_workers: Number of dataloader workers
        transform: Optional image transform (used if train/val_transform not specified)
        train_transform: Optional training-specific transform with augmentation
        val_transform: Optional validation transform (no augmentation)
        goal_frame: How to select goal frame
        include_proprio: Whether to include proprioception
        use_augmentation: Whether to use data augmentation for training
        include_hard_negatives: Whether to sample hard negative frames (FR-6)
        num_hard_negatives: Number of hard negatives to sample per episode
        
    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    import torchvision.transforms as T
    
    index_dir = Path(index_dir)
    
    # Build transforms if not provided
    if train_transform is None and use_augmentation:
        # Training augmentation: Color jitter, small random crop, etc.
        # These help with generalization without being too aggressive
        train_transform = T.Compose([
            T.RandomResizedCrop(224, scale=(0.9, 1.0), ratio=(0.95, 1.05)),
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.02),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        logger.info("🎨 Using training augmentation: RandomResizedCrop + ColorJitter")
    elif train_transform is None:
        train_transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    if val_transform is None:
        # Validation: only resize and normalize (no augmentation)
        val_transform = T.Compose([
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    loaders = {}
    for split in ["train", "val", "test"]:
        index_path = index_dir / f"{split}_index.jsonl"
        
        if not index_path.exists():
            logger.warning(f"⚠️ Index not found: {index_path}")
            continue
        
        # Use appropriate transform for each split
        if split == "train":
            split_transform = train_transform
        else:
            split_transform = val_transform
        
        dataset = GoalHeadDataset(
            index_path=str(index_path),
            data_path=data_path,
            transform=split_transform,
            goal_frame=goal_frame,
            include_proprio=include_proprio,
            include_hard_negatives=include_hard_negatives if split == "train" else False,
            num_hard_negatives=num_hard_negatives,
        )
        
        loaders[split] = GoalHeadDataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
        )
    
    return (
        loaders.get("train"),
        loaders.get("val"),
        loaders.get("test"),
    )
