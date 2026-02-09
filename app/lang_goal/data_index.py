# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Data indexing module for language-conditioned goal planning.

This module handles:
- Scanning parquet files for MetaWorld data
- Creating episode-level indices with task information
- Generating language instructions from task names
- Train/val/test splitting
"""

import argparse
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from datasets import load_dataset

from app.lang_goal.env_utils import get_git_info, get_paths, validate_all_env_vars
from src.utils.logging import get_logger
from src.utils.yaml_utils import expand_env_vars

logger = get_logger(__name__)


class TaskToTextMapper:
    """Maps task names to natural language instructions."""
    
    def __init__(self, mapping_file: str = None):
        """
        Initialize the task-to-text mapper.
        
        Args:
            mapping_file: Path to YAML file with task->text mappings.
                         If None, uses default from assets/lang_goal/task_to_text.yaml
        """
        self.mapping = {}
        self.templates = {}
        
        if mapping_file is None:
            # Try to find default mapping file
            default_paths = [
                Path(__file__).parent.parent.parent / "assets" / "lang_goal" / "task_to_text.yaml",
                Path(os.environ.get("JEPAWM_HOME", ".")) / "jepa-wms" / "assets" / "lang_goal" / "task_to_text.yaml",
            ]
            for path in default_paths:
                if path.exists():
                    mapping_file = str(path)
                    break
        
        if mapping_file and Path(mapping_file).exists():
            self._load_mapping(mapping_file)
            logger.info(f"✅ Loaded {len(self.mapping)} task-to-text mappings from {mapping_file}")
        else:
            logger.warning("⚠️ No task-to-text mapping file found, using template generation only")
    
    def _load_mapping(self, filepath: str):
        """Load mapping from YAML file."""
        with open(filepath, "r") as f:
            data = yaml.safe_load(f)
        
        # Separate templates from mappings
        if "_template" in data:
            self.templates = data.pop("_template")
        
        self.mapping = data
    
    def get_text(self, task_name: str) -> str:
        """
        Get natural language instruction for a task.
        
        Args:
            task_name: Name of the task (e.g., "drawer-open", "mw-reach-v2")
            
        Returns:
            Natural language instruction string
        """
        # Direct lookup
        if task_name in self.mapping:
            return self.mapping[task_name]
        
        # Try without version suffix
        base_name = re.sub(r"-v\d+$", "", task_name)
        if base_name in self.mapping:
            return self.mapping[base_name]
        
        # Try without "mw-" prefix
        if task_name.startswith("mw-"):
            without_prefix = task_name[3:]
            if without_prefix in self.mapping:
                return self.mapping[without_prefix]
            base_without_prefix = re.sub(r"-v\d+$", "", without_prefix)
            if base_without_prefix in self.mapping:
                return self.mapping[base_without_prefix]
        
        # Fall back to template generation
        return self._generate_from_template(task_name)
    
    def _generate_from_template(self, task_name: str) -> str:
        """Generate instruction from task name using patterns."""
        # Clean task name
        clean_name = task_name.replace("mw-", "").replace("-v2", "").replace("-v1", "")
        
        # Try pattern matching
        if "patterns" in self.templates:
            for pattern_info in self.templates["patterns"]:
                match = re.match(pattern_info["pattern"], clean_name)
                if match:
                    return pattern_info["template"].format(*match.groups())
        
        # Default template
        default_template = self.templates.get("default", "Complete the {task_name} task")
        
        # Make task name more readable
        readable_name = clean_name.replace("-", " ").replace("_", " ")
        
        return default_template.format(task_name=readable_name)


class MetaWorldDataIndexer:
    """Creates and manages data indices for MetaWorld dataset."""
    
    def __init__(
        self,
        data_path: str,
        output_path: str,
        text_mapper: TaskToTextMapper = None,
        split_seed: int = 42,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        """
        Initialize the data indexer.
        
        Args:
            data_path: Path to MetaWorld parquet data
            output_path: Path to save index files
            text_mapper: TaskToTextMapper instance
            split_seed: Random seed for splitting
            train_ratio: Fraction of data for training
            val_ratio: Fraction of data for validation
            test_ratio: Fraction of data for testing
        """
        self.data_path = Path(data_path)
        self.output_path = Path(output_path)
        self.text_mapper = text_mapper or TaskToTextMapper()
        self.split_seed = split_seed
        
        # Validate ratios
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Split ratios must sum to 1.0"
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        # Validate data path
        if not self.data_path.exists():
            raise FileNotFoundError(
                f"Data path does not exist: {self.data_path}\n"
                f"Please ensure MetaWorld data is downloaded to {self.data_path}"
            )
        
        # Create output directory
        self.output_path.mkdir(parents=True, exist_ok=True)
        
        self.index = []
        self.metadata = {}
    
    def scan_data(self) -> List[Dict]:
        """
        Scan parquet files and extract episode information.
        
        Returns:
            List of episode dictionaries with metadata
        """
        logger.info(f"📂 Scanning data at {self.data_path}...")
        
        # Load dataset using HuggingFace datasets
        try:
            ds = load_dataset("parquet", data_dir=str(self.data_path), split="train")
            logger.info(f"✅ Found {len(ds)} episodes")
        except Exception as e:
            raise RuntimeError(
                f"Failed to load dataset from {self.data_path}: {e}\n"
                f"Expected format: parquet files with columns [video, states, actions, rewards, task]"
            )
        
        episodes = []
        task_counts = {}
        
        for idx in range(len(ds)):
            row = ds[idx]
            
            # Extract episode information
            task_name = row.get("task", "unknown")
            
            # Count states/actions to determine episode length
            states = row.get("states", [])
            actions = row.get("actions", [])
            num_frames = len(actions) if actions is not None else 0
            
            # Get language instruction
            text_instruction = self.text_mapper.get_text(task_name)
            
            episode_info = {
                "episode_id": f"ep_{idx:06d}",
                "dataset_idx": idx,
                "task_name": task_name,
                "task_id": self._get_task_id(task_name),
                "num_frames": num_frames,
                "text_instruction": text_instruction,
                "has_proprio": "states" in row and row["states"] is not None,
                "has_rewards": "rewards" in row and row["rewards"] is not None,
            }
            
            episodes.append(episode_info)
            task_counts[task_name] = task_counts.get(task_name, 0) + 1
        
        # Log statistics
        logger.info(f"📊 Task distribution:")
        for task, count in sorted(task_counts.items()):
            logger.info(f"   {task}: {count} episodes")
        
        self.index = episodes
        return episodes
    
    def _get_task_id(self, task_name: str) -> int:
        """Get numeric task ID from task name."""
        # Create consistent task ID mapping
        if not hasattr(self, "_task_id_map"):
            self._task_id_map = {}
        
        clean_name = task_name.replace("mw-", "").replace("-v2", "").replace("-v1", "")
        
        if clean_name not in self._task_id_map:
            self._task_id_map[clean_name] = len(self._task_id_map)
        
        return self._task_id_map[clean_name]
    
    def split_data(self) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """
        Split episodes into train/val/test sets.
        
        Returns:
            Tuple of (train_episodes, val_episodes, test_episodes)
        """
        if not self.index:
            raise ValueError("No data indexed. Call scan_data() first.")
        
        logger.info(f"🔀 Splitting data with seed={self.split_seed}...")
        
        np.random.seed(self.split_seed)
        
        # Group by task for stratified splitting
        task_episodes = {}
        for ep in self.index:
            task = ep["task_name"]
            if task not in task_episodes:
                task_episodes[task] = []
            task_episodes[task].append(ep)
        
        train_episodes = []
        val_episodes = []
        test_episodes = []
        
        for task, episodes in task_episodes.items():
            # Shuffle episodes within task
            shuffled = episodes.copy()
            np.random.shuffle(shuffled)
            
            n = len(shuffled)
            n_train = int(n * self.train_ratio)
            n_val = int(n * self.val_ratio)
            
            train_episodes.extend(shuffled[:n_train])
            val_episodes.extend(shuffled[n_train:n_train + n_val])
            test_episodes.extend(shuffled[n_train + n_val:])
        
        # Assign split labels
        for ep in train_episodes:
            ep["split"] = "train"
        for ep in val_episodes:
            ep["split"] = "val"
        for ep in test_episodes:
            ep["split"] = "test"
        
        logger.info(f"📊 Split sizes: train={len(train_episodes)}, "
                   f"val={len(val_episodes)}, test={len(test_episodes)}")
        
        return train_episodes, val_episodes, test_episodes
    
    def save_index(self, train: List[Dict], val: List[Dict], test: List[Dict]) -> Dict[str, str]:
        """
        Save index files to disk.
        
        Args:
            train, val, test: Split episode lists
            
        Returns:
            Dictionary of output file paths
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save as JSONL (one JSON object per line)
        paths = {}
        for split_name, episodes in [("train", train), ("val", val), ("test", test)]:
            filepath = self.output_path / f"{split_name}_index.jsonl"
            with open(filepath, "w") as f:
                for ep in episodes:
                    f.write(json.dumps(ep) + "\n")
            paths[split_name] = str(filepath)
            logger.info(f"💾 Saved {split_name} index: {filepath}")
        
        # Save combined index as CSV for easy viewing
        all_episodes = train + val + test
        df = pd.DataFrame(all_episodes)
        csv_path = self.output_path / "full_index.csv"
        df.to_csv(csv_path, index=False)
        paths["csv"] = str(csv_path)
        logger.info(f"💾 Saved full index CSV: {csv_path}")
        
        # Save metadata
        self.metadata = {
            "created_at": timestamp,
            "data_path": str(self.data_path),
            "split_seed": self.split_seed,
            "train_ratio": self.train_ratio,
            "val_ratio": self.val_ratio,
            "test_ratio": self.test_ratio,
            "num_train": len(train),
            "num_val": len(val),
            "num_test": len(test),
            "num_total": len(all_episodes),
            "tasks": list(set(ep["task_name"] for ep in all_episodes)),
            "git_info": get_git_info(),
        }
        
        metadata_path = self.output_path / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(self.metadata, f, indent=2)
        paths["metadata"] = str(metadata_path)
        logger.info(f"💾 Saved metadata: {metadata_path}")
        
        return paths


def load_index(index_path: str, split: str = "train") -> List[Dict]:
    """
    Load a saved index file.
    
    Args:
        index_path: Path to index directory
        split: Which split to load ("train", "val", "test")
        
    Returns:
        List of episode dictionaries
    """
    filepath = Path(index_path) / f"{split}_index.jsonl"
    
    if not filepath.exists():
        raise FileNotFoundError(f"Index file not found: {filepath}")
    
    episodes = []
    with open(filepath, "r") as f:
        for line in f:
            if line.strip():
                episodes.append(json.loads(line))
    
    logger.info(f"✅ Loaded {len(episodes)} episodes from {filepath}")
    return episodes


def main():
    """Main entry point for data indexing."""
    parser = argparse.ArgumentParser(description="Create data index for language-goal planning")
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Path to MetaWorld data (overrides config/env var)",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
        help="Path to save index (overrides config/env var)",
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=42,
        help="Random seed for train/val/test split",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.7,
        help="Fraction for training",
    )
    parser.add_argument(
        "--val_ratio",
        type=float,
        default=0.15,
        help="Fraction for validation",
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        default=0.15,
        help="Fraction for testing",
    )
    
    args = parser.parse_args()
    
    # Load config if provided
    config = {}
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config = yaml.safe_load(f)
        # Expand environment variables in config
        config = expand_env_vars(config)
        logger.info(f"📂 Loaded config from {args.config}")
    
    # Validate environment variables
    env_paths = get_paths()
    
    # Determine paths (CLI > config > env var)
    data_path = (
        args.data_path or
        config.get("data_path") or
        str(env_paths["metaworld_data"] / "data")
    )
    
    output_path = (
        args.output_path or
        config.get("output_path") or
        str(env_paths["lang_goal_index"])
    )
    
    # Get split parameters
    split_seed = args.split_seed or config.get("split_seed", 42)
    train_ratio = args.train_ratio or config.get("train_ratio", 0.7)
    val_ratio = args.val_ratio or config.get("val_ratio", 0.15)
    test_ratio = args.test_ratio or config.get("test_ratio", 0.15)
    
    logger.info("=" * 60)
    logger.info("Data Index Generation")
    logger.info("=" * 60)
    logger.info(f"Data path: {data_path}")
    logger.info(f"Output path: {output_path}")
    logger.info(f"Split seed: {split_seed}")
    logger.info(f"Train/Val/Test ratio: {train_ratio}/{val_ratio}/{test_ratio}")
    logger.info("=" * 60)
    
    # Create indexer and run
    indexer = MetaWorldDataIndexer(
        data_path=data_path,
        output_path=output_path,
        split_seed=split_seed,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    
    # Scan data
    indexer.scan_data()
    
    # Split data
    train, val, test = indexer.split_data()
    
    # Save index
    saved_paths = indexer.save_index(train, val, test)
    
    logger.info("=" * 60)
    logger.info("✅ Data indexing complete!")
    logger.info(f"Output files:")
    for name, path in saved_paths.items():
        logger.info(f"   {name}: {path}")
    logger.info("=" * 60)
    
    return saved_paths


if __name__ == "__main__":
    main()
