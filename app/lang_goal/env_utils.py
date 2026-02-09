# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
Environment variable utilities for language-goal experiments.
"""

import os
import sys
from pathlib import Path

from src.utils.logging import get_logger

logger = get_logger(__name__)


# Default environment variable names and their descriptions
ENV_VARS = {
    "JEPAWM_HOME": "JEPA-WM project home directory",
    "JEPAWM_DSET": "Dataset root directory",
    "JEPAWM_LOGS": "Logs and outputs directory",
    "JEPAWM_CKPT": "Checkpoints directory",
    "JEPAWM_OSSCKPT": "OSS checkpoints directory",
}


def get_env_var(var_name: str, required: bool = True, default: str = None) -> str:
    """
    Get an environment variable with proper error handling.
    
    Args:
        var_name: Name of the environment variable
        required: If True, raise error when not set
        default: Default value if not required and not set
        
    Returns:
        The environment variable value
        
    Raises:
        SystemExit: If required and not set (with helpful message)
    """
    value = os.environ.get(var_name, default)
    
    if value is None and required:
        description = ENV_VARS.get(var_name, var_name)
        logger.error(f"❌ Environment variable '{var_name}' is not set.")
        logger.error(f"   Description: {description}")
        logger.error("")
        logger.error("   Please set the following environment variables:")
        logger.error("")
        logger.error("   export JEPAWM_HOME=$HOME/workspace/jepa_lang_goal")
        logger.error("   export JEPAWM_DSET=$HOME/datasets/jepa_lang_goal")
        logger.error("   export JEPAWM_LOGS=$HOME/logs/jepa_lang_goal")
        logger.error("   export JEPAWM_CKPT=$HOME/checkpoints/jepa_lang_goal")
        logger.error("   export JEPAWM_OSSCKPT=$HOME/oss_checkpoints/jepa_lang_goal")
        logger.error("")
        sys.exit(1)
    
    if value is not None:
        logger.info(f"✅ {var_name}={value}")
    
    return value


def validate_all_env_vars(required_vars: list = None) -> dict:
    """
    Validate and return all required environment variables.
    
    Args:
        required_vars: List of required variable names. 
                       If None, uses default ENV_VARS keys.
                       
    Returns:
        Dictionary of variable name -> value
    """
    if required_vars is None:
        required_vars = list(ENV_VARS.keys())
    
    logger.info("🔍 Validating environment variables...")
    env_values = {}
    
    for var_name in required_vars:
        env_values[var_name] = get_env_var(var_name, required=True)
    
    logger.info("✅ All environment variables validated.")
    return env_values


def get_paths() -> dict:
    """
    Get all standard paths for language-goal experiments.
    
    Returns:
        Dictionary with path objects for common directories.
    """
    env = validate_all_env_vars()
    
    return {
        "home": Path(env["JEPAWM_HOME"]),
        "dataset": Path(env["JEPAWM_DSET"]),
        "logs": Path(env["JEPAWM_LOGS"]),
        "checkpoints": Path(env["JEPAWM_CKPT"]),
        "oss_checkpoints": Path(env["JEPAWM_OSSCKPT"]),
        # Language-goal specific paths
        "lang_goal_logs": Path(env["JEPAWM_LOGS"]) / "lang_goal",
        "lang_goal_ckpt": Path(env["JEPAWM_CKPT"]) / "lang_goal",
        "lang_goal_index": Path(env["JEPAWM_LOGS"]) / "lang_goal" / "index",
        "pretrained": Path(env["JEPAWM_CKPT"]) / "pretrained",
        "metaworld_data": Path(env["JEPAWM_DSET"]) / "Metaworld",
    }


def ensure_dirs(paths: dict) -> None:
    """
    Ensure all directories in paths dict exist.
    
    Args:
        paths: Dictionary of path name -> Path object
    """
    for name, path in paths.items():
        if isinstance(path, Path):
            path.mkdir(parents=True, exist_ok=True)
            logger.info(f"📁 Ensured directory: {path}")


def get_git_info() -> dict:
    """
    Get git repository information for reproducibility.
    
    Returns:
        Dictionary with commit hash, branch, and dirty status.
    """
    import subprocess
    
    info = {
        "commit_hash": "unknown",
        "branch": "unknown",
        "is_dirty": False,
        "diff_summary": None,
    }
    
    try:
        # Get commit hash
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.environ.get("JEPAWM_HOME", "."),
        )
        if result.returncode == 0:
            info["commit_hash"] = result.stdout.strip()
        
        # Get branch name
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            cwd=os.environ.get("JEPAWM_HOME", "."),
        )
        if result.returncode == 0:
            info["branch"] = result.stdout.strip()
        
        # Check if dirty
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            capture_output=True,
            text=True,
            cwd=os.environ.get("JEPAWM_HOME", "."),
        )
        if result.returncode == 0:
            info["is_dirty"] = len(result.stdout.strip()) > 0
            if info["is_dirty"]:
                info["diff_summary"] = result.stdout.strip()
                
    except Exception as e:
        logger.warning(f"⚠️ Could not get git info: {e}")
    
    return info
