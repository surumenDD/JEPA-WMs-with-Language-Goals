
import os
import sys
import numpy as np
import torch
from pathlib import Path
from omegaconf import OmegaConf

# Add project root to path
project_root = Path("/home/ryo52/workspace/jepa_lang_goal/jepa-wms")
sys.path.insert(0, str(project_root))

from evals.simu_env_planning.envs.init import make_env

def check_rendering():
    # Load a minimal config (assuming we can reuse one or create a dummy one)
    # We'll valid config path from the log: configs/lang_goal/mw_lang_goal_eval_clip_patch.yaml
    config_path = project_root / "configs/lang_goal/mw_lang_goal_eval_clip_patch.yaml"
    
    print(f"Loading config from {config_path}")
    with open(config_path, "r") as f:
        cfg = OmegaConf.create(f)
        
    # Ensure EGL
    os.environ["MUJOCO_GL"] = "egl"
    
    print("Initializing Environment...")
    try:
        env = make_env(cfg)
    except Exception as e:
        print(f"Failed to make env: {e}")
        return

    print("Testing Rendering across seeds...")
    
    prev_image = None
    
    for seed in [1, 2, 3]:
        print(f"\n--- Seed {seed} ---")
        try:
            obs, info = env.reset(seed=seed)
            # Metaworld/Gym render
            img = env.render()
            
            if img is None:
                print("❌ Render returned None!")
                continue
                
            print(f"Image shape: {img.shape}, dtype: {img.dtype}")
            print(f"Image min: {img.min()}, max: {img.max()}, mean: {img.mean():.4f}, std: {img.std():.4f}")
            
            if img.max() == 0:
                print("❌ Image is completely BLACK (all zeros).")
            
            if prev_image is not None:
                diff = np.abs(img.astype(np.float32) - prev_image.astype(np.float32)).mean()
                print(f"Difference from previous seed: {diff:.4f}")
                if diff == 0:
                    print("❌ Image is IDENTICAL to previous seed.")
                else:
                    print("✅ Image Changed.")
            
            prev_image = img.copy()
            
        except Exception as e:
            print(f"Error during execution: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    check_rendering()
