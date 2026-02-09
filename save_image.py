
import os
import torch  # Critical: imports GL libs correctly
import numpy as np
from PIL import Image

# Set backend
os.environ["MUJOCO_GL"] = "egl"

import metaworld

try:
    print("Importing Metaworld...")
    ml1 = metaworld.ML1('reach-v2')
    env = ml1.train_classes['reach-v2']()
    task = [task for task in ml1.train_tasks if task.env_name == 'reach-v2'][0]
    env.set_task(task)
    
    # Apply similar camera settings as MetaWorldWrapper to reproduce issue
    # "corner2" is usually the camera used.
    # The wrapper did: self.env.model.cam_pos[2] = [0.75, 0.075, 0.7]
    # We need to find which index "corner2" is.
    try:
        corner2_id = env.model.camera_name2id("corner2")
        print(f"Corner2 ID: {corner2_id}")
        # The wrapper line: self.env.model.cam_pos[2] = ... might be HARDCODING index 2!
        # If corner2 is NOT index 2, then it's modifying a different camera 
        # but rendering with corner2.
        
        # Let's check what's at index 2
        print(f"Camera at index 2 pos: {env.model.cam_pos[2]}")
        
        # Let's try to mimic the wrapper modification
        # NOTE: logic in wrapper was self.env.model.cam_pos[2] = ...
        # If corner2_id != 2, this is a BUG in the wrapper!
        
    except Exception as e:
        print(f"Could not check cameras: {e}")

    env.reset()
    
    # Render with mode 'rgb_array'
    # We need to ensure we use the same camera name "corner2"
    # metaworld envs usually default to topview or something else?
    
    # We will try to render with 'corner2' specifically if possible
    # But usually env.render() uses default.
    
    img = env.render(mode='rgb_array')
    
    if img is None:
        print("Render returned None")
    else:
        print(f"Render success! Shape: {img.shape}")
        print(f"Stats - Min: {img.min()}, Max: {img.max()}, Mean: {img.mean()}")
        
        # Save image
        im = Image.fromarray(img)
        im.save("/home/ryo52/workspace/jepa_lang_goal/jepa-wms/debug_image_default.png")
        print("Saved debug_image_default.png")
        
    # Now try to reproduce the Wrapper logic if we can
    # But just getting ANY image is a start.

except Exception as e:
    print(f"CRITICAL FAILURE: {e}")
    import traceback
    traceback.print_exc()
