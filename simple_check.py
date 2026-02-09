
import os
import metaworld
import numpy as np

# Set backend
os.environ["MUJOCO_GL"] = "egl"

try:
    print("Importing Metaworld...")
    ml1 = metaworld.ML1('reach-v2')
    print("Metaworld imported.")
    
    env = ml1.train_classes['reach-v2']()
    task = [task for task in ml1.train_tasks if task.env_name == 'reach-v2'][0]
    env.set_task(task)
    
    print("Env created. Resetting...")
    obs = env.reset()
    print("Reset complete. Rendering...")
    
    # Try offscreen render
    # Metaworld (mujoco-py) usually uses env.render(mode='rgb_array') or similar
    try:
        img = env.render(mode='rgb_array')
        if img is None:
             # Some versions of metaworld/mujoco-py return None if offscreen not configured right
             # or they might render to window by default if mode not specified
             print("Render returned None with mode='rgb_array'")
             
             # Try without mode (might open window and hang if no display)
             # img = env.render() 
        else:
            print(f"Render success! Shape: {img.shape}")
            print(f"Mean: {img.mean()}, Std: {img.std()}, Max: {img.max()}")
            if img.max() == 0:
                print("WARNING: Image is black.")
    except Exception as e:
        print(f"Render failed: {e}")

except Exception as e:
    print(f"CRITICAL FAILURE: {e}")
    import traceback
    traceback.print_exc()
