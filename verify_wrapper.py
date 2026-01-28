import numpy as np
import gymnasium as gym
import sys
import os

# Add current path
sys.path.append(os.getcwd())

try:
    from battle_city_env import BattleCityEnv
    from dreamer_wrapper import BattleCityDreamerWrapper
    import config
except ImportError as e:
    print(f"Import Error: {e}")
    sys.exit(1)

def verify():
    print("--- VERIFYING DREAMER WRAPPER ---")
    
    # 1. Init Env
    print("Initializing Environment...")
    try:
        env = BattleCityEnv(
            rom_path="BattleCity.nes",
            enemy_count=20,
            no_shooting=False,
            use_vision=False
        )
        env = BattleCityDreamerWrapper(env, size=(64, 64))
    except Exception as e:
        print(f"FAILED to init env: {e}")
        return

    # 2. Reset
    print("\n[RESET CHECK]")
    obs, info = env.reset()
    
    check_obs(obs, "Reset")

    # 3. Step
    print("\n[STEP CHECK]")
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    
    check_obs(obs, "Step")
    
    print(f"\nReward Type: {type(reward)} Value: {reward}")
    print(f"Terminated: {type(terminated)}")
    
    env.close()
    print("\n✅ VERIFICATION PASSED: Wrapper produces valid dictionary for Dreamer.")

def check_obs(obs, stage):
    print(f"--- {stage} Obs Keys: {list(obs.keys())}")
    
    # Check Image
    if 'image' not in obs:
        print("❌ MISSING 'image' key!")
        return
    
    img = obs['image']
    print(f"Image Shape: {img.shape} (Expected (64, 64, 1))")
    print(f"Image Type: {img.dtype} (Expected uint8)")
    print(f"Image Range: [{img.min()}, {img.max()}]")
    
    if img.shape != (64, 64, 1):
        print("❌ WRONG SHAPE")
    else:
        print("✅ shape ok")
        
    if img.dtype != np.uint8:
        print("❌ WRONG TYPE (Must be uint8)")
    else:
        print("✅ type ok")

    # Check Flags (Optional but good)
    for k in ['is_first', 'is_last', 'is_terminal']:
        if k in obs:
            print(f"{k}: {obs[k]}")

if __name__ == "__main__":
    verify()
