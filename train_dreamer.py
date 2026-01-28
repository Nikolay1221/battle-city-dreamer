"""
Battle City x DreamerV3 (SheepRL Implementation)
Uses PyTorch-based sheeprl library for stable DreamerV3 training.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys

# Fix path issues
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

print(f"Working directory: {PROJECT_ROOT}")

def main():
    print("--- BATTLE CITY x DREAMER V3 (SheepRL) ---")
    
    import gymnasium as gym
    import numpy as np
    from battle_city_env import BattleCityEnv
    from dreamer_wrapper import BattleCityDreamerWrapper
    
    # Register custom environment
    class BattleCityGymEnv(gym.Env):
        metadata = {'render_modes': ['rgb_array']}
        
        def __init__(self, render_mode='rgb_array'):
            super().__init__()
            self._env = BattleCityEnv(
                rom_path=os.path.join(PROJECT_ROOT, "BattleCity.nes"),
                enemy_count=20,
                no_shooting=False,
                use_vision=False
            )
            self._wrapper = BattleCityDreamerWrapper(self._env, size=(64, 64))
            
            # SheepRL expects standard gymnasium spaces
            self.observation_space = gym.spaces.Box(0, 255, (64, 64, 1), dtype=np.uint8)
            self.action_space = gym.spaces.Discrete(self._env.action_space.n)
            self.render_mode = render_mode
            
        def reset(self, seed=None, options=None):
            obs, info = self._wrapper.reset(seed=seed)
            return obs['image'], info
            
        def step(self, action):
            obs, reward, terminated, truncated, info = self._wrapper.step(action)
            return obs['image'], float(reward), terminated, truncated, info
        
        def render(self):
            return self._env.raw_env.screen
            
        def close(self):
            self._env.close()
    
    # Register
    from gymnasium.envs.registration import register
    
    try:
        register(
            id='BattleCity-v0',
            entry_point=lambda: BattleCityGymEnv(),
            max_episode_steps=100000,
        )
        print("✅ Environment registered: BattleCity-v0")
    except Exception as e:
        print(f"Already registered or error: {e}")
    
    # Test environment works
    print("\nTesting environment...")
    test_env = BattleCityGymEnv()
    obs, info = test_env.reset()
    print(f"Obs shape: {obs.shape}, dtype: {obs.dtype}")
    
    for i in range(5):
        action = test_env.action_space.sample()
        obs, reward, term, trunc, info = test_env.step(action)
        print(f"Step {i}: reward={reward:.3f}, done={term or trunc}")
    
    test_env.close()
    print("✅ Environment test passed!\n")
    
    # Launch SheepRL training
    print("=== Starting SheepRL DreamerV3 Training ===")
    print("Run this command in terminal:\n")
    
    cmd = f"""python -m sheeprl exp=dreamer_v3 \\
  env.id=BattleCity-v0 \\
  algo.cnn_keys.encoder=[obs] \\
  algo.mlp_keys.encoder=[] \\
  algo.world_model.encoder.cnn_channels=[32,64,128] \\
  fabric.devices=1 \\
  buffer.size=100000 \\
  algo.total_steps=500000 \\
  metric.log_every=1000 \\
  checkpoint.every=50000
"""
    print(cmd)
    
    # Or run directly
    try:
        import subprocess
        subprocess.run(cmd.replace('\\\n', ' ').split(), cwd=PROJECT_ROOT)
    except Exception as e:
        print(f"\nCouldn't auto-run sheeprl: {e}")
        print("Please run the command above manually.")

if __name__ == "__main__":
    main()
