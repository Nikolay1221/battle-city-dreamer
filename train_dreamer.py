"""
Battle City x DreamerV3 Training Script
Uses the official dreamerv3.main() entry point with custom environment.
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys

# Ensure we're in the right directory
if os.path.exists('/content/battle_city'):
    os.chdir('/content/battle_city')

# Add current dir to path
sys.path.insert(0, os.getcwd())

def main():
    print("--- BATTLE CITY x DREAMER V3 ---")
    
    # Import after path setup
    import gymnasium as gym
    import numpy as np
    from battle_city_env import BattleCityEnv
    from dreamer_wrapper import BattleCityDreamerWrapper
    
    # Create a simple Gym-compatible wrapper
    class BattleCityGymEnv(gym.Env):
        """Gym-style wrapper for DreamerV3 compatibility."""
        
        def __init__(self):
            super().__init__()
            self._env = BattleCityEnv(
                rom_path="BattleCity.nes",
                enemy_count=20,
                no_shooting=False,
                use_vision=False
            )
            self._wrapper = BattleCityDreamerWrapper(self._env, size=(64, 64))
            
            # Dreamer expects specific spaces
            self.observation_space = gym.spaces.Dict({
                'image': gym.spaces.Box(0, 255, (64, 64, 1), dtype=np.uint8),
            })
            self.action_space = self._env.action_space
            
        def reset(self, seed=None, options=None):
            obs, info = self._wrapper.reset(seed=seed)
            # Return only image for simplicity
            return {'image': obs['image']}, info
            
        def step(self, action):
            obs, reward, terminated, truncated, info = self._wrapper.step(action)
            return {'image': obs['image']}, reward, terminated, truncated, info
            
        def close(self):
            self._env.close()
    
    # Register environment
    gym.register(
        id='BattleCity-v0',
        entry_point=lambda: BattleCityGymEnv(),
    )
    
    print("Environment registered as 'BattleCity-v0'")
    
    # Run DreamerV3 using command-line interface
    import dreamerv3
    from dreamerv3 import embodied
    
    # Configure
    config = embodied.Config(dreamerv3.Agent.configs['defaults'])
    config = config.update(dreamerv3.Agent.configs['size12m'])  # Small model
    config = config.update({
        'logdir': './logs/dreamer',
        'run.train_ratio': 32,
        'run.steps': 1e6,
        'batch_size': 16,
        'encoder.mlp_keys': '$^',
        'decoder.mlp_keys': '$^', 
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
    })
    
    # Parse any CLI args
    config = embodied.Flags(config).parse()
    
    # Setup logging
    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    
    print(f"Logdir: {config.logdir}")
    
    # Create environment
    def make_env(index):
        env = BattleCityGymEnv()
        env = embodied.envs.from_gymnasium.FromGymnasium(env)
        return env
    
    # Create parallel envs
    env = embodied.BatchEnv([lambda i=i: make_env(i) for i in range(2)], parallel=False)
    
    # Create agent
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    
    # Setup replay buffer
    replay = embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=logdir / 'replay',
    )
    
    # Training loop
    print("Starting training...")
    
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    
    embodied.run.train(agent, env, replay, logger=None, args=args)
    
    print("Training complete!")
    env.close()

if __name__ == "__main__":
    main()
