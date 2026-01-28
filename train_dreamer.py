"""
Battle City x DreamerV3 Training Script
"""
import warnings
warnings.filterwarnings('ignore')

import os
import sys

# Ensure we're in the right directory
if os.path.exists('/content/battle_city'):
    os.chdir('/content/battle_city')

sys.path.insert(0, os.getcwd())

def main():
    print("--- BATTLE CITY x DREAMER V3 ---")
    
    # Imports
    import gymnasium as gym
    import numpy as np
    import dreamerv3
    import embodied  # Separate package!
    
    from battle_city_env import BattleCityEnv
    from dreamer_wrapper import BattleCityDreamerWrapper
    
    # Gym wrapper
    class BattleCityGymEnv(gym.Env):
        def __init__(self):
            super().__init__()
            self._env = BattleCityEnv(
                rom_path="BattleCity.nes",
                enemy_count=20,
                no_shooting=False,
                use_vision=False
            )
            self._wrapper = BattleCityDreamerWrapper(self._env, size=(64, 64))
            self.observation_space = gym.spaces.Dict({
                'image': gym.spaces.Box(0, 255, (64, 64, 1), dtype=np.uint8),
            })
            self.action_space = self._env.action_space
            
        def reset(self, seed=None, options=None):
            obs, info = self._wrapper.reset(seed=seed)
            return {'image': obs['image']}, info
            
        def step(self, action):
            obs, reward, terminated, truncated, info = self._wrapper.step(action)
            return {'image': obs['image']}, reward, terminated, truncated, info
            
        def close(self):
            self._env.close()
    
    print("Creating environment...")
    
    # Config
    config = embodied.Config(dreamerv3.Agent.configs['defaults'])
    config = config.update(dreamerv3.Agent.configs['size12m'])
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
    
    config = embodied.Flags(config).parse()
    
    logdir = embodied.Path(config.logdir)
    logdir.mkdir()
    print(f"Logdir: {config.logdir}")
    
    # Environment factory
    def make_env(index):
        env = BattleCityGymEnv()
        # Use from_gymnasium adapter
        from embodied.envs import from_gymnasium
        env = from_gymnasium.FromGymnasium(env)
        return env
    
    # Batch envs (single process for stability)
    env = embodied.BatchEnv([lambda i=i: make_env(i) for i in range(2)], parallel=False)
    
    print(f"Obs space: {env.obs_space}")
    print(f"Act space: {env.act_space}")
    
    # Agent
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    
    # Replay
    replay = embodied.replay.Replay(
        length=config.batch_length,
        capacity=config.replay.size,
        directory=logdir / 'replay',
    )
    
    # Logger
    logger = embodied.Logger(logdir, [
        embodied.logger.TerminalOutput(),
    ])
    
    # Training args
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * config.batch_length,
    )
    
    print("Starting training...")
    embodied.run.train(agent, env, replay, logger, args)
    
    print("Done!")
    env.close()

if __name__ == "__main__":
    main()
