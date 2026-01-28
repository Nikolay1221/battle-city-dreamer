import warnings
import os
import argparse
import functools

# Filter warnings to keep output clean
warnings.filterwarnings('ignore')

try:
    import gymnasium as gym
    import dreamerv3
    import embodied
    from battle_city_env import BattleCityEnv
    from dreamer_wrapper import BattleCityDreamerWrapper
except ImportError as e:
    print(f"Error importing modules: {e}")
    print("Please ensure dreamerv3, embodied, gymnasium, and project files are installed/present.")
    exit(1)

def main():
    print("--- BATTLE CITY x DREAMER V3 ---")
    
    # 1. Configuration
    # 'small' is optimized for single GPU (T4/L4)
    config = dreamerv3.configs['small']
    config = config.update({
        'logdir': 'logs/dreamer',
        'run.train_ratio': 64,  # Replay ratio
        'run.steps': 5000000,   # Total steps
        'batch_size': 16,
        'jax.policy_devices': [0],
        'jax.train_devices': [0],
        'jax.platform': 'gpu',
        'encoder.mlp_keys': '$^', 
        'decoder.mlp_keys': '$^',
        'encoder.cnn_keys': 'image',
        'decoder.cnn_keys': 'image',
    })
    
    # Parse generic flags if needed (e.g. override from CLI)
    # config = embodied.Flags(config).parse() 
    
    print(f"Logdir: {config.logdir}")
    
    # 2. Environment Factory
    def make_env(index):
        # Create base env
        env = BattleCityEnv(
            rom_path="BattleCity.nes",
            enemy_count=20,
            no_shooting=False,
            use_vision=False # We use tactical map
        )
        # Wrap for Dreamer (64x64, Dict obs)
        env = BattleCityDreamerWrapper(env, size=(64, 64))
        
        # Wrap for Embodied (converts Gym API -> Embodied API)
        # embodied expects 'from_gym' to handle the interface adaption
        # We pass 'checks=False' because we might rely on our own wrapper's dict
        env = embodied.envs.from_gym(env, checks=False)
        return env

    # 3. Create Parallel Environments
    # Dreamer likes batching. For Colab, 1-4 envs is good.
    num_envs = 4 
    env = embodied.BatchEnv([functools.partial(make_env, i) for i in range(num_envs)], parallel='process')

    # 4. Initialize Agent
    agent = dreamerv3.Agent(env.obs_space, env.act_space, config)
    
    # 5. Setup Utils
    logger = embodied.Logger(embodied.Path(config.logdir),
                             [embodied.logger.TerminalOutput(),
                              embodied.logger.JSONLOutput(config.logdir),
                              embodied.logger.TensorBoardOutput(config.logdir)])
    
    args = embodied.Config(
        **config.run,
        logdir=config.logdir,
        batch_steps=config.batch_size * 16, # approx
        from_checkpoint=None
    )

    # 6. Training Loop (using embodied.run.train logic simplified)
    print("Starting Training Loop...")
    embodied.run.train(agent, env, args, logger=logger)
    
    # Clean up
    env.close()

if __name__ == "__main__":
    main()
