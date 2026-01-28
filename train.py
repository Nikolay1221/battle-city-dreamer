import gymnasium as gym
import os
import time
import math
import cv2
import numpy as np
import pickle
from collections import deque
from stable_baselines3 import PPO
import torch
import torch.nn as nn

# SB3 Contrib is optional
try:
    from sb3_contrib import RecurrentPPO
except ImportError:
    RecurrentPPO = None

from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from battle_city_env import BattleCityEnv
from battle_city_env import BattleCityEnv
# from simulation_env import SimulationBattleCityEnv  # Moved locally to avoid ImportError
import config 

# --- CALLBACKS ---

class KLAdaptiveLRCallback(BaseCallback):
    """
    Proportional KL-Adaptive Learning Rate Controller.
    Adjusts LR to keep approx_kl close to target_kl.
    Formula: new_lr = old_lr * (target_kl / approx_kl)
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.current_lr = config.LR_START
        self.min_lr = config.LR_MIN
        self.max_lr = config.LR_MAX
        self.target_kl = config.TARGET_KL

    def _on_step(self) -> bool:
        return True

    def on_rollout_start(self) -> None:
        # Get approx_kl from the last update
        if not hasattr(self.logger, 'name_to_value'):
            return

        approx_kl = self.logger.name_to_value.get("train/approx_kl", None)
        
        if approx_kl is not None and approx_kl > 1e-6: # Avoid division by zero
            # Proportional adjustment
            # We clamp the ratio to avoid wild swings (e.g., max 2x change per step)
            ratio = self.target_kl / approx_kl
            ratio = max(0.5, min(2.0, ratio)) 
            
            self.current_lr *= ratio
            
            # Clip to global bounds
            self.current_lr = max(self.min_lr, min(self.max_lr, self.current_lr))
            
            if self.verbose > 0:
                print(f"KL {approx_kl:.5f} | Target {self.target_kl} | Ratio {ratio:.2f} -> New LR {self.current_lr:.2e}")
            
            # Apply to optimizer
            self.model.learning_rate = self.current_lr
            for param_group in self.model.policy.optimizer.param_groups:
                param_group['lr'] = self.current_lr
                
            self.logger.record("train/learning_rate", self.current_lr)

class ConsoleLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        # Global accumulators (ONLY FOR REAL ENV)
        self.total_episodes = 0
        self.cum_reward = 0.0
        self.cum_kills = 0.0
        self.cum_explore = 0.0
        self.cum_length = 0.0

    def _on_step(self) -> bool:
        # Check for completed episodes in the current step
        # 'dones' tells us if an env finished, 'infos' contains the stats
        for info in self.locals['infos']:
            if 'episode' in info:
                # Check if this is a REAL environment (env_type == 1)
                # If 'env_type' is missing, assume it's real (legacy support)
                is_real = info.get('env_type', 1) == 1
                
                if is_real:
                    ep_data = info['episode']
                    self.total_episodes += 1
                    self.cum_reward += ep_data['r']
                    self.cum_length += ep_data['l']
                    self.cum_kills += ep_data.get('kills', 0)
                    self.cum_explore += ep_data.get('exploration_pct', 0)

        if self.num_timesteps % 1000 == 0:
            # --- ROLLING STATS (Last 100 episodes - FILTERED) ---
            roll_rew = "N/A"
            roll_kills = "N/A"
            roll_explore = "N/A"
            
            if len(self.model.ep_info_buffer) > 0:
                # Filter buffer for real envs only
                # Note: ep_info_buffer stores the raw info dicts
                real_eps = [ep for ep in self.model.ep_info_buffer if ep.get('env_type', 1) == 1]
                
                if len(real_eps) > 0:
                    roll_rew = np.mean([ep['r'] for ep in real_eps])
                    roll_kills = np.mean([ep.get('kills', 0) for ep in real_eps])
                    roll_explore = np.mean([ep.get('exploration_pct', 0) for ep in real_eps])
                
                    # Log Rolling to TensorBoard
                    self.logger.record("rollout/mean_kills", roll_kills)
                    self.logger.record("rollout/mean_explore_pct", roll_explore)

            # --- GLOBAL STATS (Since start of script) ---
            glob_rew = 0.0
            glob_kills = 0.0
            glob_explore = 0.0
            glob_len = 0.0
            
            if self.total_episodes > 0:
                glob_rew = self.cum_reward / self.total_episodes
                glob_kills = self.cum_kills / self.total_episodes
                glob_explore = self.cum_explore / self.total_episodes
                glob_len = self.cum_length / self.total_episodes

            # Log Global to TensorBoard
            self.logger.record("global/mean_reward", glob_rew)
            self.logger.record("global/mean_kills", glob_kills)
            self.logger.record("global/mean_explore_pct", glob_explore)
            self.logger.record("global/mean_ep_length", glob_len)
            self.logger.record("global/total_episodes", self.total_episodes)

            # --- CONSOLE OUTPUT ---
            # Format: [Step] | R: Roll(Glob) | K: Roll(Glob) | Exp: Roll(Glob)%
            r_str = f"{roll_rew:.1f}" if isinstance(roll_rew, float) else roll_rew
            k_str = f"{roll_kills:.1f}" if isinstance(roll_kills, float) else roll_kills
            e_str = f"{roll_explore:.1f}" if isinstance(roll_explore, float) else roll_explore
            
            print(f"[{self.num_timesteps}] "
                  f"Rew: {r_str}({glob_rew:.1f}) | "
                  f"Kills: {k_str}({glob_kills:.1f}) | "
                  f"Exp: {e_str}%({glob_explore:.1f}%) | "
                  f"Len: {glob_len:.0f}", 
                  end='\r')

        return True

class RenderCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.windows_initialized = False 
        
        if getattr(config, 'HEADLESS_MODE', False):
             self.windows_initialized = "HEADLESS"
        
    def _on_step(self) -> bool:
        if self.windows_initialized == "HEADLESS": return True
        
        try:
            # Access the VecEnv
            vec_env = self.model.get_env()
            
            # Call 'get_tactical_rgb' on ALL envs
            grid_images = vec_env.env_method("get_tactical_rgb")
            
            if not grid_images: return True

            # Calculate Grid Dimensions
            n_envs = len(grid_images)
            
            # Force 4 columns layout for better fit
            cols = 4
            rows = int(np.ceil(n_envs / cols))
            
            # Fill empty slots
            while len(grid_images) < rows * cols:
                grid_images.append(np.zeros((52, 52, 3), dtype=np.uint8))

            # Stitch Rows
            final_rows = []
            for r in range(rows):
                row_imgs = grid_images[r*cols : (r+1)*cols]
                # Horizontal stack
                row_line = np.hstack(row_imgs)
                final_rows.append(row_line)
            
            # Vertical stack
            full_grid = np.vstack(final_rows)

            # Resize for visibility (Scale x6 for 52px -> 312px blocks)
            # Increased to x6 for better visibility
            scale = 6
            h, w = full_grid.shape[:2]
            
            # Convert RGB to BGR for OpenCV
            full_grid_bgr = cv2.cvtColor(full_grid, cv2.COLOR_RGB2BGR)
            
            display_grid = cv2.resize(full_grid_bgr, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)
            
            # Add thick grid lines between environments
            for i in range(1, cols):
                x = i * 52 * scale
                cv2.line(display_grid, (x, 0), (x, h*scale), (255, 255, 255), 2)
            for i in range(1, rows):
                y = i * 52 * scale
                cv2.line(display_grid, (0, y), (w*scale, y), (255, 255, 255), 2)

            # Add internal cell grid (subtle) - every 4 cells (16 pixels)
            for i in range(w // 52 * 52 + 1):
                x = i * scale
                if x % (52 * scale) != 0:
                    # Draw grid every 4 cells (to mimic 16x16 blocks)
                    if i % 4 == 0:
                        cv2.line(display_grid, (x, 0), (x, h*scale), (40, 40, 40), 1)
            for i in range(h // 52 * 52 + 1):
                y = i * scale
                if y % (52 * scale) != 0:
                    if i % 4 == 0:
                        cv2.line(display_grid, (0, y), (w*scale, y), (40, 40, 40), 1)

            if not self.windows_initialized:
                cv2.namedWindow("BATTLE CITY - 8 PARALLEL WORLDS", cv2.WINDOW_AUTOSIZE)
                self.windows_initialized = True

            cv2.imshow("BATTLE CITY - 8 PARALLEL WORLDS", display_grid)
            cv2.waitKey(1)

        except Exception as e:
            pass # print(f"Render Error: {e}")
        return True

# --- TRAIN FUNCTION ---

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class ResBlock(nn.Module):
    """
    Residual Block for deeper reasoning without gradient vanishing.
    Input -> Conv -> ReLU -> Conv -> (+ Input) -> ReLU
    """
    def __init__(self, n_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(n_channels, n_channels, kernel_size=3, padding=1)
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(x + self.conv(x))

class CustomTacticalCNN(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0] 
        
        # 1. Initial Feature Extraction (Scale down)
        self.initial_conv = nn.Sequential(
            nn.Conv2d(n_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), # 26x26
            nn.ReLU(),
        )
        
        # 2. Deep Reasoning (ResNet Blocks) - Keep size 26x26
        self.res_blocks = nn.Sequential(
            ResBlock(128),
            ResBlock(128),
            ResBlock(128)
        )
        
        # 3. Final Compression
        self.final_conv = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1), # 13x13
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1), # 7x7
            nn.ReLU(),
            nn.Flatten()
        )

        # Compute shape
        with torch.no_grad():
            sample = torch.as_tensor(observation_space.sample()[None]).float()
            x = self.initial_conv(sample)
            x = self.res_blocks(x)
            n_flatten = self.final_conv(x).shape[1]

        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        x = self.initial_conv(observations)
        x = self.res_blocks(x)
        x = self.final_conv(x)
        return self.linear(x)

def make_configured_env(rank, variant_name, seed=0):
    """
    Factory function for specific environment configuration using ENV_VARIANTS.
    """
    def _init():
        variant = config.ENV_VARIANTS.get(variant_name, config.ENV_VARIANTS["STANDARD"])
        
        # 1. Virtual Environment
        if variant == "VIRTUAL":
            try:
                from simulation_env import SimulationBattleCityEnv
            except ImportError:
                 raise ImportError("Virtual Environment not found.")
            
            env = SimulationBattleCityEnv(render_mode='rgb_array')
            env = Monitor(env, info_keywords=('kills', 'exploration_pct', 'env_type'))
            return env

        # 2. Real NES Environment
        # Extract kwargs from variant dict
        # Default fallback
        if isinstance(variant, str): variant = config.ENV_VARIANTS["STANDARD"]

        # Reward Injection
        reward_profile_name = variant.get("reward_profile", "DEFAULT")
        reward_config = config.REWARD_VARIANTS.get(reward_profile_name, None)

        env = BattleCityEnv(
            render_mode='rgb_array',
            stack_size=config.STACK_SIZE,
            target_stage=getattr(config, 'TARGET_STAGE', None),
            enemy_count=variant.get("enemy_count", 20),
            no_shooting=variant.get("no_shooting", False),
            reward_config=reward_config,
            exploration_trigger=variant.get("exploration_trigger", None)
        )
        env.reset(seed=seed + rank)
        
        # Monitor
        # Env Type: 0 = Training/Partial, 1 = Full Combat
        is_full_combat = (env.enemy_count >= 20 and not env.no_shooting)
        return Monitor(env, info_keywords=('kills', 'exploration_pct', 'env_type'))
    return _init

def train():
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.LOG_DIR, exist_ok=True)

    print(f"--- BATTLE CITY TACTICAL TRAINING (RESNET ENHANCED) ---")
    
    # Environment Setup
    train_mode = getattr(config, 'TRAIN_MODE', 'STANDARD')
    print(f">> TRAIN MODE: {train_mode}")
    
    env_fns = []
    
    if train_mode == "HYBRID":
        # Parse HYBRID_CONFIG
        hybrid_conf = getattr(config, 'HYBRID_CONFIG', {"STANDARD": config.NUM_CPU})
        
        current_idx = 0
        # Iterate config keys
        for mode_name, count in hybrid_conf.items():
            print(f"   - {count} envs: {mode_name}")
            for _ in range(count):
                if current_idx >= config.NUM_CPU: break
                env_fns.append(make_configured_env(current_idx, mode_name))
                current_idx += 1
                
        if len(env_fns) != config.NUM_CPU:
            print(f"WARNING: Hybrid config count ({len(env_fns)}) != NUM_CPU ({config.NUM_CPU}). Filling rest with STANDARD.")
            while len(env_fns) < config.NUM_CPU:
                env_fns.append(make_configured_env(len(env_fns), "STANDARD"))
                
    else:
        # Uniform Mode
        print(f"   - All {config.NUM_CPU} envs: {train_mode}")
        for i in range(config.NUM_CPU):
            env_fns.append(make_configured_env(i, train_mode))

    if config.NUM_CPU > 1:
        env = SubprocVecEnv(env_fns)
    else:
        env = DummyVecEnv(env_fns)

    env = VecNormalize(env, norm_obs=False, norm_reward=True)

    # 2. Setup Model
    if config.USE_RECURRENT and RecurrentPPO is not None:
        print("Using RecurrentPPO (LSTM)...")
        ModelClass = RecurrentPPO
        policy_type = "CnnLstmPolicy"
    else:
        print("Using Standard PPO...")
        ModelClass = PPO
        policy_type = "CnnPolicy"

    # REDUCED CAPACITY TO FIT GPU MEMORY (512 instead of 1024)
    policy_kwargs = dict(
        features_extractor_class=CustomTacticalCNN,
        features_extractor_kwargs=dict(features_dim=512), 
        net_arch=dict(pi=[512, 512], vf=[512, 512]),   
        activation_fn=torch.nn.ReLU,
    )
    
    # Check for saved models
    final_path = f"{config.MODEL_DIR}/battle_city_final.zip"
    interrupted_path = f"{config.MODEL_DIR}/battle_city_interrupted.zip"
    
    model = None
    reset_timesteps = True
    
    # Attempt Load
    load_path = interrupted_path if os.path.exists(interrupted_path) else (final_path if os.path.exists(final_path) else None)
    
    if load_path:
        print(f"Loading model from {load_path}...")
        try:
            model = ModelClass.load(load_path, env=env)
            # Update LR schedule with new params
            model.learning_rate = config.LR_START # Initial value
            
            # RecurrentPPO might have slightly different param names, but ent_coef is standard
            model.ent_coef = getattr(config, 'ENT_COEF', 0.01)
            model.batch_size = config.BATCH_SIZE
            model.n_steps = config.N_STEPS
            reset_timesteps = False
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Incompatible model found ({e}). Starting FRESH with new tactical input.")
            if os.path.exists(load_path):
                 os.rename(load_path, f"{load_path}.old")
            model = None
    
    if model is None:
        print(f"Creating NEW {ModelClass.__name__} Model...")
        model = ModelClass(
            policy_type,
            env,
            verbose=1,
            tensorboard_log=config.LOG_DIR,
            learning_rate=config.LR_START, # Constant init, callback will adjust
            n_steps=config.N_STEPS,
            batch_size=config.BATCH_SIZE,
            n_epochs=getattr(config, 'N_EPOCHS', 10),
            gamma=config.GAMMA,
            gae_lambda=0.95,
            clip_range=config.CLIP_RANGE,
            ent_coef=getattr(config, 'ENT_COEF', 0.01),
            policy_kwargs=policy_kwargs,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )

    # 3. Train
    callbacks = [
        CheckpointCallback(save_freq=config.CHECKPOINT_FREQ, save_path=config.MODEL_DIR, name_prefix="ppo_resnet"),
        ConsoleLoggerCallback(),
        # CurriculumCallback REMOVED - Single Stage Training
        RenderCallback(),
        KLAdaptiveLRCallback(verbose=1)
    ]

    try:
        model.learn(
            total_timesteps=config.TOTAL_TIMESTEPS,
            callback=callbacks,
            progress_bar=True,
            reset_num_timesteps=reset_timesteps
        )
        model.save(final_path)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        model.save(interrupted_path)
        print("Saved emergency backup.")
    finally:
        env.close()

if __name__ == "__main__":
    train()