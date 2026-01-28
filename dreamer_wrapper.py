import gymnasium as gym
import numpy as np
import cv2
from battle_city_env import BattleCityEnv
import config

class BattleCityDreamerWrapper(gym.Wrapper):
    """
    Wraps BattleCityEnv to be compatible with DreamerV3.
    Expects dict observations: {'image': ..., 'is_first': ..., 'is_last': ..., 'is_terminal': ...}
    """
    def __init__(self, env, size=(64, 64)):
        super().__init__(env)
        self._size = size
        
        # Dreamer expects 0-255 uint8 images
        # We can use Grayscale (H, W, 1) or RGB (H, W, 3)
        # Tactical Map is naturally grayscale, but we can make it 1-channel
        self.observation_space = gym.spaces.Dict({
            'image': gym.spaces.Box(0, 255, (size[0], size[1], 1), dtype=np.uint8),
            'is_first': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_last': gym.spaces.Box(0, 1, (), dtype=bool),
            'is_terminal': gym.spaces.Box(0, 1, (), dtype=bool),
        })
        self._action_space = env.action_space
        
    @property
    def action_space(self):
        return self._action_space
        
    def _process_frame(self, frame):
        # Resize to target size (64x64)
        # Frame is likely 52x52 uint8 from _get_tactical_map
        resized = cv2.resize(frame, self._size, interpolation=cv2.INTER_NEAREST)
        return resized[..., None] # Add channel dim -> (64, 64, 1)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Get raw single frame, not stack (Dreamer handles memory itself)
        # BattleCityEnv returns stack by default via _get_obs
        # We need access to the latest single frame
        raw_frame = self.env.unwrapped.frames[-1] 
        
        done = terminated or truncated
        
        dreamer_obs = {
            'image': self._process_frame(raw_frame),
            'is_first': False,
            'is_last': done,
            'is_terminal': terminated,
        }
        
        return dreamer_obs, reward, terminated, truncated, info

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        raw_frame = self.env.unwrapped.frames[-1]
        
        dreamer_obs = {
            'image': self._process_frame(raw_frame),
            'is_first': True,
            'is_last': False,
            'is_terminal': False,
        }
        return dreamer_obs, info

def make_dreamer_env(task, size=(64, 64), **kwargs):
    # Helper to create and wrap
    import config
    # Ensure standard config used
    env = BattleCityEnv(
        rom_path=config.ROM_PATH,
        render_mode='rgb_array',
        use_vision=False, # We use tactical map
        stack_size=1, # Dreamer needs 1 frame, it has its own RNN
        **kwargs
    )
    env = BattleCityDreamerWrapper(env, size=size)
    return env
