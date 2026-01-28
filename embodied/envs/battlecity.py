"""
Battle City environment wrapper for DreamerV3.

Uses the existing BattleCityEnv from battle_city_env.py and adapts it
to the embodied.Env interface required by DreamerV3.
"""

import os
import sys
import threading

import elements
import embodied
import numpy as np

# Add parent directory to path to import battle_city_env
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))


class BattleCity(embodied.Env):

    LOCK = threading.Lock()

    def __init__(
        self,
        task='stage0',
        repeat=4,
        size=(52, 52),
        gray=True,
        length=108000,
        seed=None,
    ):
        """
        Initialize Battle City environment.

        Args:
            task: Stage identifier (e.g., 'stage0', 'stage1')
            repeat: Frame skip (action repeat)
            size: Observation size (width, height)
            gray: Use grayscale observations
            length: Maximum episode length
            seed: Random seed
        """
        self.repeat = repeat
        self.size = size
        self.gray = gray
        self.length = length
        self.rng = np.random.default_rng(seed)

        # Parse stage from task name
        if task.startswith('stage'):
            self.stage = int(task[5:]) if len(task) > 5 else 0
        else:
            self.stage = 0

        # Import and create the base environment
        from battle_city_env import BattleCityEnv
        
        with self.LOCK:
            self._env = BattleCityEnv(
                render_mode='rgb_array',
                stack_size=1,  # DreamerV3 handles temporal stacking
                target_stage=self.stage,
                enemy_count=20,
                no_shooting=False,
            )

        self.duration = None
        self.done = True

    @property
    def obs_space(self):
        shape = (*self.size, 1 if self.gray else 3)
        return {
            'image': elements.Space(np.uint8, shape),
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
        }

    @property
    def act_space(self):
        return {
            'action': elements.Space(np.int32, (), 0, 10),  # 10 discrete actions
            'reset': elements.Space(bool),
        }

    def step(self, action):
        if action['reset'] or self.done:
            return self._reset()

        reward = 0.0
        terminal = False
        last = False

        act = int(action['action'])
        
        # Frame skip with reward accumulation
        for _ in range(self.repeat):
            obs, r, terminated, truncated, info = self._env.step(act)
            reward += r
            self.duration += 1

            if terminated:
                terminal = True
                last = True
            if truncated or self.duration >= self.length:
                last = True
            if terminal or last:
                break

        self.done = last
        return self._obs(reward, is_last=last, is_terminal=terminal)

    def _reset(self):
        with self.LOCK:
            self._env.reset()
        self.duration = 0
        self.done = False
        return self._obs(0.0, is_first=True)

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False):
        # Get tactical map from our environment (52x52 grayscale)
        image = self._env.get_tactical_rgb()  # Returns (52, 52, 3)
        
        # Resize if needed
        if self.size != (52, 52):
            from PIL import Image as PILImage
            image = PILImage.fromarray(image)
            image = image.resize(self.size, PILImage.BILINEAR)
            image = np.array(image)

        # Convert to grayscale if needed
        if self.gray:
            # Use standard luminance weights
            weights = np.array([0.299, 0.587, 0.114])
            image = (image * weights).sum(-1).astype(np.uint8)[:, :, None]

        return dict(
            image=image,
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
        )

    def close(self):
        try:
            self._env.close()
        except Exception:
            pass
