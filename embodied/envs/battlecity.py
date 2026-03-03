"""
Battle City environment wrapper for DreamerV3.

Uses the existing BattleCityEnv from battle_city_env.py and adapts it
to the embodied.Env interface required by DreamerV3.

Supports two observation modes:
  - image: 64x64 grayscale tactical map (CNN encoder)
  - ram: raw 233 bytes from NES RAM (MLP encoder)
"""

import os
import sys
import threading

# Add parent directory to path to import battle_city_env
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

# Apply nes-py NumPy 2.0 compatibility patch BEFORE importing BattleCityEnv
import nes_py_patch  # noqa: F401

import elements
import embodied
import numpy as np
import cv2


class BattleCity(embodied.Env):

    LOCK = threading.Lock()
    RAM_SIZE = 233  # First 233 bytes of NES RAM (0x00-0xE8)

    def __init__(
        self,
        task='stage0',
        repeat=4,
        size=(64, 64),
        gray=True,
        length=108000,
        seed=None,
        use_ram=False,
        logdir=None,
        video_every=50,
    ):
        self.repeat = repeat
        self.size = size
        self.gray = gray
        self.length = length
        self.use_ram = use_ram or ('ram' in task)
        self.rng = np.random.default_rng(seed)

        # Parse stage from task name (supports: 'stage0', 'ram_stage0')
        task_clean = task.replace('ram_', '')
        if task_clean.startswith('stage'):
            self.stage = int(task_clean[5:]) if len(task_clean) > 5 else 0
        else:
            self.stage = 0
        
        # Fallback: use TARGET_STAGE from config.py if task didn't specify a stage
        if self.stage == 0:
            try:
                import config as cfg
                self.stage = getattr(cfg, 'TARGET_STAGE', 0)
            except ImportError:
                pass
        
        print(f"[BattleCity] Stage = {self.stage}")

        # Import and create the base environment
        from battle_city_env import BattleCityEnv

        with self.LOCK:
            self._env = BattleCityEnv(
                render_mode='rgb_array',
                start_level=self.stage,
            )

        self.duration = None
        self.done = True

        # --- Continuous Multi-Base Logic ---
        self._max_lives = 3
        self._current_lives = self._max_lives

        # --- Episode Metrics ---
        self._ep_kills = 0
        self._ep_deaths = 0
        self._ep_base_lost = 0
        self._ep_exploration = 0.0
        self._ep_reward = 0.0
        self._ep_max_kill_reward = 0.0

        # --- Video Recording ---
        self._video_every = video_every
        self._episode_count = 0
        self._global_step = 0
        self._video_frames = []
        self._video_dir = None
        if logdir and video_every > 0:
            # Use parent logdir for all envs (e.g. logs_dreamer_ram/video/)
            parent_logdir = os.path.dirname(str(logdir))
            self._video_dir = os.path.join(parent_logdir, 'video')
            os.makedirs(self._video_dir, exist_ok=True)

    @property
    def obs_space(self):
        spaces = {
            'reward': elements.Space(np.float32),
            'is_first': elements.Space(bool),
            'is_last': elements.Space(bool),
            'is_terminal': elements.Space(bool),
            # --- Per-step log metrics (DreamerV3 auto-aggregates avg/max/sum) ---
            'log/kills': elements.Space(np.float32),
            'log/deaths': elements.Space(np.float32),
            'log/base_lost': elements.Space(np.float32),
            'log/exploration': elements.Space(np.float32),
            'log/ep_reward': elements.Space(np.float32),
            'log/lives': elements.Space(np.float32),
            'log/enemies_alive': elements.Space(np.float32),
            'log/proximity': elements.Space(np.float32),
        }
        if self.use_ram:
            spaces['ram'] = elements.Space(np.float32, (self.RAM_SIZE,))
        else:
            shape = (*self.size, 1 if self.gray else 3)
            spaces['image'] = elements.Space(np.uint8, shape)
        return spaces

    @property
    def act_space(self):
        return {
            'move': elements.Space(np.int32, (), 0, 5),
            'fire': elements.Space(np.int32, (), 0, 2),
            # duration: 0=1 frame, 1=2 frames, ..., 7=8 frames
            # Agent decides how long to hold the button press
            'duration': elements.Space(np.int32, (), 0, 8),
            'reset': elements.Space(bool),
        }

    def step(self, action):
        if action['reset'] or self.done:
            return self._reset()

        reward = 0.0
        terminal = False
        last = False

        mov = int(action['move'])
        fire = int(action['fire'])
        act = np.array([mov, fire], dtype=np.int32)

        # Temporal Action Repeat: agent decides how long to hold the button
        # duration index: 0=1 frame, 1=2 frames, ..., 7=8 frames
        DURATION_MAP = [1, 2, 3, 4, 5, 6, 7, 8]
        dur_idx = int(action.get('duration', 3))  # default = 4 frames (index 3)
        n_repeat = DURATION_MAP[min(dur_idx, len(DURATION_MAP) - 1)]

        # Action repeat: execute the same action for N frames
        total_reward = 0.0
        terminated = False
        info = {}
        for _ in range(n_repeat):
            obs, rew, terminated, info = self._env.step(act)
            
            # --- Update total logged kills (always count kills for stats) ---
            current_inner_kills = info.get('kills', 0)
            if current_inner_kills > self._prev_inner_kills:
                new_kills = current_inner_kills - self._prev_inner_kills
                self._ep_kills += new_kills
                self._life_kills += new_kills
                self._prev_inner_kills = current_inner_kills
                
            total_reward += rew
            self.duration += 1
            self._life_duration += 1
            self._global_step += 1

            # Capture cropped playfield frame for video
            if self._video_dir and self._episode_count % self._video_every == 0:
                try:
                    screen = self._env.raw_env.screen
                    frame = screen[16:224, 16:224].copy()  # 208x208 playfield only
                    self._video_frames.append(frame)
                except Exception:
                    pass

            if terminated:
                break
        
        reward = total_reward
        self._life_reward += reward

        # --- Continuous Play Logic (4 Games in 1 Episode) ---
        if terminated:
            # We add a call to save the video here for each INDIVIDUAL game (3 lives / 1 base)
            # This ensures we get footage of the normal game as the user requested.
            self._save_video()
            
            # Update meta-kills offset for the next game
            self._meta_kills_offset = 0 # RESET OFFSET SO KILLS ALWAYS SCALE 1-20
            self._prev_inner_kills = 0
            
            # Additional penalty based on consecutive losses REMOVED
            terminal = True
            last = True


        if self.duration >= self.length:
            last = True

        # --- Extract metrics from info ---
        self._ep_exploration = info.get('exploration_pct', self._ep_exploration)
        self._ep_reward += reward

        # Detect death event (Fixing the missing flags from battle_city_env.py)
        if info.get('game_over', False):
            self._ep_deaths += 1
            
        if info.get('base_destroyed', False):
            self._ep_base_lost += 1

        self.done = last
        return self._obs(
            reward, is_last=last, is_terminal=terminal, info=info)

    def _reset(self):
        # Save video from previous episode if applicable
        self._save_video()
        self._episode_count += 1

        with self.LOCK:
            self._env.reset()
        
        self.duration = 0
        self.done = False
        self._video_frames = []
        
        # Reset episode metrics
        self._ep_kills = 0
        self._ep_deaths = 0
        self._ep_base_lost = 0
        self._ep_exploration = 0.0
        self._ep_reward = 0.0
        self._ep_max_kill_reward = 0.0
        
        # Meta-kill tracker for continuous reward scaling
        self._meta_kills_offset = 0
        self._prev_inner_kills = 0

        # Capture first cropped frame
        if self._video_dir and self._episode_count % self._video_every == 0:
            try:
                screen = self._env.raw_env.screen
                frame = screen[16:224, 16:224].copy()
                self._video_frames.append(frame)
            except Exception:
                pass

        return self._obs(0.0, is_first=True)

    def _save_video(self):
        """Save collected frames as MP4 video."""
        if not self._video_dir or len(self._video_frames) < 2:
            return
        try:
            kills = self._ep_kills
            ep_len = self.duration
            score = int(self._ep_reward)
            step = self._global_step
            ep_num = self._episode_count
            fname = f"ep{ep_num:05d}_step{step}_kills{kills}_score{score}_len{ep_len}.mp4"
            fpath = os.path.join(self._video_dir, fname)

            h, w = self._video_frames[0].shape[:2]
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            # Changed video resolution to 128x128 to match new observation size
            writer = cv2.VideoWriter(fpath, fourcc, 60, (128, 128))
            for frame in self._video_frames:
                small = cv2.resize(frame, (128, 128), interpolation=cv2.INTER_AREA)
                bgr = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
                writer.write(bgr)
            writer.release()
            print(f"Video saved: {fname}")
        except Exception as e:
            print(f"Video save error: {e}")
        self._video_frames = []

    def _extract_ram(self):
        """Extract first 233 bytes of NES RAM as normalized float32 vector."""
        raw_ram = self._env.raw_env.ram[:self.RAM_SIZE]
        return raw_ram.astype(np.float32)

    def _obs(self, reward, is_first=False, is_last=False, is_terminal=False,
             info=None):
        ram = self._env.raw_env.ram

        # Count alive enemies
        enemies_alive = 0
        for i in range(2, 8):
            if 0xA0 + i < 0x100 and int(ram[0xA0 + i]) >= 128:
                enemies_alive += 1

        lives = int(ram[0x51]) if int(ram[0x51]) < 10 else 0

        obs = dict(
            reward=np.float32(reward),
            is_first=is_first,
            is_last=is_last,
            is_terminal=is_terminal,
            # --- Metrics (logged per step, aggregated per episode) ---
            **{
                'log/kills': np.float32(self._ep_kills),
                'log/deaths': np.float32(self._ep_deaths),
                'log/base_lost': np.float32(self._ep_base_lost),
                'log/exploration': np.float32(self._ep_exploration),
                'log/ep_reward': np.float32(self._ep_reward),
                'log/lives': np.float32(lives),
                'log/enemies_alive': np.float32(enemies_alive),
                'log/proximity': np.float32(getattr(self._env, 'cumulative_proximity', 0.0)),
            },
        )

        if self.use_ram:
            obs['ram'] = self._extract_ram()
        else:
            # Crop playfield directly (same as video) and resize
            screen = self._env.raw_env.screen
            image = screen[16:224, 16:224].copy()  # 208x208 playfield
            image = cv2.resize(image, self.size, interpolation=cv2.INTER_AREA)
            if self.gray:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)[:, :, None]
            obs['image'] = image

        return obs

    def close(self):
        self._save_video()  # Save any remaining video
        try:
            self._env.close()
        except Exception:
            pass
