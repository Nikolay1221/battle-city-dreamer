import gym
from gymnasium import spaces as gymnasium_spaces
import numpy as np
import cv2
from nes_py import NESEnv
from nes_py.wrappers import JoypadSpace
from config import RLConfig


# Define simple actions: Move and Fire
# Battle City controls: arrows for movement, A/B for fire.
# We'll combine movement + fire for efficiency.
COMPLEX_MOVEMENT = [
    ['NOOP'],
    ['up'],
    ['down'],
    ['left'],
    ['right'],
    ['A'],
    ['up', 'A'],
    ['down', 'A'],
    ['left', 'A'],
    ['right', 'A'],
]

class MultiDiscreteActionSpaceWrapper(gym.ActionWrapper):
    """
    Wraps the BattleCityEnv JoypadSpace (which has 10 discrete actions)
    into a MultiDiscrete([5, 2]) action space.
    Action[0] Movement: 0=NOOP, 1=up, 2=down, 3=left, 4=right
    Action[1] Fire: 0=NOOP, 1=A
    """
    def __init__(self, env):
        super().__init__(env)
        self.action_space = gymnasium_spaces.MultiDiscrete([5, 2])
        
    def action(self, act):
        # act is [mov_idx, fire_idx]
        mov, fire = act[0], act[1]
        
        if fire == 0:
            return mov
        else:
            if mov == 0:
                return 5
            else:
                return mov + 5

class BattleCityEnv(gym.Wrapper):
    def __init__(self, render_mode=None, is_visible=False, start_level=None):
        # Create base NES environment
        env = NESEnv(RLConfig.GAME_PATH)
        
        # Apply Joypad wrapper to discrete actions
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        
        # Apply MultiDiscrete wrapper to separate movement and fire
        env = MultiDiscreteActionSpaceWrapper(env)
        
        # Use gym.Wrapper
        super().__init__(env)
        
        # Store render_mode locally. 
        # We don't set it on env because JoypadSpace might block it (property)
        # Store render_mode locally. 
        self.render_mode = render_mode
        self.is_visible = is_visible
        self.start_level = start_level
        # Fix for Gymnasium expecting string keys dict instead of module reference from nested gym envs
        self.__dict__['metadata'] = {'render_modes': ['human', 'rgb_array']}
        
        # Explicity convert old gym spaces to new gymnasium spaces 
        # so EnvCompatibility doesn't choke later.
        self.observation_space = gymnasium_spaces.Box(low=0, high=255, shape=(128, 128, 1), dtype=np.uint8)
        
        # We applied MultiDiscreteActionSpaceWrapper which sets correct gymnasium MultiDiscrete space.
        # No need to overwrite it with Discrete(n) here anymore, let's just make sure action_space is carried over correctly.
        if hasattr(self.env, 'action_space'):
             self.action_space = self.env.action_space

        
        # Trackers
        self.prev_kills = [0] * 4
        self.cumulative_kills = 0
        self.prev_lives = 0
        self.steps_in_episode = 0
        
    def get_tactical_rgb(self):
        """Returns a downscaled 52x52 RGB tactical map for play.py."""
        screen = self.raw_env.screen
        board = screen[16:224, 16:224]  # 208x208 playfield
        thumb = cv2.resize(board, (52, 52), interpolation=cv2.INTER_NEAREST)
        return thumb

    def _get_obs(self):
        """Returns the current screen, cropped to playfield, grayscale, resized to 128x128."""
        # Get raw screen from nes_py (240, 256, 3)
        screen = self.env.screen.copy()
        
        # Crop to playfield only (remove right sidebar + borders)
        # Playfield: 13x13 tiles × 16px = 208x208, starts at ~(16, 8)
        cropped = screen[16:224, 16:224]  # (208, 208, 3)
        
        # Convert to Grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)
        
        # Resize to 128x128 (больше деталей чем 84x84)
        resized = cv2.resize(gray, (128, 128), interpolation=cv2.INTER_LINEAR)
        
        return np.expand_dims(resized, axis=-1)

    def reset(self, **kwargs):
        # Handle Gymnasium vs Gym API differences
        # nes-py doesn't accept 'seed' or 'options' in reset()
        if 'seed' in kwargs:
            # We can use the seed to seed numpy if we want, but NESEnv might not support it directly
            pass 
        kwargs.pop('seed', None)
        kwargs.pop('options', None)


        # We ignore the pixel observation returned by reset
        _ = self.env.reset(**kwargs)
        
        # Skip Title Screen (Full 5-step sequence)
        # NOTE: We use self.raw_env.step() to send RAW NES button presses,
        # bypassing JoypadSpace and MultiDiscrete wrappers.
        
        # 1. Wait for Logo & Title
        for _ in range(120): self.raw_env.step(0) 
            
        # 2. Press START (Skip Title, go to Player Select)
        for _ in range(15): self.raw_env.step(8) 
        for _ in range(40): self.raw_env.step(0) 

        # Force Level Selection if specified
        if self.start_level is not None:
             target_val = self.start_level # RAM 1 = Stage 1
             self.raw_env.ram[RLConfig.ADDR_STAGE] = target_val
        
        # 3. Press START (Select 1 Player, go to Stage Select)
        for _ in range(15): self.raw_env.step(8)
        for _ in range(40): self.raw_env.step(0)

        # Re-enforce Level Selection just in case
        if self.start_level is not None:
             self.raw_env.ram[RLConfig.ADDR_STAGE] = self.start_level

        # 4. Press START (Start Stage)
        for _ in range(15): self.raw_env.step(8)
        
        # 5. Wait for Curtain to fully open and level to draw
        for _ in range(100): self.raw_env.step(0)
             
        # Reset trackers
        # Snapshot RAM kills after boot — treats any residual values as baseline (not new kills)
        # Snapshot RAM kills after boot
        self.prev_enemies_left = self.env.ram[0x80]
        self.cumulative_kills = 0
        self.death_count = 0
        self.steps_in_episode = 0
        # Initialize lives to ACTUAL value from RAM
        self.prev_lives = self.env.ram[0x51]
        
        # Reset Proximity Reward Tracker
        self.prev_min_dist = self._get_nearest_dist()
        self.prev_px = self.env.ram[RLConfig.ADDR_PLAYER_X]
        self.prev_py = self.env.ram[RLConfig.ADDR_PLAYER_Y]
        
        # Exploration Tracker
        self.visited_cells = set()

        # Proximity Reward Logic
        self.prev_min_dist = 999.0
        self.prev_px = 0
        self.prev_py = 0
        
        # Base Latch: 
        # RAM[0x68] is 0 at boot, 80 when active, 0 when destroyed.
        # We must wait for it to become non-zero (active) before checking for 0 (destroyed).
        self.base_active = False 
        
        # Load Game Over template (REMOVED - Using RAM Logic)
        self.game_over_template = None
            
        return self._get_obs()

    def _get_nearest_dist(self):
        """Calculates distance to nearest active enemy."""
        # Player coordinates (Center)
        # RAM gives Top-Left. Sprite is 16x16. Center is +8.
        px = float(self.env.ram[RLConfig.ADDR_PLAYER_X]) + 8.0
        py = float(self.env.ram[RLConfig.ADDR_PLAYER_Y]) + 8.0
        
        min_dist = 999.0
        found = False

        # Iterate Enemy Slots 2 to 7
        for i in range(2, 8):
            # Check Status (Alive >= 128)
            status = self.env.ram[RLConfig.ADDR_ENEMY_STATUS_BASE + i]
            if status < 128:
                continue
            
            # Enemy Coordinates
            ex = float(self.env.ram[RLConfig.ADDR_COORD_X_BASE + i]) + 8.0
            ey = float(self.env.ram[RLConfig.ADDR_COORD_Y_BASE + i]) + 8.0
            
            # Simple 0,0 check (sometimes happens during init)
            if self.env.ram[RLConfig.ADDR_COORD_X_BASE + i] == 0 and self.env.ram[RLConfig.ADDR_COORD_Y_BASE + i] == 0:
                continue
            
            # Euclidian Distance
            d = np.sqrt((ex - px)**2 + (ey - py)**2)
            if d < min_dist:
                min_dist = d
                found = True
                
        return min_dist if found else 999.0

    def _check_game_over(self):
        if self.game_over_template is None:
            return False
            
        import cv2
        # Get frame in BGR (nes-py render is RGB)
        # NES native resolution is 256x240
        frame_rgb = self.env.render(mode='rgb_array')
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        
        # Safety Check: Dimensions
        # If template is larger than frame (e.g. 512x480 vs 256x240), we must downscale template
        fh, fw = frame_bgr.shape[:2]
        th, tw = self.game_over_template.shape[:2]
        
        template_to_use = self.game_over_template
        if th > fh or tw > fw:
             # Assuming 2x scale difference (common)
             scale_x = fw / tw
             scale_y = fh / th
             scale = min(scale_x, scale_y)
             new_w = int(tw * scale)
             new_h = int(th * scale)
             template_to_use = cv2.resize(self.game_over_template, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Template Matching
        try:
            res = cv2.matchTemplate(frame_bgr, template_to_use, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            
            # Threshold 0.75 (slightly lower for robustness)
            if max_val > 0.75:
                return True
        except Exception as e:
            # print(f"Game Over Check Error: {e}")
            pass
            
        return False

    def step(self, action):
        self.steps_in_episode += 1
        # We ignore the pixel observation returned by step
        _, reward, done, info = self.env.step(action)
        
        # Check for Visual Game Over
        if not done:
             # RAM Check for Base Destruction (0x68)
             # Logic: 0 (Boot) -> 80 (Active) -> 0 (Destroyed)
             base_status = self.env.ram[RLConfig.ADDR_BASE]
             
             if not self.base_active:
                 if base_status != 0:
                     self.base_active = True
             else:
                 # Base was active, now checking if destroyed
                 if base_status == 0:
                     done = True
                     info['base_destroyed'] = True
                     
                 # INVULNERABLE BASE CHEAT
                 # Uses the game's native "Shovel" (Лопата) bonus mechanically
                 import config
                 if getattr(config, 'INVULNERABLE_BASE', False) and not done:
                     # 0x45 is the HQArmour_Timer. If it falls low, we re-apply the Shovel
                     if self.env.ram[0x45] < 5:
                         self.env.ram[0x88] = 2 # Shovel Powerup (ID 2)
                         self.env.ram[0x86] = self.env.ram[0x90] # X to Player
                         self.env.ram[0x87] = self.env.ram[0x98] # Y to Player
                     else:
                         self.env.ram[0x45] = 20 # Freeze timer so it never disappears
        
        # Check for Lives (0x51)
        # If lives == 0, it means Game Over (or about to be)
        if not done:
            curr_lives = self.env.ram[RLConfig.ADDR_LIVES]
            if curr_lives == 0:
                done = True
                info['game_over'] = True
        
        # Tracks metrics for info dict
        curr_enemies_left = self.env.ram[0x80]
        curr_lives = self.env.ram[RLConfig.ADDR_LIVES]
        
        # Address 0x80 tracks remaining enemies from 20 down to 0
        # If it decreases, it means we killed an enemy
        if 0 <= curr_enemies_left < self.prev_enemies_left and self.prev_enemies_left <= 20:
            new_kills = self.prev_enemies_left - curr_enemies_left
            self.cumulative_kills += new_kills
        
        # HUGE BONUS for Level Completion (20 Kills)
        # In Battle City, winning a stage happens when 0 enemies are left
        if not done and curr_enemies_left == 0 and self.prev_enemies_left > 0:
             done = True
             info['is_success'] = True
        
        # Track deaths
        if curr_lives < self.prev_lives:
            self.death_count += 1
            
        # Exploration tracker (just for info)
        px = self.env.ram[RLConfig.ADDR_PLAYER_X]
        py = self.env.ram[RLConfig.ADDR_PLAYER_Y]
        
        if 24 <= px <= 216 and 24 <= py <= 216:
            grid_x = (px - 24) // 16
            grid_y = (py - 24) // 16
            cell_id = (grid_x, grid_y)
            if cell_id not in self.visited_cells:
                self.visited_cells.add(cell_id)
        # --- Reward Calculation ---
        kill_reward = 0.0
        death_penalty = 0.0
        
        # Base (Eagle) is always at the bottom-center of the map
        # Pixel coordinates: approximately X=112, Y=208
        BASE_X = 112
        BASE_Y = 208
        # Max possible Manhattan distance from base to far corner (~320px)
        MAX_DIST = 320.0
        
        dist_to_base = abs(int(px) - BASE_X) + abs(int(py) - BASE_Y)
        # Proximity factor: 1.0 at base, 0.0 at max distance (linear gradient)
        proximity_factor = max(0.0, 1.0 - dist_to_base / MAX_DIST)
        # Keep near_base flag for meta-kill gating in Dreamer wrapper
        near_base = dist_to_base <= 80
        
        # Linear kill reward: 1st kill = +1, 2nd = +2, ..., 20th = +20
        # Scaled by proximity to base: further away = smaller reward
        if 0 <= curr_enemies_left < self.prev_enemies_left and self.prev_enemies_left <= 20:
            new_kills = self.prev_enemies_left - curr_enemies_left
            for k in range(new_kills):
                kill_num = self.cumulative_kills - new_kills + k + 1
                kill_reward += float(kill_num) * proximity_factor  # scaled by distance
        
        # Linear death penalty: 1st death = -1, 2nd = -2, 3rd = -3
        if curr_lives < self.prev_lives:
            death_penalty = -float(self.death_count)  # death_count already incremented above
        
        # Bonus for level completion
        level_bonus = 10.0 if info.get('is_success', False) else 0.0
        
        # Penalty for base destruction
        base_penalty = -5.0 if info.get('base_destroyed', False) else 0.0
        
        custom_reward = kill_reward + death_penalty + level_bonus + base_penalty
        
        # Update trackers
        self.prev_enemies_left = curr_enemies_left
        self.prev_lives = curr_lives
        
        # Info for debugging
        info['kills'] = self.cumulative_kills
        info['lives'] = curr_lives
        info['exploration'] = len(self.visited_cells)
        info['near_base'] = near_base
        info['dist_to_base'] = dist_to_base
        

        
        return self._get_obs(), custom_reward, done, info

    def render(self, **kwargs):
        # Explicit render handling
        # if self.is_visible: ... removed for headless optimization
             
        # Also return frame if needed by callbacks (so VideoRecorder works if we use it)
        mode = kwargs.get('mode', self.render_mode)
        if mode == 'rgb_array':
             return self.env.render(mode='rgb_array')

    @property
    def raw_env(self):
        """Expose the underlying NESEnv for RAM access."""
        e = self.env
        while hasattr(e, 'env'):
            e = e.env
        return e

    @property
    def render_mode(self):
        return getattr(self, "_render_mode", None)

    @render_mode.setter
    def render_mode(self, value):
        self._render_mode = value

