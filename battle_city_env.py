import gymnasium as gym
from gymnasium import spaces
from nes_py import NESEnv
from nes_py.wrappers import JoypadSpace
import numpy as np
from collections import deque
import os
import cv2 
import config 

class BattleCityEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 60}

    def __init__(self, rom_path=config.ROM_PATH, render_mode=None, use_vision=False, stack_size=4, target_stage=None, enemy_count=20, no_shooting=False, reward_config=None, exploration_trigger=None):
        super().__init__()
        
        self.rom_path = rom_path
        self.render_mode = render_mode
        self.USE_VISION = use_vision
        self.STACK_SIZE = stack_size
        self.target_stage = target_stage
        
        self.MAX_STEPS = 100_000_000 
        self.steps_in_episode = 0
        
        # New Mechanics
        self.enemy_count = enemy_count
        self.no_shooting = no_shooting
        self.exploration_trigger = exploration_trigger 
        self.ambush_triggered = False
        
        if not os.path.exists(self.rom_path):
            raise FileNotFoundError(f"ROM file not found at: {self.rom_path}")

        self.raw_env = NESEnv(self.rom_path)
        
        # Define Actions
        self.actions_list = [
            ['NOOP'],
            ['up'], ['down'], ['left'], ['right'],
            ['A'], # Fire
            ['up', 'A'], ['down', 'A'], ['left', 'A'], ['right', 'A']
        ]
        
        self.env = JoypadSpace(self.raw_env, self.actions_list)
        self.action_space = spaces.Discrete(len(self.actions_list))

        # Tactical Grid Settings (High Res: 52x52)
        self.GRID_SIZE = 52 
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.GRID_SIZE, self.GRID_SIZE, self.STACK_SIZE), dtype=np.uint8)
        
        # Color definitions (RGB) for visual scanning
        self.TILE_COLORS = {
            "brick": np.array([228, 92, 16]), 
            "brick_dark": np.array([168, 16, 0]), 
            "empty": np.array([0, 0, 0]),
            "eagle": np.array([60, 90, 60]), 
            "steel": np.array([124, 124, 124]),
            "player": np.array([232, 208, 32]), 
            "enemy_silver": np.array([180, 180, 180]),
            "enemy_red": np.array([168, 0, 32]),
            "bullet": np.array([255, 255, 255]) 
        }
        self.known_labels = list(self.TILE_COLORS.keys())
        self.known_colors = np.array(list(self.TILE_COLORS.values()), dtype=np.float32)

        # ID Mapping - GRAYSCALE INTENSITY
        self.ID_MAP = {
            "empty": 0,
            "brick": 200,   
            "steel": 255,   
            "eagle": 255,   
            "player": 150,  
            "enemy": 80,    
            "bullet": 255   
        }
        
        self.ram_stack = deque(maxlen=self.STACK_SIZE)
        self.frames = deque(maxlen=self.STACK_SIZE)

        # Rewards Configuration
        # Rewards Configuration
        # If reward_config is passed (from Multi-Env setup), use it. Else default to config.py defaults.
        if reward_config:
            self.rew_kill = reward_config.get('kill', 1.0)
            self.rew_death = reward_config.get('death', -1.0)
            self.rew_base = reward_config.get('base', -20.0)
            self.rew_explore = reward_config.get('explore', 0.01)
            self.rew_win = reward_config.get('win', 20.0)
            self.rew_time = reward_config.get('time', -0.005) # Default if missing in config
            self.rew_dist = reward_config.get('distance', 0.01)
        else:
            self.rew_kill = getattr(config, 'REW_KILL', 1.0)
            self.rew_death = getattr(config, 'REW_DEATH', -1.0)
            self.rew_base = -100.0 # Make it EXTREMELY painful
            self.rew_explore = getattr(config, 'REW_EXPLORE', 0.05) # Boost exploration
            self.rew_win = getattr(config, 'REW_WIN', 50.0)
            self.rew_time = -0.0001 # Make "existing" almost free
            self.rew_dist = getattr(config, 'REW_DISTANCE', 0.01)

        # CONDITIONAL WIN REWARD
        # Only grant win reward if this is a "Full" game (20 enemies)
        if self.enemy_count >= 20:
             self.rew_win = getattr(config, 'REW_WIN', 50.0)
        else:
             self.rew_win = 0.0 # No reward for clearing a partial/empty level
        
        self.rew_stuck = -0.001 
        self.rew_brick = 0.1 # BIG Reward for breaking walls to encourage movement
        
        # self.rew_dist is loaded from config above (line 91 or 99)
        self.prev_min_dist = 999.0

        # RAM Addresses
        self.ADDR_LIVES = 0x51
        self.ADDR_STATE = 0x92
        self.ADDR_BASE_STATUS = 0x68 # NEW: Base Latch Address
        self.ADDR_ENEMIES_LEFT = 0x80 # Enemies remaining to spawn
        self.ADDR_ENEMIES_ON_SCREEN = 0xA0 # Enemies currently active
        self.ADDR_KILLS = [0x73, 0x74, 0x75, 0x76] 
        self.ADDR_SCORE = [0x70, 0x71, 0x72] 
        self.ADDR_BONUS = 0x62
        self.ADDR_STAGE = 0x85
        self.ADDR_MAP = 0x0731
        self.ADDR_BASE_TILE = 0x07D3
        self.ADDR_X_ARR = 0x90 
        self.ADDR_Y_ARR = 0x98
        
        self.prev_lives = 3
        self.prev_kill_sum = 0
        self.prev_score_sum = 0
        self.prev_x = 0
        self.prev_y = 0
        self.idle_steps = 0
        self.visited_sectors = set()
        
        self.episode_kills = 0 
        self.level_cleared = False
        self.base_active_latch = False # NEW: Latch for base status
        
        # --- EXPERIMENTAL REWARDS ---
        self.last_kill_step = -9999
        self.kill_streak = 0
        
        # Base Coordinates (Approx, in 52x52 grid or higher? Use RAM)
        # Base is at 120, 216 roughly. 
        # In RAM coords (0x90, 0x98): X~120, Y~208-216?
        # Let's use scalar distance.
        self.BASE_X = 120
        self.BASE_Y = 216
        self.prev_enemies = [] # Track enemy state for defense logic
        
        # Path-Based Reward Cache
        self.cached_path_length = 999
        self.path_recalc_counter = 0
        self.PATH_RECALC_INTERVAL = 5  # Recalculate every 5 steps


    def _get_tactical_map(self):
        """Creates a 52x52 GRAYSCALE matrix representing the game state (VISUAL ONLY)."""
        grid = np.zeros((self.GRID_SIZE, self.GRID_SIZE), dtype=np.uint8)
        frame_rgb = self.raw_env.screen
        
        # 1. SCAN TERRAIN (Visual)
        screen_area = frame_rgb[16:224, 16:224]
        tiles = screen_area.reshape(self.GRID_SIZE, 4, self.GRID_SIZE, 4, 3).transpose(0, 2, 1, 3, 4)
        tiles_flat = tiles.reshape(self.GRID_SIZE, self.GRID_SIZE, 16, 3).astype(np.float32)
        
        diffs = tiles_flat[..., np.newaxis, :] - self.known_colors
        dists = np.sum(diffs**2, axis=-1)
        labels = np.argmin(dists, axis=-1)
        mask_bg = np.max(tiles_flat, axis=-1) > 40
        
        # Indices
        idx_brick = self.known_labels.index("brick")
        idx_brick_dark = self.known_labels.index("brick_dark")
        idx_steel = self.known_labels.index("steel")
        idx_eagle = self.known_labels.index("eagle")
        idx_player = self.known_labels.index("player")
        idx_enemy_s = self.known_labels.index("enemy_silver")
        idx_enemy_r = self.known_labels.index("enemy_red")
        idx_bullet = self.known_labels.index("bullet")
        
        # Count pixels per cell
        count_brick = np.sum(((labels == idx_brick) | (labels == idx_brick_dark)) & mask_bg, axis=2)
        count_steel = np.sum((labels == idx_steel) & mask_bg, axis=2)
        count_eagle = np.sum((labels == idx_eagle) & mask_bg, axis=2)
        count_player = np.sum((labels == idx_player) & mask_bg, axis=2)
        count_enemy = np.sum(((labels == idx_enemy_s) | (labels == idx_enemy_r)) & mask_bg, axis=2)
        count_bullet = np.sum((labels == idx_bullet) & mask_bg, axis=2)
        
        # Apply Grayscale Values
        grid[count_brick > 1] = self.ID_MAP["brick"]
        grid[count_steel > 2] = self.ID_MAP["steel"]
        grid[count_eagle > 1] = self.ID_MAP["eagle"]
        
        # Dynamic objects (lower threshold)
        grid[count_player > 0] = self.ID_MAP["player"]
        grid[count_enemy > 0] = self.ID_MAP["enemy"]
        grid[count_bullet > 0] = self.ID_MAP["bullet"]
        
        return grid

    def _get_obs(self):
        current_map = self._get_tactical_map()
        while len(self.frames) < self.STACK_SIZE:
            self.frames.append(current_map)
        self.frames.append(current_map)
        obs_stack = np.stack(self.frames, axis=-1)
        return obs_stack

    def get_tactical_rgb(self):
        if not self.frames: return np.zeros((52, 52, 3), dtype=np.uint8)
        grid = self.frames[-1]
        img = cv2.cvtColor(grid, cv2.COLOR_GRAY2RGB)
        return img

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.raw_env.reset()
        self.steps_in_episode = 0
        self.episode_score = 0.0
        self.episode_kills = 0
        self.level_cleared = False
        self.base_active_latch = False # Reset Latch
        self.ambush_triggered = False # Reset ambush
        self.visited_sectors = set()
        self.frames.clear()
        
        # Start Sequence
        for _ in range(80): self.raw_env.step(0)
        for _ in range(10): self.raw_env.step(8)
        for _ in range(30): self.raw_env.step(0)
        for _ in range(10): self.raw_env.step(8)
        for _ in range(30): self.raw_env.step(0)
        for _ in range(10): self.raw_env.step(8)
        for _ in range(60): self.raw_env.step(0)

        # --- CUSTOM ENEMY COUNT RAM HACK ---
        if 0 < self.enemy_count < 20:
            # Note: 0x80 is "Enemies Remaining to Spawn". 
            # The game starts with 20. If we set it to (N-2), it will spawn roughly N.
            # (Because 2-3 are usually already on screen or pending).
            # This is an approximation.
            target = max(0, self.enemy_count - 3) 
            self.raw_env.ram[self.ADDR_ENEMIES_LEFT] = target
            # Also clear any on screen if we want very few? No, let them spawn.

        # Init state
        self.prev_lives = int(self.raw_env.ram[self.ADDR_LIVES])
        self.prev_x = int(self.raw_env.ram[self.ADDR_X_ARR])
        self.prev_y = int(self.raw_env.ram[self.ADDR_Y_ARR])
        self.prev_stage = int(self.raw_env.ram[self.ADDR_STAGE]) # Track Stage
        
        self.prev_kill_sum = sum([int(self.raw_env.ram[addr]) for addr in self.ADDR_KILLS])
        self.prev_score_sum = sum([int(self.raw_env.ram[addr]) for addr in self.ADDR_SCORE])

        self.idle_steps = 0
        self.prev_min_dist = 999.0 # Reset distance tracker
        
        # Track bricks for wall-breaking rewards
        grid = self._get_tactical_map()
        self.prev_brick_count = np.sum(grid == self.ID_MAP["brick"])
        
        self.frames.clear()
        return self._get_obs(), {}

    def step(self, action):
        nes_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        
        repeat = getattr(config, 'FRAME_SKIP', 4)
        
        ram = self.raw_env.ram
        old_x, old_y = int(ram[self.ADDR_X_ARR]), int(ram[self.ADDR_Y_ARR])

        # --- MODE: No Shooting ---
        if self.no_shooting:
            # Remap fire actions to movement only
            if action == 5:
                action = 0
            elif action >= 6 and action <= 9:
                action = action - 5

        for _ in range(repeat):
            obs, r, d, i = self.env.step(action)
            nes_reward += r 
            if d:
                terminated = True
                break
        
        # (Old Limit Logic Removed - Moved to End with Ambush)
        # -------------------------------------------------
        
        self.steps_in_episode += 1 
        ram = self.raw_env.ram
        reward = 0 
        
        # 0. State Tracking for Rewards
        current_enemies = []
        for i in range(1, 7): # Slots 1-6
            e_hp = int(ram[0x60 + i]) if 0x60 + i < 0x100 else 0
            e_x  = int(ram[0x90 + i]) if 0x90 + i < 0x100 else 0
            e_y  = int(ram[0x98 + i]) if 0x98 + i < 0x100 else 0
            current_enemies.append({'hp': e_hp, 'x': e_x, 'y': e_y})

        # 1. Kill Rewards
        curr_kill_sum = sum([int(ram[addr]) for addr in self.ADDR_KILLS])
        diff = curr_kill_sum - self.prev_kill_sum
        
        info['reward_events'] = [] # For visualization
        
        if diff > 0:
            reward += self.rew_kill * diff
            self.episode_kills += diff
            info['reward_events'].append(f"KILL (+{self.rew_kill})")
            
            # --- KILL MILESTONES (VOLUME BONUS) ---
            # ... (Milestone logic same as before) ...
            
            milestones = {
                2: 1.0,  # "Double Kill" equivalent
                5: 2.0,  # "Rampage"
                8: 5.0   # "Dominating"
            }
            
            if self.episode_kills in milestones:
                bonus = milestones[self.episode_kills]
                reward += bonus
                info['reward_events'].append(f"KILL MILESTONE {self.episode_kills} (+{bonus})")

            # --- BASE DEFENSE BONUS (Advanced) ---
            # If we have history (not first step)
            if hasattr(self, 'prev_enemies') and self.prev_enemies:
                for i in range(6):
                    prev = self.prev_enemies[i]
                    curr = current_enemies[i]
                    
                    # Logic: Enemy WAS alive, NOW is dead (or respawning/gone)
                    # Note: When dead, HP becomes 0.
                    if prev['hp'] > 0 and curr['hp'] == 0:
                        # This guy died. Was he a threat?
                        dist = np.sqrt((prev['x'] - self.BASE_X)**2 + (prev['y'] - self.BASE_Y)**2)
                        if dist < 60:
                            reward += 0.5
                            info['reward_events'].append("DEFENDER (+0.5)")
            
        # Update history unconditionally
        self.prev_enemies = current_enemies
        self.prev_kill_sum = curr_kill_sum
        
        # --- ANTI-HACK MEASURES (ROBUSTNESS) ---
        
        # 1. ABANDONMENT PENALTY
        # If an enemy is threatening the base (dist < 50) AND player is far away (dist > 80),
        # punish the player for not defending.
        min_enemy_dist = 999.0
        if current_enemies:
            # Filter alive ones
            alive_dists = [np.sqrt((e['x'] - self.BASE_X)**2 + (e['y'] - self.BASE_Y)**2) for e in current_enemies if e['hp'] > 0]
            if alive_dists:
                min_enemy_dist = min(alive_dists)
        
        p_x, p_y = int(ram[self.ADDR_X_ARR]), int(ram[self.ADDR_Y_ARR])
        player_dist = np.sqrt((p_x - self.BASE_X)**2 + (p_y - self.BASE_Y)**2)
        
        if min_enemy_dist < 50 and player_dist > 80:
             reward -= 0.05
             # Don't spam log, maybe just once in a while or implicit
        
        # 2. IDLE / COWARDICE PENALTY
        # If we are idling (no explore, no kills) for too long, we punish before reset.
        if self.idle_steps > 3000:
             truncated = True
             reward -= 5.0 # "Boredom" penalty
             info['game_over_reason'] = 'idle_timeout'
             info['reward_events'].append("IDLE TIMEOUT (-5.0)")

        # CHECK FOR VICTORY
        # ...
        curr_stage = int(ram[self.ADDR_STAGE])
        
        # 1. TIME PENALTY (HUNGER)
        # Force agent to solve the level quickly. 1000 steps = -5.0 score.
        if self.enemy_count >= 20 and not self.level_cleared:
             reward += self.rew_time
        
        # 2. PATH-BASED REWARD (Smart Magnet)
        # Instead of Euclidean distance, use BFS path length
        self.path_recalc_counter += 1
        
        if self.path_recalc_counter >= self.PATH_RECALC_INTERVAL:
            self.path_recalc_counter = 0
            # Find nearest enemy and calculate path length
            new_path_length = self._get_path_length_to_nearest_enemy(p_x, p_y, current_enemies)
            
            if new_path_length < self.cached_path_length and new_path_length < 200:
                reward += self.rew_dist  # Moved closer along path!
                info['reward_events'].append(f"PATH (+{self.rew_dist})")
            
            self.cached_path_length = new_path_length

        # Condition 1: Kills Limit (Only if full game)
        # If enemy_count < 20, we don't grant win for killing all (because there aren't 20)
        # unless we explicitly want to logic that out. 
        # Currently requested: "REW_WIN = 20.0 начислять не будем" if < 20.
        if self.enemy_count >= 20: 
            if self.episode_kills >= 20 and not self.level_cleared:
                reward += self.rew_win
                self.level_cleared = True
                terminated = True 
                info['is_success'] = True
                info['win_reason'] = 'kills_limit'

        # Condition 2: Stage Changed
        # This means the internal game logic is happy (all enemies dead).
        if curr_stage != self.prev_stage:
             # Only grant the BIG reward if it was a full game
             if not self.level_cleared and self.enemy_count >= 20:
                 reward += self.rew_win
                 self.level_cleared = True
                 info['is_success'] = True
                 info['win_reason'] = 'stage_cleared'
             
             # Terminate regardless
             terminated = True
             
        self.prev_stage = curr_stage
        
        # 2. Death
        curr_lives = int(ram[self.ADDR_LIVES])
        if curr_lives < 10 and self.prev_lives < 10:
             if curr_lives < self.prev_lives:
                reward += self.rew_death
                info['reward_events'].append(f"DIED ({self.rew_death})")
        self.prev_lives = curr_lives
        
        # 3. Game Over Checks
        if curr_lives == 0:
            terminated = True
            info['game_over_reason'] = 'no_lives'

        base_status = int(ram[self.ADDR_BASE_STATUS])
        if base_status != 0: self.base_active_latch = True
        if self.base_active_latch and base_status == 0:
             terminated = True
             reward -= 100.0 # EXTREME PENALTY for base destruction
             info['game_over_reason'] = 'base_destroyed'
             info['reward_events'].append("CRITICAL FAILURE: BASE DESTROYED (-100.0)")
             
        # 3.5 BRICK DESTRUCTION REWARD
        # (Compare current world to previous to see if we opened a path)
        curr_grid = self._get_tactical_map()
        curr_brick_count = np.sum(curr_grid == self.ID_MAP["brick"])
        if curr_brick_count < self.prev_brick_count:
            reward += 0.05 * (self.prev_brick_count - curr_brick_count)
            # info['reward_events'].append(f"BREACH (+0.05)")
        self.prev_brick_count = curr_brick_count

        # 4. EXPLORATION REWARD
        curr_x, curr_y = int(ram[self.ADDR_X_ARR]), int(ram[self.ADDR_Y_ARR])
        
        if abs(curr_x - old_x) > 2 or abs(curr_y - old_y) > 2:
             sec_x, sec_y = curr_x // 16, curr_y // 16
             if (sec_x, sec_y) not in self.visited_sectors:
                 reward += self.rew_explore
                 self.visited_sectors.add((sec_x, sec_y))
                 info['reward_events'].append(f"EXPLORE (+{self.rew_explore})")
             self.idle_steps = 0
        else:
             if action != 0: 
                 reward += self.rew_stuck
             self.idle_steps += 1
        
        # --- AMBUSH LOGIC & ENEMY CONTROL ---
        
        # 1. AMBUSH STATE MANAGEMENT
        if self.exploration_trigger is not None:
            explore_pct = len(self.visited_sectors) / 240.0
            
            if not self.ambush_triggered:
                if explore_pct >= self.exploration_trigger:
                    # --- TRIGGER ACTIVATED ---
                    self.ambush_triggered = True
                    info['reward_events'].append("AMBUSH STARTED! (ENEMIES ARRIVING)")
                    
                    # Teleport existing "held" enemies to battle positions
                    # Spawn X Coords: 0, 128, 192 (Approximate standard spawns)
                    spawn_x = [0, 128, 192]
                    for i in range(1, 7):
                        if ram[0x60 + i] > 0: # If alive (was held)
                             target_x = spawn_x[(i-1) % 3]
                             self.raw_env.ram[0x90 + i] = target_x # X
                             self.raw_env.ram[0x98 + i] = 0        # Y (Top)
                
                else:
                    # --- PRE-AMBUSH SUPPRESSION ---
                    # Keep enemies alive but trapped/hidden at (0,0)
                    for i in range(1, 7):
                         # Slots 1-4 have HP. Slots 5-6 might be coords-only (stars/glitches).
                         should_suppress = False
                         
                         if i <= 4:
                             if ram[0x60 + i] > 0: should_suppress = True
                         else:
                             should_suppress = True # Unconditional for ghosts
                             
                         if should_suppress:
                              self.raw_env.ram[0x90 + i] = 0
                              self.raw_env.ram[0x98 + i] = 0
        
        # 2. STANDARD LIMIT (For non-Ambush limited modes like VERY_EASY)
        # Only apply if we are NOT in pre-ambush suppression (since suppression handles it).
        # OR if Ambush is active, we might still want to limit to 5 enemies?
        # Yes, obey self.enemy_count.
        
        apply_standard_limit = True
        if self.exploration_trigger is not None and not self.ambush_triggered:
             apply_standard_limit = False # Suppression already handling it
             
        if apply_standard_limit and self.enemy_count < 6:
             for i in range(self.enemy_count + 1, 7):
                 if 0x90 + i < 0x100: 
                      self.raw_env.ram[0x90 + i] = 0
                      self.raw_env.ram[0x98 + i] = 0
                      #Kill excess beyond limit (standard behavior)
                      self.raw_env.ram[0x60 + i] = 0
             
             # Special Case: If 0 enemies, we also prevent idle timeouts
             if self.enemy_count == 0:
                 self.idle_steps = 0 
        
        info['exploration_pct'] = len(self.visited_sectors) / 240.0 # Corrected total
        info['trigger_pct'] = self.exploration_trigger if self.exploration_trigger else 0.0
        info['nes_reward'] = nes_reward
        info['kills'] = self.episode_kills 
        # Env Type: 0 = Peaceful/Sim/Simplified, 1 = Full Standard Combat
        info['env_type'] = 1 if (self.enemy_count >= 20 and not self.no_shooting) else 0


        self.prev_x, self.prev_y = curr_x, curr_y
        
        if self.steps_in_episode >= self.MAX_STEPS:
            truncated = True 

        self.episode_score += reward
        info['score'] = self.episode_score
        
        # --- ENEMY DETECTION FOR VISUALIZATION ---
        player_cv, enemies_data = self._detect_enemies()
        info['player_cv'] = player_cv
        info['enemy_positions'] = enemies_data
        info['enemies_detected'] = len(enemies_data)
        
        if enemies_data:
            p_x, p_y = int(ram[0x90]), int(ram[0x98])
            dists = [np.sqrt((ex - p_x)**2 + (ey - p_y)**2) for (ex, ey, *_) in enemies_data]
            info['closest_enemy_dist'] = min(dists)
        else:
            info['closest_enemy_dist'] = 999.0
        
        return self._get_obs(), reward, terminated, truncated, info

    def render(self, mode='human'):
        try:
            return self.env.render(mode=mode)
        except TypeError:
            return self.env.render()

    def cheat_clear_enemies(self):
        """Debug tool: Clear all enemies by manipulating RAM."""
        # 1. Reset spawn counter
        self.raw_env.ram[self.ADDR_ENEMIES_LEFT] = 0
        # 2. Reset on-screen counter
        self.raw_env.ram[self.ADDR_ENEMIES_ON_SCREEN] = 0
        # 3. Optional: Zero out enemy HPs 
        for i in range(1, 5):
            self.raw_env.ram[0x60 + i] = 0
            
    def close(self):
        self.env.close()

    # --- ENEMY DETECTION HELPER ---
    def _detect_enemies(self):
        """
        Detects enemies using RAM (100% Accurate).
        RAM Map:
        X Coords: 0x90 (Player), 0x91-0x94 (Enemies)
        Y Coords: 0x98 (Player), 0x99-0x9C (Enemies)
        """
        ram = self.raw_env.ram
        screen = self.raw_env.screen
        
        # 1. Player (Slot 0)
        px = int(ram[0x90])
        py = int(ram[0x98])
        player_pos = (px + 8, py + 8)  # Center of tank
        
        # 2. Enemies (Slots 1-4)
        enemies = []
        
        for i in range(1, 6):  # 5 slots (Battle City can use slot 5 sometimes)
            ex = int(ram[0x90 + i])
            ey = int(ram[0x98 + i])
            
            # Skip empty slots (coords at 0,0)
            if ex == 0 and ey == 0:
                continue
            
            # --- LoS Check (Raycast) ---
            is_visible = False
            
            # GAME MECHANIC: Axis Alignment (Tanks shoot straight)
            dx = abs(ex - px)
            dy = abs(ey - py)
            is_aligned = (dx < 12) or (dy < 12)
            
            if is_aligned:
                is_visible = True
                # DENSER SAMPLING (10 points)
                for t in np.linspace(0.1, 0.9, 10):
                    sx = int(px + (ex - px) * t)
                    sy = int(py + (ey - py) * t)
                    if 0 <= sx < 256 and 0 <= sy < 240:
                        pixel = screen[sy, sx]
                        if pixel[0] > 60:  # Wall (Red channel high)
                            is_visible = False
                            break
            
            # Format: (x_center, y_center, slot_id, confidence, is_visible)
            enemies.append((ex + 8, ey + 8, i, 1.0, is_visible))
            
        return player_pos, enemies

    def _get_path_length_to_nearest_enemy(self, p_x, p_y, enemies):
        """
        Calculate BFS path length from player to nearest enemy.
        Returns path length (in grid cells) or 999 if no path found.
        """
        if not enemies:
            return 999
        
        try:
            grid_map = self._get_tactical_map()
        except:
            return 999
        
        # Find nearest enemy by Euclidean (to pick target)
        min_dist = 999.0
        target_x, target_y = 0, 0
        
        for e in enemies:
            if e['hp'] > 0 or (e['x'] != 0 or e['y'] != 0):
                d = np.sqrt((e['x'] - p_x)**2 + (e['y'] - p_y)**2)
                if d < min_dist:
                    min_dist = d
                    target_x, target_y = e['x'], e['y']
        
        if target_x == 0 and target_y == 0:
            return 999
        
        # Convert to grid coords (52x52 over 208x208 play area)
        # Map area: 16:224 pixels. Grid: 52 cells -> 4px per cell.
        gx_start = max(0, min(51, int((p_x - 16) / 4)))
        gy_start = max(0, min(51, int((p_y - 16) / 4)))
        gx_end   = max(0, min(51, int((target_x - 16) / 4)))
        gy_end   = max(0, min(51, int((target_y - 16) / 4)))
        
        # Quick BFS
        queue = [(gx_start, gy_start, 0)]  # x, y, distance
        visited = set([(gx_start, gy_start)])
        
        while queue:
            cx, cy, dist = queue.pop(0)
            
            if cx == gx_end and cy == gy_end:
                return dist
            
            if dist > 100:  # Limit search depth
                continue
            
            for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                nx, ny = cx + dx, cy + dy
                if 0 <= nx < 52 and 0 <= ny < 52:
                    if (nx, ny) not in visited:
                        val = grid_map[ny, nx]
                        # Passable: 0 (empty), 80 (enemy), 150 (player), 200 (brick)
                        if val == 0 or val == 80 or val == 150 or val == 200:
                            visited.add((nx, ny))
                            queue.append((nx, ny, dist + 1))
        
        return 999  # No path found
