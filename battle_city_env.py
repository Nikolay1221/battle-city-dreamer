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

        # SIMPLE REWARD SYSTEM
        # ========================================
        # Kill: +1, Death: -1, Base Lost: -20
        # That's it. No tricks.
        # ========================================
        
        self.rew_kill = 1.0        # +1 за убийство врага
        self.rew_death = -1.0      # -1 за смерть (симметрично с kill)
        self.rew_base_lost = -20.0 # -20 за потерю базы

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
        # Enemies are at slots 2-7 (Indices in RAM arrays)
        # Slot 0 = P1, Slot 1 = P2, Slots 2-7 = Enemies
        for i in range(2, 8): 
            # Check Status at 0xA0 + i
            # Status >= 0x80 means active tank. < 0x80 means empty or exploding.
            if 0xA0 + i < 0x100:
                st = int(ram[0xA0 + i])
                e_hp = 1 if st >= 128 else 0 
                # Optional: Check Armor at 0xA8 + i for armored tanks?
                # But simple Alive/Dead is enough for rewards.
            else:
                e_hp = 0
                
            e_x  = int(ram[0x90 + i]) if 0x90 + i < 0x100 else 0
            e_y  = int(ram[0x98 + i]) if 0x98 + i < 0x100 else 0
            current_enemies.append({'hp': e_hp, 'x': e_x, 'y': e_y, 'id': i})

        # ========================================
        # OPTIMIZED REWARD LOGIC
        # ========================================
        
        info['reward_events'] = [] # For visualization
        
        # --- REWARD 1: KILLS (+1 per kill) ---
        curr_kill_sum = sum([int(ram[addr]) for addr in self.ADDR_KILLS])
        diff = curr_kill_sum - self.prev_kill_sum
        
        if diff > 0:
            reward += self.rew_kill * diff  # +1.0 per kill
            self.episode_kills += diff
            info['reward_events'].append(f"KILL (+{self.rew_kill * diff})")
            
        self.prev_kill_sum = curr_kill_sum
        
        # --- REWARD 2: DEATH (-1 per death) ---
        curr_lives = int(ram[self.ADDR_LIVES])
        if curr_lives < 10 and self.prev_lives < 10:
             if curr_lives < self.prev_lives:
                reward += self.rew_death  # -2.0 per death
                info['reward_events'].append(f"DIED ({self.rew_death})")
        self.prev_lives = curr_lives
        
        # ========================================
        # GAME STATE CHECKS (no reward, just termination)
        # ========================================
        
        curr_stage = int(ram[self.ADDR_STAGE])
        
        # Victory: killed 20 enemies or stage changed
        if self.enemy_count >= 20 and self.episode_kills >= 20 and not self.level_cleared:
            self.level_cleared = True
            terminated = True 
            info['is_success'] = True
            info['win_reason'] = 'kills_limit'

        if curr_stage != self.prev_stage:
             if not self.level_cleared and self.enemy_count >= 20:
                 self.level_cleared = True
                 info['is_success'] = True
                 info['win_reason'] = 'stage_cleared'
             terminated = True
             
        self.prev_stage = curr_stage
        
        # Game Over: no lives
        if curr_lives == 0:
            terminated = True
            info['game_over_reason'] = 'no_lives'

        # Game Over: base destroyed
        base_status = int(ram[self.ADDR_BASE_STATUS])
        if base_status != 0: self.base_active_latch = True
        if self.base_active_latch and base_status == 0:
             terminated = True
             reward += self.rew_base_lost  # -5.0 for losing base!
             info['game_over_reason'] = 'base_destroyed'
             info['reward_events'].append(f"BASE DESTROYED ({self.rew_base_lost})")
             
        # Track position for game mechanics (idle detection for truncation)
        curr_x, curr_y = int(ram[self.ADDR_X_ARR]), int(ram[self.ADDR_Y_ARR])
        if abs(curr_x - old_x) > 2 or abs(curr_y - old_y) > 2:
             self.visited_sectors.add((curr_x // 16, curr_y // 16))
             self.idle_steps = 0
        else:
             self.idle_steps += 1
             
        # Idle timeout (truncation, no penalty)
        if self.idle_steps > 5000:
             truncated = True
             info['game_over_reason'] = 'idle_timeout'
        
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
                    for i in range(2, 8): # Slots 2-7
                        if ram[0xA0 + i] >= 128: # If alive
                             target_x = spawn_x[(i-2) % 3]
                             self.raw_env.ram[0x90 + i] = target_x # X
                             self.raw_env.ram[0x98 + i] = 0        # Y (Top)
                
                else:
                    # --- PRE-AMBUSH SUPPRESSION ---
                    # Keep enemies alive but trapped/hidden at (0,0)
                    for i in range(2, 8):
                         # Slots 2-5 have HP/Status. 
                         should_suppress = False
                         
                         if ram[0xA0 + i] >= 128: should_suppress = True
                             
                         if should_suppress:
                              self.raw_env.ram[0x90 + i] = 0
                              self.raw_env.ram[0x98 + i] = 0
        
        # 2. STANDARD LIMIT
        apply_standard_limit = True
        if self.exploration_trigger is not None and not self.ambush_triggered:
             apply_standard_limit = False # Suppression already handling it
             
        if apply_standard_limit and self.enemy_count < 6:
             # Kill excess slots
             # Slots 2..7. If we want N enemies max.
             # e.g. Count=2. We allow slots 2,3. Kill 4,5,6,7.
             limit_idx = 2 + self.enemy_count 
             for i in range(limit_idx, 8):
                 if 0xA0 + i < 0x100: 
                      self.raw_env.ram[0x90 + i] = 0
                      self.raw_env.ram[0x98 + i] = 0
                      self.raw_env.ram[0xA0 + i] = 0 # Kill status
             
             # Special Case: If 0 enemies
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
        self.raw_env.ram[0x64] = 0 # TanksOnScreen address is actually not 0xA0, usually managed by engine.
        # But we can kill individual tanks.
        for i in range(2, 8):
            self.raw_env.ram[0xA0 + i] = 0 # Status = 0 (Dead)
            self.raw_env.ram[0x90 + i] = 0 # X=0
            self.raw_env.ram[0x98 + i] = 0 # Y=0
            
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
        
        # 2. Enemies (Slots 2-7) - Battle City Standard
        enemies = []
        
        for i in range(2, 8):  # Slots 2-7
            # Check Status (Alive >= 128)
            status = int(ram[0xA0 + i]) if 0xA0 + i < 0x100 else 0
            if status < 128:
                 continue
                 
            ex = int(ram[0x90 + i])
            ey = int(ram[0x98 + i])
            
            # Skip empty slots (coords at 0,0) - Redundant if Status checked, but safe
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
            
            
            # Format: (x_topleft, y_topleft, slot_id, status_byte, is_visible)
            # Send raw coords (no +8)
            enemies.append((ex, ey, i, status, is_visible))
            
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
