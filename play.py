import gymnasium as gym
import pygame
import numpy as np
import sys
import os
import pandas as pd
from collections import deque
import time

def has_line_of_sight(p1, p2, grid):
    x0, y0 = p1
    x1, y1 = p2
    
    steps = max(abs(x1-x0), abs(y1-y0))
    if steps == 0: return True
    
    # Check all points along the line
    for i in range(1, steps + 1): 
        t = i / steps
        x = int(x0 + (x1 - x0) * t + 0.5)
        y = int(y0 + (y1 - y0) * t + 0.5)
        
        if 0 <= y < 52 and 0 <= x < 52:
             val = grid[y, x]
             # Passable: 0(Empty), 80(Enemy), 150(Player), 200(Brick)
             if not (val == 0 or val == 80 or val == 150 or val == 200):
                 return False
    return True

def get_smoothed_path(path, grid_map):
    if not path or len(path) < 3: return path
    
    smoothed = [path[0]]
    idx = 0
    # Greedy simplification
    while idx < len(path) - 1:
        found = False
        # Look for furthest visible node
        for i in range(len(path)-1, idx, -1):
            if i == idx + 1: break # Neighbor always visible
            
            if has_line_of_sight(path[idx], path[i], grid_map):
                smoothed.append(path[i])
                idx = i
                found = True
                break
        
        if not found:
            smoothed.append(path[idx+1])
            idx += 1
            
    return smoothed

# Ensure we can import battle_city_env
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import config
from battle_city_env import BattleCityEnv

def main():
    print("Initializing Battle City Visualizer...")
    
    pygame.init()
    pygame.font.init()
    font_mono = pygame.font.SysFont("Courier New", 14, bold=True)
    font_ui = pygame.font.SysFont("Arial", 16, bold=True)
    
    # Init Env with Modes
    modes = list(config.ENV_VARIANTS.keys())
    # Filter out VIRTUAL for play.py
    modes = [m for m in modes if m != "VIRTUAL"]
    current_mode_idx = 0
    
    def make_env_for_mode(mode_name):
        variant = config.ENV_VARIANTS[mode_name]
        reward_profile = variant.get("reward_profile", "DEFAULT")
        reward_config = config.REWARD_VARIANTS.get(reward_profile, None)
        
        print(f"Switching to Mode: {mode_name}")
        print(f" - Enemies: {variant.get('enemy_count')}")
        print(f" - Profile: {reward_profile}")
        
        return BattleCityEnv(
            render_mode='rgb_array', 
            use_vision=False, 
            enemy_count=variant.get("enemy_count", 20),
            no_shooting=variant.get("no_shooting", False),
            reward_config=reward_config,
            exploration_trigger=variant.get("exploration_trigger", None)
        )

    # Initial Start (Default to PROFILE_EXPLORER if available, else STANDARD)
    start_mode = "PROFILE_EXPLORER" if "PROFILE_EXPLORER" in modes else "STANDARD"
    current_mode_idx = modes.index(start_mode)
    env = make_env_for_mode(start_mode)
    obs, info = env.reset()
    
    frame = env.raw_env.screen.copy()
    h, w, c = frame.shape
    
    SCALE = 3
    SIDE_PANEL = 500
    screen = pygame.display.set_mode((w * SCALE + SIDE_PANEL, h * SCALE))
    pygame.display.set_caption("Battle City - Clean Tactical View")
    
    clock = pygame.time.Clock()
    running = True
    
    # Path caching
    cached_path = []
    path_recalc_counter = 0
    PATH_RECALC_INTERVAL = 1  # Recalculate EVERY frame

    msg_log = deque(maxlen=50) # Increased history
    reward_log = deque(maxlen=20) # Лог наград
    popups = [] # List of [text, timer, color]
    panel_scroll = 0 # Прокрутка всего бокового меню (в пикселях)
    total_episode_reward = 0.0 # Накопленная награда за эпизод

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.MOUSEWHEEL:
                panel_scroll -= event.y * 30 # Прокрутка всего меню (30px за тик)
                if panel_scroll < 0: panel_scroll = 0
                if panel_scroll > 600: panel_scroll = 600  # Максимум прокрутки
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key == pygame.K_m:
                    # Switch Mode
                    current_mode_idx = (current_mode_idx + 1) % len(modes)
                    new_mode = modes[current_mode_idx]
                    env.close()
                    env = make_env_for_mode(new_mode)
                    obs, info = env.reset()
                    msg_log.append(f"SWITCHED MODE: {new_mode}")
                elif event.key == pygame.K_k: 
                    print("CHEAT: Clearing Enemies!")
                    # env.cheat_clear_enemies() # Not implemented in base env yet
                    pass 
                elif event.key == pygame.K_j:
                    pass
        
        # Input
        keys = pygame.key.get_pressed()
        action = 0
        
        # Menu
        raw_action = 0
        if keys[pygame.K_RETURN]: raw_action |= 0x08
        if keys[pygame.K_TAB]:    raw_action |= 0x04
        
        if raw_action > 0:
            obs, reward, done, info = env.raw_env.step(raw_action)
            terminated, truncated = done, False
        else:
            up, down = keys[pygame.K_UP], keys[pygame.K_DOWN]
            left, right = keys[pygame.K_LEFT], keys[pygame.K_RIGHT]
            fire = keys[pygame.K_z]
            
            if up:
                if fire: action = 6
                else:    action = 1
            elif down:
                if fire: action = 7
                else:    action = 2
            elif left:
                if fire: action = 8
                else:    action = 3
            elif right:
                if fire: action = 9
                else:    action = 4
            elif fire:
                action = 5
                
            obs, reward, terminated, truncated, info = env.step(action)
            
            # Накапливаем награду и логируем события
            total_episode_reward += reward
            events = info.get('reward_events', [])
            for e in events:
                # Определяем цвет по типу награды
                if 'KILL' in e:
                    reward_log.append(('🎯 ' + e, (0, 255, 100)))  # Зелёный
                elif 'DIED' in e:
                    reward_log.append(('💀 ' + e, (255, 80, 80)))  # Красный
                elif 'BASE' in e:
                    reward_log.append(('🦅 ' + e, (255, 0, 0)))  # Ярко-красный

        if terminated or truncated:
            msg_log.append(f"EPISODE END (Score: {info.get('score', 0):.1f}, Reward: {total_episode_reward:.2f})")
            reward_log.append((f"=== EPISODE: {total_episode_reward:.2f} ===", (255, 255, 0)))
            total_episode_reward = 0.0  # Сброс
            env.reset()

        # Render Game
        frame = env.raw_env.screen
        surf = pygame.surfarray.make_surface(frame.swapaxes(0,1))
        surf = pygame.transform.scale(surf, (w * SCALE, h * SCALE))
        screen.blit(surf, (0, 0))
        
        # --- DRAW BOUNDING BOXES (DEBUG) ---
        ram = env.raw_env.ram
        
        # Draw Player (from info or RAM)
        # ALWAYS get raw coords for BFS pathfinding
        px, py = ram[0x90], ram[0x98]
        
        if 'player_cv' in info:
            px_c, py_c = info['player_cv']
            # Correction: info adds +8, so we subtract 8 to get raw visual center
            cx, cy = px_c - 8, py_c - 8 
        else:
            cx, cy = px, py # Raw coord seems to be visual center
        
        # SAVE player center (before cx, cy get overwritten by enemy loop!)
        player_cx, player_cy = int(cx), int(cy)
        
        # Draw Player (Blue Dot)
        pygame.draw.circle(screen, (0, 0, 255), (player_cx*SCALE, player_cy*SCALE), 4)
        lbl_player = font_mono.render("PLAYER", True, (0, 255, 255))
        screen.blit(lbl_player, (player_cx*SCALE - 15, player_cy*SCALE - 25))
        
        # Draw Enemies (from info with LoS)
        if 'enemy_positions' in info:
            for enemy_data in info['enemy_positions']:
                ex, ey, slot_id, st_val, is_visible = enemy_data
                # Draw Point (Red Dot at Center)
                # Correction: Visual center seems to match raw coord
                cx, cy = ex, ey
                pygame.draw.circle(screen, (255, 0, 0), (int(cx*SCALE), int(cy*SCALE)), 4)
                
                # Draw Label
                status = "LoS" if is_visible else "X"
                if isinstance(st_val, (int, float)): # Check if it is status
                    lbl = font_mono.render(f"#{slot_id} ST:{int(st_val):02X}", True, (255, 255, 0))
                else:
                    lbl = font_mono.render(f"#{slot_id} {status}", True, (255, 255, 0))
                
                screen.blit(lbl, (cx*SCALE - 10, cy*SCALE - 25))
        else:
            # Fallback: Direct RAM (Slots 2-7)
            for i in range(2, 8):
                if ram[0xA0 + i] >= 128:
                    ex, ey = ram[0x90 + i], ram[0x98 + i]
                    # Correction: Visual center seems to match raw coord (or close enough)
                    cx, cy = ex, ey
                    pygame.draw.circle(screen, (255, 0, 0), (cx*SCALE, cy*SCALE), 4)
                    lbl = font_mono.render(f"#{i}", True, (255, 255, 0))
                    screen.blit(lbl, (cx*SCALE - 10, cy*SCALE - 25))
        
        # Render Side Panel Background
        pygame.draw.rect(screen, (30, 30, 30), (w*SCALE, 0, SIDE_PANEL, h*SCALE))
        x_start = w*SCALE + 20
        y_pos = 20 - panel_scroll  # Применяем прокрутку
        
        # --- TACTICAL MAP ---
        screen.blit(font_ui.render("TACTICAL MAP (26x26)", True, (255, 255, 255)), (x_start, y_pos))
        y_pos += 30
        
        tactical_rgb = env.get_tactical_rgb()
        cell_size = 12
        map_surf = pygame.surfarray.make_surface(tactical_rgb.swapaxes(0,1))
        map_surf = pygame.transform.scale(map_surf, (26 * cell_size, 26 * cell_size))
        
        screen.blit(map_surf, (x_start, y_pos))
        
        # Grid Lines
        for i in range(27):
            pygame.draw.line(screen, (50, 50, 50),
                             (x_start, y_pos + i * cell_size),
                             (x_start + 26 * cell_size, y_pos + i * cell_size))
            pygame.draw.line(screen, (50, 50, 50),
                             (x_start + i * cell_size, y_pos),
                             (x_start + i * cell_size, y_pos + 26 * cell_size))

        y_pos += 26 * cell_size + 20
        
        # --- REWARD POPUPS MANAGER ---
        events = info.get('reward_events', [])
        if events:
            for e in events:
                msg_log.append(f">>> {e} <<<")
                # Add to visual popups: [Text, Timer(frames), Color]
                col = (255, 255, 0) # Default Yellow
                if "DEFENDER" in e: col = (100, 255, 100) # Green for Defense
                if "MILESTONE" in e: col = (255, 100, 255) # Purple for Milestones
                popups.append([e, 120, col]) # 2 Seconds (60fps * 2)

        # Draw Popups (Kill Feed Style - Top Left, Stacking Down)
        # Filter dead ones
        popups = [[txt, t-1, c] for txt, t, c in popups if t > 0]
        
        # Show max 5 recent popups
        visible_popups = popups[-5:] 
        
        for i, (txt, t, c) in enumerate(visible_popups):
            # i=0 is the oldest valid, i=4 is newest
            # Let's stack them downwards: Oldest at top? Or Newest at top?
            # Standard kill feed: Newest at bottom usually, or Newest at top.
            # Let's put Newest at Bottom of the list (which is index -1).
            
            label = font_ui.render(txt, True, c)
            # Text Outline
            outline = font_ui.render(txt, True, (0,0,0))
            
            # Position: Top Left
            p_x = 10
            p_y = 10 + (i * 25) # Stack downwards
            
            # Simple background box for readability
            bg_rect = pygame.Rect(p_x - 2, p_y - 2, label.get_width() + 4, label.get_height() + 4)
            pygame.draw.rect(screen, (0, 0, 0, 180), bg_rect) # Semi-transparent black? Pygame rect doesn't support alpha directly without surface. Just black for now.
            
            screen.blit(label, (p_x, p_y))

        
        # --- RAM INSPECTOR ---
        ram = env.raw_env.ram
        
        screen.blit(font_ui.render("RAM INSPECTOR:", True, (255, 255, 0)), (x_start, y_pos))
        y_pos += 25
        
        # DEBUG: Enemies Control
        enemies_left = ram[0x80]
        enemies_on_screen = ram[0xA0]
        screen.blit(font_mono.render(f"ENEMIES LEFT (0x80): {enemies_left}", True, (255, 100, 255)), (x_start, y_pos))
        y_pos += 20
        screen.blit(font_mono.render(f"ON SCREEN (0xA0):    {enemies_on_screen}", True, (255, 100, 255)), (x_start, y_pos))
        y_pos += 20

        # 1. Player
        px, py = ram[0x90], ram[0x98]
        p_dir = ram[0x99] # Direction
        screen.blit(font_mono.render(f"PLAYER: XY({px:03d},{py:03d}) DIR({p_dir})", True, (200, 255, 200)), (x_start, y_pos))
        y_pos += 20
        
        # 2. Base
        # 2. Base
        base_status = ram[0x68]
        latch = getattr(env, 'base_active_latch', False)
        
        if latch:
             status_txt = f"LATCHED (ACTIVE)"
             status_col = (255, 255, 0)
        else:
             status_txt = "WAITING..."
             status_col = (100, 100, 100)
             
        if base_status == 0 and latch:
             base_txt = f"BASE: DESTROYED (0x{base_status:02X})"
             base_col = (255, 0, 0)
        elif base_status != 0:
             base_txt = f"BASE: ALIVE (0x{base_status:02X})"
             base_col = (0, 255, 0)
        else:
             base_txt = f"BASE: INIT (0x{base_status:02X})"
             base_col = (100, 100, 255)

        screen.blit(font_mono.render(base_txt, True, base_col), (x_start, y_pos))
        y_pos += 15
        screen.blit(font_mono.render(status_txt, True, status_col), (x_start, y_pos))
        y_pos += 20
        
        # 3. Enemies
        screen.blit(font_mono.render("ENEMIES (HP | X, Y):", True, (200, 200, 200)), (x_start, y_pos))
        y_pos += 20
        
        active_enemies = 0
        # Use slots 2-7 (Enemies)
        for i in range(2, 8): 
            st = ram[0xA0 + i] # Status (>=128 Alive)
            ex, ey = ram[0x90 + i], ram[0x98 + i]
            
            is_alive = (st >= 128)
            
            if is_alive:
                active_enemies += 1
                txt = f"E#{i-1} [{ex:03},{ey:03}] ST:{st:02X}"
                col = (255, 100, 100)
            else:
                txt = f"E#{i-1} DEAD/EMPTY"
                col = (80, 80, 80)
                
            screen.blit(font_mono.render(txt, True, col), (x_start, y_pos))
            y_pos += 18
            
        y_pos += 10
        
        # 4. Magnet (Distance) Debug & PATHFINDING
        # Calculate min distance exactly as in env
        import numpy as np
        min_dist = 999.0
        nearest_idx = -1
        target_ex, target_ey = 0, 0
        
        dist_debug_lines = []
        for i in range(2, 8):  # Slots 2-7
            ex, ey = float(ram[0x90 + i]), float(ram[0x98 + i])
            st = int(ram[0xA0 + i])
            
            # Alive check: Status >= 128 AND Valid Coords
            if st < 128 or (int(ex) == 0 and int(ey) == 0):
                dist_debug_lines.append(f"#{i-1}: --- (ST:{st:02X})")
                continue
            
            # Base Defense Logic: Target enemy closest to BASE (Eagle)
            # Eagle is roughly at (12*8, 24*8) = (96, 192). Center ~ (104, 200).
            bx, by = 104.0, 200.0
            
            # Distance to Base
            d_base = np.sqrt((ex - bx)**2 + (ey - by)**2)
            
            dist_debug_lines.append(f"#{i-1}: {d_base:.1f}")
            
            if d_base < min_dist:
                min_dist = d_base
                nearest_idx = i
                target_ex, target_ey = ex, ey
        
        if nearest_idx != -1:
             # Get explicit values for display
             t_ex, t_ey = float(ram[0x90 + nearest_idx]), float(ram[0x98 + nearest_idx])
             mag_txt = f"MAGNET: {min_dist:.1f} | P({px},{py})->E#{nearest_idx}({int(t_ex)},{int(t_ey)})"
             mag_col = (100, 255, 255) # Cyan
             
             # --- DIRECT LINE VISUALIZATION (Fast!) ---
             # Draw lines from PLAYER to ALL alive enemies
             player_screen_x = player_cx * SCALE
             player_screen_y = player_cy * SCALE
             
             # Collect all enemies with distances
             enemy_lines = []
             for i in range(2, 8):  # All enemy slots
                 st = int(ram[0xA0 + i])
                 ex_i, ey_i = int(ram[0x90 + i]), int(ram[0x98 + i])
                 
                 # Skip dead/empty
                 if st < 128 or (ex_i == 0 and ey_i == 0):
                     continue
                 
                 # Calculate distance (line length)
                 dx = ex_i - player_cx
                 dy = ey_i - player_cy
                 dist = (dx*dx + dy*dy) ** 0.5
                 
                 enemy_lines.append((i, ex_i, ey_i, dist))
             
             # Find closest enemy (shortest line)
             closest_idx = -1
             if enemy_lines:
                 closest_idx = min(enemy_lines, key=lambda x: x[3])[0]
             
             # Draw all lines
             for (idx, ex_i, ey_i, dist) in enemy_lines:
                 enemy_screen_x = ex_i * SCALE
                 enemy_screen_y = ey_i * SCALE
                 
                 # Closest = Green (priority), Others = Gray
                 if idx == closest_idx:
                     line_color = (0, 255, 0)  # Green
                     line_width = 3
                 else:
                     line_color = (80, 80, 80)  # Gray
                     line_width = 1
                 
                 pygame.draw.line(screen, line_color, 
                                  (player_screen_x, player_screen_y),
                                  (enemy_screen_x, enemy_screen_y), 
                                  line_width)
                 
                 # Show distance label near midpoint
                 mid_x = (player_screen_x + enemy_screen_x) // 2
                 mid_y = (player_screen_y + enemy_screen_y) // 2
                 dist_lbl = font_mono.render(f"{dist:.0f}", True, line_color)
                 screen.blit(dist_lbl, (mid_x, mid_y))

        else:
             mag_txt = "MAGNET: NO TARGET"
             mag_col = (100, 100, 100)
             
        # Show all distances
        for dbg_line in dist_debug_lines:
            screen.blit(font_mono.render(dbg_line, True, (150, 150, 150)), (x_start, y_pos))
            y_pos += 15
        y_pos += 5
             
        screen.blit(font_mono.render(mag_txt, True, mag_col), (x_start, y_pos))
        y_pos += 25 # Fix overlap
        
        # --- STATS ---
        lives = ram[0x51]
        kills = sum([ram[0x73+i] for i in range(4)])
        stage = ram[0x85]
        
        screen.blit(font_ui.render(f"LIVES: {lives} | KILLS: {kills} | STAGE: {stage} | MODE: {modes[current_mode_idx]}", True, (255, 255, 255)), (x_start, y_pos))
        y_pos += 30
        
        # --- EXPLORATION PROGRESS ---
        explore_pct = info.get('exploration_pct', 0.0)
        trigger_pct = info.get('trigger_pct', 0.0)
        
        # Draw Bar
        bar_w = 200
        bar_h = 15
        pygame.draw.rect(screen, (50, 50, 50), (x_start, y_pos, bar_w, bar_h))
        pygame.draw.rect(screen, (0, 100, 255), (x_start, y_pos, int(bar_w * explore_pct), bar_h))
        
        # Draw Trigger Marker
        if trigger_pct > 0:
             trig_x = x_start + int(bar_w * trigger_pct)
             pygame.draw.line(screen, (255, 0, 0), (trig_x, y_pos - 5), (trig_x, y_pos + bar_h + 5), 2)
             msg = f"EXPLORE: {explore_pct*100:.1f}% (Trig: {trigger_pct*100:.0f}%)"
        else:
             msg = f"EXPLORE: {explore_pct*100:.1f}%"
             
        screen.blit(font_mono.render(msg, True, (200, 200, 255)), (x_start + bar_w + 10, y_pos))
        y_pos += 25
        
        # --- REWARDS LOG (Прокручиваемый) ---
        steps = env.steps_in_episode
        time_reward = steps * 0.001
        screen.blit(font_ui.render(f"REWARDS: Total={total_episode_reward:+.2f} | ⏱️ TIME={time_reward:+.2f}", True, (255, 215, 0)), (x_start, y_pos))
        y_pos += 20
        
        # Показываем последние N наград
        reward_list = list(reward_log)
        visible_rewards = reward_list[-8:]  # Последние 8 записей
        
        for (txt, col) in visible_rewards:
            screen.blit(font_mono.render(txt, True, col), (x_start, y_pos))
            y_pos += 16
        
        y_pos += 10
        
        # --- EVENT LOG ---
        screen.blit(font_ui.render("EVENT LOG", True, (200, 200, 0)), (x_start, y_pos))
        y_pos += 20

        # Показываем последние 10 сообщений
        visible_logs = list(msg_log)[-10:]
        for msg in visible_logs:
            screen.blit(font_mono.render(msg, True, (150, 150, 150)), (x_start, y_pos))
            y_pos += 20

        pygame.display.flip()
        clock.tick(30)

    env.close()
    pygame.quit()

if __name__ == "__main__":
    main()