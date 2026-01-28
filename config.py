# --- SYSTEM & HARDWARE ---
import multiprocessing
import os

NUM_CPU = 8  # Используем все ядра для сбора опыта

# --- HYBRID CONFIGURATION ---
NUM_VIRTUAL = 2

HEADLESS_MODE = False

# --- TRAINING DURATION ---
TOTAL_TIMESTEPS = 6_000_000

# --- OBSERVATION ---
STACK_SIZE = 4
FRAME_SKIP = 4
ROM_PATH = "BattleCity.nes"
TARGET_STAGE = 0

USE_RECURRENT = False
USE_TRANSFORMER = False

# --- GLOBAL DEFAULT REWARDS ---
REW_KILL = 1.0
REW_DEATH = -1.0
REW_BASE = -100.0
REW_EXPLORE = 0.01
REW_WIN = 20.0
REW_TIME = -0.005 # Штраф за каждый шаг (Голод), чтобы не стоял на месте
REW_DISTANCE = 0.01 # Бонус за сближение с врагом (Магнит)

# --- REWARD PROFILES ---
# Allows different environments to incentivize different behaviors
REWARD_VARIANTS = {
    "DEFAULT": {
        # Balanced: Map control (0.03 * 400 cells = 12 pts) is roughly equal to ~12 Kills.
        # Encourages moving out of base.
        "kill": 1.0, "death": -1.0, "base": -100.0, "explore": 0.05, "win": 20.0, "time": -0.0001, "distance": 0.01
    },
    "EXPLORER": {
        "kill": 0.1, "death": -0.5, "base": -5.0, "explore": 0.05, "win": 10.0, "time": -0.005, "distance": 0.01
    },
    "SURVIVOR": {
        "kill": 1.0, "death": -5.0, "base": -30.0, "explore": 0.0, "win": 50.0, "time": 0.0, "distance": 0.0
    },
    "AGGRESSIVE": {
        "kill": 1.0, "death": -1.0, "base": -20.0, "explore": 0.1, "win": 20.0, "time": -0.01, "distance": 0.02
    }
}

# --- TRAINING MODES ---
# Define presets for environment variants
# Added 'reward_profile' key
ENV_VARIANTS = {
    "STANDARD":        {"enemy_count": 20, "no_shooting": False, "reward_profile": "DEFAULT"},
    "PEACEFUL":        {"enemy_count": 0,  "no_shooting": True,  "reward_profile": "EXPLORER"}, # Focus on moving
    "TARGET_PRACTICE": {"enemy_count": 0,  "no_shooting": False, "reward_profile": "DEFAULT"},
    "VERY_EASY":       {"enemy_count": 2,  "no_shooting": False, "reward_profile": "AGGRESSIVE"}, # Learn to kill!
    "LIGHT_COMBAT":    {"enemy_count": 5,  "no_shooting": False, "reward_profile": "DEFAULT"},
    # --- FULL COMBAT PROFILES (Standard Spawning, Different Goals) ---
    "PROFILE_AGGRESSIVE": {"enemy_count": 20, "no_shooting": False, "reward_profile": "AGGRESSIVE"}, # Kill everything
    "PROFILE_EXPLORER":   {"enemy_count": 20, "no_shooting": False, "reward_profile": "EXPLORER"},   # Focus on map
    "PROFILE_SURVIVOR":   {"enemy_count": 20, "no_shooting": False, "reward_profile": "SURVIVOR"},   # Focus on survival
    
    "VIRTUAL":         "VIRTUAL" 
}

TRAIN_MODE = "HYBRID"

# Hybrid Distribution (Sum must be 16)
# Multi-Objective Training:
HYBRID_CONFIG = {
    "STANDARD": 8,  # #1-4: Killer Squad
}
# --- OBSOLETE FLAGS ---
USE_VIRTUAL = False
USE_HYBRID = False

# --- SAVING ---
CHECKPOINT_FREQ = 2_000
MODEL_DIR = "models"
LOG_DIR = "logs"

# --- PPO HYPERPARAMETERS (КОНФИГ ДЛЯ ЧИСТОГО СТАРТА) ---
# Adaptive Learning Rate Settings
# Для старта с нуля 0.00005 - это слишком мало. Агент не поймет связи действий и наград.
LR_START = 0.0003  # (3e-4) Золотой стандарт Карпатого. Идеально для старта.
LR_MIN = 0.0001  # (1e-4) Если станет слишком сложно, он притормозит.
LR_MAX = 0.0005  # (5e-4) Позволяем разгоняться, если пойдет хорошо.
TARGET_KL = 0.03   

# --- MEMORY TRICK (VRAM SAVER) ---
# N_STEPS = 4096 критически важен для старта с нуля!
# В начале агент делает рандом. Чем больше буфер, тем выше шанс,
# что в рандоме попадется убийство врага, и сеть научится.
N_STEPS       = 2048    
BATCH_SIZE    = 1024 
# Если вылетит ошибка, ставь 128, но 256 лучше.

N_EPOCHS = 2  # Учимся усердно на каждом куске данных.
ENT_COEF = 0.01  # Увеличили в 5 раз (0.01 -> 0.05), чтобы он НЕ успокаивался на 3 фрагах, а искал способ убить всех 20.
# Ему нужно много экспериментировать в начале.
GAMMA = 0.99
CLIP_RANGE = 0.2  # Даем свободу изменений.
ALLOW_NEW_MODEL = True