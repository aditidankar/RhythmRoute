"""
Configuration for the EDGE model.
"""
from pathlib import Path

# --- Directories ---
ROOT_DIR = Path(__file__).parent
DATA_DIR = ROOT_DIR / "data"
PROCESSED_DATA_DIR = DATA_DIR / "dataset_backups"
RENDER_DIR = ROOT_DIR / "renders"
CHECKPOINTS_DIR = ROOT_DIR / "checkpoints"

# --- Preprocessing ---
# Path to the directory where the baseline features will be saved.
# BASELINE_FEATURES_DIR = "/path/to/your/baseline_features"

# Path to the file containing the mean of the trajectory data.
TRAJ_MEAN_PATH = "jb_mean.pt"

# Path to the file containing the standard deviation of the trajectory data.
TRAJ_STD_PATH = "jb_std.pt"

# Path to the file containing the normalizer for the pose data.
# NORMALIZER_PATH = "/path/to/your/normalizer.pkl"

# --- Model ---
# FEATURE_TYPE = "jukebox" 
# POS_DIM = 3
# ROT_DIM = 24 * 6
# REPR_DIM = POS_DIM + ROT_DIM + 4
# MUSIC_FEATURE_DIM = 35 if FEATURE_TYPE == "baseline" else 4800
# HORIZON_SECONDS = 5
# FPS = 30
# HORIZON = HORIZON_SECONDS * FPS

# --- Training ---
# WANDB_PJ_NAME = "EDGE_Trajectory"
# BATCH_SIZE = 128
# EPOCHS = 200
# FORCE_RELOAD = False
# NO_CACHE = False
# SAVE_INTERVAL = 10
# EMA_INTERVAL = 1
# LEARNING_RATE = 4e-4
# WEIGHT_DECAY = 0.02
# IS_TRAINING = True

# --- Checkpoint ---
# CHECKPOINT = ""
