import random
import torch
import numpy as np
import os

# --- Project Root ---
# Determines the absolute path of the project's root directory
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
DATASETS_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, '../datasets'))

# Default configurations
SIZE = 1250
PATIENCE = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHT_DECAY = 1e-4
SAMPLING_RATE = 250 # Hz

# Experiment configurations
EXPERIMENT_TYPE = 'multitask' # 'single' or 'multitask'
MODEL_NAMES = ["vanet", "lightnet", "heavynet"]
# DEFAULT_SEEDS = [42, 123, 367] # Default seeds if none are provided via command line
DEFAULT_SEEDS = [2, 12, 367] 
LEARNING_RATE = 3e-4
BATCH_SIZE = 128
EPOCHS = 2
NUM_WORKERS = 2
PERSISTENT_WORKERS = False

# Paths
DATASET = 'mit'  # 'iegm' or 'mit'
BASE_PATH = os.path.join(PROJECT_ROOT, "database")
ROOT_OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output")

# Pretext Task Configuration
QRS_COMPLEX_PROPORTION = 0.4

# Multi-task learning is now handled by uncertainty weighting in the model

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False