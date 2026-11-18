"""Configuration and design parameters for HybridTrainer."""

from pathlib import Path

# Paths relative to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
BASELINE_DIR = PROJECT_ROOT / "Baseline"
DATA_DIR = PROJECT_ROOT / "Data"
OUTPUTS_DIR = BASELINE_DIR / "outputs"
SYNTHETIC_DATA_DIR = PROJECT_ROOT / "Synthetic_User_Data"

# Ensure outputs directory exists
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
SYNTHETIC_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Dataset paths
FITREC_JSON_PATH = DATA_DIR / "FitRec" / "endomondoHR_proper.json"
PAMAP2_PROTOCOL_DIR = (
    DATA_DIR / "PAMAP2" / "PAMAP2_Dataset" / "PAMAP2_Dataset" / "Protocol"
)
WESAD_DIR = DATA_DIR / "WESAD" / "WESAD"

# Random seeds for strict separation
SEED_DEMO = 42
SEED_TRAIN = 1000
SEED_VAL = 2000
SEED_TEST = 3000

# Reward weights (proposal formula: R = w_c*C + w_s*S - w_e*|RPE-RPE_tar| - w_f*F)
WEIGHT_COMPLETION = 1.0      # w_c
WEIGHT_SATISFACTION = 0.5    # w_s
WEIGHT_RPE_ERROR = 0.2       # w_e
WEIGHT_FATIGUE = 0.5         # w_f

# Constraint bounds and tolerances
DURATION_TOLERANCE_PCT = 0.10  # 10% duration tolerance
MIN_WORKOUT_DURATION_MIN = 10
MAX_WORKOUT_DURATION_MIN = 120
INTENSITY_MIN = 1.0
INTENSITY_MAX = 10.0

# Ollama settings
OLLAMA_BASE_URL = "http://localhost:11434"
OLLAMA_MODEL = "qwen3.5:0.8b"
OLLAMA_TIMEOUT_SECONDS = 45

# Tabular RL hyperparameters
RL_LEARNING_RATE = 0.1
RL_DISCOUNT_FACTOR = 0.95
RL_EPSILON_START = 1.0
RL_EPSILON_END = 0.05
RL_EPSILON_DECAY = 0.995
RL_DEFAULT_TRAIN_EPISODES = 500
RL_STEPS_PER_EPISODE = 5
RL_QTABLE_PATH = OUTPUTS_DIR / "rl_qtable.json"
