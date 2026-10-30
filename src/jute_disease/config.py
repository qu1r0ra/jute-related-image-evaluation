from pathlib import Path

# Project paths
ROOT_DIR = Path(__file__).resolve(strict=True).parents[2]

DATA_DIR = ROOT_DIR / "data"
BY_CLASS_DIR = DATA_DIR / "by_class"
ML_SPLIT_DIR = DATA_DIR / "ml_split"
PLANTDOC_DIR = DATA_DIR / "external" / "plantdoc"
PLANTVILLAGE_DIR = DATA_DIR / "external" / "plantvillage"
SOURCE_INVENTORY_DIR = DATA_DIR / "recovered_original_split" / "ml_split"
CONFIGS_DIR = ROOT_DIR / "configs"
BENCHMARK_CONFIG = CONFIGS_DIR / "benchmark.json"
CURATION_RULES_PATH = ROOT_DIR / "configs" / "data" / "curation_rules.json"
IDENTITY_REVIEWS_PATH = ROOT_DIR / "configs" / "data" / "identity_reviews.json"


ARTIFACTS_DIR = ROOT_DIR / "artifacts"
LOGS_DIR = ARTIFACTS_DIR / "logs"
CHECKPOINTS_DIR = ARTIFACTS_DIR / "checkpoints"
ML_MODELS_DIR = ARTIFACTS_DIR / "ml_models"
ML_FEATURES_DIR = ARTIFACTS_DIR / "features"
FREEZE_DIR = ARTIFACTS_DIR / "freeze"
MANIFEST_DIR = FREEZE_DIR / "target"
IMAGE_CACHE_DIR = ARTIFACTS_DIR / "image_cache"

RUNS_DIR = ARTIFACTS_DIR / "runs"

RESULTS_DIR = ROOT_DIR / "results"
SUBMITTED_DIR = RESULTS_DIR / "submitted"
SUBMITTED_LOGS_DIR = SUBMITTED_DIR / "logs"
SUBMITTED_FIGURES_DIR = SUBMITTED_DIR / "figures"
ML_ASSETS_DIR = SUBMITTED_FIGURES_DIR / "ml"
DL_ASSETS_DIR = SUBMITTED_FIGURES_DIR / "dl"

# Data configurations
IMAGE_SIZE = 256
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"]
SPLITS = {"train": 0.7, "val": 0.15, "test": 0.15}

# Execution defaults
BATCH_SIZE = 32
NUM_WORKERS = 4
SEEDS = (42, 1337, 7)
DEFAULT_SEED = SEEDS[0]
DEFAULT_LR = 3e-4
DEFAULT_WEIGHT_DECAY = 0.01

# Task configurations
CLASS_NAMES = (
    "Cercospora Leaf Spot",
    "Dieback",
    "General Damage",
    "Healthy",
    "Mosaic",
    "Stem Rot",
)
NUM_CLASSES = len(CLASS_NAMES)

# Weights & Biases
WANDB_ENTITY = "grade-descent"
WANDB_PROJECT = "jute-disease-detection"

# Visualization defaults
DPI = 300
