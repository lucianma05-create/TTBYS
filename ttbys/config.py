"""Central configuration for TTBYS.

All shared paths, model directories and constants live here.
Fill in the model paths below before running any script.
"""
import os
from pathlib import Path

# ------------------------------------------------------------------
# Paths
# ------------------------------------------------------------------
PACKAGE_DIR = Path(__file__).resolve().parent   # ttbys/
REPO_ROOT = PACKAGE_DIR.parent                  # repository root
DATA_PATH = REPO_ROOT / "data" / "ToM_BPD.json"

OUTPUT_DIR = REPO_ROOT / "results"              # generated results (gitignored)
FIRST_THINK_OUTPUT = OUTPUT_DIR / "first_think_result.json"
SECOND_THINK_OUTPUT = OUTPUT_DIR / "second_think_result.json"
THIRD_THINK_OUTPUT = OUTPUT_DIR / "third_think_result.json"

# ------------------------------------------------------------------
# Environment (.env) loading
# ------------------------------------------------------------------
# Loads local configuration from the repository root's .env file (see
# .env.example). Existing environment variables take precedence.
try:
    from dotenv import load_dotenv
    load_dotenv(REPO_ROOT / ".env")
except ImportError:
    pass

# ------------------------------------------------------------------
# GPU selection (this machine is shared; GPU 4 is the designated one).
# Restricts CUDA visibility so local inference never lands on busy GPUs.
# CUDA_DEVICE_ORDER=PCI_BUS_ID makes the device index match nvidia-smi
# numbering (without it, the small RTX A400 ends up at index 4).
# Externally set variables take precedence; set CUDA_VISIBLE_DEVICES
# to "" to use all GPUs.
# ------------------------------------------------------------------
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4")

# ------------------------------------------------------------------
# Model paths (set in .env or fill in directly here)
# ------------------------------------------------------------------
# LLM used by First/Second/Third Think (desire classification, belief
# generation, strategy prediction).
MODEL_DIR = os.environ.get("TTBYS_MODEL_DIR", "")
# LLM used by agent.py for generating persuasive responses.
# Falls back to MODEL_DIR when empty.
GEN_MODEL_DIR = os.environ.get("TTBYS_GEN_MODEL_DIR", "")
# Sentence-embedding model used for experience retrieval.
EMBEDDING_MODEL = os.environ.get(
    "TTBYS_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"
)

# ------------------------------------------------------------------
# Evaluation API (used by second_think_eval.py for GPT-based belief
# scoring). Configure via .env — never commit real keys.
# ------------------------------------------------------------------
EVAL_API_HOST = os.environ.get("EVAL_API_HOST", "")   # e.g. "https://api.openai.com/v1"
EVAL_API_KEY = os.environ.get("EVAL_API_KEY", "")
EVAL_MODEL = os.environ.get("EVAL_MODEL", "gpt-4o-mini")

# ------------------------------------------------------------------
# Dataset / experiment settings
# ------------------------------------------------------------------
# The first MAX_EVAL_DIALOGS dialogs are used for evaluation; the
# remaining dialogs build the experience knowledge base (KB).
MAX_EVAL_DIALOGS = 100
TOP_K = 5

# ------------------------------------------------------------------
# Task constants
# ------------------------------------------------------------------
STRATEGIES = [
    "Expression of views",
    "Logical appeal",
    "Enhancement of views",
    "Task inquiry",
    "Personal story",
    "Affirmation and reassurance",
    "Reflection of feelings",
    "Supplying information",
    "Giving Examples",
]

STRATEGY_TOKENS = ["V", "L", "E", "T", "P", "A", "R", "I", "G"]  # single-letter labels

# Dataset annotations contain harmless capitalization variants (for example,
# "expression of views" and "Giving examples").  Keep one canonical label
# space so retrieval and evaluation do not silently discard those examples.
_STRATEGY_BY_CASEFOLD = {strategy.casefold(): strategy for strategy in STRATEGIES}


def normalize_strategy(strategy):
    """Return a canonical strategy label, or ``None`` when it is unknown."""
    if not isinstance(strategy, str):
        return None
    return _STRATEGY_BY_CASEFOLD.get(strategy.strip().casefold())

DESIRE_ORDER = [-1, 0, 1]                      # unwilling / uncertain / willing
DESIRE_MAP = {-1: "unwilling", 0: "uncertain", 1: "willing"}
DESIRE_TOKEN_MAP = {-1: " A", 0: " B", 1: " C"}

# ------------------------------------------------------------------
# Interactive agent (agent.py)
# ------------------------------------------------------------------
TASK_DESCRIPTION = "Insert your persuasion task or context here."
INITIAL_STRATEGY = "Supplying information"
INITIAL_TEXT = "[Supplying information] Here is some initial persuasive context..."
MAX_AGENT_CHARS = 300   # max length of a generated agent utterance
