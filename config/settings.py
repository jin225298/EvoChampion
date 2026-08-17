"""
Configuration settings for the EvoChampion system.
All configurable values are loaded from environment variables with sensible defaults.

Architecture: Three-layer design
  1. Agent Layer   — decision-making nodes (teacher, searcher, evaluator, etc.)
  2. Tool Layer    — reusable tools exposing standardised interfaces
  3. Data/Server Layer — training backend, model cache, dataset storage
"""

import os
from pathlib import Path
from typing import Literal
import importlib

try:
    load_dotenv = importlib.import_module("dotenv").load_dotenv
except ImportError:
    def load_dotenv(*args, **kwargs):
        return False

_PROJECT_ROOT = Path(__file__).parent.parent.absolute()
_env_file = _PROJECT_ROOT / ".env"
if _env_file.exists():
    load_dotenv(_env_file, override=False)


# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = _PROJECT_ROOT
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# =============================================================================
# Generic Benchmark Configuration
# =============================================================================
# The benchmark dataset used for evaluation (e.g. "openai/gsm8k", "MMLU/elementary_mathematics")
BENCHMARK_DATASET_ID = os.getenv("BENCHMARK_DATASET_ID", "")
BENCHMARK_SUBSET = os.getenv("BENCHMARK_SUBSET", "main")
BENCHMARK_SPLIT = os.getenv("BENCHMARK_SPLIT", "train")
BENCHMARK_EVAL_SPLIT = os.getenv("BENCHMARK_EVAL_SPLIT", "test")

# Field mappings: how to extract question/answer from the dataset rows
BENCHMARK_QUESTION_KEY = os.getenv("BENCHMARK_QUESTION_KEY", "question")
BENCHMARK_ANSWER_KEY = os.getenv("BENCHMARK_ANSWER_KEY", "answer")
BENCHMARK_FORMAT = os.getenv("BENCHMARK_FORMAT")

# Instruction prefix prepended to every inference prompt
INSTRUCTION_PREFIX = os.getenv("INSTRUCTION_PREFIX", "请解答下面的题目。\n")

# Whether to stratify the global probe by module (requires BENCHMARK_FORMAT=gsm8k or module inference)
GLOBAL_PROBE_STRATIFIED_BY_MODULE = os.getenv("GLOBAL_PROBE_STRATIFIED_BY_MODULE", "true").lower() == "true"

# =============================================================================
# Domain Configuration
# =============================================================================
# Domain: "math" (default) or "code"
DOMAIN = os.getenv("DOMAIN", "math").strip().lower()

# Answer verifier type: "auto" (use code execution when DOMAIN=code) or "symbolic"
ANSWER_VERIFIER_TYPE = os.getenv("ANSWER_VERIFIER_TYPE", "auto").strip().lower()


# =============================================================================
# Model Configuration
# =============================================================================
BASE_MODEL_NAME = os.getenv("BASE_MODEL_NAME", "")
CHAMPION_MODEL_PATH = os.getenv("CHAMPION_MODEL_PATH", BASE_MODEL_NAME)
CANDIDATE_MODEL_DIR = os.getenv("CANDIDATE_MODEL_DIR", str(ARTIFACTS_DIR / "candidates"))
# Candidate model retention. 0 keeps the existing behavior (keep every candidate).
# Positive values keep only the current champion plus this many recent rollback/prune candidates.
CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED = int(
    os.getenv("CANDIDATE_RETENTION_KEEP_RECENT_NON_PROMOTED", "0")
)
AGENT_BASE_MODEL_NAME = os.getenv("AGENT_BASE_MODEL_NAME", BASE_MODEL_NAME)

# LLM Agent toggle — when enabled, LLM makes decisions instead of deterministic rules
USE_LLM_AGENTS = os.getenv("USE_LLM_AGENTS", "1").lower() in ("1", "true", "yes")
LLM_AGENT_MAX_NEW_TOKENS = int(os.getenv("LLM_AGENT_MAX_NEW_TOKENS", "2048"))
LLM_AGENT_LEAF_MAX_NEW_TOKENS = int(os.getenv("LLM_AGENT_LEAF_MAX_NEW_TOKENS", "512"))
LLM_JUDGE_MAX_NEW_TOKENS = int(os.getenv("LLM_JUDGE_MAX_NEW_TOKENS", str(LLM_AGENT_MAX_NEW_TOKENS)))
USE_AGENT_INSPECTION = os.getenv("USE_AGENT_INSPECTION", "0").lower() in ("1", "true", "yes")
CONTEXT_COMPACTION_ENABLED = os.getenv("CONTEXT_COMPACTION_ENABLED", "1").lower() in ("1", "true", "yes")
CONTEXT_COMPACTION_TRIGGER_CHARS = int(os.getenv("CONTEXT_COMPACTION_TRIGGER_CHARS", "2000"))
CONTEXT_COMPACTION_PRETTY_MAX_BYTES = int(os.getenv("CONTEXT_COMPACTION_PRETTY_MAX_BYTES", "102400"))

# Training
TRAINING_CONFIG_TEMPLATE = os.getenv(
    "TRAINING_CONFIG_TEMPLATE",
    str(CONFIG_DIR / "templates" / "llama_factory_sft.yaml")
)
LLAMA_FACTORY_ROOT = os.getenv("LLAMA_FACTORY_ROOT", "/path/to/LLaMA-Factory")
TRAIN_FINETUNING_TYPE = os.getenv("TRAIN_FINETUNING_TYPE", "full")
TRAINING_TIMEOUT_SECONDS = int(os.getenv("TRAINING_TIMEOUT_SECONDS", "3600"))
TRAIN_LF_EVAL_ENABLED = os.getenv("TRAIN_LF_EVAL_ENABLED", "true").lower() in ("1", "true", "yes")
TRAIN_EVAL_STRATEGY = os.getenv("TRAIN_EVAL_STRATEGY", "steps")
TRAIN_EVAL_STEPS = int(os.getenv("TRAIN_EVAL_STEPS", "20"))
TRAIN_EVAL_BATCH_SIZE = int(os.getenv("TRAIN_EVAL_BATCH_SIZE", "4"))
TRAIN_LOAD_BEST_MODEL_AT_END = os.getenv("TRAIN_LOAD_BEST_MODEL_AT_END", "true").lower() in ("1", "true", "yes")
TRAIN_SAVE_STEPS = int(os.getenv("TRAIN_SAVE_STEPS", "50"))
TRAIN_SAVE_TOTAL_LIMIT = int(os.getenv("TRAIN_SAVE_TOTAL_LIMIT", "2"))
TRAIN_SAVE_ONLY_MODEL = os.getenv("TRAIN_SAVE_ONLY_MODEL", "false").lower() in ("1", "true", "yes")
TRAIN_RESUME_ENABLED = os.getenv("TRAIN_RESUME_ENABLED", "true").lower() in ("1", "true", "yes")
TRAIN_PACKING = os.getenv("TRAIN_PACKING", "false").lower() in ("1", "true", "yes")
TRAIN_NEAT_PACKING = os.getenv("TRAIN_NEAT_PACKING", "false").lower() in ("1", "true", "yes")
TRAIN_TOKENIZED_PATH = os.getenv("TRAIN_TOKENIZED_PATH", "")
EXPORT_DEVICE = os.getenv("EXPORT_DEVICE", "cpu")
EXPORT_TEMPLATE = os.getenv("EXPORT_TEMPLATE", "")
EXPORT_SIZE_GB = int(os.getenv("EXPORT_SIZE_GB", "5"))

# LoRA config (used when finetuning_type=lora)
LORA_RANK = int(os.getenv("LORA_RANK", "8"))
LORA_ALPHA = int(os.getenv("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.getenv("LORA_DROPOUT", "0.05"))
LORA_TARGET_MODULES = os.getenv("LORA_TARGET_MODULES", "q_proj,v_proj")


# =============================================================================
# Search Configuration
# =============================================================================
SEARCH_DATASET_REPO_LIMIT = int(os.getenv("SEARCH_DATASET_REPO_LIMIT", "100"))
SEARCH_TIMEOUT_SECONDS = int(os.getenv("SEARCH_TIMEOUT_SECONDS", "60"))
# When HF API is unreachable, offline fallback mode: "empty" | "predefined" (predefined uses SEARCH_FALLBACK_DATASETS)
SEARCH_FALLBACK_MODE = os.getenv("SEARCH_FALLBACK_MODE", "empty")
SEARCH_FALLBACK_DATASETS = os.getenv("SEARCH_FALLBACK_DATASETS", "")


# =============================================================================
# Dataset Review Configuration
# =============================================================================
# Wall-clock timeout for one dataset review. 0 disables subprocess timeout and
# runs inline, useful for narrow unit tests.
DATASET_REVIEW_PER_REF_TIMEOUT_SECONDS = float(os.getenv("DATASET_REVIEW_PER_REF_TIMEOUT_SECONDS", "300"))
DATASET_REVIEW_TIMEOUT_KILL_GRACE_SECONDS = float(os.getenv("DATASET_REVIEW_TIMEOUT_KILL_GRACE_SECONDS", "2"))
# Failed/stuck datasets are skipped for this many seconds after enough failures.
DATASET_REVIEW_FAILURE_BACKOFF_SECONDS = float(os.getenv("DATASET_REVIEW_FAILURE_BACKOFF_SECONDS", "1800"))
DATASET_REVIEW_FAILURES_BEFORE_BACKOFF = int(os.getenv("DATASET_REVIEW_FAILURES_BEFORE_BACKOFF", "1"))


# =============================================================================
# Dataset Cleaner Codegen Configuration
# =============================================================================
DATA_CLEANER_PROVIDER = os.getenv("DATA_CLEANER_PROVIDER", "off").strip().lower()
if DATA_CLEANER_PROVIDER not in {"off", "deepseek", "local"}:
    DATA_CLEANER_PROVIDER = "off"
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com").rstrip("/")
DEEPSEEK_CLEANER_MODEL = os.getenv("DEEPSEEK_CLEANER_MODEL", "deepseek-v4-flash")
DATA_CLEANER_MAX_REPAIR_ATTEMPTS = int(os.getenv("DATA_CLEANER_MAX_REPAIR_ATTEMPTS", "2"))
DATA_CLEANER_REQUEST_TIMEOUT_SECONDS = float(os.getenv("DATA_CLEANER_REQUEST_TIMEOUT_SECONDS", "120"))


# =============================================================================
# Filter Configuration
# =============================================================================
FILTER_TARGET_QUESTIONS_PER_ROUND = int(os.getenv("FILTER_TARGET_QUESTIONS_PER_ROUND", "1000"))
_FILTER_SAMPLING_METHOD_RAW = os.getenv("FILTER_SAMPLING_METHOD", "stratified")
if _FILTER_SAMPLING_METHOD_RAW == "random":
    FILTER_SAMPLING_METHOD: Literal["random", "stratified", "difficulty_balanced"] = "random"
elif _FILTER_SAMPLING_METHOD_RAW == "difficulty_balanced":
    FILTER_SAMPLING_METHOD = "difficulty_balanced"
else:
    FILTER_SAMPLING_METHOD = "stratified"
FILTER_DEDUPLICATE = os.getenv("FILTER_DEDUPLICATE", "true").lower() == "true"
FILTER_DROP_MASTERED = os.getenv("FILTER_DROP_MASTERED", "true").lower() == "true"
# Ratio of samples to bias toward the target dynamic-difficulty bucket (easy/medium/hard)
TARGET_BUCKET_RATIO = float(os.getenv("TARGET_BUCKET_RATIO", "0.7"))
SCREENING_ENTRY_MAX_QUESTIONS = int(os.getenv("SCREENING_ENTRY_MAX_QUESTIONS", "1000"))
DATASET_PROFILE_WINDOW_SIZE = int(os.getenv(
    "DATASET_PROFILE_WINDOW_SIZE",
    os.getenv(
        "DATASET_WINDOW_SIZE",
        os.getenv("MAX_QUESTIONS_PER_DATASET", str(SCREENING_ENTRY_MAX_QUESTIONS)),
    ),
))
MAX_PROFILE_WINDOWS_PER_ROUND = int(os.getenv("MAX_PROFILE_WINDOWS_PER_ROUND", "10"))
MAX_PROFILE_ITEMS_PER_ROUND = int(os.getenv(
    "MAX_PROFILE_ITEMS_PER_ROUND",
    str(max(1, MAX_PROFILE_WINDOWS_PER_ROUND) * max(1, DATASET_PROFILE_WINDOW_SIZE)),
))
SCREENING_MIN_NEXT_WINDOW_REMAINDER = int(os.getenv(
    "SCREENING_MIN_NEXT_WINDOW_REMAINDER",
    str(SCREENING_ENTRY_MAX_QUESTIONS),
))


# =============================================================================
# Rollout Configuration
# =============================================================================
ROLLOUT_TIMES = int(os.getenv("ROLLOUT_TIMES", "8"))
ROLLOUT_MAX_CONCURRENT = int(os.getenv("ROLLOUT_MAX_CONCURRENT", "4"))
ROLLOUT_TEMPERATURE = float(os.getenv("ROLLOUT_TEMPERATURE", "0.7"))
ROLLOUT_TOP_P = float(os.getenv("ROLLOUT_TOP_P", "0.95"))
DIFFICULTY_LATE_ROUND = int(os.getenv("DIFFICULTY_LATE_ROUND", "5"))
DIFFICULTY_EARLY_EASY_MIN = int(os.getenv("DIFFICULTY_EARLY_EASY_MIN", "5"))
DIFFICULTY_EARLY_MEDIUM_MIN = int(os.getenv("DIFFICULTY_EARLY_MEDIUM_MIN", "2"))
DIFFICULTY_LATE_EASY_MIN = int(os.getenv("DIFFICULTY_LATE_EASY_MIN", "8"))
DIFFICULTY_LATE_MEDIUM_MIN = int(os.getenv("DIFFICULTY_LATE_MEDIUM_MIN", "4"))
MASTERED_CORRECT_RATIO = float(os.getenv("MASTERED_CORRECT_RATIO", "0.75"))
_MASTERED_CORRECT_THRESHOLD_DEFAULT = max(1, int(ROLLOUT_TIMES * MASTERED_CORRECT_RATIO + 0.999999))
MASTERED_CORRECT_THRESHOLD = int(
    os.getenv("MASTERED_CORRECT_THRESHOLD", str(_MASTERED_CORRECT_THRESHOLD_DEFAULT))
)


# =============================================================================
# Evaluation Configuration
# =============================================================================
EVAL_OLD_SKILL_GATE = float(os.getenv("EVAL_OLD_SKILL_GATE", "0.95"))
EVAL_NEW_SKILL_GATE = float(os.getenv("EVAL_NEW_SKILL_GATE", "0.60"))
EVAL_PROBE_ACC_GATE = float(os.getenv("EVAL_PROBE_ACC_GATE", "0.75"))
HOLDOUT_EVAL_SIZE = int(os.getenv("HOLDOUT_EVAL_SIZE", "50"))
HOLDOUT_EVAL_SEED = int(os.getenv("HOLDOUT_EVAL_SEED", "42"))
HOLDOUT_ERROR_TOLERANCE = float(os.getenv("HOLDOUT_ERROR_TOLERANCE", "0.05"))
FROZEN_DEGRADE_TOLERANCE = float(os.getenv("FROZEN_DEGRADE_TOLERANCE", "0.03"))
REWARD_FORGETTING_PENALTY_WEIGHT = float(os.getenv("REWARD_FORGETTING_PENALTY_WEIGHT", "0.20"))
EARLY_TERMINATION_ROLLBACK_STREAK = int(os.getenv("EARLY_TERMINATION_ROLLBACK_STREAK", "4"))
TEST_ROLLOUT_TIMES = int(os.getenv("TEST_ROLLOUT_TIMES", "3"))
TEST_FORGETTING_TOLERANCE = float(os.getenv("TEST_FORGETTING_TOLERANCE", "0.10"))

# Evaluation batch size limits
EVAL_TEST_MAX_ITEMS = int(os.getenv("EVAL_TEST_MAX_ITEMS", "120"))
EVAL_COTEST_MAX_ITEMS = int(os.getenv("EVAL_COTEST_MAX_ITEMS", "60"))
EVAL_MASTERED_MAX_ITEMS = int(os.getenv("EVAL_MASTERED_MAX_ITEMS", "50"))

# External probe evaluation
USE_EXTERNAL_PROBE_GATE = os.getenv("USE_EXTERNAL_PROBE_GATE", "true").lower() == "true"
EXTERNAL_PROBE_PATH = os.getenv("EXTERNAL_PROBE_PATH", str(DATA_DIR / "external_probe.jsonl"))
EXTERNAL_PROBE_MANIFEST_PATH = os.getenv(
    "EXTERNAL_PROBE_MANIFEST_PATH",
    str(DATA_DIR / "external_probe_manifest.json"),
)
EXTERNAL_PROBE_EVAL_MAX_ITEMS = int(os.getenv("EXTERNAL_PROBE_EVAL_MAX_ITEMS", "300"))

# LLM-as-judge for soft scoring
USE_LLM_AS_JUDGE = os.getenv("USE_LLM_AS_JUDGE", "1").lower() in ("1", "true", "yes")
LLM_JUDGE_MAX_ITEMS = int(os.getenv("LLM_JUDGE_MAX_ITEMS", "0"))
LLM_JUDGE_BATCH_SIZE = int(os.getenv("LLM_JUDGE_BATCH_SIZE", "64"))

# Probe Set
PROBE_SET_SIZE_PER_CLASS = int(os.getenv("PROBE_SET_SIZE_PER_CLASS", "50"))
PROBE_BANK_PATH = os.getenv("PROBE_BANK_PATH", str(CONFIG_DIR / "probe_bank_seed.json"))
GLOBAL_PROBE_SOURCE = os.getenv("GLOBAL_PROBE_SOURCE", "benchmark_holdout")
GLOBAL_PROBE_SIZE = int(os.getenv("GLOBAL_PROBE_SIZE", "120"))
FROZEN_PROBE_SIZE = int(os.getenv("FROZEN_PROBE_SIZE", str(GLOBAL_PROBE_SIZE)))
FROZEN_PROBE_EVAL_METHOD = os.getenv("FROZEN_PROBE_EVAL_METHOD", "local")
GLOBAL_PROBE_SPLIT = os.getenv("GLOBAL_PROBE_SPLIT", BENCHMARK_EVAL_SPLIT)
GLOBAL_PROBE_HOLDOUT_OFFSET = int(os.getenv("GLOBAL_PROBE_HOLDOUT_OFFSET", "0"))
PROBE_EVAL_MAX_ITEMS = int(os.getenv("PROBE_EVAL_MAX_ITEMS", "120"))
PROBE_DIFFICULTY_ROLLOUTS = int(os.getenv("PROBE_DIFFICULTY_ROLLOUTS", str(ROLLOUT_TIMES)))
FIXED_BENCHMARK_SOURCE = os.getenv("FIXED_BENCHMARK_SOURCE", "benchmark_eval")
FIXED_BENCHMARK_SIZE = int(os.getenv("FIXED_BENCHMARK_SIZE", str(FROZEN_PROBE_SIZE)))
FIXED_BENCHMARK_SEED = int(os.getenv("FIXED_BENCHMARK_SEED", "20260511"))

# MathBench/OpenCompass frozen probe. When FROZEN_PROBE_EVAL_METHOD is
# mathbench_opencompass, evaluator uses this external score as probe_acc_frozen.
MATHBENCH_OPENCOMPASS_ROOT = os.getenv("MATHBENCH_OPENCOMPASS_ROOT", "")
MATHBENCH_OPENCOMPASS_PYTHON = os.getenv("MATHBENCH_OPENCOMPASS_PYTHON", "")
MATHBENCH_DATASET = os.getenv("MATHBENCH_DATASET", "mathbench_gen")
MATHBENCH_DATASET_FILTER = os.getenv("MATHBENCH_DATASET_FILTER", "")
MATHBENCH_SUMMARIZER = os.getenv("MATHBENCH_SUMMARIZER", "mathbench_v1_2024")
MATHBENCH_WORK_DIR = os.getenv("MATHBENCH_WORK_DIR", "")
MATHBENCH_MAX_SEQ_LEN = int(os.getenv("MATHBENCH_MAX_SEQ_LEN", "2048"))
MATHBENCH_MAX_OUT_LEN = int(os.getenv("MATHBENCH_MAX_OUT_LEN", "512"))
MATHBENCH_HF_BATCH_SIZE = int(os.getenv("MATHBENCH_HF_BATCH_SIZE", "8"))
MATHBENCH_NUM_GPUS = int(os.getenv("MATHBENCH_NUM_GPUS", "1"))
MATHBENCH_NO_BATCH_PADDING = os.getenv("MATHBENCH_NO_BATCH_PADDING", "0").lower() in ("1", "true", "yes")
MATHBENCH_MODEL_KWARGS = os.getenv("MATHBENCH_MODEL_KWARGS", "device_map='auto'")
MATHBENCH_TOKENIZER_KWARGS = os.getenv(
    "MATHBENCH_TOKENIZER_KWARGS",
    "padding_side='left' truncation='left' use_fast=False",
)
MATHBENCH_EXTRA_ARGS = os.getenv("MATHBENCH_EXTRA_ARGS", "")


# =============================================================================
# Replay Buffer Configuration
# =============================================================================
REPLAY_BUFFER_MAX_SIZE = int(os.getenv("REPLAY_BUFFER_MAX_SIZE", "1000"))
REPLAY_BUFFER_SAMPLE_RATIO = float(os.getenv("REPLAY_BUFFER_SAMPLE_RATIO", "0.10"))
REPLAY_BUFFER_MIN_SAMPLE_SIZE = int(os.getenv("REPLAY_BUFFER_MIN_SAMPLE_SIZE", "300"))


# =============================================================================
# Data Split Configuration
# =============================================================================
TRAIN_SPLIT_RATIO = float(os.getenv("TRAIN_SPLIT_RATIO", "0.56"))
COTEST_SPLIT_RATIO = float(os.getenv("COTEST_SPLIT_RATIO", "0.02"))
TEST_SPLIT_RATIO = float(os.getenv("TEST_SPLIT_RATIO", "0.17"))
PROBE_SPLIT_RATIO = float(os.getenv("PROBE_SPLIT_RATIO", "0.05"))
LF_VAL_SPLIT_RATIO = float(os.getenv("LF_VAL_SPLIT_RATIO", "0.05"))
LF_VAL_MIN_QUESTIONS = int(os.getenv("LF_VAL_MIN_QUESTIONS", "8"))
LF_VAL_MAX_QUESTIONS = int(os.getenv("LF_VAL_MAX_QUESTIONS", "64"))
LF_VAL_MIN_TRAIN_REMAINING = int(os.getenv("LF_VAL_MIN_TRAIN_REMAINING", "30"))
TEST_BUFFER_SAMPLE_RATIO = float(os.getenv("TEST_BUFFER_SAMPLE_RATIO", "0.20"))
TEST_BUFFER_MAX_SIZE = int(os.getenv("TEST_BUFFER_MAX_SIZE", "500"))
PROBE_POOL_MAX_SIZE = int(os.getenv("PROBE_POOL_MAX_SIZE", "200"))
PROBE_POOL_INTAKE_RATIO = float(os.getenv("PROBE_POOL_INTAKE_RATIO", os.getenv("PROBE_POOL_SAMPLE_RATIO", "0.05")))
PROBE_POOL_EVAL_SAMPLE_RATIO = float(os.getenv("PROBE_POOL_EVAL_SAMPLE_RATIO", "0.20"))
DATASET_SHARD_COUNT = int(os.getenv("DATASET_SHARD_COUNT", "10"))
DATASET_SHARD_SIZE = int(os.getenv("DATASET_SHARD_SIZE", "500"))
DATASET_CACHE_MODE = os.getenv("DATASET_CACHE_MODE", "").strip().lower()
DATASET_OFFSET_CACHE_MODE = os.getenv("DATASET_OFFSET_CACHE_MODE", "0").lower() in ("1", "true", "yes", "on")
DATASET_SHARD_SELECTION_POLICY = os.getenv("DATASET_SHARD_SELECTION_POLICY", "single")
TEST_BUFFER_OVERFLOW_TO_TRAIN = os.getenv("TEST_BUFFER_OVERFLOW_TO_TRAIN", "true").lower() == "true"
DATA_HARD_RATIO_MERGE_THRESHOLD = float(os.getenv("DATA_HARD_RATIO_MERGE_THRESHOLD", "0.80"))
DATA_MIN_TRAIN_QUESTIONS_PER_ROUND = int(os.getenv("DATA_MIN_TRAIN_QUESTIONS_PER_ROUND", "64"))
DATA_WINDOW_RETRY_LIMIT = int(os.getenv("DATA_WINDOW_RETRY_LIMIT", "2"))

# Rollout repetitions per item for cold-start full-dataset difficulty classification
ROLLOUT_REPETITIONS = int(os.getenv("ROLLOUT_REPETITIONS", "5"))
# Probe accuracy improvement threshold to trigger re-rollout of deferred items
PROBE_IMPROVEMENT_REFRESH_THRESHOLD = float(os.getenv("PROBE_IMPROVEMENT_REFRESH_THRESHOLD", "0.10"))
DEFERRED_REFRESH_INTERVAL_ROUNDS = int(os.getenv("DEFERRED_REFRESH_INTERVAL_ROUNDS", "3"))


# =============================================================================
# Training Gates
# =============================================================================
MIN_NEW_SKILL_GAIN = float(os.getenv("MIN_NEW_SKILL_GAIN", "0.03"))
PROMOTION_TOTAL_RELATIVE_GAIN = float(os.getenv("PROMOTION_TOTAL_RELATIVE_GAIN", "0.10"))


# =============================================================================
# Evolution Loop
# =============================================================================
MAX_ROUNDS = int(os.getenv("MAX_ROUNDS", "10"))
EVOLVE_TRACE_ID = os.getenv("EVOLVE_TRACE_ID", "")
DATA_WINDOW_SIZE = int(os.getenv("MAX_QUESTIONS_PER_DATASET", os.getenv("DATA_WINDOW_SIZE", str(DATASET_PROFILE_WINDOW_SIZE))))
_LEGACY_MCTS_EXPLORATION_C = os.getenv("MCTS_EXPLORATION_C")
MCTS_EXPLORATION_C_NODE = float(os.getenv("MCTS_EXPLORATION_C_NODE", _LEGACY_MCTS_EXPLORATION_C or "0.1"))
MCTS_EXPLORATION_C_ACTION = float(os.getenv("MCTS_EXPLORATION_C_ACTION", _LEGACY_MCTS_EXPLORATION_C or "0.15"))
MCTS_EXPLORATION_C = MCTS_EXPLORATION_C_NODE
MCTS_TUNER_COLD_START_EDGES = int(os.getenv("MCTS_TUNER_COLD_START_EDGES", "5"))
MCTS_QUERY_COLD_START_EDGES = int(os.getenv("MCTS_QUERY_COLD_START_EDGES", "5"))
MCTS_QUERY_CANDIDATE_TOP_K = int(os.getenv("MCTS_QUERY_CANDIDATE_TOP_K", "3"))
PARAMETER_MASTER_CARD_TOP_NODES = int(os.getenv("PARAMETER_MASTER_CARD_TOP_NODES", "5"))
PARAMETER_MASTER_CARD_TOP_EDGES_PER_NODE = int(os.getenv("PARAMETER_MASTER_CARD_TOP_EDGES_PER_NODE", "1"))
MCTS_QUERY_HIGH_FORGETTING_THRESHOLD = float(os.getenv("MCTS_QUERY_HIGH_FORGETTING_THRESHOLD", "0.15"))
MCTS_QUERY_LOW_PROBE_THRESHOLD = float(os.getenv("MCTS_QUERY_LOW_PROBE_THRESHOLD", str(EVAL_PROBE_ACC_GATE)))
MCTS_QUERY_LOW_NEW_SKILL_THRESHOLD = float(os.getenv("MCTS_QUERY_LOW_NEW_SKILL_THRESHOLD", str(EVAL_NEW_SKILL_GATE)))
MCTS_QUERY_LOW_COTEST_THRESHOLD = float(os.getenv("MCTS_QUERY_LOW_COTEST_THRESHOLD", "0.50"))
MCTS_QUERY_ROLLBACK_STREAK_THRESHOLD = int(os.getenv("MCTS_QUERY_ROLLBACK_STREAK_THRESHOLD", "2"))
MCTS_QUERY_ACTION_BIAS_MAX = float(os.getenv("MCTS_QUERY_ACTION_BIAS_MAX", "0.25"))
MCTS_MUTATION_SCALE_MIN = float(os.getenv("MCTS_MUTATION_SCALE_MIN", "0.5"))
MCTS_MUTATION_SCALE_MAX = float(os.getenv("MCTS_MUTATION_SCALE_MAX", "2.0"))
REWARD_GATE_BONUS = float(os.getenv("REWARD_GATE_BONUS", "0.05"))
REWARD_STOP_BONUS = float(os.getenv("REWARD_STOP_BONUS", "0.15"))
MCTS_ACTION_SPACE_PATH = os.getenv("MCTS_ACTION_SPACE_PATH", str(CONFIG_DIR / "mcts_action_space.json"))
MCTS_CONTINUOUS_ACTION_ENABLED = os.getenv("MCTS_CONTINUOUS_ACTION_ENABLED", "true").lower() in ("1", "true", "yes")
MCTS_CONTINUOUS_CANDIDATES = int(os.getenv("MCTS_CONTINUOUS_CANDIDATES", "32"))
MCTS_CONTINUOUS_BANDWIDTH = float(os.getenv("MCTS_CONTINUOUS_BANDWIDTH", "0.35"))
MCTS_CONTINUOUS_EXPLORATION_BETA = float(os.getenv("MCTS_CONTINUOUS_EXPLORATION_BETA", "0.15"))
MCTS_CONTINUOUS_EPOCH_COST_WEIGHT = float(os.getenv("MCTS_CONTINUOUS_EPOCH_COST_WEIGHT", "0.02"))
BENCHMARK_HOLDOUT_OFFSET = int(os.getenv(
    "BENCHMARK_HOLDOUT_OFFSET",
    str(GLOBAL_PROBE_HOLDOUT_OFFSET),
))


# =============================================================================
# Classification Configuration
# =============================================================================
USE_GLINER = os.getenv("USE_GLINER", "1").lower() in ("1", "true", "yes")
GLINER_MODEL_NAME = os.getenv("GLINER_MODEL_NAME", "urchade/gliner_medium-v2.1")
# Target labels for classification — empty by default, filled by prompt_designer dynamically
CLASSIFIER_TARGET_LABELS_RAW = os.getenv("CLASSIFIER_TARGET_LABELS", "")
CLASSIFIER_TARGET_LABELS: list[str] = (
    [label.strip() for label in CLASSIFIER_TARGET_LABELS_RAW.split(",") if label.strip()]
    if CLASSIFIER_TARGET_LABELS_RAW
    else []
)
CLASSIFIER_MIN_CONFIDENCE = float(os.getenv("CLASSIFIER_MIN_CONFIDENCE", "0.7"))


# =============================================================================
# Harness Configuration
# =============================================================================
HARNESS_MAX_RETRIES = int(os.getenv("HARNESS_MAX_RETRIES", "3"))
HARNESS_TIMEOUT_SECONDS = int(os.getenv("HARNESS_TIMEOUT_SECONDS", "3600"))
ENFORCE_HARNESS_CONTRACTS = os.getenv("ENFORCE_HARNESS_CONTRACTS", "true").lower() == "true"


# =============================================================================
# Logging Configuration
# =============================================================================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


# =============================================================================
# Utility Functions
# =============================================================================
def load_harness_dataset_repo_limit() -> int:
    """Load the maximum number of dataset repos to retrieve from search."""
    return SEARCH_DATASET_REPO_LIMIT


def load_filter_round_quota() -> int:
    """Load the target number of questions per training round."""
    return FILTER_TARGET_QUESTIONS_PER_ROUND


def get_session_dir(trace_id: str) -> Path:
    """Get or create the session-specific directory."""
    session_dir = ARTIFACTS_DIR / f"session_{trace_id}"
    session_dir.mkdir(parents=True, exist_ok=True)
    return session_dir


def get_classifier_labels() -> list[str]:
    """Get current classifier target labels, with GSM8K-compatible defaults if empty."""
    runtime_labels = [
        label.strip()
        for label in os.getenv("CLASSIFIER_TARGET_LABELS", "").split(",")
        if label.strip()
    ]
    if runtime_labels:
        return runtime_labels
    if CLASSIFIER_TARGET_LABELS:
        return list(CLASSIFIER_TARGET_LABELS)
    # GSM8K-compatible default labels (kept for backward compatibility)
    return [
        "ratio_rate",
        "money_cost",
        "counting_combinatorics",
        "geometry_measurement",
        "fraction_percent",
        "work_time",
        "age_numbers",
        "arithmetic_misc",
        "unknown",
    ]




# =============================================================================
# Inference Configuration (performance optimization)
# =============================================================================
INFERENCE_MAX_NEW_TOKENS = int(os.getenv("INFERENCE_MAX_NEW_TOKENS", "96"))
ROLLOUT_MAX_NEW_TOKENS = int(os.getenv("ROLLOUT_MAX_NEW_TOKENS", "1024"))
EVAL_MAX_NEW_TOKENS = int(os.getenv("EVAL_MAX_NEW_TOKENS", "1024"))
INFERENCE_BATCH_SIZE = int(os.getenv("INFERENCE_BATCH_SIZE", "8"))
INFERENCE_TRACE_ENABLED = os.getenv("INFERENCE_TRACE_ENABLED", "1").lower() in ("1", "true", "yes")
INFERENCE_TRACE_MAX_TEXT_CHARS = int(os.getenv("INFERENCE_TRACE_MAX_TEXT_CHARS", "4000"))
USE_VLLM = os.getenv("USE_VLLM", "1").lower() in ("1", "true", "yes")
USE_VLLM_FOR_LOCAL_CHECKPOINTS = os.getenv("USE_VLLM_FOR_LOCAL_CHECKPOINTS", "1").lower() in ("1", "true", "yes")
# Legacy rollback switch for the old isolated local-checkpoint helper. The
# default path is Ray named actors so concurrent rollout workers share one
# candidate vLLM engine/KV cache.
USE_VLLM_LOCAL_CHECKPOINT_SUBPROCESS = os.getenv("USE_VLLM_LOCAL_CHECKPOINT_SUBPROCESS", "0").lower() in ("1", "true", "yes")
VLLM_LOCAL_CHECKPOINT_SUBPROCESS_TIMEOUT_SECONDS = int(os.getenv("VLLM_LOCAL_CHECKPOINT_SUBPROCESS_TIMEOUT_SECONDS", "3600"))
VLLM_MAX_CACHED_ENGINES = int(os.getenv("VLLM_MAX_CACHED_ENGINES", "1"))
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.82"))
VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION = float(os.getenv("VLLM_PER_ENGINE_GPU_MEMORY_UTILIZATION", "0"))
VLLM_ENFORCE_EAGER = os.getenv("VLLM_ENFORCE_EAGER", "0").lower() in ("1", "true", "yes")
VLLM_ENABLE_PREFIX_CACHING = os.getenv("VLLM_ENABLE_PREFIX_CACHING", "1").lower() in ("1", "true", "yes")
VLLM_ENABLE_CHUNKED_PREFILL = os.getenv("VLLM_ENABLE_CHUNKED_PREFILL", "1").lower() in ("1", "true", "yes")
VLLM_MAX_NUM_SEQS = int(os.getenv("VLLM_MAX_NUM_SEQS", "64"))
VLLM_MAX_NUM_BATCHED_TOKENS = int(os.getenv("VLLM_MAX_NUM_BATCHED_TOKENS", "8192"))
VLLM_RAY_INFLIGHT_PER_CALL = int(os.getenv("VLLM_RAY_INFLIGHT_PER_CALL", "8"))
VLLM_RAY_GENERATE_TIMEOUT_SECONDS = int(os.getenv("VLLM_RAY_GENERATE_TIMEOUT_SECONDS", "1800"))
VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS = float(os.getenv("VLLM_RAY_GENERATE_TIMEOUT_PER_PROMPT_SECONDS", "3"))
DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR = os.getenv(
    "DISABLE_HF_FALLBACK_AFTER_VLLM_MEMORY_ERROR",
    "1",
).lower() in ("1", "true", "yes")
VLLM_RAY_ACTOR_NUM_GPUS = float(os.getenv("VLLM_RAY_ACTOR_NUM_GPUS", "1"))
VLLM_RAY_ACTOR_MAX_RESTARTS = int(os.getenv("VLLM_RAY_ACTOR_MAX_RESTARTS", "1"))
VLLM_RAY_ACTOR_MAX_TASK_RETRIES = int(os.getenv("VLLM_RAY_ACTOR_MAX_TASK_RETRIES", "0"))
VLLM_RAY_ACTOR_MAX_CONCURRENCY = int(os.getenv("VLLM_RAY_ACTOR_MAX_CONCURRENCY", "8"))
RAY_ADDRESS = os.getenv("RAY_ADDRESS", "local")
RAY_NAMESPACE = os.getenv("RAY_NAMESPACE", "evochampion")
