from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
PACKAGED_DATA_DIR = PROJECT_ROOT / "data" / "app" / "screenlot-demo"
FULL_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "movielens" / "ml-32m"


def _resolve_path(env_name: str, default: Path) -> Path:
    raw_value = os.getenv(env_name, "").strip()
    return Path(raw_value).expanduser() if raw_value else default


def resolve_default_data_dir() -> Path:
    raw_value = os.getenv("SCREENLOT_DATA_DIR", "").strip()
    if raw_value:
        candidate = Path(raw_value).expanduser()
        if candidate.exists():
            return candidate
    if PACKAGED_DATA_DIR.exists():
        return PACKAGED_DATA_DIR
    return FULL_DATA_DIR


DEFAULT_DATA_DIR = resolve_default_data_dir()
ARTIFACTS_DIR = _resolve_path(
    "SCREENLOT_ARTIFACT_DIR",
    PROJECT_ROOT / "artifacts" / "screenlot",
)
MODEL_ARTIFACT_PATH = _resolve_path(
    "SCREENLOT_MODEL_ARTIFACT",
    ARTIFACTS_DIR / "hybrid_recommender.joblib",
)
MODEL_CARD_PATH = _resolve_path(
    "SCREENLOT_MODEL_CARD",
    ARTIFACTS_DIR / "model_card.json",
)
BENCHMARK_JSON_PATH = _resolve_path(
    "SCREENLOT_BENCHMARK_JSON",
    ARTIFACTS_DIR / "benchmark_results.json",
)
BENCHMARK_CSV_PATH = _resolve_path(
    "SCREENLOT_BENCHMARK_CSV",
    ARTIFACTS_DIR / "benchmark_results.csv",
)
STATE_DIR = _resolve_path(
    "SCREENLOT_STATE_DIR",
    PROJECT_ROOT / ".screenlot",
)
FEEDBACK_DIR = STATE_DIR / "feedback"
FEEDBACK_LOG_PATH = _resolve_path(
    "SCREENLOT_FEEDBACK_LOG",
    FEEDBACK_DIR / "events.jsonl",
)


def ensure_runtime_dirs() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FEEDBACK_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
