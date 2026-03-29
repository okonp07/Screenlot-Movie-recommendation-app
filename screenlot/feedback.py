from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime in the app.
    pd = None

from .runtime import FEEDBACK_LOG_PATH, ensure_runtime_dirs


VALID_ACTIONS = {"like", "save", "not_for_me"}


def _empty_frame() -> Any:
    if pd is None:
        return []
    return pd.DataFrame(
        columns=[
            "timestamp",
            "action",
            "movie_id",
            "title",
            "score",
            "reason",
            "favorite_titles",
            "selected_model",
            "rank",
        ]
    )


def append_feedback(
    *,
    action: str,
    movie_id: int,
    title: str,
    score: float,
    reason: str,
    favorite_titles: list[str],
    selected_model: str,
    rank: int,
    feedback_path: Path = FEEDBACK_LOG_PATH,
) -> Path:
    normalized_action = action.strip().lower()
    if normalized_action not in VALID_ACTIONS:
        raise ValueError(f"Unsupported feedback action: {action}")

    ensure_runtime_dirs()
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "action": normalized_action,
        "movie_id": int(movie_id),
        "title": title,
        "score": float(score),
        "reason": reason,
        "favorite_titles": list(favorite_titles),
        "selected_model": selected_model,
        "rank": int(rank),
    }
    feedback_path.parent.mkdir(parents=True, exist_ok=True)
    with feedback_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")
    return feedback_path


def load_feedback_frame(
    feedback_path: Path = FEEDBACK_LOG_PATH,
) -> Any:
    if pd is None or not feedback_path.exists():
        return _empty_frame()
    try:
        frame = pd.read_json(feedback_path, lines=True)
    except ValueError:
        return _empty_frame()
    if frame.empty:
        return _empty_frame()
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce", utc=True)
    return frame


def feedback_summary(feedback_path: Path = FEEDBACK_LOG_PATH) -> dict[str, Any]:
    frame = load_feedback_frame(feedback_path)
    if pd is None or getattr(frame, "empty", True):
        return {
            "total_events": 0,
            "action_counts": {},
            "top_titles": [],
            "latest_timestamp": None,
        }

    action_counts = (
        frame["action"].value_counts().to_dict()
        if "action" in frame.columns
        else {}
    )
    top_titles = []
    if {"title", "action"}.issubset(frame.columns):
        grouped = (
            frame.groupby(["title", "action"], as_index=False)
            .size()
            .sort_values("size", ascending=False)
            .head(10)
        )
        top_titles = grouped.to_dict(orient="records")

    latest_timestamp = None
    if "timestamp" in frame.columns and frame["timestamp"].notna().any():
        latest_timestamp = frame["timestamp"].max().isoformat()

    return {
        "total_events": int(len(frame)),
        "action_counts": {str(key): int(value) for key, value in action_counts.items()},
        "top_titles": top_titles,
        "latest_timestamp": latest_timestamp,
    }

