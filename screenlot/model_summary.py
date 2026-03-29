from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .runtime import MODEL_CARD_PATH


MODEL_DISPLAY_NAMES = {
    "popularity": "Popularity Baseline",
    "item_knn": "Item-to-Item Collaborative",
    "matrix_factorization": "Matrix Factorization",
    "hybrid": "ScreenLot Hybrid",
    "screenlot_serving": "Live ScreenLot Engine",
}


def _ranking_signature(metrics: dict[str, Any]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics.get("ndcg@20", 0.0)),
        float(metrics.get("recall@20", 0.0)),
        float(metrics.get("map@20", 0.0)),
        float(metrics.get("ndcg@10", 0.0)),
        float(metrics.get("recall@10", 0.0)),
    )


def humanize_model_name(model_name: str | None) -> str:
    if not model_name:
        return "Unknown"
    return MODEL_DISPLAY_NAMES.get(model_name, model_name.replace("_", " ").title())


def load_model_card(model_card_path: Path = MODEL_CARD_PATH) -> dict[str, Any] | None:
    if not model_card_path.exists():
        return None
    return json.loads(model_card_path.read_text(encoding="utf-8"))


def benchmark_rows(model_card: dict[str, Any] | None, split: str = "test") -> list[dict[str, Any]]:
    if not model_card:
        return []
    key = "validation_benchmarks" if split == "validation" else "test_benchmarks"
    return list(model_card.get(key, []))


def benchmark_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {
        str(row.get("model")): row
        for row in rows
        if row.get("model")
    }


def serving_benchmark_row(model_card: dict[str, Any] | None, split: str = "test") -> dict[str, Any] | None:
    rows = benchmark_rows(model_card, split=split)
    lookup = benchmark_lookup(rows)
    selected_model_name = str((model_card or {}).get("selected_model_name") or "hybrid")
    return lookup.get("screenlot_serving") or lookup.get(selected_model_name)


def leaderboard_rows(
    model_card: dict[str, Any] | None,
    split: str = "test",
    include_serving: bool = False,
) -> list[dict[str, Any]]:
    rows = benchmark_rows(model_card, split=split)
    filtered = []
    for row in rows:
        model_name = str(row.get("model", ""))
        if not include_serving and model_name == "screenlot_serving":
            continue
        filtered.append(row)
    return sorted(filtered, key=_ranking_signature, reverse=True)


def model_snapshot(model_card: dict[str, Any] | None) -> dict[str, Any] | None:
    if not model_card:
        return None

    selected_model_name = str(model_card.get("selected_model_name") or "hybrid")
    serving_row = serving_benchmark_row(model_card, split="test")
    validation_row = serving_benchmark_row(model_card, split="validation")
    if serving_row is None:
        return None

    leaderboard = leaderboard_rows(model_card, split="test", include_serving=False)
    runner_up = None
    for row in leaderboard:
        if str(row.get("model")) != selected_model_name:
            runner_up = row
            break

    hybrid_details = model_card.get("hybrid_details", {})
    training_config = model_card.get("training_config", {})
    return {
        "selected_model_name": selected_model_name,
        "selected_model_label": humanize_model_name(selected_model_name),
        "model_label": hybrid_details.get("model_label", humanize_model_name(selected_model_name)),
        "score_strategy": hybrid_details.get("score_strategy", "unknown"),
        "serving_row": serving_row,
        "validation_row": validation_row,
        "runner_up_row": runner_up,
        "training_config": training_config,
    }
