from __future__ import annotations

from collections.abc import Mapping, Sequence
import math


def recall_at_k(recommended_items: Sequence[int], relevant_item: int, k: int) -> float:
    return 1.0 if relevant_item in recommended_items[:k] else 0.0


def ndcg_at_k(recommended_items: Sequence[int], relevant_item: int, k: int) -> float:
    try:
        rank = recommended_items[:k].index(relevant_item) + 1
    except ValueError:
        return 0.0
    return 1.0 / math.log2(rank + 1.0)


def map_at_k(recommended_items: Sequence[int], relevant_item: int, k: int) -> float:
    try:
        rank = recommended_items[:k].index(relevant_item) + 1
    except ValueError:
        return 0.0
    return 1.0 / float(rank)


def coverage_at_k(
    rankings: Mapping[int, Sequence[int]],
    catalog_size: int,
    k: int,
) -> float:
    if catalog_size <= 0:
        return 0.0
    unique_items = {
        item_id
        for recommended_items in rankings.values()
        for item_id in recommended_items[:k]
    }
    return len(unique_items) / float(catalog_size)


def novelty_at_k(
    rankings: Mapping[int, Sequence[int]],
    item_probabilities: Mapping[int, float],
    k: int,
) -> float:
    values: list[float] = []
    for recommended_items in rankings.values():
        for item_id in recommended_items[:k]:
            probability = max(float(item_probabilities.get(item_id, 1e-12)), 1e-12)
            values.append(-math.log2(probability))
    return sum(values) / float(len(values)) if values else 0.0


def diversity_at_k(
    rankings: Mapping[int, Sequence[int]],
    item_genres: Mapping[int, frozenset[str]],
    k: int,
) -> float:
    pairwise_scores: list[float] = []
    for recommended_items in rankings.values():
        limited = recommended_items[:k]
        for left_index, left_item in enumerate(limited):
            for right_item in limited[left_index + 1 :]:
                left_genres = item_genres.get(left_item, frozenset())
                right_genres = item_genres.get(right_item, frozenset())
                if not left_genres and not right_genres:
                    continue
                union = left_genres.union(right_genres)
                if not union:
                    continue
                jaccard = len(left_genres.intersection(right_genres)) / float(len(union))
                pairwise_scores.append(1.0 - jaccard)
    return sum(pairwise_scores) / float(len(pairwise_scores)) if pairwise_scores else 0.0

