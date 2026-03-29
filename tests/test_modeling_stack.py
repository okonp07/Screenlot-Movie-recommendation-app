from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from screenlot.modeling import (
    build_user_profiles,
    leave_last_two_split,
    load_ratings_frame,
    sample_user_ids_for_target_rows,
    select_best_model_name,
)
from screenlot.ranking_metrics import coverage_at_k, map_at_k, ndcg_at_k, novelty_at_k, recall_at_k


class ModelingStackTests(unittest.TestCase):
    def test_leave_last_two_split(self) -> None:
        ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 1},
                {"userId": 1, "movieId": 11, "rating": 4.5, "timestamp": 2},
                {"userId": 1, "movieId": 12, "rating": 5.0, "timestamp": 3},
                {"userId": 2, "movieId": 20, "rating": 4.0, "timestamp": 1},
                {"userId": 2, "movieId": 21, "rating": 4.5, "timestamp": 2},
                {"userId": 2, "movieId": 22, "rating": 5.0, "timestamp": 3},
            ]
        )
        splits = leave_last_two_split(ratings, min_user_interactions=3)
        self.assertEqual(set(splits.validation["movieId"]), {11, 21})
        self.assertEqual(set(splits.test["movieId"]), {12, 22})
        self.assertEqual(set(splits.train["movieId"]), {10, 20})

    def test_build_user_profiles(self) -> None:
        train = pd.DataFrame(
            [
                {"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 1},
                {"userId": 1, "movieId": 11, "rating": 5.0, "timestamp": 2},
                {"userId": 1, "movieId": 12, "rating": 2.0, "timestamp": 3},
            ]
        )
        profiles = build_user_profiles(train, positive_threshold=3.5)
        profile = profiles[1]
        self.assertIn(10, profile.seen_items)
        self.assertIn(11, profile.seed_items)
        self.assertTrue(np.all(profile.seed_weights > 0))

    def test_ranking_metrics(self) -> None:
        ranking = [10, 20, 30]
        self.assertEqual(recall_at_k(ranking, 20, 2), 1.0)
        self.assertGreater(ndcg_at_k(ranking, 20, 3), 0.0)
        self.assertEqual(map_at_k(ranking, 40, 3), 0.0)
        rankings = {1: [10, 20], 2: [20, 30]}
        self.assertGreater(coverage_at_k(rankings, catalog_size=5, k=2), 0.0)
        self.assertGreater(novelty_at_k(rankings, {10: 0.1, 20: 0.2, 30: 0.05}, k=2), 0.0)

    def test_sampling_preserves_complete_user_histories(self) -> None:
        ratings = pd.DataFrame(
            [
                {"userId": 1, "movieId": 10, "rating": 4.0, "timestamp": 1},
                {"userId": 1, "movieId": 11, "rating": 4.5, "timestamp": 2},
                {"userId": 2, "movieId": 20, "rating": 3.5, "timestamp": 1},
                {"userId": 2, "movieId": 21, "rating": 4.0, "timestamp": 2},
                {"userId": 2, "movieId": 22, "rating": 4.5, "timestamp": 3},
                {"userId": 3, "movieId": 30, "rating": 5.0, "timestamp": 1},
            ]
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            csv_path = Path(temp_dir) / "ratings.csv"
            ratings.to_csv(csv_path, index=False)
            sampled_user_ids = sample_user_ids_for_target_rows(
                ratings_path=csv_path,
                target_rows=3,
                min_user_interactions=2,
                seed=7,
                chunksize=2,
            )
            sampled = load_ratings_frame(
                ratings_path=csv_path,
                sampled_user_ids=sampled_user_ids,
                chunksize=2,
            )

        self.assertTrue(sampled_user_ids.issubset({1, 2}))
        self.assertGreaterEqual(len(sampled), 3)
        self.assertEqual(set(sampled["userId"]), sampled_user_ids)
        for user_id in sampled_user_ids:
            expected_count = int((ratings["userId"] == user_id).sum())
            actual_count = int((sampled["userId"] == user_id).sum())
            self.assertEqual(actual_count, expected_count)

    def test_select_best_model_name_prefers_rank_quality(self) -> None:
        benchmark_rows = [
            {"model": "item_knn", "ndcg@20": 0.051, "recall@20": 0.12, "map@20": 0.03, "ndcg@10": 0.03, "recall@10": 0.07},
            {"model": "matrix_factorization", "ndcg@20": 0.054, "recall@20": 0.126, "map@20": 0.034, "ndcg@10": 0.043, "recall@10": 0.082},
            {"model": "hybrid", "ndcg@20": 0.045, "recall@20": 0.112, "map@20": 0.028, "ndcg@10": 0.032, "recall@10": 0.056},
        ]
        self.assertEqual(select_best_model_name(benchmark_rows), "matrix_factorization")


if __name__ == "__main__":
    unittest.main()
