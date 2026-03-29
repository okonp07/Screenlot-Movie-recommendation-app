from __future__ import annotations

import unittest

from screenlot.model_summary import humanize_model_name, leaderboard_rows, model_snapshot


class ModelSummaryTests(unittest.TestCase):
    def test_humanize_model_name(self) -> None:
        self.assertEqual(humanize_model_name("matrix_factorization"), "Matrix Factorization")
        self.assertEqual(humanize_model_name("screenlot_serving"), "Live ScreenLot Engine")

    def test_model_snapshot_prefers_serving_row(self) -> None:
        model_card = {
            "selected_model_name": "hybrid",
            "training_config": {"max_rows": 750000, "elapsed_seconds": 120.0},
            "validation_benchmarks": [
                {"model": "matrix_factorization", "recall@20": 0.12, "ndcg@20": 0.05, "evaluated_users": 100},
                {"model": "hybrid", "recall@20": 0.14, "ndcg@20": 0.06, "evaluated_users": 100},
            ],
            "test_benchmarks": [
                {"model": "matrix_factorization", "recall@20": 0.11, "ndcg@20": 0.05, "evaluated_users": 50},
                {"model": "hybrid", "recall@20": 0.13, "ndcg@20": 0.06, "evaluated_users": 50},
                {"model": "screenlot_serving", "recall@20": 0.13, "ndcg@20": 0.06, "evaluated_users": 50},
            ],
            "hybrid_details": {"model_label": "ScreenLot hybrid reranker (validation-tuned blend)", "score_strategy": "blend"},
        }
        snapshot = model_snapshot(model_card)
        self.assertIsNotNone(snapshot)
        self.assertEqual(snapshot["selected_model_name"], "hybrid")
        self.assertEqual(snapshot["serving_row"]["model"], "screenlot_serving")
        self.assertEqual(snapshot["runner_up_row"]["model"], "matrix_factorization")

    def test_leaderboard_rows_sorts_best_first(self) -> None:
        model_card = {
            "test_benchmarks": [
                {"model": "item_knn", "ndcg@20": 0.05, "recall@20": 0.12, "map@20": 0.03},
                {"model": "hybrid", "ndcg@20": 0.06, "recall@20": 0.14, "map@20": 0.04},
                {"model": "screenlot_serving", "ndcg@20": 0.06, "recall@20": 0.14, "map@20": 0.04},
            ]
        }
        rows = leaderboard_rows(model_card, split="test")
        self.assertEqual(rows[0]["model"], "hybrid")
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
