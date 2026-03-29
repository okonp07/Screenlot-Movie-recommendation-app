from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from screenlot.feedback import append_feedback, feedback_summary, load_feedback_frame


class FeedbackTests(unittest.TestCase):
    def test_append_and_summarize_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            feedback_path = Path(temp_dir) / "feedback.jsonl"
            append_feedback(
                action="like",
                movie_id=1,
                title="Toy Story (1995)",
                score=0.81,
                reason="Shared genres.",
                favorite_titles=["Jumanji (1995)", "Toy Story (1995)"],
                selected_model="ScreenLot hybrid reranker",
                rank=1,
                feedback_path=feedback_path,
            )
            append_feedback(
                action="save",
                movie_id=2,
                title="Babe (1995)",
                score=0.62,
                reason="Behavior match.",
                favorite_titles=["Toy Story (1995)", "Jumanji (1995)"],
                selected_model="ScreenLot hybrid reranker",
                rank=2,
                feedback_path=feedback_path,
            )

            frame = load_feedback_frame(feedback_path)
            summary = feedback_summary(feedback_path)

        self.assertEqual(len(frame), 2)
        self.assertEqual(summary["total_events"], 2)
        self.assertEqual(summary["action_counts"]["like"], 1)
        self.assertEqual(summary["action_counts"]["save"], 1)


if __name__ == "__main__":
    unittest.main()
