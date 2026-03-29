from __future__ import annotations

from dataclasses import dataclass
import io
import pickle
from pathlib import Path
from typing import Any
import zipfile

import joblib
import numpy as np

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime in the app.
    pd = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
except ImportError:  # pragma: no cover - handled at runtime in the app.
    TfidfVectorizer = None
    cosine_similarity = None

from .content import MODEL_ARCHIVE
from .modeling import TRAINED_HYBRID_ARTIFACT


@dataclass
class RecommendationEngineStatus:
    collaborative_model_loaded: bool
    collaborative_model_name: str


def _require_runtime() -> None:
    if pd is None or TfidfVectorizer is None or cosine_similarity is None:
        raise RuntimeError(
            "ScreenLot needs pandas and scikit-learn to build recommendations."
        )


def load_archived_collaborative_model(model_archive: Path = MODEL_ARCHIVE) -> tuple[Any | None, str]:
    if not model_archive.exists():
        return None, "No collaborative model archive found"

    candidates = (
        "Model pickles/SVDpp_model.pkl",
        "Model pickles/SVD.pkl",
    )

    try:
        with zipfile.ZipFile(model_archive) as archive:
            for candidate in candidates:
                if candidate in archive.namelist():
                    with archive.open(candidate) as handle:
                        model = pickle.load(io.BytesIO(handle.read()))
                    return model, Path(candidate).name
    except Exception as exc:  # pragma: no cover - runtime dependency issue.
        return None, f"Collaborative model unavailable: {exc}"

    return None, "No supported pickle found in archive"


def load_trained_screenlot_artifact(artifact_path: Path = TRAINED_HYBRID_ARTIFACT) -> tuple[Any | None, str]:
    if not artifact_path.exists():
        return None, "No trained ScreenLot artifact found"
    try:
        model = joblib.load(artifact_path)
        label = model.model_label() if hasattr(model, "model_label") else "trained ScreenLot artifact"
        return model, f"Using {label}: {artifact_path.name}"
    except Exception as exc:  # pragma: no cover - runtime dependency issue.
        return None, f"Trained artifact unavailable: {exc}"


class ScreenLotHybridRecommender:
    def __init__(self, catalog: Any, collaborative_model: Any | None = None) -> None:
        _require_runtime()

        self.catalog = catalog.copy().reset_index(drop=True)
        self.catalog["normalized_title"] = self.catalog["title"].fillna("").str.casefold()
        self.catalog["feature_text"] = self.catalog.apply(self._build_feature_text, axis=1)
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self.feature_matrix = self.vectorizer.fit_transform(self.catalog["feature_text"])
        self.collaborative_model = collaborative_model

        self._title_to_index = {}
        for index, title in enumerate(self.catalog["normalized_title"]):
            self._title_to_index.setdefault(title, []).append(index)

    @staticmethod
    def _clean_cell(row: Any, column: str) -> str:
        if column not in row or pd.isna(row[column]):
            return ""
        return str(row[column]).replace("|", " ").replace(",", " ")

    def _build_feature_text(self, row: Any) -> str:
        director_column = next(
            (column for column in row.index if column in {"director", "directors"}),
            None,
        )
        cast_column = next(
            (column for column in row.index if column in {"cast", "stars", "actors"}),
            None,
        )
        overview_column = next(
            (
                column
                for column in row.index
                if column in {"plot", "overview", "description", "storyline"}
            ),
            None,
        )

        parts = [
            self._clean_cell(row, "title"),
            self._clean_cell(row, "genres"),
        ]

        if director_column:
            parts.append(self._clean_cell(row, director_column))
        if cast_column:
            parts.append(self._clean_cell(row, cast_column))
        if overview_column:
            parts.append(self._clean_cell(row, overview_column))

        return " ".join(part for part in parts if part).strip()

    def resolve_titles(self, titles: list[str]) -> list[int]:
        indices = []
        for title in titles:
            normalized = title.casefold().strip()
            matches = self._title_to_index.get(normalized)
            if not matches:
                raise ValueError(f"'{title}' was not found in the loaded catalog.")
            indices.append(matches[0])
        return indices

    def status(self) -> RecommendationEngineStatus:
        return RecommendationEngineStatus(
            collaborative_model_loaded=self.collaborative_model is not None,
            collaborative_model_name=(
                type(self.collaborative_model).__name__
                if self.collaborative_model is not None
                else "Content and popularity blend only"
            ),
        )

    def _content_scores(self, favorite_indices: list[int]) -> np.ndarray:
        profile = self.feature_matrix[favorite_indices].mean(axis=0)
        return cosine_similarity(profile, self.feature_matrix).ravel()

    def _popularity_scores(self) -> np.ndarray:
        mean_rating = self.catalog["mean_rating"].fillna(self.catalog["mean_rating"].median())
        rating_count = self.catalog["rating_count"].fillna(0)

        rating_span = float(mean_rating.max() - mean_rating.min()) or 1.0
        rating_score = (mean_rating - mean_rating.min()) / rating_span
        count_score = np.log1p(rating_count) / max(float(np.log1p(rating_count).max()), 1.0)

        return (0.65 * rating_score + 0.35 * count_score).to_numpy()

    def _collaborative_scores(self, user_id: int | None, movie_ids: np.ndarray) -> np.ndarray:
        if self.collaborative_model is None or user_id is None:
            return np.zeros(len(movie_ids))

        predictions = np.array(
            [
                float(self.collaborative_model.predict(int(user_id), int(movie_id)).est)
                for movie_id in movie_ids
            ]
        )
        return np.clip((predictions - 0.5) / 4.5, 0.0, 1.0)

    def _explanation(self, row: Any, selected_titles: list[str], selected_genres: set[str]) -> str:
        row_genres = {
            genre.strip()
            for genre in str(row.get("genres", "")).split("|")
            if genre.strip() and genre.strip() != "(no genres listed)"
        }
        overlap = sorted(row_genres.intersection(selected_genres))
        if overlap:
            return (
                f"Shares {', '.join(overlap[:3])} signals with your selections "
                f"({', '.join(selected_titles[:2])})."
            )

        director = row.get("director") or row.get("directors")
        if director and not pd.isna(director):
            return f"Matches the metadata profile for your picks and features director context from {director}."

        return "Ranks highly on the hybrid blend of metadata similarity, ratings strength, and recommendation score."

    def _explanation_tags(self, row: Any, selected_genres: set[str]) -> str:
        tags: list[str] = []
        row_genres = {
            genre.strip()
            for genre in str(row.get("genres", "")).split("|")
            if genre.strip() and genre.strip() != "(no genres listed)"
        }
        overlap = sorted(row_genres.intersection(selected_genres))
        if overlap:
            tags.append("Shared genres")
        if row.get("director") or row.get("directors"):
            tags.append("Metadata match")
        if float(row.get("rating_count", 0) or 0) > 0:
            tags.append("Crowd-approved")
        tags.append("Hybrid blend")
        return "|".join(dict.fromkeys(tags))

    def _explanation_details(self, row: Any, selected_titles: list[str], selected_genres: set[str]) -> str:
        details: list[str] = []
        row_genres = {
            genre.strip()
            for genre in str(row.get("genres", "")).split("|")
            if genre.strip() and genre.strip() != "(no genres listed)"
        }
        overlap = sorted(row_genres.intersection(selected_genres))
        if overlap:
            details.append(f"Shared genres with {', '.join(selected_titles[:2])}: {', '.join(overlap[:3])}.")
        if row.get("director") or row.get("directors"):
            details.append("Metadata context helped this title rank well.")
        if float(row.get("rating_count", 0) or 0) > 0:
            details.append(
                f"Catalog signal: average rating {float(row.get('mean_rating', 0.0) or 0.0):.2f} across "
                f"{int(row.get('rating_count', 0) or 0):,} ratings."
            )
        return " ".join(details[:3]).strip()

    def recommend_from_titles(
        self,
        favorite_titles: list[str],
        top_n: int = 10,
        user_id: int | None = None,
        genre_filter: list[str] | None = None,
        min_year: int | None = None,
    ) -> Any:
        if len(favorite_titles) < 2:
            raise ValueError("Select at least two titles to build a useful recommendation profile.")

        favorite_indices = self.resolve_titles(favorite_titles)
        content_scores = self._content_scores(favorite_indices)
        popularity_scores = self._popularity_scores()
        collaborative_scores = self._collaborative_scores(
            user_id=user_id,
            movie_ids=self.catalog["movie_id"].to_numpy(),
        )

        combined_score = (
            0.55 * content_scores
            + 0.20 * popularity_scores
            + 0.25 * collaborative_scores
        )

        results = self.catalog.copy()
        results["score"] = combined_score
        results = results.drop(index=favorite_indices).copy()

        if genre_filter:
            selected = {genre.casefold() for genre in genre_filter}
            results = results[
                results["genres"]
                .fillna("")
                .apply(
                    lambda value: bool(
                        {
                            genre.strip().casefold()
                            for genre in str(value).split("|")
                            if genre.strip()
                        }.intersection(selected)
                    )
                )
            ]

        if min_year is not None and "release_year" in results.columns:
            results = results[results["release_year"].fillna(0) >= min_year]

        selected_genres = {
            genre.strip()
            for title in favorite_titles
            for genre in str(
                self.catalog.iloc[self.resolve_titles([title])[0]].get("genres", "")
            ).split("|")
            if genre.strip()
        }

        results = results.sort_values(
            by=["score", "mean_rating", "rating_count"],
            ascending=[False, False, False],
        )
        results = results.drop_duplicates(subset=["title"]).head(top_n).copy()
        results["reason"] = results.apply(
            lambda row: self._explanation(row, favorite_titles, selected_genres), axis=1
        )
        results["explanation_tags"] = results.apply(
            lambda row: self._explanation_tags(row, selected_genres),
            axis=1,
        )
        results["explanation_details"] = results.apply(
            lambda row: self._explanation_details(row, favorite_titles, selected_genres),
            axis=1,
        )
        results["model_source"] = "fallback_hybrid"

        display_columns = [
            "movie_id",
            "title",
            "genres",
            "release_year",
            "mean_rating",
            "rating_count",
            "score",
            "reason",
            "explanation_tags",
            "explanation_details",
            "model_source",
        ]
        existing_columns = [column for column in display_columns if column in results.columns]
        return results[existing_columns]
