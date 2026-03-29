from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize

from .open_data import PROCESSED_SCREENLOT_DIR, RAW_MOVIELENS_32M_DIR
from .ranking_metrics import (
    coverage_at_k,
    diversity_at_k,
    map_at_k,
    ndcg_at_k,
    novelty_at_k,
    recall_at_k,
)
from .runtime import (
    ARTIFACTS_DIR,
    BENCHMARK_CSV_PATH,
    BENCHMARK_JSON_PATH,
    MODEL_ARTIFACT_PATH,
    MODEL_CARD_PATH,
)


TRAINED_HYBRID_ARTIFACT = MODEL_ARTIFACT_PATH
BENCHMARK_JSON = BENCHMARK_JSON_PATH
BENCHMARK_CSV = BENCHMARK_CSV_PATH
MODEL_CARD_JSON = MODEL_CARD_PATH

RATING_COLUMNS = ["userId", "movieId", "rating", "timestamp"]
RATING_DTYPES = {
    "userId": "int32",
    "movieId": "int32",
    "rating": "float32",
    "timestamp": "int64",
}

DEFAULT_BLEND_WEIGHTS = {
    "item_knn_score": 0.22,
    "mf_score": 0.34,
    "content_score": 0.15,
    "popularity_score": 0.06,
    "mean_rating_norm": 0.04,
    "rating_count_norm": 0.04,
    "freshness_score": 0.02,
    "genre_overlap": 0.08,
    "director_overlap": 0.03,
    "series_overlap": 0.02,
}


def ensure_artifact_dir() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    evaluation_users: np.ndarray


@dataclass
class UserProfile:
    user_id: int
    seen_items: set[int]
    seed_items: list[int]
    seed_weights: np.ndarray


def _split_pipe_values(value: Any) -> frozenset[str]:
    text = "" if value is None else str(value)
    if not text or text.lower() == "nan":
        return frozenset()
    return frozenset(
        part.strip()
        for part in text.split("|")
        if part.strip() and part.strip().lower() != "nan"
    )


def _normalize_scores(values: np.ndarray) -> np.ndarray:
    if values.size == 0:
        return values
    minimum = float(values.min())
    maximum = float(values.max())
    span = maximum - minimum
    if span <= 0:
        return np.zeros_like(values, dtype=np.float32)
    return ((values - minimum) / span).astype(np.float32)


def _ranking_signature(metrics: dict[str, float]) -> tuple[float, float, float, float, float]:
    return (
        float(metrics.get("ndcg@20", 0.0)),
        float(metrics.get("recall@20", 0.0)),
        float(metrics.get("map@20", 0.0)),
        float(metrics.get("ndcg@10", 0.0)),
        float(metrics.get("recall@10", 0.0)),
    )


def select_best_model_name(benchmark_rows: list[dict[str, float]]) -> str:
    if not benchmark_rows:
        return "hybrid"
    return str(max(benchmark_rows, key=_ranking_signature).get("model", "hybrid"))


def load_processed_catalog(
    catalog_path: Path = PROCESSED_SCREENLOT_DIR / "catalog.csv",
) -> pd.DataFrame:
    catalog = pd.read_csv(catalog_path, low_memory=False)
    catalog["movie_id"] = pd.to_numeric(catalog["movie_id"], errors="coerce").astype("Int64")
    catalog = catalog.dropna(subset=["movie_id"]).copy()
    catalog["movie_id"] = catalog["movie_id"].astype("int32")
    numeric_columns = ("mean_rating", "rating_count", "release_year")
    for column in numeric_columns:
        if column in catalog.columns:
            catalog[column] = pd.to_numeric(catalog[column], errors="coerce")
    if "last_rating_at" in catalog.columns:
        catalog["last_rating_at"] = pd.to_datetime(catalog["last_rating_at"], errors="coerce", utc=True)
    else:
        catalog["last_rating_at"] = pd.NaT
    catalog["genres"] = catalog.get("genres", "").fillna("")
    catalog["wikidata_genres"] = catalog.get("wikidata_genres", "").fillna("")
    catalog["directors"] = catalog.get("directors", "").fillna("")
    catalog["series"] = catalog.get("series", "").fillna("")
    catalog["countries"] = catalog.get("countries", "").fillna("")
    catalog["languages"] = catalog.get("languages", "").fillna("")
    catalog["wikidata_description"] = catalog.get("wikidata_description", "").fillna("")
    catalog["title"] = catalog.get("title", "").fillna("")
    return catalog


def load_ratings_frame(
    ratings_path: Path = RAW_MOVIELENS_32M_DIR / "ratings.csv",
    max_rows: int = 0,
    sampled_user_ids: set[int] | None = None,
    chunksize: int = 1_000_000,
) -> pd.DataFrame:
    if sampled_user_ids:
        frames: list[pd.DataFrame] = []
        for chunk in pd.read_csv(
            ratings_path,
            usecols=RATING_COLUMNS,
            dtype=RATING_DTYPES,
            chunksize=chunksize,
        ):
            filtered = chunk[chunk["userId"].isin(sampled_user_ids)]
            if not filtered.empty:
                frames.append(filtered)
        if frames:
            return pd.concat(frames, ignore_index=True)
        return pd.DataFrame(
            {
                column: pd.Series(dtype=dtype)
                for column, dtype in RATING_DTYPES.items()
            }
        )

    nrows = None if max_rows <= 0 else max_rows
    return pd.read_csv(
        ratings_path,
        usecols=RATING_COLUMNS,
        dtype=RATING_DTYPES,
        nrows=nrows,
    )


def sample_user_ids_for_target_rows(
    ratings_path: Path = RAW_MOVIELENS_32M_DIR / "ratings.csv",
    target_rows: int = 0,
    min_user_interactions: int = 1,
    seed: int = 42,
    chunksize: int = 1_000_000,
) -> set[int]:
    user_counts: pd.Series | None = None
    for chunk in pd.read_csv(
        ratings_path,
        usecols=["userId"],
        dtype={"userId": "int32"},
        chunksize=chunksize,
    ):
        chunk_counts = chunk["userId"].value_counts(sort=False)
        user_counts = (
            chunk_counts.astype("int64")
            if user_counts is None
            else user_counts.add(chunk_counts, fill_value=0)
        )

    if user_counts is None or user_counts.empty:
        return set()

    eligible = user_counts[user_counts >= min_user_interactions]
    if eligible.empty:
        return set()

    user_ids = eligible.index.to_numpy(dtype=np.int32)
    counts = eligible.to_numpy(dtype=np.int64)
    if target_rows <= 0 or int(counts.sum()) <= target_rows:
        return {int(user_id) for user_id in user_ids.tolist()}

    rng = np.random.default_rng(seed)
    order = rng.permutation(len(user_ids))
    selected_user_ids: list[int] = []
    cumulative = 0
    for index in order:
        selected_user_ids.append(int(user_ids[index]))
        cumulative += int(counts[index])
        if cumulative >= target_rows:
            break
    return set(selected_user_ids)


def filter_active_entities(
    ratings: pd.DataFrame,
    min_user_interactions: int = 5,
    min_item_interactions: int = 20,
) -> pd.DataFrame:
    filtered = ratings.copy()
    for _ in range(2):
        user_counts = filtered["userId"].value_counts()
        item_counts = filtered["movieId"].value_counts()
        filtered = filtered[
            filtered["userId"].isin(user_counts[user_counts >= min_user_interactions].index)
            & filtered["movieId"].isin(item_counts[item_counts >= min_item_interactions].index)
        ].copy()
    return filtered


def sample_users(
    ratings: pd.DataFrame,
    max_users: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    if max_users <= 0:
        return ratings.copy()
    user_ids = ratings["userId"].drop_duplicates().to_numpy()
    if len(user_ids) <= max_users:
        return ratings.copy()
    rng = np.random.default_rng(seed)
    sampled = rng.choice(user_ids, size=max_users, replace=False)
    return ratings[ratings["userId"].isin(sampled)].copy()


def leave_last_two_split(
    ratings: pd.DataFrame,
    min_user_interactions: int = 5,
) -> DatasetSplits:
    ratings = ratings.sort_values(["userId", "timestamp", "movieId"]).copy()
    user_counts = ratings["userId"].value_counts()
    eligible_users = user_counts[user_counts >= min_user_interactions].index.to_numpy()
    eligible_mask = ratings["userId"].isin(eligible_users)

    eligible_rows = ratings.loc[eligible_mask].copy()
    eligible_rows["rank_from_end"] = eligible_rows.groupby("userId").cumcount(ascending=False)

    validation = eligible_rows[eligible_rows["rank_from_end"] == 1].drop(columns=["rank_from_end"]).copy()
    test = eligible_rows[eligible_rows["rank_from_end"] == 0].drop(columns=["rank_from_end"]).copy()
    train = pd.concat(
        [
            ratings.loc[~eligible_mask],
            eligible_rows[eligible_rows["rank_from_end"] > 1].drop(columns=["rank_from_end"]),
        ],
        ignore_index=True,
    )

    return DatasetSplits(
        train=train,
        validation=validation,
        test=test,
        evaluation_users=eligible_users.astype(np.int32),
    )


def sample_holdout_rows(
    rows: pd.DataFrame,
    max_users: int = 0,
    seed: int = 42,
) -> pd.DataFrame:
    if max_users <= 0 or rows["userId"].nunique() <= max_users:
        return rows.copy()
    rng = np.random.default_rng(seed)
    user_ids = rows["userId"].drop_duplicates().to_numpy()
    sampled_users = rng.choice(user_ids, size=max_users, replace=False)
    return rows[rows["userId"].isin(sampled_users)].copy()


def build_user_profiles(
    train: pd.DataFrame,
    positive_threshold: float = 3.5,
    max_history: int = 30,
) -> dict[int, UserProfile]:
    profiles: dict[int, UserProfile] = {}
    for user_id, frame in train.groupby("userId", sort=False):
        ordered = frame.sort_values("timestamp", ascending=False)
        seen_items = set(ordered["movieId"].astype(int).tolist())
        positives = ordered[ordered["rating"] >= positive_threshold].head(max_history)
        if positives.empty:
            positives = ordered.head(min(max_history, len(ordered)))
        seed_items = positives["movieId"].astype(int).tolist()
        ratings = positives["rating"].astype(np.float32).to_numpy()
        decay = 1.0 / np.log2(np.arange(len(seed_items), dtype=np.float32) + 2.0)
        weights = np.maximum(ratings - positive_threshold + 1.0, 0.2) * decay
        profiles[int(user_id)] = UserProfile(
            user_id=int(user_id),
            seen_items=seen_items,
            seed_items=seed_items,
            seed_weights=weights.astype(np.float32),
        )
    return profiles


class PopularityRecommender:
    name = "popularity"

    def __init__(self, shrinkage: float = 25.0) -> None:
        self.shrinkage = shrinkage
        self.item_scores: pd.Series | None = None
        self.popularity_probabilities: dict[int, float] = {}
        self.rank_order: list[int] = []

    def fit(self, train: pd.DataFrame) -> "PopularityRecommender":
        stats = train.groupby("movieId")["rating"].agg(["mean", "count", "sum"]).reset_index()
        global_mean = float(train["rating"].mean())
        adjusted = (stats["sum"] + self.shrinkage * global_mean) / (stats["count"] + self.shrinkage)
        log_count = np.log1p(stats["count"].to_numpy(dtype=np.float32))
        score = 0.7 * _normalize_scores(adjusted.to_numpy(dtype=np.float32)) + 0.3 * _normalize_scores(log_count)
        self.item_scores = pd.Series(score, index=stats["movieId"].astype(int).to_numpy())
        self.rank_order = self.item_scores.sort_values(ascending=False).index.astype(int).tolist()
        total_count = float(stats["count"].sum())
        self.popularity_probabilities = {
            int(movie_id): float(count) / total_count
            for movie_id, count in zip(stats["movieId"], stats["count"], strict=True)
        }
        return self

    def score_candidates(self, candidate_ids: list[int]) -> dict[int, float]:
        if self.item_scores is None:
            return {}
        return {
            int(item_id): float(self.item_scores.get(item_id, 0.0))
            for item_id in candidate_ids
        }

    def top_items(self, exclude_items: set[int], top_k: int) -> list[int]:
        results: list[int] = []
        for item_id in self.rank_order:
            if item_id in exclude_items:
                continue
            results.append(int(item_id))
            if len(results) >= top_k:
                break
        return results

    def recommend_for_user(self, user_profile: UserProfile, top_k: int) -> list[int]:
        return self.top_items(user_profile.seen_items, top_k)


class ContentSimilarityModel:
    name = "content"

    def __init__(self, max_features: int = 40000) -> None:
        self.max_features = max_features
        self.catalog: pd.DataFrame | None = None
        self.vectorizer: TfidfVectorizer | None = None
        self.matrix: csr_matrix | None = None
        self.item_ids: np.ndarray | None = None
        self.item_to_index: dict[int, int] = {}

    @staticmethod
    def _feature_text(row: pd.Series) -> str:
        parts = [
            row.get("title", ""),
            row.get("genres", ""),
            row.get("wikidata_genres", ""),
            row.get("directors", ""),
            row.get("countries", ""),
            row.get("languages", ""),
            row.get("series", ""),
            row.get("wikidata_description", ""),
        ]
        return " ".join(str(part).replace("|", " ") for part in parts if str(part).strip())

    def fit(self, catalog: pd.DataFrame) -> "ContentSimilarityModel":
        self.catalog = catalog.copy().drop_duplicates(subset=["movie_id"]).reset_index(drop=True)
        self.catalog["feature_text"] = self.catalog.apply(self._feature_text, axis=1)
        self.vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=self.max_features,
            min_df=2,
        )
        matrix = self.vectorizer.fit_transform(self.catalog["feature_text"])
        self.matrix = normalize(matrix, norm="l2", copy=False)
        self.item_ids = self.catalog["movie_id"].astype(int).to_numpy()
        self.item_to_index = {item_id: index for index, item_id in enumerate(self.item_ids.tolist())}
        return self

    def score_all(self, seed_items: list[int], seed_weights: np.ndarray) -> np.ndarray:
        if self.matrix is None:
            return np.array([], dtype=np.float32)
        matched_items = [item_id for item_id in seed_items if item_id in self.item_to_index]
        indices = [self.item_to_index[item_id] for item_id in matched_items]
        if not indices:
            return np.zeros(self.matrix.shape[0], dtype=np.float32)
        weights = np.asarray(
            [seed_weights[seed_items.index(item_id)] for item_id in matched_items],
            dtype=np.float32,
        )
        profile = np.asarray(
            self.matrix[indices].multiply(weights[:, None]).sum(axis=0),
            dtype=np.float32,
        ).ravel()
        norm = float(np.linalg.norm(profile))
        if norm > 0:
            profile = profile / norm
        scores = self.matrix @ profile
        scores = np.asarray(scores).ravel()
        return scores.astype(np.float32)

    def top_items(self, seed_items: list[int], seed_weights: np.ndarray, exclude_items: set[int], top_k: int) -> list[int]:
        scores = self.score_all(seed_items, seed_weights)
        if scores.size == 0 or self.item_ids is None:
            return []
        order = np.argsort(scores)[::-1]
        results: list[int] = []
        for index in order:
            item_id = int(self.item_ids[index])
            if item_id in exclude_items:
                continue
            results.append(item_id)
            if len(results) >= top_k:
                break
        return results

    def score_candidates(
        self,
        seed_items: list[int],
        seed_weights: np.ndarray,
        candidate_ids: list[int],
    ) -> dict[int, float]:
        scores = self.score_all(seed_items, seed_weights)
        if scores.size == 0:
            return {int(item_id): 0.0 for item_id in candidate_ids}
        return {
            int(item_id): float(scores[self.item_to_index[item_id]])
            for item_id in candidate_ids
            if item_id in self.item_to_index
        }

    def recommend_for_user(self, user_profile: UserProfile, top_k: int) -> list[int]:
        return self.top_items(user_profile.seed_items, user_profile.seed_weights, user_profile.seen_items, top_k)


class ItemKNNRecommender:
    name = "item_knn"

    def __init__(self, n_neighbors: int = 120, positive_threshold: float = 3.5) -> None:
        self.n_neighbors = n_neighbors
        self.positive_threshold = positive_threshold
        self.item_user_matrix: csr_matrix | None = None
        self.nearest_neighbors: NearestNeighbors | None = None
        self.item_ids: np.ndarray | None = None
        self.item_to_index: dict[int, int] = {}

    def fit(self, train: pd.DataFrame) -> "ItemKNNRecommender":
        positive = train[train["rating"] >= self.positive_threshold].copy()
        if positive.empty:
            positive = train.copy()
        user_ids = np.sort(positive["userId"].unique())
        item_ids = np.sort(positive["movieId"].unique())
        user_to_idx = {int(user_id): index for index, user_id in enumerate(user_ids.tolist())}
        item_to_idx = {int(item_id): index for index, item_id in enumerate(item_ids.tolist())}
        row_indices = positive["movieId"].map(item_to_idx).to_numpy()
        col_indices = positive["userId"].map(user_to_idx).to_numpy()
        weights = np.maximum(positive["rating"].to_numpy(dtype=np.float32) - self.positive_threshold + 1.0, 0.25)
        matrix = csr_matrix(
            (weights, (row_indices, col_indices)),
            shape=(len(item_ids), len(user_ids)),
            dtype=np.float32,
        )
        self.item_user_matrix = normalize(matrix, norm="l2", copy=False)
        self.nearest_neighbors = NearestNeighbors(
            metric="cosine",
            algorithm="brute",
            n_neighbors=min(self.n_neighbors + 1, len(item_ids)),
        )
        self.nearest_neighbors.fit(self.item_user_matrix)
        self.item_ids = item_ids.astype(np.int32)
        self.item_to_index = item_to_idx
        return self

    def candidate_scores(
        self,
        seed_items: list[int],
        seed_weights: np.ndarray,
        top_k_candidates: int = 400,
    ) -> dict[int, float]:
        if self.nearest_neighbors is None or self.item_user_matrix is None or self.item_ids is None:
            return {}
        scores: dict[int, float] = {}
        for seed_index, item_id in enumerate(seed_items):
            item_index = self.item_to_index.get(int(item_id))
            if item_index is None:
                continue
            distances, indices = self.nearest_neighbors.kneighbors(
                self.item_user_matrix[item_index],
                return_distance=True,
            )
            for distance, candidate_index in zip(distances[0], indices[0], strict=True):
                candidate_item = int(self.item_ids[candidate_index])
                if candidate_item == int(item_id):
                    continue
                similarity = max(1.0 - float(distance), 0.0)
                scores[candidate_item] = scores.get(candidate_item, 0.0) + similarity * float(seed_weights[seed_index])
        if not scores:
            return {}
        ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k_candidates]
        return {int(item_id): float(score) for item_id, score in ordered}

    def top_items(self, user_profile: UserProfile, top_k: int) -> list[int]:
        scores = self.candidate_scores(user_profile.seed_items, user_profile.seed_weights, top_k_candidates=max(top_k * 20, 200))
        results: list[int] = []
        for item_id, _score in sorted(scores.items(), key=lambda item: item[1], reverse=True):
            if item_id in user_profile.seen_items:
                continue
            results.append(int(item_id))
            if len(results) >= top_k:
                break
        return results

    def score_candidates(self, user_profile: UserProfile, candidate_ids: list[int]) -> dict[int, float]:
        scores = self.candidate_scores(user_profile.seed_items, user_profile.seed_weights, top_k_candidates=max(len(candidate_ids) * 3, 400))
        return {int(item_id): float(scores.get(item_id, 0.0)) for item_id in candidate_ids}

    def recommend_for_user(self, user_profile: UserProfile, top_k: int) -> list[int]:
        return self.top_items(user_profile, top_k)


class MatrixFactorizationRecommender:
    name = "matrix_factorization"

    def __init__(self, n_components: int = 64, positive_threshold: float = 3.0) -> None:
        self.n_components = n_components
        self.positive_threshold = positive_threshold
        self.user_ids: np.ndarray | None = None
        self.item_ids: np.ndarray | None = None
        self.user_to_index: dict[int, int] = {}
        self.item_to_index: dict[int, int] = {}
        self.user_factors: np.ndarray | None = None
        self.item_factors: np.ndarray | None = None

    def fit(self, train: pd.DataFrame) -> "MatrixFactorizationRecommender":
        user_ids = np.sort(train["userId"].unique())
        item_ids = np.sort(train["movieId"].unique())
        self.user_ids = user_ids.astype(np.int32)
        self.item_ids = item_ids.astype(np.int32)
        self.user_to_index = {int(user_id): index for index, user_id in enumerate(self.user_ids.tolist())}
        self.item_to_index = {int(item_id): index for index, item_id in enumerate(self.item_ids.tolist())}

        row_indices = train["userId"].map(self.user_to_index).to_numpy()
        col_indices = train["movieId"].map(self.item_to_index).to_numpy()
        values = np.maximum(train["rating"].to_numpy(dtype=np.float32) - self.positive_threshold, 0.0) + 0.05
        matrix = csr_matrix(
            (values, (row_indices, col_indices)),
            shape=(len(self.user_ids), len(self.item_ids)),
            dtype=np.float32,
        )
        model = TruncatedSVD(
            n_components=min(self.n_components, max(2, min(matrix.shape) - 1)),
            random_state=42,
        )
        self.user_factors = model.fit_transform(matrix).astype(np.float32)
        self.item_factors = model.components_.T.astype(np.float32)
        return self

    def _profile_vector(self, user_profile: UserProfile) -> np.ndarray:
        if self.user_factors is None or self.item_factors is None:
            return np.array([], dtype=np.float32)
        user_index = self.user_to_index.get(int(user_profile.user_id))
        if user_index is not None:
            return self.user_factors[user_index]
        indices = [self.item_to_index[item_id] for item_id in user_profile.seed_items if item_id in self.item_to_index]
        if not indices:
            return np.zeros(self.item_factors.shape[1], dtype=np.float32)
        weights = np.asarray([user_profile.seed_weights[user_profile.seed_items.index(item_id)] for item_id in user_profile.seed_items if item_id in self.item_to_index], dtype=np.float32)
        return np.average(self.item_factors[indices], axis=0, weights=weights).astype(np.float32)

    def top_items(self, user_profile: UserProfile, top_k: int) -> list[int]:
        if self.item_factors is None or self.item_ids is None:
            return []
        profile_vector = self._profile_vector(user_profile)
        if profile_vector.size == 0:
            return []
        scores = self.item_factors @ profile_vector
        order = np.argsort(scores)[::-1]
        results: list[int] = []
        for index in order:
            item_id = int(self.item_ids[index])
            if item_id in user_profile.seen_items:
                continue
            results.append(item_id)
            if len(results) >= top_k:
                break
        return results

    def score_candidates(self, user_profile: UserProfile, candidate_ids: list[int]) -> dict[int, float]:
        if self.item_factors is None:
            return {}
        profile_vector = self._profile_vector(user_profile)
        if profile_vector.size == 0:
            return {int(item_id): 0.0 for item_id in candidate_ids}
        scores: dict[int, float] = {}
        for item_id in candidate_ids:
            item_index = self.item_to_index.get(int(item_id))
            if item_index is None:
                scores[int(item_id)] = 0.0
                continue
            scores[int(item_id)] = float(np.dot(self.item_factors[item_index], profile_vector))
        return scores

    def recommend_for_user(self, user_profile: UserProfile, top_k: int) -> list[int]:
        return self.top_items(user_profile, top_k)


class TrainedScreenLotHybridRecommender:
    name = "hybrid"

    def __init__(
        self,
        catalog: pd.DataFrame,
        popularity_model: PopularityRecommender,
        content_model: ContentSimilarityModel,
        item_knn_model: ItemKNNRecommender,
        matrix_factorization_model: MatrixFactorizationRecommender,
    ) -> None:
        self.catalog = catalog.copy().drop_duplicates(subset=["movie_id"]).reset_index(drop=True)
        self.catalog["movie_id"] = self.catalog["movie_id"].astype(int)
        self.popularity_model = popularity_model
        self.content_model = content_model
        self.item_knn_model = item_knn_model
        self.matrix_factorization_model = matrix_factorization_model
        self.calibrator: LogisticRegression | None = None
        self.score_strategy = "blend"
        self.blend_weights = dict(DEFAULT_BLEND_WEIGHTS)
        self.strategy_selection_metrics: dict[str, dict[str, float]] = {}
        self.serving_model_name = "hybrid"
        self.validation_benchmarks: list[dict[str, float]] = []
        self.feature_columns = [
            "item_knn_score",
            "mf_score",
            "content_score",
            "popularity_score",
            "mean_rating_norm",
            "rating_count_norm",
            "freshness_score",
            "genre_overlap",
            "director_overlap",
            "series_overlap",
        ]

        self.catalog_by_movie = self.catalog.set_index("movie_id", drop=False)
        self.movie_ids = self.catalog["movie_id"].astype(int).tolist()
        self.title_to_movie_ids: dict[str, list[int]] = {}
        for movie_id, title in zip(self.catalog["movie_id"], self.catalog["title"], strict=True):
            self.title_to_movie_ids.setdefault(str(title).casefold().strip(), []).append(int(movie_id))

        self.genre_sets = {
            int(row.movie_id): _split_pipe_values(row.genres).union(_split_pipe_values(row.wikidata_genres))
            for row in self.catalog.itertuples(index=False)
        }
        self.director_sets = {
            int(row.movie_id): _split_pipe_values(row.directors)
            for row in self.catalog.itertuples(index=False)
        }
        self.series_sets = {
            int(row.movie_id): _split_pipe_values(row.series)
            for row in self.catalog.itertuples(index=False)
        }

        last_rating = self.catalog["last_rating_at"]
        timestamps = last_rating.map(lambda value: value.timestamp() if pd.notna(value) else np.nan).to_numpy(dtype=np.float64)
        self.rating_count_max = max(float(self.catalog["rating_count"].fillna(0).max()), 1.0)
        self.mean_rating_min = float(self.catalog["mean_rating"].fillna(0).min())
        self.mean_rating_max = float(self.catalog["mean_rating"].fillna(0).max())
        self.freshness_min = float(np.nanmin(timestamps)) if np.isfinite(np.nanmin(timestamps)) else 0.0
        self.freshness_max = float(np.nanmax(timestamps)) if np.isfinite(np.nanmax(timestamps)) else 1.0

    def model_label(self) -> str:
        if self.serving_model_name != "hybrid":
            return f"ScreenLot validation-selected {self.serving_model_name.replace('_', ' ')} ranker"
        if self.score_strategy == "logistic":
            return "ScreenLot hybrid reranker (logistic validation scorer)"
        return "ScreenLot hybrid reranker (validation-tuned blend)"

    def configure_serving_model(
        self,
        model_name: str,
        validation_benchmarks: list[dict[str, float]] | None = None,
    ) -> "TrainedScreenLotHybridRecommender":
        supported = {"hybrid", "matrix_factorization", "item_knn", "popularity"}
        self.serving_model_name = model_name if model_name in supported else "hybrid"
        if validation_benchmarks is not None:
            self.validation_benchmarks = list(validation_benchmarks)
        return self

    def _selected_base_model(self) -> Any | None:
        if self.serving_model_name == "matrix_factorization":
            return self.matrix_factorization_model
        if self.serving_model_name == "item_knn":
            return self.item_knn_model
        if self.serving_model_name == "popularity":
            return self.popularity_model
        return None

    def _resolve_titles(self, favorite_titles: list[str]) -> list[int]:
        item_ids: list[int] = []
        for title in favorite_titles:
            matches = self.title_to_movie_ids.get(title.casefold().strip())
            if not matches:
                raise ValueError(f"'{title}' was not found in the trained ScreenLot catalog.")
            item_ids.append(int(matches[0]))
        return item_ids

    def _user_profile_from_titles(self, favorite_titles: list[str]) -> UserProfile:
        seed_items = self._resolve_titles(favorite_titles)
        weights = np.linspace(1.0, 0.7, num=len(seed_items), dtype=np.float32)
        return UserProfile(
            user_id=-1,
            seen_items=set(seed_items),
            seed_items=seed_items,
            seed_weights=weights,
        )

    def _candidate_pool(self, user_profile: UserProfile, top_per_model: int = 250) -> list[int]:
        candidates = set(self.popularity_model.top_items(user_profile.seen_items, top_per_model))
        candidates.update(self.content_model.top_items(user_profile.seed_items, user_profile.seed_weights, user_profile.seen_items, top_per_model))
        candidates.update(self.item_knn_model.top_items(user_profile, top_per_model))
        candidates.update(self.matrix_factorization_model.top_items(user_profile, top_per_model))
        candidates.difference_update(user_profile.seen_items)
        return list(candidates)

    def _row_metadata_features(self, movie_id: int, user_profile: UserProfile) -> tuple[float, float, float, float, float, float]:
        row = self.catalog_by_movie.loc[movie_id]
        mean_rating = float(row.get("mean_rating", 0.0) or 0.0)
        rating_count = float(row.get("rating_count", 0.0) or 0.0)
        mean_rating_norm = 0.0 if self.mean_rating_max <= self.mean_rating_min else (mean_rating - self.mean_rating_min) / (self.mean_rating_max - self.mean_rating_min)
        rating_count_norm = np.log1p(rating_count) / np.log1p(self.rating_count_max)
        last_rating_at = row.get("last_rating_at")
        last_ts = last_rating_at.timestamp() if pd.notna(last_rating_at) else np.nan
        if np.isnan(last_ts) or self.freshness_max <= self.freshness_min:
            freshness_score = 0.0
        else:
            freshness_score = (last_ts - self.freshness_min) / (self.freshness_max - self.freshness_min)

        seed_genres = set().union(*(self.genre_sets.get(item_id, frozenset()) for item_id in user_profile.seed_items))
        seed_directors = set().union(*(self.director_sets.get(item_id, frozenset()) for item_id in user_profile.seed_items))
        seed_series = set().union(*(self.series_sets.get(item_id, frozenset()) for item_id in user_profile.seed_items))
        candidate_genres = self.genre_sets.get(movie_id, frozenset())
        candidate_directors = self.director_sets.get(movie_id, frozenset())
        candidate_series = self.series_sets.get(movie_id, frozenset())

        genre_overlap = (
            len(candidate_genres.intersection(seed_genres)) / float(len(seed_genres))
            if seed_genres
            else 0.0
        )
        director_overlap = (
            len(candidate_directors.intersection(seed_directors)) / float(len(seed_directors))
            if seed_directors
            else 0.0
        )
        series_overlap = 1.0 if candidate_series.intersection(seed_series) else 0.0
        return (
            mean_rating_norm,
            rating_count_norm,
            freshness_score,
            genre_overlap,
            director_overlap,
            series_overlap,
        )

    def _feature_frame(
        self,
        user_profile: UserProfile,
        candidate_ids: list[int],
    ) -> pd.DataFrame:
        popularity_scores = self.popularity_model.score_candidates(candidate_ids)
        item_knn_scores = self.item_knn_model.score_candidates(user_profile, candidate_ids)
        mf_scores = self.matrix_factorization_model.score_candidates(user_profile, candidate_ids)
        content_scores = self.content_model.score_candidates(user_profile.seed_items, user_profile.seed_weights, candidate_ids)

        records = []
        for movie_id in candidate_ids:
            mean_rating_norm, rating_count_norm, freshness_score, genre_overlap, director_overlap, series_overlap = self._row_metadata_features(movie_id, user_profile)
            records.append(
                {
                    "movie_id": int(movie_id),
                    "item_knn_score": float(item_knn_scores.get(movie_id, 0.0)),
                    "mf_score": float(mf_scores.get(movie_id, 0.0)),
                    "content_score": float(content_scores.get(movie_id, 0.0)),
                    "popularity_score": float(popularity_scores.get(movie_id, 0.0)),
                    "mean_rating_norm": float(mean_rating_norm),
                    "rating_count_norm": float(rating_count_norm),
                    "freshness_score": float(freshness_score),
                    "genre_overlap": float(genre_overlap),
                    "director_overlap": float(director_overlap),
                    "series_overlap": float(series_overlap),
                }
            )
        return pd.DataFrame.from_records(records)

    def _blend_candidates(self, seed: int = 42, random_trials: int = 24) -> list[dict[str, float]]:
        manual_candidates = [
            dict(DEFAULT_BLEND_WEIGHTS),
            {
                "item_knn_score": 0.18,
                "mf_score": 0.44,
                "content_score": 0.12,
                "popularity_score": 0.05,
                "mean_rating_norm": 0.04,
                "rating_count_norm": 0.04,
                "freshness_score": 0.02,
                "genre_overlap": 0.07,
                "director_overlap": 0.02,
                "series_overlap": 0.02,
            },
            {
                "item_knn_score": 0.32,
                "mf_score": 0.28,
                "content_score": 0.13,
                "popularity_score": 0.06,
                "mean_rating_norm": 0.03,
                "rating_count_norm": 0.04,
                "freshness_score": 0.02,
                "genre_overlap": 0.08,
                "director_overlap": 0.02,
                "series_overlap": 0.02,
            },
            {
                "item_knn_score": 0.20,
                "mf_score": 0.30,
                "content_score": 0.20,
                "popularity_score": 0.05,
                "mean_rating_norm": 0.03,
                "rating_count_norm": 0.03,
                "freshness_score": 0.02,
                "genre_overlap": 0.10,
                "director_overlap": 0.04,
                "series_overlap": 0.03,
            },
        ]
        rng = np.random.default_rng(seed)
        alpha = np.asarray([2.0, 3.4, 1.4, 0.9, 0.7, 0.8, 0.4, 1.1, 0.5, 0.4], dtype=np.float32)
        for _ in range(random_trials):
            sample = rng.dirichlet(alpha)
            manual_candidates.append(
                {
                    column: float(weight)
                    for column, weight in zip(self.feature_columns, sample.tolist(), strict=True)
                }
            )
        unique_candidates: list[dict[str, float]] = []
        seen_keys: set[tuple[float, ...]] = set()
        for weights in manual_candidates:
            key = tuple(round(float(weights[column]), 6) for column in self.feature_columns)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            unique_candidates.append(weights)
        return unique_candidates

    def _blend_scores(
        self,
        feature_frame: pd.DataFrame,
        weights: dict[str, float] | None = None,
    ) -> np.ndarray:
        if feature_frame.empty:
            return np.array([], dtype=np.float32)
        active_weights = weights or self.blend_weights
        normalized_frame = pd.DataFrame(index=feature_frame.index)
        for column in self.feature_columns:
            normalized_frame[column] = _normalize_scores(
                feature_frame[column].to_numpy(dtype=np.float32)
            )
        scores = np.zeros(len(feature_frame), dtype=np.float32)
        for column in self.feature_columns:
            scores += float(active_weights.get(column, 0.0)) * normalized_frame[column].to_numpy(dtype=np.float32)
        return scores.astype(np.float32)

    def _evaluate_validation_frames(
        self,
        validation_frames: list[tuple[int, int, pd.DataFrame]],
        strategy: str,
        weights: dict[str, float] | None = None,
        k_values: tuple[int, ...] = (10, 20),
    ) -> dict[str, float]:
        rankings: dict[int, list[int]] = {}
        positives: dict[int, int] = {}
        max_k = max(k_values)

        for user_id, positive_item, feature_frame in validation_frames:
            if feature_frame.empty:
                continue
            if strategy == "logistic" and self.calibrator is not None:
                scores = self.calibrator.predict_proba(feature_frame[self.feature_columns])[:, 1].astype(np.float32)
            else:
                scores = self._blend_scores(feature_frame, weights=weights)
            order = np.argsort(scores)[::-1]
            rankings[user_id] = feature_frame.iloc[order]["movie_id"].astype(int).head(max_k).tolist()
            positives[user_id] = int(positive_item)

        metrics: dict[str, float] = {}
        if not rankings:
            for k in k_values:
                metrics[f"recall@{k}"] = 0.0
                metrics[f"ndcg@{k}"] = 0.0
                metrics[f"map@{k}"] = 0.0
            metrics["evaluated_users"] = 0
            return metrics

        for k in k_values:
            recalls = [recall_at_k(rankings[user_id], positives[user_id], k) for user_id in rankings]
            ndcgs = [ndcg_at_k(rankings[user_id], positives[user_id], k) for user_id in rankings]
            maps = [map_at_k(rankings[user_id], positives[user_id], k) for user_id in rankings]
            metrics[f"recall@{k}"] = float(np.mean(recalls))
            metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))
            metrics[f"map@{k}"] = float(np.mean(maps))
        metrics["evaluated_users"] = len(rankings)
        return metrics

    def fit_calibrator(
        self,
        validation_rows: pd.DataFrame,
        user_profiles: dict[int, UserProfile],
        negatives_per_user: int = 30,
        seed: int = 42,
    ) -> "TrainedScreenLotHybridRecommender":
        rng = np.random.default_rng(seed)
        training_frames: list[pd.DataFrame] = []
        validation_frames: list[tuple[int, int, pd.DataFrame]] = []
        popular_candidates = self.popularity_model.rank_order[:5000]

        for row in validation_rows.itertuples(index=False):
            user_id = int(row.userId)
            positive_item = int(row.movieId)
            user_profile = user_profiles.get(user_id)
            if user_profile is None:
                continue
            candidate_ids = set(self._candidate_pool(user_profile, top_per_model=120))
            candidate_ids.add(positive_item)
            if negatives_per_user > 0 and popular_candidates:
                extra = rng.choice(popular_candidates, size=min(negatives_per_user, len(popular_candidates)), replace=False)
                candidate_ids.update(int(item_id) for item_id in extra if int(item_id) not in user_profile.seen_items)
            feature_frame = self._feature_frame(user_profile, list(candidate_ids))
            validation_frames.append((user_id, positive_item, feature_frame.copy()))
            feature_frame["label"] = (feature_frame["movie_id"] == positive_item).astype(int)
            training_frames.append(feature_frame)

        if not training_frames:
            return self

        training_data = pd.concat(training_frames, ignore_index=True)
        if training_data["label"].nunique() >= 2:
            model = LogisticRegression(
                max_iter=500,
                class_weight="balanced",
                random_state=42,
            )
            model.fit(training_data[self.feature_columns], training_data["label"])
            self.calibrator = model

        strategy_results: dict[str, dict[str, float]] = {}
        best_strategy = ("blend", dict(DEFAULT_BLEND_WEIGHTS))
        best_metrics = self._evaluate_validation_frames(
            validation_frames,
            strategy="blend",
            weights=self.blend_weights,
        )
        strategy_results["blend_default"] = best_metrics

        for index, weights in enumerate(self._blend_candidates(seed=seed), start=1):
            metrics = self._evaluate_validation_frames(
                validation_frames,
                strategy="blend",
                weights=weights,
            )
            strategy_results[f"blend_{index}"] = metrics
            if _ranking_signature(metrics) > _ranking_signature(best_metrics):
                best_metrics = metrics
                best_strategy = ("blend", weights)

        if self.calibrator is not None:
            logistic_metrics = self._evaluate_validation_frames(
                validation_frames,
                strategy="logistic",
            )
            strategy_results["logistic"] = logistic_metrics
            if _ranking_signature(logistic_metrics) > _ranking_signature(best_metrics):
                best_metrics = logistic_metrics
                best_strategy = ("logistic", dict(self.blend_weights))

        self.score_strategy = best_strategy[0]
        if self.score_strategy == "blend":
            self.blend_weights = dict(best_strategy[1])
        self.strategy_selection_metrics = strategy_results
        return self

    def _predict(self, feature_frame: pd.DataFrame) -> np.ndarray:
        if feature_frame.empty:
            return np.array([], dtype=np.float32)
        if self.score_strategy == "logistic" and self.calibrator is not None:
            return self.calibrator.predict_proba(feature_frame[self.feature_columns])[:, 1].astype(np.float32)
        return self._blend_scores(feature_frame)

    def _shared_context(self, movie_id: int, user_profile: UserProfile) -> dict[str, list[str]]:
        seed_genres = set().union(*(self.genre_sets.get(item_id, frozenset()) for item_id in user_profile.seed_items))
        seed_directors = set().union(*(self.director_sets.get(item_id, frozenset()) for item_id in user_profile.seed_items))
        seed_series = set().union(*(self.series_sets.get(item_id, frozenset()) for item_id in user_profile.seed_items))
        candidate_genres = self.genre_sets.get(movie_id, frozenset())
        candidate_directors = self.director_sets.get(movie_id, frozenset())
        candidate_series = self.series_sets.get(movie_id, frozenset())
        return {
            "genres": sorted(candidate_genres.intersection(seed_genres))[:3],
            "directors": sorted(candidate_directors.intersection(seed_directors))[:2],
            "series": sorted(candidate_series.intersection(seed_series))[:1],
        }

    def _reason(self, movie_id: int, row: pd.Series, user_profile: UserProfile, favorite_titles: list[str] | None = None) -> str:
        title_context = favorite_titles[:2] if favorite_titles else []
        if float(row["series_overlap"]) > 0:
            return "Stays within the same franchise or series universe as one of your picks."
        if float(row["genre_overlap"]) > 0:
            seed_text = f" {', '.join(title_context)}" if title_context else ""
            return f"Shares genre signals with your selected favorites{seed_text}."
        if float(row["director_overlap"]) > 0:
            return "Matches director-level context from movies already in your taste profile."
        if float(row["item_knn_score"]) >= max(float(row["mf_score"]), float(row["content_score"])):
            return "People who liked similar movies in your history also tended to engage with this title."
        if float(row["mf_score"]) > float(row["content_score"]):
            return "Ranks well for users with rating behavior patterns close to your profile."
        if float(row["content_score"]) > 0:
            return "Matches the metadata signature of your selected titles."
        return "Balances popularity, quality, and discovery value inside the ScreenLot hybrid ranker."

    def _explanation_payload(
        self,
        row: pd.Series,
        user_profile: UserProfile,
        favorite_titles: list[str] | None = None,
    ) -> dict[str, str]:
        movie_id = int(row["movie_id"])
        shared = self._shared_context(movie_id, user_profile)
        tags: list[str] = []
        evidence_lines: list[str] = []

        if shared["series"]:
            tags.append("Same series")
            evidence_lines.append(f"Shared series universe: {', '.join(shared['series'])}.")
        if shared["genres"]:
            tags.append("Shared genres")
            evidence_lines.append(f"Shared genres: {', '.join(shared['genres'])}.")
        if shared["directors"]:
            tags.append("Shared director")
            evidence_lines.append(f"Shared director context: {', '.join(shared['directors'])}.")

        dominant_signal = max(
            (
                ("Similar viewers", float(row["item_knn_score"])),
                ("Behavior match", float(row["mf_score"])),
                ("Metadata match", float(row["content_score"])),
                ("Popularity lift", float(row["popularity_score"])),
            ),
            key=lambda item: item[1],
        )[0]
        tags.append(dominant_signal)

        if dominant_signal == "Similar viewers":
            evidence_lines.append("Strong nearest-neighbor signal from viewers with similar positive histories.")
        elif dominant_signal == "Behavior match":
            evidence_lines.append("Strong matrix-factorization signal from rating patterns close to your profile.")
        elif dominant_signal == "Metadata match":
            evidence_lines.append("Strong metadata similarity signal from title, genre, director, and Wikidata context.")
        else:
            evidence_lines.append("Ranks well on catalog popularity and aggregate rating strength.")

        rating_count = int(row.get("rating_count", 0) or 0)
        mean_rating = float(row.get("mean_rating", 0.0) or 0.0)
        if rating_count > 0:
            tags.append("Crowd-approved")
            evidence_lines.append(
                f"Catalog quality signal: average rating {mean_rating:.2f} across {rating_count:,} ratings."
            )
        if float(row.get("freshness_score", 0.0) or 0.0) >= 0.7:
            tags.append("Fresh interest")
            evidence_lines.append("Recent interaction activity suggests this title is still showing up in active user behavior.")

        tags = list(dict.fromkeys(tags))[:4]
        evidence_text = " ".join(evidence_lines[:3]).strip()
        return {
            "reason": self._reason(movie_id, row, user_profile, favorite_titles=favorite_titles),
            "explanation_tags": "|".join(tags),
            "explanation_details": evidence_text,
            "model_source": self.serving_model_name,
        }

    def recommend_for_user(self, user_profile: UserProfile, top_k: int) -> list[int]:
        selected_model = self._selected_base_model()
        if selected_model is not None:
            return selected_model.recommend_for_user(user_profile, top_k)
        candidate_ids = self._candidate_pool(user_profile, top_per_model=max(top_k * 12, 180))
        feature_frame = self._feature_frame(user_profile, candidate_ids)
        if feature_frame.empty:
            return self.popularity_model.top_items(user_profile.seen_items, top_k)
        feature_frame["score"] = self._predict(feature_frame)
        feature_frame = feature_frame.sort_values("score", ascending=False)
        return feature_frame["movie_id"].astype(int).head(top_k).tolist()

    def recommend_from_titles(
        self,
        favorite_titles: list[str],
        top_n: int = 10,
        user_id: int | None = None,
        genre_filter: list[str] | None = None,
        min_year: int | None = None,
    ) -> pd.DataFrame:
        user_profile = self._user_profile_from_titles(favorite_titles)
        selected_model = self._selected_base_model()
        if selected_model is not None:
            candidate_ids = selected_model.recommend_for_user(user_profile, max(top_n * 25, 250))
            if not candidate_ids:
                candidate_ids = self._candidate_pool(user_profile, top_per_model=max(top_n * 15, 220))
        else:
            candidate_ids = self._candidate_pool(user_profile, top_per_model=max(top_n * 15, 220))
        feature_frame = self._feature_frame(user_profile, candidate_ids)
        if feature_frame.empty:
            return pd.DataFrame(columns=["title", "genres", "release_year", "mean_rating", "rating_count", "score", "reason"])
        if selected_model is None:
            feature_frame["score"] = self._predict(feature_frame)
        elif self.serving_model_name == "matrix_factorization":
            candidate_scores = self.matrix_factorization_model.score_candidates(user_profile, feature_frame["movie_id"].astype(int).tolist())
            feature_frame["score"] = feature_frame["movie_id"].map(candidate_scores).astype(float)
        elif self.serving_model_name == "item_knn":
            candidate_scores = self.item_knn_model.score_candidates(user_profile, feature_frame["movie_id"].astype(int).tolist())
            feature_frame["score"] = feature_frame["movie_id"].map(candidate_scores).astype(float)
        else:
            candidate_scores = self.popularity_model.score_candidates(feature_frame["movie_id"].astype(int).tolist())
            feature_frame["score"] = feature_frame["movie_id"].map(candidate_scores).astype(float)

        merged = feature_frame.merge(self.catalog, left_on="movie_id", right_on="movie_id", how="left")
        merged = merged[~merged["movie_id"].isin(user_profile.seen_items)].copy()
        if genre_filter:
            selected = {genre.casefold() for genre in genre_filter}
            merged = merged[
                merged["genres"].fillna("").apply(
                    lambda value: bool(
                        {
                            genre.strip().casefold()
                            for genre in str(value).split("|")
                            if genre.strip()
                        }.intersection(selected)
                    )
                )
            ]
        if min_year is not None:
            merged = merged[merged["release_year"].fillna(0) >= min_year]

        merged = merged.sort_values("score", ascending=False).head(top_n).copy()
        explanation_payload = merged.apply(
            lambda row: self._explanation_payload(row, user_profile, favorite_titles=favorite_titles),
            axis=1,
        )
        explanation_frame = pd.DataFrame(explanation_payload.tolist(), index=merged.index)
        merged = pd.concat([merged, explanation_frame], axis=1)
        return merged[
            [
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
        ].reset_index(drop=True)

    def save(self, artifact_path: Path = TRAINED_HYBRID_ARTIFACT) -> Path:
        ensure_artifact_dir()
        joblib.dump(self, artifact_path, compress=3)
        return artifact_path

    @classmethod
    def load(cls, artifact_path: Path = TRAINED_HYBRID_ARTIFACT) -> "TrainedScreenLotHybridRecommender":
        return joblib.load(artifact_path)


def benchmark_model(
    model: Any,
    holdout_rows: pd.DataFrame,
    user_profiles: dict[int, UserProfile],
    catalog: pd.DataFrame,
    item_probabilities: dict[int, float],
    k_values: tuple[int, ...] = (10, 20),
    model_name: str | None = None,
) -> dict[str, float]:
    rankings: dict[int, list[int]] = {}
    positives: dict[int, int] = {}
    max_k = max(k_values)

    for row in holdout_rows.itertuples(index=False):
        user_id = int(row.userId)
        user_profile = user_profiles.get(user_id)
        if user_profile is None:
            continue
        rankings[user_id] = model.recommend_for_user(user_profile, max_k)
        positives[user_id] = int(row.movieId)

    metrics: dict[str, float] = {"model": model_name or getattr(model, "name", type(model).__name__)}
    if not rankings:
        for k in k_values:
            metrics[f"recall@{k}"] = 0.0
            metrics[f"ndcg@{k}"] = 0.0
            metrics[f"map@{k}"] = 0.0
        metrics["coverage"] = 0.0
        metrics["novelty"] = 0.0
        metrics["diversity"] = 0.0
        metrics["evaluated_users"] = 0
        return metrics

    for k in k_values:
        recalls = [recall_at_k(rankings[user_id], positives[user_id], k) for user_id in rankings]
        ndcgs = [ndcg_at_k(rankings[user_id], positives[user_id], k) for user_id in rankings]
        maps = [map_at_k(rankings[user_id], positives[user_id], k) for user_id in rankings]
        metrics[f"recall@{k}"] = float(np.mean(recalls))
        metrics[f"ndcg@{k}"] = float(np.mean(ndcgs))
        metrics[f"map@{k}"] = float(np.mean(maps))

    item_genres = {
        int(row.movie_id): _split_pipe_values(row.genres).union(_split_pipe_values(row.wikidata_genres))
        for row in catalog.itertuples(index=False)
    }
    metrics["coverage"] = coverage_at_k(rankings, catalog["movie_id"].nunique(), max_k)
    metrics["novelty"] = novelty_at_k(rankings, item_probabilities, max_k)
    metrics["diversity"] = diversity_at_k(rankings, item_genres, max_k)
    metrics["evaluated_users"] = len(rankings)
    return metrics


def save_benchmarks(
    benchmark_rows: list[dict[str, Any]],
    benchmark_json_path: Path = BENCHMARK_JSON,
    benchmark_csv_path: Path = BENCHMARK_CSV,
) -> None:
    ensure_artifact_dir()
    benchmark_frame = pd.DataFrame(benchmark_rows)
    benchmark_csv_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_json_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_frame.to_csv(benchmark_csv_path, index=False)
    benchmark_json_path.write_text(json.dumps(benchmark_rows, indent=2), encoding="utf-8")


def save_model_card(
    benchmark_rows: list[dict[str, Any]],
    artifact_path: Path,
    training_config: dict[str, Any],
    validation_benchmarks: list[dict[str, Any]] | None = None,
    selected_model_name: str | None = None,
    hybrid_details: dict[str, Any] | None = None,
) -> None:
    ensure_artifact_dir()
    model_card = {
        "artifact_path": str(artifact_path),
        "training_config": training_config,
        "selected_model_name": selected_model_name,
        "validation_benchmarks": validation_benchmarks or [],
        "test_benchmarks": benchmark_rows,
        "hybrid_details": hybrid_details or {},
    }
    MODEL_CARD_JSON.write_text(json.dumps(model_card, indent=2), encoding="utf-8")
