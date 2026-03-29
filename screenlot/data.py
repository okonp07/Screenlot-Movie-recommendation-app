from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any

try:
    import pandas as pd
except ImportError:  # pragma: no cover - handled at runtime in the app.
    pd = None

from .content import DEFAULT_DATA_DIR


PRIMARY_REQUIRED_FILES = ("ratings.csv", "movies.csv")
LEGACY_REQUIRED_FILES = ("train.csv", "movies.csv")
OPTIONAL_FILES = (
    "test.csv",
    "imdb_data.csv",
    "links.csv",
    "genome_scores.csv",
    "genome_tags.csv",
)

COLUMN_ALIASES = {
    "movieid": "movie_id",
    "userid": "user_id",
    "imdbid": "imdb_id",
    "tmdbid": "tmdb_id",
    "avg_rating": "mean_rating",
    "average_rating": "mean_rating",
    "num_ratings": "rating_count",
}


@dataclass
class DataBundle:
    train: Any
    movies: Any
    test: Any = None
    imdb: Any = None
    links: Any = None
    genome_scores: Any = None
    genome_tags: Any = None


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError(
            "pandas is required to load the ScreenLot datasets. Install the project "
            "requirements before launching the app."
        )


def standardize_columns(frame: Any) -> Any:
    renamed = {}
    for column in frame.columns:
        normalized = re.sub(r"[^a-z0-9]+", "_", str(column).strip().lower()).strip("_")
        renamed[column] = COLUMN_ALIASES.get(normalized, normalized)
    return frame.rename(columns=renamed)


def dataset_status(data_dir: Path = DEFAULT_DATA_DIR) -> dict[str, bool]:
    expected_files = {
        *PRIMARY_REQUIRED_FILES,
        *LEGACY_REQUIRED_FILES,
        *OPTIONAL_FILES,
    }
    return {
        file_name: (data_dir / file_name).exists()
        for file_name in sorted(expected_files)
    }


def missing_required_files(data_dir: Path = DEFAULT_DATA_DIR) -> list[str]:
    if all((data_dir / file_name).exists() for file_name in PRIMARY_REQUIRED_FILES):
        return []
    if all((data_dir / file_name).exists() for file_name in LEGACY_REQUIRED_FILES):
        return []
    return list(PRIMARY_REQUIRED_FILES)


def _read_if_present(path: Path) -> Any:
    return pd.read_csv(path) if path.exists() else None


def load_data_bundle(data_dir: Path = DEFAULT_DATA_DIR) -> DataBundle:
    _require_pandas()
    missing = missing_required_files(data_dir)
    if missing:
        raise FileNotFoundError(
            "Missing required dataset files: " + ", ".join(missing)
        )

    ratings_path = (
        data_dir / "ratings.csv"
        if (data_dir / "ratings.csv").exists()
        else data_dir / "train.csv"
    )

    return DataBundle(
        train=standardize_columns(pd.read_csv(ratings_path)),
        movies=standardize_columns(pd.read_csv(data_dir / "movies.csv")),
        test=(
            standardize_columns(_read_if_present(data_dir / "test.csv"))
            if (data_dir / "test.csv").exists()
            else None
        ),
        imdb=(
            standardize_columns(_read_if_present(data_dir / "imdb_data.csv"))
            if (data_dir / "imdb_data.csv").exists()
            else None
        ),
        links=(
            standardize_columns(_read_if_present(data_dir / "links.csv"))
            if (data_dir / "links.csv").exists()
            else None
        ),
        genome_scores=(
            standardize_columns(_read_if_present(data_dir / "genome_scores.csv"))
            if (data_dir / "genome_scores.csv").exists()
            else None
        ),
        genome_tags=(
            standardize_columns(_read_if_present(data_dir / "genome_tags.csv"))
            if (data_dir / "genome_tags.csv").exists()
            else None
        ),
    )


def _extract_release_year(title: Any) -> Any:
    years = title.fillna("").astype(str).str.extract(r"\((\d{4})\)\s*$")[0]
    return pd.to_numeric(years, errors="coerce")


def build_catalog(bundle: DataBundle) -> Any:
    _require_pandas()

    movies = bundle.movies.copy()
    if "movie_id" not in movies.columns:
        raise ValueError("movies.csv must contain a movie_id column.")

    catalog = movies.copy()

    if bundle.imdb is not None and "movie_id" in bundle.imdb.columns:
        catalog = catalog.merge(bundle.imdb, on="movie_id", how="left", suffixes=("", "_imdb"))

    if "title" not in catalog.columns:
        raise ValueError("movies.csv must contain a title column.")

    if "genres" not in catalog.columns:
        catalog["genres"] = "(no genres listed)"

    rating_summary = (
        bundle.train.groupby("movie_id", dropna=False)["rating"]
        .agg(mean_rating="mean", rating_count="count")
        .reset_index()
    )
    catalog = catalog.merge(rating_summary, on="movie_id", how="left")

    if "release_year" not in catalog.columns:
        catalog["release_year"] = _extract_release_year(catalog["title"])

    catalog["mean_rating"] = pd.to_numeric(catalog["mean_rating"], errors="coerce")
    catalog["rating_count"] = pd.to_numeric(catalog["rating_count"], errors="coerce").fillna(0)
    catalog["genres"] = catalog["genres"].fillna("(no genres listed)")
    catalog["title"] = catalog["title"].fillna("Untitled")

    return catalog


def available_genres(catalog: Any) -> list[str]:
    _require_pandas()
    exploded = (
        catalog["genres"]
        .fillna("")
        .astype(str)
        .str.split("|")
        .explode()
        .str.strip()
    )
    return sorted({genre for genre in exploded.tolist() if genre and genre != "(no genres listed)"})


def dataset_metrics(bundle: DataBundle, catalog: Any) -> dict[str, int]:
    _require_pandas()
    return {
        "ratings": int(len(bundle.train)),
        "movies": int(catalog["movie_id"].nunique()),
        "users": int(bundle.train["user_id"].nunique()) if "user_id" in bundle.train.columns else 0,
        "catalog_rows": int(len(catalog)),
    }
