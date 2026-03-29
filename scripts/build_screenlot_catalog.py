from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from screenlot.open_data import (  # noqa: E402
    INTERIM_WIKIDATA_DIR,
    PROCESSED_SCREENLOT_DIR,
    RAW_MOVIELENS_32M_DIR,
    ensure_data_dirs,
    extract_release_year,
    format_imdb_id,
)


def read_csv_by_key(path: Path, key_name: str) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row[key_name]: row for row in reader}


def aggregate_rating_stats(ratings_path: Path) -> tuple[dict[str, dict[str, float]], dict[str, int], dict[str, int]]:
    rating_stats: dict[str, dict[str, float]] = defaultdict(lambda: {"count": 0, "sum": 0.0, "min_ts": 0, "max_ts": 0})
    user_rating_counts: dict[str, int] = defaultdict(int)
    rating_distribution: dict[str, int] = defaultdict(int)

    with ratings_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            movie_id = row["movieId"]
            user_id = row["userId"]
            rating = float(row["rating"])
            timestamp = int(row["timestamp"])

            stats = rating_stats[movie_id]
            stats["count"] += 1
            stats["sum"] += rating
            stats["min_ts"] = timestamp if stats["min_ts"] == 0 else min(int(stats["min_ts"]), timestamp)
            stats["max_ts"] = max(int(stats["max_ts"]), timestamp)

            user_rating_counts[user_id] += 1
            rating_distribution[row["rating"]] += 1

    return rating_stats, user_rating_counts, rating_distribution


def aggregate_tag_counts(tags_path: Path) -> dict[str, int]:
    tag_counts: dict[str, int] = defaultdict(int)
    if not tags_path.exists():
        return tag_counts

    with tags_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            tag_counts[row["movieId"]] += 1
    return tag_counts


def isoformat_timestamp(timestamp: int) -> str:
    if not timestamp:
        return ""
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a ScreenLot-ready catalog from MovieLens 32M and Wikidata.")
    parser.add_argument("--movies-path", type=Path, default=RAW_MOVIELENS_32M_DIR / "movies.csv")
    parser.add_argument("--links-path", type=Path, default=RAW_MOVIELENS_32M_DIR / "links.csv")
    parser.add_argument("--ratings-path", type=Path, default=RAW_MOVIELENS_32M_DIR / "ratings.csv")
    parser.add_argument("--tags-path", type=Path, default=RAW_MOVIELENS_32M_DIR / "tags.csv")
    parser.add_argument(
        "--wikidata-path",
        type=Path,
        default=INTERIM_WIKIDATA_DIR / "wikidata_movie_metadata.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=PROCESSED_SCREENLOT_DIR)
    args = parser.parse_args()

    ensure_data_dirs()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    movies = read_csv_by_key(args.movies_path, "movieId")
    links = read_csv_by_key(args.links_path, "movieId")
    wikidata = read_csv_by_key(args.wikidata_path, "movie_id") if args.wikidata_path.exists() else {}

    rating_stats, user_rating_counts, rating_distribution = aggregate_rating_stats(args.ratings_path)
    tag_counts = aggregate_tag_counts(args.tags_path)

    catalog_rows = []
    for movie_id, movie in movies.items():
        link = links.get(movie_id, {})
        imdb_id = format_imdb_id(link.get("imdbId", ""))
        wikidata_row = wikidata.get(movie_id, {})
        stats = rating_stats.get(movie_id, {"count": 0, "sum": 0.0, "min_ts": 0, "max_ts": 0})
        rating_count = int(stats["count"])
        mean_rating = round(stats["sum"] / rating_count, 6) if rating_count else ""

        catalog_rows.append(
            {
                "movie_id": movie_id,
                "title": movie.get("title", ""),
                "genres": movie.get("genres", ""),
                "release_year": extract_release_year(movie.get("title", "")),
                "imdb_id": imdb_id,
                "tmdb_id": link.get("tmdbId", "") or "",
                "wikidata_id": wikidata_row.get("wikidata_id", ""),
                "wikidata_label": wikidata_row.get("wikidata_label", ""),
                "wikidata_description": wikidata_row.get("wikidata_description", ""),
                "release_date": wikidata_row.get("release_date", ""),
                "countries": wikidata_row.get("countries", ""),
                "languages": wikidata_row.get("languages", ""),
                "directors": wikidata_row.get("directors", ""),
                "screenwriters": wikidata_row.get("screenwriters", ""),
                "cast_members": wikidata_row.get("cast_members", ""),
                "series": wikidata_row.get("series", ""),
                "wikidata_genres": wikidata_row.get("genres", ""),
                "mean_rating": str(mean_rating),
                "rating_count": str(rating_count),
                "unique_rater_count": str(rating_count),
                "tag_count": str(tag_counts.get(movie_id, 0)),
                "first_rating_at": isoformat_timestamp(int(stats["min_ts"])),
                "last_rating_at": isoformat_timestamp(int(stats["max_ts"])),
            }
        )

    catalog_rows.sort(key=lambda row: int(row["movie_id"]))

    catalog_path = args.output_dir / "catalog.csv"
    fieldnames = list(catalog_rows[0].keys()) if catalog_rows else []
    write_csv(catalog_path, fieldnames, catalog_rows)

    top_movies_path = args.output_dir / "top_movies.csv"
    top_movies = sorted(
        catalog_rows,
        key=lambda row: (
            float(row["mean_rating"]) if row["mean_rating"] else 0.0,
            int(row["rating_count"]),
        ),
        reverse=True,
    )[:500]
    write_csv(top_movies_path, fieldnames, top_movies)

    rating_distribution_path = args.output_dir / "rating_distribution.csv"
    write_csv(
        rating_distribution_path,
        ["rating", "count"],
        [
            {"rating": rating, "count": str(count)}
            for rating, count in sorted(rating_distribution.items(), key=lambda item: float(item[0]))
        ],
    )

    user_activity_path = args.output_dir / "user_activity_summary.csv"
    user_activity_rows = sorted(
        (
            {"user_id": user_id, "rating_count": str(count)}
            for user_id, count in user_rating_counts.items()
        ),
        key=lambda row: int(row["rating_count"]),
        reverse=True,
    )[:5000]
    write_csv(user_activity_path, ["user_id", "rating_count"], user_activity_rows)

    profile = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "movie_count": len(catalog_rows),
        "wikidata_rows": len(wikidata),
        "ratings_source": str(args.ratings_path),
        "links_source": str(args.links_path),
        "movies_source": str(args.movies_path),
        "wikidata_source": str(args.wikidata_path) if args.wikidata_path.exists() else "",
    }
    (args.output_dir / "dataset_profile.json").write_text(json.dumps(profile, indent=2), encoding="utf-8")

    print(f"Catalog written to {catalog_path}")
    print(f"Wikidata-enriched rows available for {len(wikidata)} movies")
    print(f"Rating distribution written to {rating_distribution_path}")


if __name__ == "__main__":
    main()
