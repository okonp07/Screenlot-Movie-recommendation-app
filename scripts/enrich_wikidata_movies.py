from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
import time
import ssl
import urllib.error
import urllib.parse
import urllib.request
import json


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from screenlot.open_data import (  # noqa: E402
    INTERIM_WIKIDATA_DIR,
    RAW_MOVIELENS_32M_DIR,
    WIKIDATA_SPARQL_ENDPOINT,
    WIKIDATA_USER_AGENT,
    ensure_data_dirs,
    format_imdb_id,
)


OUTPUT_COLUMNS = [
    "movie_id",
    "imdb_id",
    "tmdb_id",
    "wikidata_id",
    "wikidata_label",
    "wikidata_description",
    "release_date",
    "genres",
    "directors",
    "cast_members",
    "screenwriters",
    "countries",
    "languages",
    "series",
]


def load_links(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            imdb_id = format_imdb_id(row.get("imdbId", ""))
            if not imdb_id:
                continue
            rows.append(
                {
                    "movie_id": row["movieId"],
                    "imdb_id": imdb_id,
                    "tmdb_id": row.get("tmdbId", "") or "",
                }
            )
    return rows


def load_completed_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return {row["imdb_id"] for row in reader if row.get("imdb_id")}


def chunked(values: list[dict[str, str]], size: int) -> list[list[dict[str, str]]]:
    return [values[index : index + size] for index in range(0, len(values), size)]


def build_query(imdb_ids: list[str]) -> str:
    values = " ".join(f'"{imdb_id}"' for imdb_id in imdb_ids)
    return f"""
PREFIX bd: <http://www.bigdata.com/rdf#>
PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX wikibase: <http://wikiba.se/ontology#>
PREFIX schema: <http://schema.org/>

SELECT ?imdb_id ?film ?filmLabel ?filmDescription
       (SAMPLE(?release_date_raw) AS ?release_date)
       (GROUP_CONCAT(DISTINCT ?genreLabel; separator="|") AS ?genres)
       (GROUP_CONCAT(DISTINCT ?directorLabel; separator="|") AS ?directors)
       (GROUP_CONCAT(DISTINCT ?countryLabel; separator="|") AS ?countries)
       (GROUP_CONCAT(DISTINCT ?languageLabel; separator="|") AS ?languages)
       (GROUP_CONCAT(DISTINCT ?seriesLabel; separator="|") AS ?series)
WHERE {{
  VALUES ?imdb_id {{ {values} }}
  ?film wdt:P345 ?imdb_id .
  OPTIONAL {{ ?film schema:description ?filmDescription FILTER(LANG(?filmDescription) = "en") }}
  OPTIONAL {{ ?film wdt:P577 ?release_date_raw }}
  OPTIONAL {{ ?film wdt:P136 ?genre . ?genre rdfs:label ?genreLabel FILTER(LANG(?genreLabel) = "en") }}
  OPTIONAL {{ ?film wdt:P57 ?director . ?director rdfs:label ?directorLabel FILTER(LANG(?directorLabel) = "en") }}
  OPTIONAL {{ ?film wdt:P495 ?country . ?country rdfs:label ?countryLabel FILTER(LANG(?countryLabel) = "en") }}
  OPTIONAL {{ ?film wdt:P364 ?language . ?language rdfs:label ?languageLabel FILTER(LANG(?languageLabel) = "en") }}
  OPTIONAL {{ ?film wdt:P179 ?series_item . ?series_item rdfs:label ?seriesLabel FILTER(LANG(?seriesLabel) = "en") }}
  SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
}}
GROUP BY ?imdb_id ?film ?filmLabel ?filmDescription
""".strip()


def execute_query(
    query: str,
    retries: int = 3,
    pause_seconds: float = 2.0,
    insecure: bool = False,
) -> dict:
    payload = urllib.parse.urlencode({"query": query, "format": "json"}).encode("utf-8")
    request = urllib.request.Request(
        WIKIDATA_SPARQL_ENDPOINT,
        data=payload,
        headers={
            "Accept": "application/sparql-results+json",
            "Content-Type": "application/x-www-form-urlencoded",
            "User-Agent": WIKIDATA_USER_AGENT,
        },
        method="POST",
    )

    last_error: Exception | None = None
    ssl_context = ssl._create_unverified_context() if insecure else None
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=120, context=ssl_context) as response:
                return json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as exc:
            last_error = exc
            time.sleep(pause_seconds * attempt)

    raise RuntimeError(f"Wikidata query failed after {retries} attempts: {last_error}")


def value_from(binding: dict, key: str) -> str:
    return binding.get(key, {}).get("value", "")


def wikidata_id_from_uri(uri: str) -> str:
    return uri.rsplit("/", 1)[-1] if uri else ""


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    file_exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=OUTPUT_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Incrementally enrich MovieLens movies with Wikidata metadata.")
    parser.add_argument(
        "--links-path",
        type=Path,
        default=RAW_MOVIELENS_32M_DIR / "links.csv",
        help="Path to the MovieLens links.csv file.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=INTERIM_WIKIDATA_DIR / "wikidata_movie_metadata.csv",
        help="Output CSV path for the enriched metadata.",
    )
    parser.add_argument("--batch-size", type=int, default=100, help="Number of IMDb ids to send per SPARQL request.")
    parser.add_argument("--sleep-seconds", type=float, default=1.0, help="Pause between query batches.")
    parser.add_argument("--limit", type=int, default=0, help="Optional number of new rows to enrich in this run.")
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification for environments with local certificate issues.",
    )
    args = parser.parse_args()

    ensure_data_dirs()

    links_rows = load_links(args.links_path)
    completed_ids = load_completed_ids(args.output_path)
    pending_rows = [row for row in links_rows if row["imdb_id"] not in completed_ids]
    if args.limit > 0:
        pending_rows = pending_rows[: args.limit]

    links_by_imdb = {row["imdb_id"]: row for row in links_rows}
    total = len(pending_rows)
    if total == 0:
        print("No pending Wikidata enrichments.")
        return

    for index, batch in enumerate(chunked(pending_rows, args.batch_size), start=1):
        imdb_ids = [row["imdb_id"] for row in batch]
        payload = execute_query(build_query(imdb_ids), insecure=args.insecure)
        results = []
        matched_ids = set()

        for binding in payload["results"]["bindings"]:
            imdb_id = value_from(binding, "imdb_id")
            matched_ids.add(imdb_id)
            source_row = links_by_imdb.get(imdb_id, {})
            results.append(
                {
                    "movie_id": source_row.get("movie_id", ""),
                    "imdb_id": imdb_id,
                    "tmdb_id": source_row.get("tmdb_id", ""),
                    "wikidata_id": wikidata_id_from_uri(value_from(binding, "film")),
                    "wikidata_label": value_from(binding, "filmLabel"),
                    "wikidata_description": value_from(binding, "filmDescription"),
                    "release_date": value_from(binding, "release_date"),
                    "genres": value_from(binding, "genres"),
                    "directors": value_from(binding, "directors"),
                    "cast_members": value_from(binding, "cast_members"),
                    "screenwriters": value_from(binding, "screenwriters"),
                    "countries": value_from(binding, "countries"),
                    "languages": value_from(binding, "languages"),
                    "series": value_from(binding, "series"),
                }
            )

        # Preserve progress for titles that do not resolve in Wikidata.
        unresolved_rows = [
            {
                "movie_id": row["movie_id"],
                "imdb_id": row["imdb_id"],
                "tmdb_id": row["tmdb_id"],
                "wikidata_id": "",
                "wikidata_label": "",
                "wikidata_description": "",
                "release_date": "",
                "genres": "",
                "directors": "",
                "cast_members": "",
                "screenwriters": "",
                "countries": "",
                "languages": "",
                "series": "",
            }
            for row in batch
            if row["imdb_id"] not in matched_ids
        ]
        write_rows(args.output_path, results + unresolved_rows)
        print(f"Processed batch {index}: {min(index * args.batch_size, total)} / {total}")
        time.sleep(args.sleep_seconds)


if __name__ == "__main__":
    main()
