from __future__ import annotations

from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"

RAW_MOVIELENS_DIR = RAW_DIR / "movielens"
RAW_MOVIELENS_32M_DIR = RAW_MOVIELENS_DIR / "ml-32m"
INTERIM_WIKIDATA_DIR = INTERIM_DIR / "wikidata"
PROCESSED_SCREENLOT_DIR = PROCESSED_DIR / "screenlot"

MOVIELENS_32M_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip"
MOVIELENS_32M_MD5_URL = "https://files.grouplens.org/datasets/movielens/ml-32m.zip.md5"
MOVIELENS_32M_README_URL = "https://files.grouplens.org/datasets/movielens/ml-32m-README.html"

MOVIELENS_ARCHIVE = RAW_MOVIELENS_DIR / "ml-32m.zip"
MOVIELENS_MD5_FILE = RAW_MOVIELENS_DIR / "ml-32m.zip.md5"
MOVIELENS_HTML_README = RAW_MOVIELENS_DIR / "ml-32m-README.html"

WIKIDATA_SPARQL_ENDPOINT = "https://query.wikidata.org/sparql"
WIKIDATA_USER_AGENT = "ScreenLotOpenDataPipeline/0.1 (contact: local-development)"


def ensure_data_dirs() -> None:
    for path in (
        RAW_MOVIELENS_DIR,
        INTERIM_WIKIDATA_DIR,
        PROCESSED_SCREENLOT_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def format_imdb_id(value: object) -> str:
    raw = str(value).strip()
    if not raw or raw.lower() == "nan":
        return ""

    if raw.startswith("tt"):
        suffix = raw[2:]
        return f"tt{suffix.zfill(7) if suffix.isdigit() and len(suffix) < 7 else suffix}"

    digits = re.sub(r"\D", "", raw)
    if not digits:
        return ""
    if len(digits) < 7:
        digits = digits.zfill(7)
    return f"tt{digits}"


def extract_release_year(title: str) -> str:
    match = re.search(r"\((\d{4})\)\s*$", title or "")
    return match.group(1) if match else ""

