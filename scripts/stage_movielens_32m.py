from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from screenlot.open_data import (  # noqa: E402
    MOVIELENS_32M_MD5_URL,
    MOVIELENS_32M_README_URL,
    MOVIELENS_32M_URL,
    MOVIELENS_ARCHIVE,
    MOVIELENS_HTML_README,
    MOVIELENS_MD5_FILE,
    RAW_MOVIELENS_32M_DIR,
    ensure_data_dirs,
)


def download_file(url: str, destination: Path) -> None:
    request = urllib.request.Request(
        url,
        headers={"User-Agent": "ScreenLotOpenDataPipeline/0.1"},
    )
    with urllib.request.urlopen(request) as response, destination.open("wb") as output:
        shutil.copyfileobj(response, output)


def compute_md5(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def expected_md5(md5_file: Path) -> str:
    return md5_file.read_text().strip().split()[0]


def extract_archive(archive_path: Path, target_dir: Path) -> None:
    with zipfile.ZipFile(archive_path) as archive:
        archive.extractall(target_dir.parent)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download and verify the official MovieLens 32M dataset.")
    parser.add_argument("--force-download", action="store_true", help="Download even if the archive already exists.")
    parser.add_argument("--skip-download", action="store_true", help="Skip the download step and only verify/extract.")
    parser.add_argument("--force-extract", action="store_true", help="Re-extract even if the target folder already exists.")
    args = parser.parse_args()

    ensure_data_dirs()

    if not args.skip_download:
        if args.force_download or not MOVIELENS_ARCHIVE.exists():
            download_file(MOVIELENS_32M_URL, MOVIELENS_ARCHIVE)
        if args.force_download or not MOVIELENS_MD5_FILE.exists():
            download_file(MOVIELENS_32M_MD5_URL, MOVIELENS_MD5_FILE)
        if args.force_download or not MOVIELENS_HTML_README.exists():
            download_file(MOVIELENS_32M_README_URL, MOVIELENS_HTML_README)

    if not MOVIELENS_ARCHIVE.exists():
        raise FileNotFoundError(f"Expected archive at {MOVIELENS_ARCHIVE}")
    if not MOVIELENS_MD5_FILE.exists():
        raise FileNotFoundError(f"Expected checksum file at {MOVIELENS_MD5_FILE}")

    actual = compute_md5(MOVIELENS_ARCHIVE)
    expected = expected_md5(MOVIELENS_MD5_FILE)
    if actual != expected:
        raise RuntimeError(f"MD5 mismatch for {MOVIELENS_ARCHIVE.name}: expected {expected}, got {actual}")

    if args.force_extract or not RAW_MOVIELENS_32M_DIR.exists():
        extract_archive(MOVIELENS_ARCHIVE, RAW_MOVIELENS_32M_DIR)

    print("MovieLens 32M is staged successfully.")
    print(f"Archive: {MOVIELENS_ARCHIVE}")
    print(f"Dataset: {RAW_MOVIELENS_32M_DIR}")
    print(f"MD5: {actual}")


if __name__ == "__main__":
    main()

