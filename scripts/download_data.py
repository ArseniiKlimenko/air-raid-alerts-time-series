#!/usr/bin/env python3
"""Download the air-raid sirens dataset with 24h caching.

Usage
-----
    python scripts/download_data.py [--force]

Caching
-------
If ``data/raw/sirens.csv`` exists and was modified within the last
``cache_max_age_hours`` (default: 24), the download is skipped.
Use ``--force`` to bypass the cache.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Add project root to sys.path so we can import our package
_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root / "src"))

from air_raid_analysis.config import settings  # noqa: E402  (after sys.path setup)

logger = logging.getLogger(__name__)


def _is_cache_fresh(path: Path, max_age_hours: int) -> bool:
    """Return True if file exists and is younger than *max_age_hours*."""
    if not path.exists():
        return False
    age_seconds = time.time() - path.stat().st_mtime
    age_hours = age_seconds / 3600
    logger.info("Cache age: %.1f hours (max: %d)", age_hours, max_age_hours)
    return age_hours < max_age_hours


def download_dataset(force: bool = False) -> Path:
    """Download the CSV dataset from GitHub.

    Parameters
    ----------
    force : bool
        If True, ignore the cache and re-download.

    Returns
    -------
    Path
        Path to the downloaded CSV file.
    """
    import requests

    target = settings.raw_csv_path

    if not force and _is_cache_fresh(target, settings.cache_max_age_hours):
        logger.info("Cache is fresh — skipping download. Use --force to override.")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Downloading dataset from %s …", settings.dataset_url)
    max_attempts = 3
    response = None
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(settings.dataset_url, timeout=60, stream=True)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == max_attempts:
                raise
            backoff = 2 ** (attempt - 1)
            logger.warning(
                "Download attempt %d/%d failed (%s); retrying in %ds …",
                attempt, max_attempts, exc, backoff,
            )
            time.sleep(backoff)

    content_type = response.headers.get("Content-Type", "")
    if "text/plain" not in content_type and "text/csv" not in content_type:
        logger.warning(
            "Unexpected Content-Type: %s. Expected text/csv or text/plain",
            content_type,
        )

    total_bytes = 0
    with open(target, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            total_bytes += len(chunk)

    if total_bytes < 10240:
        target.unlink(missing_ok=True)
        raise ValueError(f"Downloaded file is suspiciously small ({total_bytes} bytes). Aborting.")

    size_mb = total_bytes / (1024 * 1024)
    logger.info("Downloaded %.2f MB → %s", size_mb, target)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description="Download air-raid sirens dataset.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force download even if cache is fresh",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s │ %(levelname)s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    try:
        path = download_dataset(force=args.force)
        print(f"\n✅ Dataset ready: {path}")
    except Exception as e:
        logger.error("Download failed: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()
