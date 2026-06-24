"""Data loader: CSV reading with per-row Pydantic validation.

Handles encoding detection, datetime parsing, and collects
all validation errors without aborting on the first bad row.
"""

from __future__ import annotations

import logging
import zoneinfo
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from pydantic import ValidationError as PydanticValidationError

from air_raid_analysis.config import settings
from air_raid_analysis.models import (
    AlertRecord,
    RawAlertRow,
    ValidatedDataset,
    AlertValidationError,
)

logger = logging.getLogger(__name__)

# Datetime formats we try, in priority order
_DT_FORMATS = [
    "%Y-%m-%d %H:%M:%S%z",       # 2022-03-15 10:30:00+02:00
    "%Y-%m-%dT%H:%M:%S%z",       # ISO 8601
    "%Y-%m-%d %H:%M:%S",         # naive
    "%Y-%m-%dT%H:%M:%S",         # naive ISO
    "%d.%m.%Y %H:%M:%S",         # DD.MM.YYYY
    "%d.%m.%Y %H:%M",            # DD.MM.YYYY HH:MM
]


def _parse_datetime(raw: str | None) -> datetime | None:
    """Try multiple datetime formats; return None if all fail."""
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None

    raw = raw.strip()

    kyiv_tz = zoneinfo.ZoneInfo("Europe/Kyiv")

    for fmt in _DT_FORMATS:
        try:
            dt = datetime.strptime(raw, fmt)
            # Ensure UTC awareness. If naive, assume Europe/Kyiv then convert to UTC.
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=kyiv_tz).astimezone(timezone.utc)
            return dt
        except ValueError:
            continue

    # Fallback: let pandas try
    try:
        dt = pd.to_datetime(raw)
        if dt.tzinfo is None:
            dt = dt.tz_localize("Europe/Kyiv").tz_convert("UTC")
        else:
            dt = dt.tz_convert("UTC")
        return dt.to_pydatetime()
    except (ValueError, TypeError, OverflowError):
        return None


def _read_csv(path: Path) -> pd.DataFrame:
    """Read CSV with encoding auto-detection.

    Tries UTF-8 first, falls back to cp1251 (common for Ukrainian data),
    then latin-1 as a last resort.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1251", "latin-1"]

    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc, dtype=str, keep_default_na=False)
            logger.info("Read CSV with encoding '%s': %d rows, %d columns", enc, len(df), len(df.columns))
            return df
        except UnicodeDecodeError:
            logger.debug("Encoding '%s' failed for %s, trying next …", enc, path)
            continue

    raise ValueError(
        f"Could not read {path} with any of the encodings: {encodings}"
    )


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names to snake_case expected by our models."""
    col_map = {
        "region": "region",
        "Region": "region",
        "oblast": "region",
        "district": "district",
        "District": "district",
        "municipality": "municipality",
        "Municipality": "municipality",
        "level": "level",
        "Level": "level",
        "started_at": "started_at",
        "start": "started_at",
        "Start": "started_at",
        "alert_start": "started_at",
        "finished_at": "finished_at",
        "end": "finished_at",
        "End": "finished_at",
        "alert_end": "finished_at",
    }

    renamed = {}
    for col in df.columns:
        col_stripped = col.strip()
        if col_stripped in col_map:
            renamed[col] = col_map[col_stripped]
        else:
            renamed[col] = col_stripped.lower().replace(" ", "_")

    df = df.rename(columns=renamed)
    return df


def load_and_validate(path: Path | None = None) -> ValidatedDataset:
    """Load CSV and validate every row through Pydantic models.

    Parameters
    ----------
    path : Path, optional
        Path to the CSV file.  Defaults to ``settings.raw_csv_path``.

    Returns
    -------
    ValidatedDataset
        Container with valid ``AlertRecord`` instances and collected errors.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist.
    ValueError
        If the CSV cannot be decoded or has no usable columns.
    """
    path = path or settings.raw_csv_path

    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found: {path}\n"
            f"Run 'python scripts/download_data.py' to fetch it."
        )

    raw_df = _read_csv(path)
    raw_df = _normalize_columns(raw_df)

    # Ensure required columns exist
    required = {"region", "started_at"}
    missing = required - set(raw_df.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {missing}. "
            f"Available columns: {list(raw_df.columns)}"
        )

    records: list[AlertRecord] = []
    errors: list[AlertValidationError] = []

    for row_index, row_dict in raw_df.to_dict(orient="index").items():
        row_index = int(row_index)  # type: ignore[arg-type]

        # Step 1: Parse raw row
        try:
            raw_record = RawAlertRow(**row_dict)
        except PydanticValidationError as e:
            errors.append(
                AlertValidationError(
                    row_index=row_index,
                    reason=f"Raw parsing failed: {e.error_count()} error(s) — {e}",
                    raw_data=row_dict,
                )
            )
            continue

        # Step 2: Parse datetimes
        started = _parse_datetime(raw_record.started_at)
        if started is None:
            errors.append(
                AlertValidationError(
                    row_index=row_index,
                    reason=f"Cannot parse started_at: '{raw_record.started_at}'",
                    raw_data=row_dict,
                )
            )
            continue

        finished = _parse_datetime(raw_record.finished_at)

        # Step 3: Build validated AlertRecord
        try:
            record = AlertRecord(
                region=raw_record.region,
                started_at=started,
                finished_at=finished,
            )
            records.append(record)
        except PydanticValidationError as e:
            errors.append(
                AlertValidationError(
                    row_index=row_index,
                    reason=str(e),
                    raw_data=row_dict,
                )
            )

    result = ValidatedDataset(records=records, errors=errors)
    logger.info("\n%s", result.summary())

    if errors:
        logger.warning(
            "First 5 validation errors:\n%s",
            "\n".join(
                f"  Row {e.row_index}: {e.reason}" for e in errors[:5]
            ),
        )

    return result
