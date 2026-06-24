"""Pydantic v2 data models for air-raid alert records.

Every CSV row is validated through ``RawAlertRow`` → ``AlertRecord`` pipeline.
Invalid rows are collected, not silently dropped.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any

from pydantic import (
    BaseModel,
    Field,
    field_validator,
    model_validator,
)

from air_raid_analysis.config import settings


# ── Enums ────────────────────────────────────────────────────────────────────

class AlertStatus(str, Enum):
    """Derived status of an alert record."""

    COMPLETED = "completed"       # Both start & end present, valid
    ACTIVE = "active"             # finished_at is missing → still ongoing
    ANOMALOUS = "anomalous"       # Duration exceeds MAX_ALERT_DURATION


# ── Raw CSV Row ──────────────────────────────────────────────────────────────

class RawAlertRow(BaseModel):
    """Loose model that mirrors CSV columns 1:1.

    Accepts messy input — empty strings, wrong types — and surfaces
    parsing errors through Pydantic's ``ValidationError``.
    """

    model_config = {"str_strip_whitespace": True}

    region: str
    district: str | None = None
    municipality: str | None = None
    level: str | None = None
    started_at: str = ""
    finished_at: str | None = None

    @field_validator("region")
    @classmethod
    def region_not_empty(cls, v: str) -> str:
        if not v:
            raise ValueError("region must not be empty")
        return v

    @field_validator("finished_at", mode="before")
    @classmethod
    def blank_to_none(cls, v: str | None) -> str | None:
        """Treat empty / whitespace-only strings as None."""
        if isinstance(v, str) and not v.strip():
            return None
        return v


# ── Validated Alert Record ───────────────────────────────────────────────────

class AlertRecord(BaseModel):
    """Fully validated, analysis-ready alert record.

    Validation rules
    ~~~~~~~~~~~~~~~~
    * ``started_at`` ≥ WAR_START (2022-02-24).
    * ``finished_at > started_at`` when both present.
    * Duration ≤ 72 h — otherwise ``status = ANOMALOUS``.
    * Missing ``finished_at`` → ``status = ACTIVE``.
    """

    model_config = {"frozen": True}

    region: str
    started_at: datetime
    finished_at: datetime | None = None
    status: AlertStatus = AlertStatus.COMPLETED
    duration_minutes: float | None = None
    is_active: bool = Field(default=False, description="True if finished_at was missing in source")

    # ── Validators ───────────────────────────────────────────────────────

    @field_validator("started_at")
    @classmethod
    def started_after_war(cls, v: datetime) -> datetime:
        """Reject timestamps before the full-scale invasion."""
        war_start = datetime.strptime(settings.war_start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        
        # Convert naive datetime to UTC to properly compare, avoiding time shift bugs.
        # This matches loader's behavior of assuming unlocalized datetimes are UTC for comparison purposes here,
        # or properly comparing if it already has a timezone.
        if v.tzinfo is None:
            v_compare = v.replace(tzinfo=timezone.utc)
        else:
            v_compare = v.astimezone(timezone.utc)

        if v_compare < war_start:
            raise ValueError(
                f"started_at ({v.isoformat()}) is before the war start "
                f"({war_start.date().isoformat()})"
            )
        return v

    @model_validator(mode="before")
    @classmethod
    def derive_fields(cls, data: dict[str, Any] | Any) -> dict[str, Any] | Any:
        """Compute ``status``, ``duration_minutes``, ``is_active`` before validation."""
        if not isinstance(data, dict):
            return data

        started = data.get("started_at")
        finished = data.get("finished_at")

        # Fallback in case they haven't been parsed yet (though loader passes datetimes)
        if isinstance(started, str):
            try:
                started = datetime.fromisoformat(started.replace("Z", "+00:00"))
            except ValueError:
                pass
        if isinstance(finished, str):
            try:
                finished = datetime.fromisoformat(finished.replace("Z", "+00:00"))
            except ValueError:
                pass

        if finished is None:
            data["is_active"] = True
            data["status"] = AlertStatus.ACTIVE
            data["duration_minutes"] = None
        elif isinstance(started, datetime) and isinstance(finished, datetime):
            if finished <= started:
                raise ValueError(
                    f"finished_at ({finished.isoformat()}) must be "
                    f"after started_at ({started.isoformat()})"
                )
            delta = finished - started
            dur_min = delta.total_seconds() / 60.0
            data["duration_minutes"] = dur_min
            data["is_active"] = False

            max_dur = timedelta(hours=settings.max_alert_duration_hours)
            if delta > max_dur:
                data["status"] = AlertStatus.ANOMALOUS
            else:
                data["status"] = AlertStatus.COMPLETED

        return data


# ── Validation Result Container ──────────────────────────────────────────────

class AlertValidationError(BaseModel):
    """One validation failure — keeps the problematic row index and reason."""

    row_index: int
    reason: str
    raw_data: dict | None = None


class ValidatedDataset(BaseModel):
    """Container returned by the loader after validating every row."""

    records: list[AlertRecord]
    errors: list[AlertValidationError] = Field(default_factory=list)

    @property
    def total_rows(self) -> int:
        return len(self.records) + len(self.errors)

    @property
    def valid_count(self) -> int:
        return len(self.records)

    @property
    def error_count(self) -> int:
        return len(self.errors)

    @property
    def active_count(self) -> int:
        return sum(1 for r in self.records if r.is_active)

    @property
    def anomalous_count(self) -> int:
        return sum(1 for r in self.records if r.status == AlertStatus.ANOMALOUS)

    def summary(self) -> str:
        """Human-readable summary of validation results."""
        lines = [
            "╔══════════════════════════════════════════════╗",
            "║        Dataset Validation Summary            ║",
            "╠══════════════════════════════════════════════╣",
            f"║  Total rows processed:  {self.total_rows:>8,}             ║",
            f"║  Valid records:         {self.valid_count:>8,}  ✓          ║",
            f"║  Validation errors:     {self.error_count:>8,}  ✗          ║",
            f"║  Active (ongoing):      {self.active_count:>8,}  ⏳         ║",
            f"║  Anomalous (>72h):      {self.anomalous_count:>8,}  ⚠          ║",
            "╚══════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)
