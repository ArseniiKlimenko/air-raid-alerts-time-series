"""Tests for Pydantic data models — validation rules."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

# Ensure src/ is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from air_raid_analysis.models import (
    AlertRecord,
    AlertStatus,
    RawAlertRow,
    ValidatedDataset,
    AlertValidationError as VError,
)


# ═══════════════════════════════════════════════════════════════════════════
#  RawAlertRow tests
# ═══════════════════════════════════════════════════════════════════════════

class TestRawAlertRow:
    def test_valid_row(self):
        row = RawAlertRow(
            region="Kyivska oblast",
            started_at="2022-03-15 10:00:00",
            finished_at="2022-03-15 11:00:00",
        )
        assert row.region == "Kyivska oblast"
        assert row.started_at == "2022-03-15 10:00:00"

    def test_empty_region_raises(self):
        with pytest.raises(ValidationError, match="region must not be empty"):
            RawAlertRow(region="", started_at="2022-03-15 10:00:00")

    def test_whitespace_region_raises(self):
        with pytest.raises(ValidationError, match="region must not be empty"):
            RawAlertRow(region="   ", started_at="2022-03-15 10:00:00")

    def test_blank_finished_at_becomes_none(self):
        row = RawAlertRow(
            region="Lvivska oblast",
            started_at="2022-03-15 10:00:00",
            finished_at="   ",
        )
        assert row.finished_at is None

    def test_optional_fields_default_to_none(self):
        row = RawAlertRow(region="Test", started_at="2022-03-15")
        assert row.district is None
        assert row.municipality is None
        assert row.level is None


# ═══════════════════════════════════════════════════════════════════════════
#  AlertRecord tests
# ═══════════════════════════════════════════════════════════════════════════

class TestAlertRecord:
    def _make_dt(self, year: int, month: int, day: int, hour: int = 0) -> datetime:
        return datetime(year, month, day, hour, tzinfo=timezone.utc)

    def test_valid_completed_alert(self):
        rec = AlertRecord(
            region="Kharkivska oblast",
            started_at=self._make_dt(2022, 5, 10, 8),
            finished_at=self._make_dt(2022, 5, 10, 9),
        )
        assert rec.status == AlertStatus.COMPLETED
        assert rec.is_active is False
        assert rec.duration_minutes == 60.0

    def test_active_alert_no_finished_at(self):
        rec = AlertRecord(
            region="Odeska oblast",
            started_at=self._make_dt(2024, 1, 15, 14),
            finished_at=None,
        )
        assert rec.status == AlertStatus.ACTIVE
        assert rec.is_active is True
        assert rec.duration_minutes is None

    def test_anomalous_alert_over_72h(self):
        rec = AlertRecord(
            region="Donetska oblast",
            started_at=self._make_dt(2023, 6, 1, 0),
            finished_at=self._make_dt(2023, 6, 5, 0),  # 96 hours
        )
        assert rec.status == AlertStatus.ANOMALOUS
        assert rec.duration_minutes == 96 * 60

    def test_finished_before_started_raises(self):
        with pytest.raises(ValidationError, match="must be after"):
            AlertRecord(
                region="Test",
                started_at=self._make_dt(2022, 5, 10, 10),
                finished_at=self._make_dt(2022, 5, 10, 9),
            )

    def test_started_before_war_raises(self):
        with pytest.raises(ValidationError, match="before the war start"):
            AlertRecord(
                region="Test",
                started_at=self._make_dt(2022, 1, 1),
                finished_at=self._make_dt(2022, 1, 1, 1),
            )

    def test_started_after_war_with_non_utc_timezone(self):
        # War start is 2022-02-24 00:00:00 UTC
        # 2022-02-24 01:00:00+03:00 is 2022-02-23 22:00:00 UTC (BEFORE war)
        dt_before = datetime.fromisoformat("2022-02-24T01:00:00+03:00")
        with pytest.raises(ValidationError, match="before the war start"):
            AlertRecord(region="Test", started_at=dt_before)

        # 2022-02-24 00:00:00-05:00 is 2022-02-24 05:00:00 UTC (AFTER war)
        dt_after = datetime.fromisoformat("2022-02-24T00:00:00-05:00")
        dt_after_end = datetime.fromisoformat("2022-02-24T02:00:00-05:00")
        rec = AlertRecord(region="Test", started_at=dt_after, finished_at=dt_after_end)
        assert rec.status == AlertStatus.COMPLETED

    def test_exactly_war_start_is_valid(self):
        rec = AlertRecord(
            region="Test",
            started_at=self._make_dt(2022, 2, 24, 5),
            finished_at=self._make_dt(2022, 2, 24, 6),
        )
        assert rec.status == AlertStatus.COMPLETED

    def test_frozen_model(self):
        rec = AlertRecord(
            region="Test",
            started_at=self._make_dt(2022, 5, 10),
            finished_at=self._make_dt(2022, 5, 10, 1),
        )
        with pytest.raises(ValidationError):
            rec.region = "Changed"


# ═══════════════════════════════════════════════════════════════════════════
#  ValidatedDataset tests
# ═══════════════════════════════════════════════════════════════════════════

class TestValidatedDataset:
    def _make_record(self, region: str = "Test", active: bool = False) -> AlertRecord:
        start = datetime(2023, 1, 1, tzinfo=timezone.utc)
        end = None if active else datetime(2023, 1, 1, 1, tzinfo=timezone.utc)
        return AlertRecord(region=region, started_at=start, finished_at=end)

    def test_counts(self):
        ds = ValidatedDataset(
            records=[self._make_record(), self._make_record(active=True)],
            errors=[VError(row_index=0, reason="bad")],
        )
        assert ds.total_rows == 3
        assert ds.valid_count == 2
        assert ds.error_count == 1
        assert ds.active_count == 1

    def test_summary_output(self):
        ds = ValidatedDataset(records=[self._make_record()])
        summary = ds.summary()
        assert "Validation Summary" in summary
        assert "1" in summary
