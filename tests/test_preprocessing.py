"""Tests for preprocessing: overlap splitting, capping, aggregation."""

from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from air_raid_analysis.models import AlertRecord, ValidatedDataset
from air_raid_analysis.preprocessing import (
    aggregate_daily_alerts,
    aggregate_daily_national,
    cap_active_alerts,
    remove_duplicates,
    split_alerts_by_day,
)

_KYIV = ZoneInfo("Europe/Kyiv")


def _dt(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=timezone.utc)


def _kyiv(year: int, month: int, day: int, hour: int = 0, minute: int = 0) -> datetime:
    """Build a UTC datetime from a Kyiv wall-clock time.

    Day-level splitting uses local (Kyiv) midnights, so tests read most
    naturally when expressed in Kyiv wall-clock and converted to UTC.
    """
    return datetime(year, month, day, hour, minute, tzinfo=_KYIV).astimezone(timezone.utc)


def _make_dataset(records: list[AlertRecord]) -> ValidatedDataset:
    return ValidatedDataset(records=records, errors=[])


# ═══════════════════════════════════════════════════════════════════════════
#  cap_active_alerts
# ═══════════════════════════════════════════════════════════════════════════

class TestCapActiveAlerts:
    def test_caps_active_alerts_to_dataset_max(self):
        """Active alert's finished_at should be set to max of completed alerts."""
        records = [
            AlertRecord(
                region="Region A",
                started_at=_dt(2023, 1, 1, 10),
                finished_at=_dt(2023, 1, 1, 12),  # completed: max finished
            ),
            AlertRecord(
                region="Region B",
                started_at=_dt(2023, 1, 1, 11),  # starts before ceiling
                finished_at=None,  # active
            ),
        ]
        ds = _make_dataset(records)
        df = cap_active_alerts(ds)

        active_row = df[df["is_active"] == True]  # noqa
        assert len(active_row) == 1
        assert active_row.iloc[0]["is_capped"] == True  # noqa: E712 (np.bool_)
        # finished_at should be set to max of completed: 2023-01-01 12:00 UTC
        assert active_row.iloc[0]["finished_at"] == pd.Timestamp("2023-01-01 12:00:00", tz="UTC")

    def test_caps_active_alerts_when_started_after_ceiling(self):
        """If active alert started after dataset ceiling, cap finished_at to its own started_at."""
        records = [
            AlertRecord(
                region="Region A",
                started_at=_dt(2023, 1, 1, 10),
                finished_at=_dt(2023, 1, 1, 12),  # completed: max finished
            ),
            AlertRecord(
                region="Region B",
                started_at=_dt(2023, 1, 2, 14),  # starts AFTER ceiling
                finished_at=None,  # active
            ),
        ]
        ds = _make_dataset(records)
        df = cap_active_alerts(ds)

        active_row = df[df["is_active"] == True]  # noqa
        # finished_at should be max(started_at, ceiling) to avoid negative duration
        assert active_row.iloc[0]["finished_at"] == pd.Timestamp("2023-01-02 14:00:00", tz="UTC")
        assert active_row.iloc[0]["duration_minutes"] == 0.0

    def test_completed_alerts_not_capped(self):
        records = [
            AlertRecord(
                region="Region A",
                started_at=_dt(2023, 1, 1, 10),
                finished_at=_dt(2023, 1, 1, 11),
            ),
        ]
        df = cap_active_alerts(_make_dataset(records))
        assert df.iloc[0]["is_capped"] == False  # noqa: E712 (np.bool_)

    def test_empty_dataset(self):
        df = cap_active_alerts(_make_dataset([]))
        assert df.empty


# ═══════════════════════════════════════════════════════════════════════════
#  split_alerts_by_day — the overlap bug fix
# ═══════════════════════════════════════════════════════════════════════════

class TestSplitAlertsByDay:
    def test_alert_within_single_day(self):
        """Alert that starts and ends on the same local day → 1 segment."""
        records = [
            AlertRecord(
                region="Region A",
                started_at=_kyiv(2023, 6, 15, 10),
                finished_at=_kyiv(2023, 6, 15, 12),
            ),
        ]
        df = cap_active_alerts(_make_dataset(records))
        split = split_alerts_by_day(df)
        assert len(split) == 1
        assert split.iloc[0]["duration_minutes"] == 120.0
        assert split.iloc[0]["date"] == pd.Timestamp("2023-06-15")

    def test_alert_crossing_local_midnight(self):
        """Alert 23:45 → 02:15 next Kyiv day → 2 segments with correct proportions."""
        records = [
            AlertRecord(
                region="Region A",
                started_at=_kyiv(2023, 6, 15, 23, 45),
                finished_at=_kyiv(2023, 6, 16, 2, 15),
            ),
        ]
        df = cap_active_alerts(_make_dataset(records))
        split = split_alerts_by_day(df)

        assert len(split) == 2

        day1 = split[split["date"] == pd.Timestamp("2023-06-15")]
        day2 = split[split["date"] == pd.Timestamp("2023-06-16")]

        assert len(day1) == 1
        assert len(day2) == 1

        # Day 1: 23:45 → 00:00 = 15 min
        assert day1.iloc[0]["duration_minutes"] == 15.0
        # Day 2: 00:00 → 02:15 = 135 min
        assert day2.iloc[0]["duration_minutes"] == 135.0

    def test_night_alert_attributed_to_local_day(self):
        """A 00:30 Kyiv alert must land on the Kyiv day, not the previous UTC day.

        00:30 Kyiv (summer, UTC+3) == 21:30 UTC of the *previous* date. With
        UTC-day boundaries this alert would be mis-attributed to the day before.
        """
        records = [
            AlertRecord(
                region="Region A",
                started_at=_kyiv(2023, 7, 10, 0, 30),
                finished_at=_kyiv(2023, 7, 10, 1, 0),
            ),
        ]
        df = cap_active_alerts(_make_dataset(records))
        split = split_alerts_by_day(df)
        assert len(split) == 1
        assert split.iloc[0]["date"] == pd.Timestamp("2023-07-10")

    def test_alert_spanning_three_days(self):
        """Alert spanning 3 local days → 3 segments."""
        records = [
            AlertRecord(
                region="Region A",
                started_at=_kyiv(2023, 6, 15, 22),
                finished_at=_kyiv(2023, 6, 17, 6),
            ),
        ]
        df = cap_active_alerts(_make_dataset(records))
        split = split_alerts_by_day(df)

        assert len(split) == 3
        total = split["duration_minutes"].sum()
        # 22:00 June 15 → 06:00 June 17 = 32 hours = 1920 min
        assert abs(total - 1920.0) < 0.01

    def test_total_duration_preserved(self):
        """Splitting must preserve total duration exactly."""
        records = [
            AlertRecord(
                region="R",
                started_at=_kyiv(2023, 3, 1, 18, 30),
                finished_at=_kyiv(2023, 3, 3, 7, 45),
            ),
        ]
        df = cap_active_alerts(_make_dataset(records))
        original_duration = (
            df.iloc[0]["finished_at"] - df.iloc[0]["started_at"]
        ).total_seconds() / 60.0

        split = split_alerts_by_day(df)
        assert abs(split["duration_minutes"].sum() - original_duration) < 0.01


# ═══════════════════════════════════════════════════════════════════════════
#  Aggregation
# ═══════════════════════════════════════════════════════════════════════════

class TestAggregation:
    def _make_split_df(self) -> pd.DataFrame:
        """Create a small split DataFrame for testing aggregation."""
        records = [
            AlertRecord(
                region="Region A",
                started_at=_dt(2023, 1, 1, 10),
                finished_at=_dt(2023, 1, 1, 11),
            ),
            AlertRecord(
                region="Region A",
                started_at=_dt(2023, 1, 1, 14),
                finished_at=_dt(2023, 1, 1, 15),
            ),
            AlertRecord(
                region="Region B",
                started_at=_dt(2023, 1, 1, 10),
                finished_at=_dt(2023, 1, 1, 10, 30),
            ),
        ]
        df = cap_active_alerts(_make_dataset(records))
        return split_alerts_by_day(df)

    def test_daily_alerts_grouping(self):
        split = self._make_split_df()
        daily = aggregate_daily_alerts(split)

        region_a = daily[daily["region"] == "Region A"]
        assert len(region_a) == 1
        assert region_a.iloc[0]["alert_count"] == 2
        assert region_a.iloc[0]["total_minutes"] == 120.0

    def test_national_fills_missing_dates(self):
        split = self._make_split_df()
        # Add an alert 3 days later to create a gap
        records2 = [
            AlertRecord(
                region="Region A",
                started_at=_dt(2023, 1, 4, 10),
                finished_at=_dt(2023, 1, 4, 11),
            ),
        ]
        df2 = cap_active_alerts(_make_dataset(records2))
        split2 = split_alerts_by_day(df2)

        combined = pd.concat([split, split2], ignore_index=True)
        national = aggregate_daily_national(combined)

        # Should have 4 days (Jan 1–4) with zeros for Jan 2–3
        assert len(national) == 4
        assert national.iloc[1]["alert_count"] == 0
        assert national.iloc[2]["alert_count"] == 0


# ═══════════════════════════════════════════════════════════════════════════
#  Deduplication
# ═══════════════════════════════════════════════════════════════════════════

class TestRemoveDuplicates:
    def test_removes_exact_duplicates(self):
        records = [
            AlertRecord(
                region="Region A",
                started_at=_dt(2023, 1, 1, 10),
                finished_at=_dt(2023, 1, 1, 11),
            ),
            AlertRecord(
                region="Region A",
                started_at=_dt(2023, 1, 1, 10),
                finished_at=_dt(2023, 1, 1, 11),
            ),
        ]
        df = cap_active_alerts(_make_dataset(records))
        deduped = remove_duplicates(df)
        assert len(deduped) == 1
