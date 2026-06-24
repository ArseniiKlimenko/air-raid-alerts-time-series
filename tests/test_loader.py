"""Tests for the CSV data loader."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest
from datetime import timezone

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from air_raid_analysis.loader import _parse_datetime, load_and_validate


# ═══════════════════════════════════════════════════════════════════════════
#  Datetime parsing
# ═══════════════════════════════════════════════════════════════════════════

class TestParseDatetime:
    def test_iso_with_timezone(self):
        dt = _parse_datetime("2022-03-15T10:30:00+02:00")
        assert dt is not None
        assert dt.year == 2022
        assert dt.month == 3

    def test_space_separated(self):
        dt = _parse_datetime("2022-03-15 10:30:00")
        assert dt is not None
        assert dt.tzinfo == timezone.utc
        assert dt.hour == 8  # 10:30 Kyiv in standard time is 08:30 UTC

    def test_dst_change(self):
        # DST in Ukraine started Mar 27 in 2022
        dt1 = _parse_datetime("2022-03-26 12:00:00")
        dt2 = _parse_datetime("2022-03-28 12:00:00")
        assert dt1.hour == 10  # UTC+2
        assert dt2.hour == 9   # UTC+3

    def test_dot_format(self):
        dt = _parse_datetime("15.03.2022 10:30:00")
        assert dt is not None

    def test_none_returns_none(self):
        assert _parse_datetime(None) is None

    def test_empty_returns_none(self):
        assert _parse_datetime("") is None

    def test_garbage_returns_none(self):
        assert _parse_datetime("not-a-date") is None


# ═══════════════════════════════════════════════════════════════════════════
#  CSV loading
# ═══════════════════════════════════════════════════════════════════════════

class TestLoadAndValidate:
    def test_valid_csv(self, tmp_path: Path):
        csv = tmp_path / "test.csv"
        csv.write_text(textwrap.dedent("""\
            region,started_at,finished_at
            Kyivska oblast,2022-03-15 10:00:00+02:00,2022-03-15 11:00:00+02:00
            Lvivska oblast,2022-04-01 14:00:00+03:00,2022-04-01 15:30:00+03:00
        """))

        result = load_and_validate(csv)
        assert result.valid_count == 2
        assert result.error_count == 0

    def test_invalid_rows_collected(self, tmp_path: Path):
        csv = tmp_path / "test.csv"
        csv.write_text(textwrap.dedent("""\
            region,started_at,finished_at
            ,2022-03-15 10:00:00,2022-03-15 11:00:00
            Good Region,2022-03-15 10:00:00,2022-03-15 11:00:00
            Another,not-a-date,2022-03-15 11:00:00
        """))

        result = load_and_validate(csv)
        # Row 0: empty region → error
        # Row 1: valid
        # Row 2: bad date → error
        assert result.valid_count == 1
        assert result.error_count == 2

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_and_validate(Path("/nonexistent/file.csv"))

    def test_active_alert_no_finished(self, tmp_path: Path):
        csv = tmp_path / "test.csv"
        csv.write_text(textwrap.dedent("""\
            region,started_at,finished_at
            Kharkivska oblast,2024-01-15 14:00:00+02:00,
        """))

        result = load_and_validate(csv)
        assert result.valid_count == 1
        assert result.records[0].is_active is True

    def test_column_name_normalization(self, tmp_path: Path):
        """Test that different column naming conventions are handled."""
        csv = tmp_path / "test.csv"
        csv.write_text(textwrap.dedent("""\
            Region,Start,End
            Test Region,2022-05-01 10:00:00,2022-05-01 11:00:00
        """))

        result = load_and_validate(csv)
        assert result.valid_count == 1

    def test_pre_war_date_rejected(self, tmp_path: Path):
        csv = tmp_path / "test.csv"
        csv.write_text(textwrap.dedent("""\
            region,started_at,finished_at
            Test,2021-01-01 10:00:00,2021-01-01 11:00:00
        """))

        result = load_and_validate(csv)
        assert result.valid_count == 0
        assert result.error_count == 1
