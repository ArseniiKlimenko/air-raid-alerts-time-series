"""Tests for the analysis engine."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from air_raid_analysis.analysis import ForecastResult, TimeSeriesAnalyzer
from air_raid_analysis.config import settings


def _national(periods: int, base: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic national daily series with weekly seasonality + noise."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-03-01", periods=periods, freq="D")
    weekly = 8 * np.sin(2 * np.pi * np.arange(periods) / 7)
    counts = np.clip(base + weekly + rng.normal(0, 2, periods), 0, None).round()
    return pd.DataFrame({"date": dates, "alert_count": counts.astype(int)})


def test_analyzer_init_zero_values():
    """Test that explicit 0 values are preserved, fixing the falsy 'or' bug."""
    analyzer = TimeSeriesAnalyzer(rolling_window=0, stl_period=0, top_n=0)
    assert analyzer.rolling_window == 0
    assert analyzer.stl_period == 0
    assert analyzer.top_n == 0


def test_decompose_series_short_fallback():
    """Test that STL gracefully returns NaNs when the series is too short."""
    analyzer = TimeSeriesAnalyzer(stl_period=7)
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=10),  # Less than 2 periods (14)
            "alert_count": [1] * 10,
        }
    )
    result = analyzer.decompose_series(df)
    assert np.isnan(result.trend).all()
    assert np.isnan(result.seasonal).all()
    assert np.isnan(result.residual).all()
    assert len(result.observed) == 10
    assert result.method == "none"


def test_decompose_multiseasonal(monkeypatch):
    """With enough data and a small annual period, MSTL is used."""
    monkeypatch.setattr(settings, "annual_period", 30)
    monkeypatch.setattr(settings, "enable_multiseasonal", True)
    analyzer = TimeSeriesAnalyzer(stl_period=7)
    df = _national(periods=200)
    result = analyzer.decompose_series(df)
    assert result.method.startswith("MSTL")
    assert "seasonal_7" in result.seasonal_components
    assert "seasonal_30" in result.seasonal_components
    assert not np.isnan(result.trend).any()


def test_detect_anomalies_flags_spike():
    analyzer = TimeSeriesAnalyzer(rolling_window=7)
    df = _national(periods=90)
    df.loc[80, "alert_count"] = 500  # inject a clear spike
    out = analyzer.detect_anomalies(df)
    assert "is_anomaly" in out.columns
    assert bool(out.loc[80, "is_anomaly"]) is True
    assert int(out["is_anomaly"].sum()) >= 1


def test_hourly_weekday_shape_and_localization():
    analyzer = TimeSeriesAnalyzer()
    # 02:30 UTC == 04:30/05:30 Kyiv depending on DST; ensure no 02:00-UTC binning.
    starts = pd.to_datetime(
        ["2023-07-02 21:30:00+00:00"], utc=True  # Sun 21:30 UTC == Mon 00:30 Kyiv (UTC+3)
    )
    alerts = pd.DataFrame({"started_at": starts})
    pivot = analyzer.analyze_hourly_weekday(alerts)
    assert pivot.shape == (7, 24)
    # 21:30 UTC Sun-night → 00:30 Mon Kyiv → weekday 0, hour 0
    assert pivot.loc[0, 0] == 1
    assert pivot.values.sum() == 1


def test_forecast_returns_result_for_sufficient_series(monkeypatch):
    monkeypatch.setattr(settings, "forecast_backtest_folds", 1)
    analyzer = TimeSeriesAnalyzer()
    df = _national(periods=120)
    result = analyzer.forecast_series(df, horizon=14)
    assert isinstance(result, ForecastResult)
    assert len(result.forecast_values) == 14
    assert len(result.forecast_dates) == 14
    assert (result.forecast_values >= 0).all()
    assert (result.upper >= result.lower).all()


def test_forecast_returns_none_for_short_series():
    analyzer = TimeSeriesAnalyzer()
    df = _national(periods=20)
    assert analyzer.forecast_series(df, horizon=14) is None
