"""Time-series analysis engine.

Provides ``TimeSeriesAnalyzer`` — a stateless class that takes preprocessed
DataFrames and returns analysis results (dicts / DataFrames) that the
visualisation layer consumes.
"""

from __future__ import annotations

import logging
import warnings
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from statsmodels.tsa.seasonal import MSTL, STL
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.stattools import acf, pacf

from air_raid_analysis.config import settings

logger = logging.getLogger(__name__)


@dataclass
class DecompositionResult:
    """Container for (M)STL decomposition output.

    ``seasonal`` is the *combined* seasonal signal (sum of all seasonal
    components). For multi-seasonal decomposition each individual component
    (e.g. weekly, annual) is also kept in ``seasonal_components``.
    """

    dates: pd.DatetimeIndex
    observed: np.ndarray
    trend: np.ndarray
    seasonal: np.ndarray
    residual: np.ndarray
    method: str = "STL"
    seasonal_components: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass
class ForecastResult:
    """Container for a SARIMAX forecast plus its backtest evaluation."""

    history_dates: pd.DatetimeIndex
    history_values: np.ndarray
    forecast_dates: pd.DatetimeIndex
    forecast_values: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    order: tuple
    seasonal_order: tuple
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass
class AutocorrelationResult:
    """Container for ACF / PACF output."""

    acf_values: np.ndarray
    acf_confint: np.ndarray
    pacf_values: np.ndarray
    pacf_confint: np.ndarray
    nlags: int


@dataclass
class BasicStats:
    """Summary statistics for the dataset."""

    total_alerts: int = 0
    total_regions: int = 0
    date_range_start: str = ""
    date_range_end: str = ""
    mean_duration_minutes: float = 0.0
    median_duration_minutes: float = 0.0
    std_duration_minutes: float = 0.0
    max_duration_minutes: float = 0.0
    min_duration_minutes: float = 0.0
    active_alerts_count: int = 0
    anomalous_alerts_count: int = 0
    top_regions: list[tuple[str, int]] = field(default_factory=list)


class TimeSeriesAnalyzer:
    """Stateless analysis engine.

    All methods accept DataFrames produced by the preprocessing module
    and return structured results.
    """

    def __init__(
        self,
        rolling_window: int | None = None,
        stl_period: int | None = None,
        top_n: int | None = None,
    ) -> None:
        self.rolling_window = rolling_window if rolling_window is not None else settings.rolling_window_days
        self.stl_period = stl_period if stl_period is not None else settings.stl_period
        self.top_n = top_n if top_n is not None else settings.top_n_regions

    # ── Basic Stats ──────────────────────────────────────────────────────

    def compute_basic_stats(self, alerts_df: pd.DataFrame) -> BasicStats:
        """Compute summary statistics from the cleaned alerts DataFrame.

        Parameters
        ----------
        alerts_df : pd.DataFrame
            Output of ``preprocessing.cap_active_alerts`` (one row per alert).
        """
        if alerts_df.empty:
            return BasicStats()

        # Duration statistics are computed on *clean* alerts only: anomalous
        # (>72h) records and capped active alerts carry artificial durations
        # that would inflate the mean / std, so they are excluded here. They
        # are still surfaced separately via the active / anomalous counts and
        # the duration-distribution chart.
        clean_mask = alerts_df["duration_minutes"].notna() & (
            alerts_df["status"] == "completed"
        )
        if "is_capped" in alerts_df.columns:
            clean_mask &= ~alerts_df["is_capped"].astype(bool)
        durations = alerts_df.loc[clean_mask, "duration_minutes"]
        region_counts = alerts_df["region"].value_counts()

        return BasicStats(
            total_alerts=len(alerts_df),
            total_regions=alerts_df["region"].nunique(),
            date_range_start=str(alerts_df["started_at"].min().date()),
            date_range_end=str(alerts_df["started_at"].max().date()),
            mean_duration_minutes=round(durations.mean(), 2) if len(durations) else 0.0,
            median_duration_minutes=round(durations.median(), 2) if len(durations) else 0.0,
            std_duration_minutes=round(durations.std(), 2) if len(durations) else 0.0,
            max_duration_minutes=round(durations.max(), 2) if len(durations) else 0.0,
            min_duration_minutes=round(durations.min(), 2) if len(durations) else 0.0,
            active_alerts_count=int(alerts_df["is_active"].sum()),
            anomalous_alerts_count=int(
                (alerts_df["status"] == "anomalous").sum()
            ),
            top_regions=[
                (region, int(count))
                for region, count in region_counts.head(self.top_n).items()
            ],
        )

    # ── STL Decomposition ────────────────────────────────────────────────

    def decompose_series(
        self,
        national_df: pd.DataFrame,
        column: str = "alert_count",
    ) -> DecompositionResult:
        """Run STL decomposition on a national-level daily time series.

        Parameters
        ----------
        national_df : pd.DataFrame
            Output of ``preprocessing.aggregate_daily_national``.
        column : str
            Column to decompose (default: ``alert_count``).

        Returns
        -------
        DecompositionResult
        """
        series = national_df.set_index("date")[column].astype(float)
        series.index = pd.DatetimeIndex(series.index, freq="D")
        n = len(series)

        # STL requires at least 2 full periods of the *shortest* season.
        min_length = self.stl_period * 2
        if n < min_length:
            logger.warning(
                "Series too short (%d) for STL with period=%d. "
                "Minimum required is %d. Returning NaN decomposition.",
                n,
                self.stl_period,
                min_length,
            )
            nans = np.full(n, np.nan)
            return DecompositionResult(
                dates=series.index,
                observed=series.values,
                trend=nans,
                seasonal=nans,
                residual=nans,
                method="none",
            )

        # Decide which seasonal periods we can afford. Annual seasonality is
        # only meaningful (and only fittable) with at least two full years.
        periods = [self.stl_period]
        if settings.enable_multiseasonal and n >= 2 * settings.annual_period:
            periods.append(settings.annual_period)

        if len(periods) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = MSTL(series, periods=periods).fit()
            seasonal_df = result.seasonal  # DataFrame: one column per period
            components = {
                f"seasonal_{p}": seasonal_df.iloc[:, i].values
                for i, p in enumerate(periods)
            }
            combined_seasonal = seasonal_df.sum(axis=1).values
            method = f"MSTL(periods={periods})"
            trend = result.trend.values
            resid = result.resid.values
        else:
            result = STL(series, period=self.stl_period, robust=True).fit()
            combined_seasonal = result.seasonal.values
            components = {f"seasonal_{self.stl_period}": combined_seasonal}
            method = f"STL(period={self.stl_period})"
            trend = result.trend.values
            resid = result.resid.values

        logger.info("Decomposition method: %s", method)
        return DecompositionResult(
            dates=series.index,
            observed=series.values,
            trend=trend,
            seasonal=combined_seasonal,
            residual=resid,
            method=method,
            seasonal_components=components,
        )

    # ── Autocorrelation ──────────────────────────────────────────────────

    def compute_autocorrelation(
        self,
        national_df: pd.DataFrame,
        column: str = "alert_count",
        nlags: int = 40,
    ) -> AutocorrelationResult:
        """Compute ACF and PACF for periodicity detection.

        Parameters
        ----------
        national_df : pd.DataFrame
            National daily series.
        column : str
            Column to analyse.
        nlags : int
            Number of lags.

        Returns
        -------
        AutocorrelationResult
        """
        series = national_df[column].astype(float).values

        # Cap nlags to series length - 1
        max_lags = min(nlags, len(series) // 2 - 1)
        if max_lags < 1:
            max_lags = 1

        acf_vals, acf_ci = acf(series, nlags=max_lags, alpha=0.05)
        pacf_vals, pacf_ci = pacf(series, nlags=max_lags, alpha=0.05)

        return AutocorrelationResult(
            acf_values=acf_vals,
            acf_confint=acf_ci,
            pacf_values=pacf_vals,
            pacf_confint=pacf_ci,
            nlags=max_lags,
        )

    # ── Rolling Statistics ───────────────────────────────────────────────

    def compute_rolling_stats(
        self,
        national_df: pd.DataFrame,
        column: str = "alert_count",
    ) -> pd.DataFrame:
        """Compute rolling mean and std for the national daily series.

        Parameters
        ----------
        national_df : pd.DataFrame
            National daily series.
        column : str
            Column to compute rolling stats for.

        Returns
        -------
        pd.DataFrame
            Original columns plus rolling_mean, rolling_std,
            upper_band, lower_band.
        """
        df = national_df.copy()
        w = self.rolling_window

        df["rolling_mean"] = df[column].rolling(window=w, min_periods=1).mean().round(2)
        df["rolling_std"] = df[column].rolling(window=w, min_periods=1).std().round(2)
        df["upper_band"] = (df["rolling_mean"] + 2 * df["rolling_std"]).round(2)
        df["lower_band"] = (df["rolling_mean"] - 2 * df["rolling_std"]).clip(lower=0).round(2)

        return df

    # ── Duration Distribution ────────────────────────────────────────────

    def analyze_duration_distribution(
        self,
        alerts_df: pd.DataFrame,
    ) -> pd.DataFrame:
        """Prepare duration data for histogram / KDE visualisation.

        Filters out NaN durations and returns a DataFrame with
        ``duration_minutes``, ``duration_hours``, ``status``, ``region``.
        """
        df = alerts_df[alerts_df["duration_minutes"].notna()].copy()
        df["duration_hours"] = (df["duration_minutes"] / 60.0).round(3)
        return df[["region", "duration_minutes", "duration_hours", "status"]].reset_index(
            drop=True
        )

    # ── Day-level anomaly detection ───────────────────────────────────────

    def detect_anomalies(
        self,
        national_df: pd.DataFrame,
        column: str = "alert_count",
    ) -> pd.DataFrame:
        """Flag days whose value exceeds a *trailing* ``±anomaly_sigma·σ`` band.

        The baseline mean/std are computed over the preceding window only
        (``shift(1)``) so that a spike never inflates its own threshold —
        otherwise a single large day would mask itself.

        Returns the national series augmented with ``rolling_mean``,
        ``rolling_std``, ``anomaly_upper`` and a boolean ``is_anomaly`` column.
        """
        df = national_df.copy()
        w = self.rolling_window
        min_p = max(3, w // 2)

        baseline = df[column].shift(1)
        df["rolling_mean"] = baseline.rolling(window=w, min_periods=min_p).mean().round(2)
        df["rolling_std"] = baseline.rolling(window=w, min_periods=min_p).std().round(2)

        sigma = settings.anomaly_sigma
        df["anomaly_upper"] = (df["rolling_mean"] + sigma * df["rolling_std"]).round(2)
        df["is_anomaly"] = (df[column] > df["anomaly_upper"]) & df["rolling_std"].gt(0)
        n_anom = int(df["is_anomaly"].sum())
        logger.info("Detected %d anomalous day(s) at %.1fσ.", n_anom, sigma)
        return df

    # ── Hour-of-day × weekday profile ─────────────────────────────────────

    def analyze_hourly_weekday(
        self,
        alerts_df: pd.DataFrame,
        tz: str | None = None,
    ) -> pd.DataFrame:
        """Count alert *starts* by (weekday, hour) in local time.

        Returns a 7×24 pivot (rows = weekday 0=Mon … 6=Sun, cols = hour
        0–23) — ideal for a "when do sirens sound" heatmap.
        """
        if alerts_df.empty:
            return pd.DataFrame()

        tzinfo = tz or settings.analysis_timezone
        starts = pd.to_datetime(alerts_df["started_at"], utc=True).dt.tz_convert(tzinfo)
        tmp = pd.DataFrame({"weekday": starts.dt.dayofweek, "hour": starts.dt.hour})

        pivot = (
            tmp.groupby(["weekday", "hour"]).size().unstack(fill_value=0)
            .reindex(index=range(7), columns=range(24), fill_value=0)
        )
        return pivot.astype(int)

    # ── Forecasting (SARIMAX + rolling-origin backtest) ───────────────────

    def _fit_sarimax(self, series: pd.Series, order: tuple, seasonal_order: tuple):
        """Fit SARIMAX with relaxed constraints; warnings suppressed."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                series,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            return model.fit(disp=False)

    def _backtest(
        self,
        series: pd.Series,
        order: tuple,
        seasonal_order: tuple,
        horizon: int,
        folds: int,
    ) -> dict[str, float]:
        """Rolling-origin backtest. Returns MAE / RMSE / MASE averaged over folds."""
        errors: list[np.ndarray] = []
        scales: list[float] = []
        season = seasonal_order[3] or 1

        for k in range(folds, 0, -1):
            cut = len(series) - k * horizon
            if cut <= 2 * season + 2:
                continue
            train = series.iloc[:cut]
            test = series.iloc[cut: cut + horizon]
            if test.empty:
                continue
            try:
                fitted = self._fit_sarimax(train, order, seasonal_order)
                pred = fitted.forecast(steps=len(test)).to_numpy()
            except Exception as exc:  # noqa: BLE001 — backtest must never crash the run
                logger.warning("Backtest fold (k=%d) failed: %s", k, exc)
                continue
            actual = test.to_numpy()
            errors.append(np.abs(pred - actual))
            # Seasonal-naive scale for MASE: mean in-sample |y_t - y_{t-season}|.
            tr = train.to_numpy()
            naive = np.abs(tr[season:] - tr[:-season]) if tr.size > season else np.array([])
            scales.append(naive.mean() if naive.size else np.nan)

        if not errors:
            return {}

        all_err = np.concatenate(errors)
        mae = float(np.mean(all_err))
        rmse = float(np.sqrt(np.mean(all_err ** 2)))
        scale = float(np.nanmean(scales)) if scales else np.nan
        mase = float(mae / scale) if scale and scale > 0 else float("nan")
        return {"mae": round(mae, 3), "rmse": round(rmse, 3), "mase": round(mase, 3)}

    def forecast_series(
        self,
        national_df: pd.DataFrame,
        column: str = "alert_count",
        horizon: int | None = None,
    ) -> ForecastResult | None:
        """Forecast the national daily series with a weekly-seasonal SARIMAX.

        Runs a rolling-origin backtest first (for honest error metrics), then
        refits on the full history to produce the forward forecast with a 95%
        interval. Returns ``None`` if the series is too short to model.
        """
        series = national_df.set_index("date")[column].astype(float)
        series.index = pd.DatetimeIndex(series.index, freq="D")
        n = len(series)

        horizon = horizon if horizon is not None else settings.forecast_horizon_days
        order = (1, 1, 1)
        seasonal_order = (1, 0, 1, 7)
        min_train = max(2 * seasonal_order[3] + horizon, 30)

        if n < min_train:
            logger.warning(
                "Series too short (%d < %d) for SARIMAX forecast — skipping.", n, min_train
            )
            return None

        folds = min(settings.forecast_backtest_folds, max(0, (n - min_train) // horizon + 1))
        metrics = self._backtest(series, order, seasonal_order, horizon, folds) if folds else {}

        try:
            fitted = self._fit_sarimax(series, order, seasonal_order)
            fc = fitted.get_forecast(steps=horizon)
            mean = np.clip(fc.predicted_mean.to_numpy(), 0, None)
            ci = fc.conf_int(alpha=0.05).to_numpy()
        except Exception as exc:  # noqa: BLE001
            logger.warning("SARIMAX fit failed: %s — skipping forecast.", exc)
            return None

        future_dates = pd.date_range(
            series.index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D"
        )
        return ForecastResult(
            history_dates=series.index,
            history_values=series.to_numpy(),
            forecast_dates=future_dates,
            forecast_values=mean,
            lower=np.clip(ci[:, 0], 0, None),
            upper=ci[:, 1],
            order=order,
            seasonal_order=seasonal_order,
            metrics=metrics,
        )

    # ── Convenience: run all analyses ────────────────────────────────────

    def run_all(self, preprocessed: dict[str, pd.DataFrame]) -> dict:
        """Execute the full analysis suite.

        Parameters
        ----------
        preprocessed : dict
            Output of ``preprocessing.preprocess()``.

        Returns
        -------
        dict
            Keys: basic_stats, decomposition, autocorrelation,
            rolling_stats, duration_distribution.
        """
        alerts = preprocessed["alerts"]
        national = preprocessed["national"]

        logger.info("Computing basic stats …")
        stats = self.compute_basic_stats(alerts)

        logger.info("Running seasonal decomposition …")
        decomposition = self.decompose_series(national)

        logger.info("Computing autocorrelation (ACF/PACF) …")
        autocorrelation = self.compute_autocorrelation(national)

        logger.info("Computing rolling statistics …")
        rolling = self.compute_rolling_stats(national)

        logger.info("Detecting day-level anomalies …")
        anomalies = self.detect_anomalies(national)

        logger.info("Analyzing hour-of-day × weekday profile …")
        hourly_weekday = self.analyze_hourly_weekday(alerts)

        logger.info("Analyzing duration distribution …")
        durations = self.analyze_duration_distribution(alerts)

        logger.info("Forecasting national series (SARIMAX) …")
        forecast = self.forecast_series(national)

        logger.info("All analyses complete.")
        return {
            "basic_stats": stats,
            "decomposition": decomposition,
            "autocorrelation": autocorrelation,
            "rolling_stats": rolling,
            "anomalies": anomalies,
            "hourly_weekday": hourly_weekday,
            "duration_distribution": durations,
            "forecast": forecast,
        }
