"""Interactive Plotly visualisations for air-raid alert analysis.

Every method produces a ``plotly.graph_objects.Figure`` and optionally
saves it as an interactive HTML file in the output directory.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from air_raid_analysis.analysis import (
    AutocorrelationResult,
    BasicStats,
    DecompositionResult,
    ForecastResult,
)
from air_raid_analysis.config import settings

_WEEKDAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]

logger = logging.getLogger(__name__)

# ── Colour palette ───────────────────────────────────────────────────────

_PALETTE = {
    "primary": "#1B98E0",
    "secondary": "#E8175D",
    "accent": "#FFB627",
    "bg_dark": "#0D1117",
    "bg_panel": "#161B22",
    "text": "#C9D1D9",
    "grid": "#21262D",
    "active_marker": "#FF6B6B",
    "anomaly_marker": "#FFA500",
}

_LAYOUT_DEFAULTS = dict(
    template="plotly_dark",
    paper_bgcolor=_PALETTE["bg_dark"],
    plot_bgcolor=_PALETTE["bg_panel"],
    font=dict(family="Inter, sans-serif", color=_PALETTE["text"], size=13),
    margin=dict(l=60, r=30, t=60, b=50),
    hovermode="x unified",
)


def _save(fig: go.Figure, name: str, output_dir: Path | None = None) -> Path:
    """Save figure as interactive HTML."""
    out = output_dir or settings.output_dir
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn")
    logger.info("Saved plot → %s", path)
    return path


class AlertVisualizer:
    """Generates interactive Plotly charts for air-raid alert analysis."""

    def __init__(self, output_dir: Path | None = None) -> None:
        self.output_dir = output_dir or settings.output_dir

    # ── 1. Daily alerts time series ──────────────────────────────────────

    def plot_daily_alerts(
        self,
        national_df: pd.DataFrame,
        save: bool = True,
    ) -> go.Figure:
        """Line chart: daily alert count across all regions."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=national_df["date"],
                y=national_df["alert_count"],
                mode="lines",
                name="Daily Alerts",
                line=dict(color=_PALETTE["primary"], width=1),
                fill="tozeroy",
                fillcolor="rgba(27,152,224,0.15)",
            )
        )

        # Mark days with active (capped) alerts
        if "has_active_alerts" in national_df.columns:
            active_days = national_df[national_df["has_active_alerts"] == True]  # noqa: E712
            if not active_days.empty:
                fig.add_trace(
                    go.Scatter(
                        x=active_days["date"],
                        y=active_days["alert_count"],
                        mode="markers",
                        name="Contains capped alerts ⏳",
                        marker=dict(
                            color=_PALETTE["active_marker"],
                            size=5,
                            symbol="diamond",
                        ),
                    )
                )

        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title="🇺🇦 Daily Air Raid Alerts — National Level",
            xaxis_title="Date",
            yaxis_title="Alert Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        if save:
            _save(fig, "daily_alerts", self.output_dir)
        return fig

    # ── 2. Heatmap: region × month ──────────────────────────────────────

    def plot_heatmap(
        self,
        heatmap_df: pd.DataFrame,
        save: bool = True,
    ) -> go.Figure:
        """Heatmap of total alert-minutes: region (rows) × month (cols)."""
        fig = go.Figure(
            go.Heatmap(
                z=heatmap_df.values,
                x=heatmap_df.columns.tolist(),
                y=heatmap_df.index.tolist(),
                colorscale=[
                    [0, "#0D1117"],
                    [0.2, "#1B3A5C"],
                    [0.4, "#1B98E0"],
                    [0.6, "#FFB627"],
                    [0.8, "#E8175D"],
                    [1.0, "#FF3366"],
                ],
                colorbar=dict(title="Minutes"),
                hovertemplate=(
                    "Region: %{y}<br>"
                    "Month: %{x}<br>"
                    "Total minutes: %{z:,.0f}<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title="🗺️ Air Raid Alert Intensity — Region × Month (minutes)",
            xaxis_title="Month",
            yaxis_title="Region",
            height=max(500, len(heatmap_df) * 28),
        )

        if save:
            _save(fig, "heatmap_region_month", self.output_dir)
        return fig

    # ── 3. STL Decomposition ────────────────────────────────────────────

    def plot_decomposition(
        self,
        decomp: DecompositionResult,
        save: bool = True,
    ) -> go.Figure:
        """4-panel STL decomposition: observed, trend, seasonal, residual."""
        fig = make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.04,
            subplot_titles=["Observed", "Trend", "Seasonal", "Residual"],
        )

        dates = decomp.dates

        components = [
            (decomp.observed, _PALETTE["primary"], "Observed"),
            (decomp.trend, _PALETTE["accent"], "Trend"),
            (decomp.seasonal, _PALETTE["secondary"], "Seasonal"),
            (decomp.residual, _PALETTE["text"], "Residual"),
        ]

        for i, (data, color, name) in enumerate(components, start=1):
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=data,
                    mode="lines",
                    name=name,
                    line=dict(color=color, width=1.2),
                ),
                row=i,
                col=1,
            )

        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title="📊 STL Decomposition — Daily Alert Count",
            height=800,
            showlegend=False,
        )

        if save:
            _save(fig, "stl_decomposition", self.output_dir)
        return fig

    # ── 4. ACF / PACF ───────────────────────────────────────────────────

    def plot_acf_pacf(
        self,
        acr: AutocorrelationResult,
        save: bool = True,
    ) -> go.Figure:
        """Side-by-side ACF and PACF bar charts."""
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=["ACF (Autocorrelation)", "PACF (Partial Autocorrelation)"],
        )

        lags = list(range(acr.nlags + 1))

        # ACF
        fig.add_trace(
            go.Bar(
                x=lags,
                y=acr.acf_values,
                marker_color=_PALETTE["primary"],
                name="ACF",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        # Confidence interval for ACF
        ci_upper = acr.acf_confint[:, 1] - acr.acf_values
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=ci_upper,
                mode="lines",
                line=dict(color=_PALETTE["secondary"], dash="dash", width=1),
                name="95% CI",
                showlegend=False,
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=-ci_upper,
                mode="lines",
                line=dict(color=_PALETTE["secondary"], dash="dash", width=1),
                showlegend=False,
            ),
            row=1,
            col=1,
        )

        # PACF
        fig.add_trace(
            go.Bar(
                x=lags,
                y=acr.pacf_values,
                marker_color=_PALETTE["accent"],
                name="PACF",
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        ci_upper_p = acr.pacf_confint[:, 1] - acr.pacf_values
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=ci_upper_p,
                mode="lines",
                line=dict(color=_PALETTE["secondary"], dash="dash", width=1),
                showlegend=False,
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Scatter(
                x=lags,
                y=-ci_upper_p,
                mode="lines",
                line=dict(color=_PALETTE["secondary"], dash="dash", width=1),
                showlegend=False,
            ),
            row=1,
            col=2,
        )

        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title="📈 Autocorrelation Analysis",
            height=400,
        )
        fig.update_xaxes(title_text="Lag (days)", row=1, col=1)
        fig.update_xaxes(title_text="Lag (days)", row=1, col=2)

        if save:
            _save(fig, "acf_pacf", self.output_dir)
        return fig

    # ── 5. Duration Distribution ─────────────────────────────────────────

    def plot_duration_distribution(
        self,
        duration_df: pd.DataFrame,
        save: bool = True,
    ) -> go.Figure:
        """Histogram of alert durations with KDE-style marginal."""
        fig = px.histogram(
            duration_df,
            x="duration_hours",
            color="status",
            nbins=80,
            marginal="violin",
            color_discrete_map={
                "completed": _PALETTE["primary"],
                "active": _PALETTE["active_marker"],
                "anomalous": _PALETTE["anomaly_marker"],
            },
            labels={"duration_hours": "Duration (hours)", "status": "Status"},
            title="⏱️ Alert Duration Distribution",
        )

        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            barmode="overlay",
            xaxis_title="Duration (hours)",
            yaxis_title="Count",
        )
        fig.update_traces(opacity=0.75)

        if save:
            _save(fig, "duration_distribution", self.output_dir)
        return fig

    # ── 6. Rolling Statistics ────────────────────────────────────────────

    def plot_rolling_stats(
        self,
        rolling_df: pd.DataFrame,
        save: bool = True,
    ) -> go.Figure:
        """Rolling mean with ±2σ confidence band."""
        fig = go.Figure()

        # Confidence band
        fig.add_trace(
            go.Scatter(
                x=pd.concat([rolling_df["date"], rolling_df["date"][::-1]]),
                y=pd.concat([rolling_df["upper_band"], rolling_df["lower_band"][::-1]]),
                fill="toself",
                fillcolor="rgba(27,152,224,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                name="±2σ Band",
                hoverinfo="skip",
            )
        )

        # Raw data
        fig.add_trace(
            go.Scatter(
                x=rolling_df["date"],
                y=rolling_df["alert_count"],
                mode="lines",
                name="Daily Count",
                line=dict(color=_PALETTE["text"], width=0.6),
                opacity=0.4,
            )
        )

        # Rolling mean
        fig.add_trace(
            go.Scatter(
                x=rolling_df["date"],
                y=rolling_df["rolling_mean"],
                mode="lines",
                name=f"{settings.rolling_window_days}-day Rolling Mean",
                line=dict(color=_PALETTE["accent"], width=2.5),
            )
        )

        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title=f"📉 {settings.rolling_window_days}-Day Rolling Mean with ±2σ Band",
            xaxis_title="Date",
            yaxis_title="Alert Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )

        if save:
            _save(fig, "rolling_stats", self.output_dir)
        return fig

    # ── 7. Top Regions Bar Chart ─────────────────────────────────────────

    def plot_top_regions(
        self,
        stats: BasicStats,
        save: bool = True,
    ) -> go.Figure:
        """Horizontal bar chart: top regions by total alert count."""
        regions = [r for r, _ in reversed(stats.top_regions)]
        counts = [c for _, c in reversed(stats.top_regions)]

        fig = go.Figure(
            go.Bar(
                x=counts,
                y=regions,
                orientation="h",
                marker=dict(
                    color=counts,
                    colorscale=[
                        [0, _PALETTE["primary"]],
                        [0.5, _PALETTE["accent"]],
                        [1, _PALETTE["secondary"]],
                    ],
                ),
                text=[f"{c:,}" for c in counts],
                textposition="outside",
            )
        )

        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title=f"🏆 Top {len(stats.top_regions)} Regions by Alert Count",
            xaxis_title="Total Alerts",
            yaxis_title="",
            height=max(400, len(regions) * 35),
        )

        if save:
            _save(fig, "top_regions", self.output_dir)
        return fig

    # ── 8. Hour-of-day × weekday heatmap ─────────────────────────────────

    def plot_hourly_weekday(
        self,
        pivot: pd.DataFrame,
        save: bool = True,
    ) -> go.Figure:
        """Heatmap of alert starts by weekday (rows) × hour-of-day (cols)."""
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=[f"{h:02d}" for h in pivot.columns.tolist()],
                y=[_WEEKDAY_NAMES[i] for i in pivot.index.tolist()],
                colorscale=[
                    [0, "#0D1117"],
                    [0.25, "#1B3A5C"],
                    [0.5, "#1B98E0"],
                    [0.75, "#FFB627"],
                    [1.0, "#E8175D"],
                ],
                colorbar=dict(title="Alerts"),
                hovertemplate=(
                    "%{y}, %{x}:00<br>Alerts started: %{z:,}<extra></extra>"
                ),
            )
        )
        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title="🕐 When Sirens Sound — Weekday × Hour (local time)",
            xaxis_title="Hour of day (Europe/Kyiv)",
            yaxis_title="",
            height=420,
        )
        if save:
            _save(fig, "hourly_weekday", self.output_dir)
        return fig

    # ── 9. Anomalous days ────────────────────────────────────────────────

    def plot_anomalies(
        self,
        anomalies_df: pd.DataFrame,
        save: bool = True,
    ) -> go.Figure:
        """Daily series with the rolling anomaly threshold and flagged days."""
        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=anomalies_df["date"],
                y=anomalies_df["anomaly_upper"],
                mode="lines",
                name=f"{settings.anomaly_sigma:g}σ threshold",
                line=dict(color=_PALETTE["accent"], width=1, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=anomalies_df["date"],
                y=anomalies_df["alert_count"],
                mode="lines",
                name="Daily Alerts",
                line=dict(color=_PALETTE["primary"], width=1),
            )
        )
        flagged = anomalies_df[anomalies_df["is_anomaly"]]
        if not flagged.empty:
            fig.add_trace(
                go.Scatter(
                    x=flagged["date"],
                    y=flagged["alert_count"],
                    mode="markers",
                    name=f"Anomaly ({len(flagged)})",
                    marker=dict(color=_PALETTE["secondary"], size=7, symbol="x"),
                )
            )

        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title="🚨 Anomalous Days — Spikes Beyond the Rolling Band",
            xaxis_title="Date",
            yaxis_title="Alert Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        if save:
            _save(fig, "anomalies", self.output_dir)
        return fig

    # ── 10. Forecast ──────────────────────────────────────────────────────

    def plot_forecast(
        self,
        forecast: ForecastResult,
        save: bool = True,
        history_tail: int = 120,
    ) -> go.Figure:
        """History + SARIMAX forecast with a 95% prediction interval."""
        fig = go.Figure()

        # Trim history so the forecast remains readable.
        h_dates = forecast.history_dates[-history_tail:]
        h_vals = forecast.history_values[-history_tail:]

        fig.add_trace(
            go.Scatter(
                x=h_dates, y=h_vals, mode="lines", name="History",
                line=dict(color=_PALETTE["primary"], width=1),
            )
        )
        # Confidence band
        fig.add_trace(
            go.Scatter(
                x=list(forecast.forecast_dates) + list(forecast.forecast_dates[::-1]),
                y=list(forecast.upper) + list(forecast.lower[::-1]),
                fill="toself",
                fillcolor="rgba(255,182,39,0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% interval",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=forecast.forecast_dates, y=forecast.forecast_values,
                mode="lines", name="Forecast",
                line=dict(color=_PALETTE["accent"], width=2.5),
            )
        )

        m = forecast.metrics
        subtitle = (
            f"  ·  backtest MAE={m['mae']:g}, RMSE={m['rmse']:g}, MASE={m['mase']:g}"
            if m else "  ·  (backtest unavailable)"
        )
        fig.update_layout(
            **_LAYOUT_DEFAULTS,
            title=f"🔮 {len(forecast.forecast_dates)}-Day Forecast — SARIMAX{subtitle}",
            xaxis_title="Date",
            yaxis_title="Alert Count",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
        )
        if save:
            _save(fig, "forecast", self.output_dir)
        return fig

    # ── Run all visualisations ───────────────────────────────────────────

    def generate_all(
        self,
        preprocessed: dict[str, pd.DataFrame],
        analysis_results: dict,
    ) -> list[Path]:
        """Generate and save all charts.

        Parameters
        ----------
        preprocessed : dict
            Output of ``preprocessing.preprocess()``.
        analysis_results : dict
            Output of ``TimeSeriesAnalyzer.run_all()``.

        Returns
        -------
        list[Path]
            Paths of all saved HTML files.
        """
        logger.info("Generating visualisations …")

        # (filename, figure) in dashboard order. Optional charts are appended
        # only when the underlying analysis produced data.
        figures: list[tuple[str, go.Figure]] = [
            ("daily_alerts", self.plot_daily_alerts(preprocessed["national"], save=False)),
            ("top_regions", self.plot_top_regions(analysis_results["basic_stats"], save=False)),
            ("heatmap_region_month", self.plot_heatmap(preprocessed["heatmap"], save=False)),
        ]

        hw = analysis_results.get("hourly_weekday")
        if hw is not None and not hw.empty:
            figures.append(("hourly_weekday", self.plot_hourly_weekday(hw, save=False)))

        figures.append(
            ("stl_decomposition", self.plot_decomposition(analysis_results["decomposition"], save=False))
        )
        figures.append(
            ("acf_pacf", self.plot_acf_pacf(analysis_results["autocorrelation"], save=False))
        )
        figures.append(
            ("rolling_stats", self.plot_rolling_stats(analysis_results["rolling_stats"], save=False))
        )

        anomalies = analysis_results.get("anomalies")
        if anomalies is not None and not anomalies.empty:
            figures.append(("anomalies", self.plot_anomalies(anomalies, save=False)))

        forecast = analysis_results.get("forecast")
        if forecast is not None:
            figures.append(("forecast", self.plot_forecast(forecast, save=False)))

        figures.append(
            ("duration_distribution",
             self.plot_duration_distribution(analysis_results["duration_distribution"], save=False))
        )

        paths = [_save(fig, name, self.output_dir) for name, fig in figures]

        dashboard = self.build_dashboard([fig for _, fig in figures])
        paths.append(dashboard)

        logger.info("All %d charts + dashboard saved to %s", len(figures), self.output_dir)
        return paths

    # ── Unified dashboard ────────────────────────────────────────────────

    def build_dashboard(self, figures: list[go.Figure]) -> Path:
        """Combine all figures into a single self-contained ``index.html``.

        Plotly.js is loaded once from the CDN; each subsequent figure is
        embedded as a plain ``<div>`` to keep the file small.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        blocks: list[str] = []
        for i, fig in enumerate(figures):
            blocks.append(
                fig.to_html(
                    full_html=False,
                    include_plotlyjs="cdn" if i == 0 else False,
                    default_width="100%",
                )
            )

        body = "\n".join(f'<section class="card">{b}</section>' for b in blocks)
        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Air Raid Alerts — Analysis Dashboard</title>
<style>
  :root {{ color-scheme: dark; }}
  body {{ margin:0; background:{_PALETTE['bg_dark']}; color:{_PALETTE['text']};
         font-family:Inter, system-ui, sans-serif; }}
  header {{ padding:28px 24px 8px; }}
  header h1 {{ margin:0; font-size:26px; }}
  header p {{ margin:6px 0 0; opacity:.7; font-size:14px; }}
  main {{ max-width:1200px; margin:0 auto; padding:16px 16px 48px; }}
  .card {{ background:{_PALETTE['bg_panel']}; border:1px solid {_PALETTE['grid']};
          border-radius:12px; margin:18px 0; padding:8px; }}
  footer {{ text-align:center; opacity:.5; font-size:12px; padding:24px; }}
</style>
</head>
<body>
<header>
  <h1>🇺🇦 Air Raid Alerts — Time Series Dashboard</h1>
  <p>Interactive analysis of air-raid sirens in Ukraine · day boundaries in Europe/Kyiv local time</p>
</header>
<main>
{body}
</main>
<footer>Generated by air-raid-analysis · plotly</footer>
</body>
</html>"""
        path = self.output_dir / "index.html"
        path.write_text(html, encoding="utf-8")
        logger.info("Saved dashboard → %s", path)
        return path
