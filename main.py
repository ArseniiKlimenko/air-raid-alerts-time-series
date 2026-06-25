#!/usr/bin/env python3
"""CLI entrypoint — orchestrates the full analysis pipeline.

Usage
-----
    python main.py                          # defaults
    python main.py --csv data/raw/my.csv    # custom input
    python main.py --output results/        # custom output dir
    python main.py --download               # download dataset first
    python main.py --rolling-window 14      # 14-day rolling window
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path

# Ensure the src/ directory is importable
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from air_raid_analysis.analysis import BasicStats, TimeSeriesAnalyzer
from air_raid_analysis.config import settings
from air_raid_analysis.loader import load_and_validate
from air_raid_analysis.preprocessing import preprocess
from air_raid_analysis.visualization import AlertVisualizer

logger = logging.getLogger(__name__)


def _build_report(stats: BasicStats, analysis_results: dict, chart_paths: list[Path]) -> str:
    """Build a plain-text summary report."""
    s = stats
    lines = [
        "=" * 60,
        "  AIR RAID ALERTS — TIME SERIES ANALYSIS REPORT",
        "=" * 60,
        "",
        f"  Date range:        {s.date_range_start} → {s.date_range_end}",
        f"  Total alerts:      {s.total_alerts:,}",
        f"  Unique regions:    {s.total_regions}",
        f"  Active (capped):   {s.active_alerts_count:,}",
        f"  Anomalous (>72h):  {s.anomalous_alerts_count:,}",
        "",
        "  Duration Statistics (minutes)",
        "  ─────────────────────────────",
        f"    Mean:    {s.mean_duration_minutes:>10.1f}",
        f"    Median:  {s.median_duration_minutes:>10.1f}",
        f"    Std Dev: {s.std_duration_minutes:>10.1f}",
        f"    Min:     {s.min_duration_minutes:>10.1f}",
        f"    Max:     {s.max_duration_minutes:>10.1f}",
        "",
        f"  Top {len(s.top_regions)} Regions by Alert Count",
        "  ─────────────────────────────",
    ]
    for rank, (region, count) in enumerate(s.top_regions, start=1):
        lines.append(f"    {rank:>2}. {region:<35s} {count:>8,}")

    # ── Anomalies ────────────────────────────────────────────────────────
    anomalies = analysis_results.get("anomalies")
    if anomalies is not None and not anomalies.empty:
        n_anom = int(anomalies["is_anomaly"].sum())
        lines += [
            "",
            f"  Anomalous Days (>{settings.anomaly_sigma:g}σ): {n_anom}",
            "  ─────────────────────────────",
        ]
        top_days = (
            anomalies[anomalies["is_anomaly"]]
            .nlargest(5, "alert_count")[["date", "alert_count"]]
        )
        for _, r in top_days.iterrows():
            lines.append(f"    • {r['date'].date()}  —  {int(r['alert_count']):>5,} alerts")

    # ── Forecast ─────────────────────────────────────────────────────────
    forecast = analysis_results.get("forecast")
    if forecast is not None:
        m = forecast.metrics
        lines += [
            "",
            f"  Forecast (SARIMAX, {len(forecast.forecast_dates)} days ahead)",
            "  ─────────────────────────────",
            f"    Horizon end:  {forecast.forecast_dates[-1].date()}",
        ]
        if m:
            lines.append(
                f"    Backtest:     MAE={m['mae']:g}  RMSE={m['rmse']:g}  MASE={m['mase']:g}"
            )

    lines += [
        "",
        "  Generated Charts",
        "  ─────────────────────────────",
    ]
    for p in chart_paths:
        lines.append(f"    • {p.name}")

    lines += ["", "=" * 60]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Time Series Analysis of Air Raid Alerts in Ukraine",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Examples:
              python main.py --download
              python main.py --csv data/raw/sirens.csv --rolling-window 14
        """),
    )
    parser.add_argument("--csv", type=Path, default=None, help="Path to CSV file")
    parser.add_argument("--output", type=Path, default=None, help="Output directory")
    parser.add_argument("--download", action="store_true", help="Download dataset first")
    parser.add_argument("--force-download", action="store_true", help="Force re-download")
    parser.add_argument("--rolling-window", type=int, default=None, help="Rolling window (days)")
    parser.add_argument("--top-n", type=int, default=None, help="Number of top regions")
    parser.add_argument("--stl-period", type=int, default=None, help="STL seasonal period (days)")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose logging")

    args = parser.parse_args()

    # Logging setup
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s │ %(levelname)-7s │ %(name)s │ %(message)s",
        datefmt="%H:%M:%S",
    )

    # Override settings if needed
    if args.output:
        settings.output_dir = args.output

    # ── Step 0: Download (optional) ──────────────────────────────────────
    if args.download or args.force_download:
        from scripts.download_data import download_dataset

        download_dataset(force=args.force_download)

    # ── Step 1: Load & validate ──────────────────────────────────────────
    csv_path = args.csv or settings.raw_csv_path
    logger.info("Loading data from %s …", csv_path)

    try:
        dataset = load_and_validate(csv_path)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1
    except ValueError as e:
        logger.error("Data format error: %s", e)
        return 1

    if dataset.valid_count == 0:
        logger.error("No valid records found. Cannot proceed.")
        return 1

    print(dataset.summary())

    # ── Step 2: Preprocess ───────────────────────────────────────────────
    logger.info("Preprocessing …")
    preprocessed = preprocess(dataset)

    # ── Step 3: Analyse ──────────────────────────────────────────────────
    logger.info("Running analysis …")
    analyzer = TimeSeriesAnalyzer(
        rolling_window=args.rolling_window,
        stl_period=args.stl_period,
        top_n=args.top_n,
    )
    results = analyzer.run_all(preprocessed)

    # ── Step 4: Visualise ────────────────────────────────────────────────
    logger.info("Generating charts …")
    viz = AlertVisualizer(output_dir=settings.output_dir)
    chart_paths = viz.generate_all(preprocessed, results)

    # ── Step 5: Report ───────────────────────────────────────────────────
    report = _build_report(results["basic_stats"], results, chart_paths)
    report_path = settings.output_dir / "report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report, encoding="utf-8")

    # ── Step 6: AI analyst (optional) ────────────────────────────────────
    # Runs only when OPENAI_API_KEY is set; otherwise this is a no-op.
    from air_raid_analysis.agent import generate_ai_insights

    logger.info("Running AI analyst …")
    insights = generate_ai_insights(report, results["basic_stats"], results.get("forecast"))
    if insights:
        (settings.output_dir / "ai_insights.md").write_text(insights, encoding="utf-8")
        # Rebuild the dashboard so the briefing appears at the top.
        viz.build_dashboard(viz.figures, insights_md=insights)

    print(f"\n{report}")
    print(f"\n✅ Report saved: {report_path}")
    print(f"✅ Charts saved: {settings.output_dir}/")
    print(f"✅ Dashboard:    {settings.output_dir / 'index.html'}")
    if insights:
        print(f"🤖 AI insights:  {settings.output_dir / 'ai_insights.md'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
