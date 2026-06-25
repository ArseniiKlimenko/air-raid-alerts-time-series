# 🇺🇦 Air Raid Alerts — Time Series Analysis

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![CI](https://github.com/ArseniiKlimenko/air-raid-alerts-time-series/actions/workflows/ci.yml/badge.svg)](https://github.com/ArseniiKlimenko/air-raid-alerts-time-series/actions/workflows/ci.yml)
[![Daily Agent Analytics](https://github.com/ArseniiKlimenko/air-raid-alerts-time-series/actions/workflows/daily_agent.yml/badge.svg)](https://github.com/ArseniiKlimenko/air-raid-alerts-time-series/actions/workflows/daily_agent.yml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A modular Python pipeline that turns the raw [Ukrainian air-raid sirens
dataset](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset) into a
fully interactive analysis dashboard — from strict per-row validation, through
local-time aggregation and seasonal decomposition, to an honestly backtested
SARIMAX forecast — and **redeploys itself daily** via an autonomous GitHub
Actions pipeline.

> **📊 [Live Dashboard](https://arseniiklimenko.github.io/air-raid-alerts-time-series/)** — rebuilt every day from fresh data
>
> **140k+ validated alerts · 25 regions · 2022-02-24 → present**

---

## Highlights

- **Strict, transparent validation** — every CSV row is validated with Pydantic v2;
  bad rows are *collected and reported*, never silently dropped.
- **Defense-honest handling of active alerts** — ongoing alerts (`finished_at = NULL`)
  are **never** filled with a median; they are flagged, capped, and excluded from
  duration statistics.
- **Local-time correctness** — alerts are split across **Europe/Kyiv** calendar days,
  not UTC, so night-time sirens are attributed to the correct Ukrainian day.
- **Multi-seasonal decomposition** — `MSTL` (weekly + annual) with an `STL` fallback.
- **Honest forecasting** — `SARIMAX` with a **rolling-origin backtest** reporting
  MAE / RMSE / **MASE** *before* the forward forecast.
- **Trailing-band anomaly detection** — spikes can't inflate their own threshold.
- **One interactive dashboard** — 10 Plotly charts combined into a single `index.html`.

## Analyses & Charts

| # | Chart | What it shows |
|:--|:--|:--|
| 1 | Daily alerts | National daily alert count over time |
| 2 | Top regions | Most-alerted oblasts |
| 3 | Region × Month heatmap | Alert-minutes intensity by region and month |
| 4 | **Hour × Weekday heatmap** | *When* sirens sound (local time) |
| 5 | **(M)STL decomposition** | Trend + weekly/annual seasonality + residual |
| 6 | ACF / PACF | Periodicity / autocorrelation structure |
| 7 | Rolling mean ±2σ | Smoothed level with volatility band |
| 8 | **Anomalous days** | Spikes beyond a trailing σ-band |
| 9 | **SARIMAX forecast** | 30-day forecast + 95% interval + backtest metrics |
| 10 | Duration distribution | Histogram + violin of alert lengths |

## Quick Start

```bash
git clone https://github.com/ArseniiKlimenko/air-raid-alerts-time-series.git
cd air-raid-alerts-time-series

# Create the environment and install (editable)
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"          # base + lint/test tooling
# pip install -e ".[dev,agent]"  # + optional LLM analyst (needs OPENAI_API_KEY)

# Fetch the dataset (cached for 24h, with retry/backoff)
python scripts/download_data.py

# Run the full pipeline → writes output/index.html + report.txt
python main.py

# Open the dashboard
open output/index.html        # macOS  (xdg-open on Linux)
```

### CLI options

```bash
python main.py --download                 # fetch dataset first
python main.py --csv data/raw/sirens.csv   # custom input
python main.py --output results/           # custom output dir
python main.py --rolling-window 14         # 14-day rolling window
python main.py --stl-period 7 --top-n 15
python main.py -v                          # verbose logging
```

Every tunable knob is also overridable via `ARA_`-prefixed environment variables
(see [`config.py`](src/air_raid_analysis/config.py)) — e.g.
`ARA_ANALYSIS_TIMEZONE`, `ARA_FORECAST_HORIZON_DAYS`, `ARA_ANOMALY_SIGMA`.

## Architecture

```
CSV ─▶ loader ─▶ preprocessing ─▶ analysis ─▶ visualization ─▶ dashboard
      (Pydantic   (Kyiv-day split, (MSTL, ACF,   (Plotly)        (index.html
       validate)   capping, agg)    anomalies,                    + report.txt)
                                    SARIMAX)
```

| Module | Responsibility |
|:--|:--|
| [`config.py`](src/air_raid_analysis/config.py) | Pydantic `BaseSettings` — all knobs, env-overridable |
| [`models.py`](src/air_raid_analysis/models.py) | `RawAlertRow` → `AlertRecord` validation models |
| [`loader.py`](src/air_raid_analysis/loader.py) | Encoding-tolerant CSV reading + per-row validation |
| [`preprocessing.py`](src/air_raid_analysis/preprocessing.py) | Active-alert capping, local-day splitting, aggregation |
| [`analysis.py`](src/air_raid_analysis/analysis.py) | Stats, MSTL, ACF/PACF, rolling, anomalies, hourly, SARIMAX |
| [`visualization.py`](src/air_raid_analysis/visualization.py) | Plotly charts + unified dashboard |

## Agentic Automation Pipeline

This repository is not a static codebase — it is a **self-operating continuous-analytics
system**. Two GitHub Actions workflows turn every push and every day into an automated
cycle with no human in the loop.

```
        ┌─────────────────────────── CI (ci.yml) ───────────────────────────┐
push/PR ─▶ checkout ─▶ setup-py 3.10 (pip cache) ─▶ ruff check ─▶ pytest ─▶ ✅ gate
        └────────────────────────────────────────────────────────────────────┘

        ┌──────────────── Daily Agent Analytics (daily_agent.yml) ───────────┐
cron ───▶ checkout ─▶ install ".[agent]" ─▶ download_data.py --force         │
07:00    ─▶ main.py  (validate → preprocess → MSTL/SARIMAX → LLM analyst)     │
Kyiv     ─▶ upload-pages-artifact(output/) ─▶ deploy-pages ─▶ 🌐 Live Dashboard│
        └────────────────────────────────────────────────────────────────────┘
```

**How it runs autonomously**

- **Trigger** — a `schedule` cron (`0 4 * * *` UTC ≈ 07:00 Kyiv) fires the agent daily;
  `workflow_dispatch` allows on-demand runs. No manual `python main.py` is ever needed.
- **Fresh data** — `download_data.py --force` pulls the latest upstream dataset on every run.
- **Analysis + agent** — `main.py` executes the full numeric pipeline, then the optional
  **LLM analyst** (`agent.py`) synthesises a grounded Markdown briefing. It is **env-gated
  on `OPENAI_API_KEY`** (injected from GitHub Secrets) and **fails open**: missing key,
  missing SDK, or a flaky API call all degrade to a skipped step — the analytics never break.
- **Zero-touch deploy** — the regenerated `output/` (dashboard + report + AI briefing) is
  published with the modern Pages flow (`upload-pages-artifact` → `deploy-pages`) using
  OIDC (`id-token: write`), so no long-lived deploy token is stored.
- **Safe concurrency** — CI cancels superseded runs per-ref; the deploy job uses a `pages`
  concurrency group so two daily runs can never race on a deployment.

The result: the dashboard you see is **rebuilt from reality every morning**, fully hands-off.

## Sample Output

```
  Date range:        2022-03-15 → 2026-06-24
  Total alerts:      140,197
  Unique regions:    25
  Anomalous (>72h):  64

  Duration (min):    mean 117.9 · median 50.5 · max 4319.2
  Anomalous days (>3σ): 57
  Forecast backtest: MAE=52.1  RMSE=70.8  MASE=1.75
```

> **On the forecast:** an honest rolling-origin backtest yields **MASE ≈ 1.75**,
> i.e. the simple `SARIMAX(1,1,1)(1,0,1)₇` model is *worse* than a seasonal-naive
> baseline on a 30-day horizon. Air-raid frequency is highly non-stationary and
> event-driven, so this is the truthful result — the value here is the rigorous
> backtest harness, not a cherry-picked accuracy number.

## Design Decisions

1. **No median imputation** for active alerts — defense data must not be invented.
2. **Local (Kyiv) day boundaries** — UTC midnight is 02:00–03:00 Kyiv; using it
   would mis-attribute the majority of (night-time) alerts.
3. **Clean duration statistics** — anomalous (>72h) and capped alerts are excluded
   from mean/std/median so they don't distort the summary.
4. **Trailing anomaly band** — the threshold uses the *preceding* window (`shift(1)`)
   so a spike never masks itself.
5. **Backtest before forecast** — accuracy is measured out-of-sample, not claimed.

## Testing

```bash
python -m pytest tests/ -q     # 50 tests
```

Coverage spans validation rules, datetime parsing, error collection, local-day
splitting (incl. midnight crossing & DST-aware attribution), duration preservation,
aggregation, MSTL, anomaly detection, hour×weekday localization, the forecast
(both the sufficient-series and too-short-series paths), and the AI analyst's
graceful-skip / Markdown-escaping behaviour.

## Development

```bash
ruff check src/ main.py scripts/ tests/   # lint
```

## Data Source

[`ukrainian-air-raid-sirens-dataset`](https://github.com/Vadimkin/ukrainian-air-raid-sirens-dataset)
by Vadym Klymenko — official air-raid alert records. The dataset is **not** committed
to this repo; it is fetched on demand by `scripts/download_data.py`.

## License

[MIT](LICENSE)
