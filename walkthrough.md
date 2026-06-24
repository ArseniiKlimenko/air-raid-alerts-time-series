# Walkthrough — Air Raid Alerts Time Series Analysis

## What Was Built

Модульний Python-проєкт для аналізу часових рядів повітряних тривог в Україні —
від валідації «брудного» CSV до прогнозу SARIMAX і єдиного інтерактивного дашборду.

## Project Structure

```
Time Series Analysis of air raid alerts in Ukraine/
├── pyproject.toml                          # Dependencies & config
├── src/air_raid_analysis/
│   ├── __init__.py
│   ├── config.py                           # Pydantic BaseSettings (+ tz, forecast, anomaly knobs)
│   ├── models.py                           # Pydantic v2 data models
│   ├── loader.py                           # CSV loader + validation
│   ├── preprocessing.py                    # Local-day splitting, capping
│   ├── analysis.py                         # MSTL, ACF, rolling, anomalies, hourly, SARIMAX
│   └── visualization.py                    # Plotly charts + unified dashboard
├── scripts/download_data.py                # Dataset downloader with caching + retry
├── main.py                                 # CLI entrypoint
└── tests/
    ├── test_models.py                      # validation rules
    ├── test_loader.py                      # parsing, error collection
    ├── test_preprocessing.py               # local-day splitting, capping, aggregation
    ├── test_analysis.py                    # MSTL, anomalies, hourly, forecast
    └── test_visualization.py               # visualizer construction
```

## Key Design Decisions

### 1. No Median Imputation (Defense-Critical)
Active alerts (`finished_at = None`) are **NOT** filled with median duration. Instead:
- Marked as `is_active = True`
- Capped to `max(finished_at)` of the completed dataset (or own `started_at` if it starts later)
- Explicitly flagged on all charts (diamond markers, separate colour)
- **Excluded** from duration mean/std/median (they carry artificial durations)

### 2. Local-Day (Europe/Kyiv) Overlap Splitting
Day boundaries are **local Kyiv midnights**, not UTC. The source data is in UTC, and
UTC midnight falls at 02:00–03:00 Kyiv time — so night-time alerts (the majority) would
otherwise be mis-attributed to the wrong calendar day. An alert crossing local midnight
(23:45 → 02:15 Kyiv) is split proportionally:
- Day 1: 15 min
- Day 2: 135 min

Total duration is always preserved (tested). The common single-day case is vectorised;
only midnight-crossing alerts use the per-row loop.

### 3. Pydantic v2 Strict Validation
Every CSV row passes through `RawAlertRow` → `AlertRecord`:
- `started_at >= 2022-02-24` (war start), timezone-aware UTC comparison
- `finished_at > started_at`
- Duration > 72h → `status = ANOMALOUS`
- Missing `finished_at` → `status = ACTIVE`
- Bad rows collected in `ValidatedDataset.errors`, never silently dropped

### 4. Multi-Seasonal Decomposition
`MSTL` captures **weekly (7) + annual (365)** seasonality when ≥ 2 years of data are
available; falls back to single-period `STL`, and to a NaN result for very short series.

### 5. Honest Forecasting
`SARIMAX(1,1,1)(1,0,1)₇` forecasts the national daily series. A **rolling-origin
backtest** reports MAE / RMSE / **MASE** (seasonal-naive scaled) *before* the forward
forecast, so accuracy is measured out-of-sample rather than claimed.

### 6. Trailing-Band Anomaly Detection
Days are flagged as anomalous when they exceed a `mean ± σ·anomaly_sigma` band computed
on the **preceding** window (`shift(1)`) — a spike never inflates its own threshold.

## Analyses & Charts

10 interactive Plotly charts, all saved as HTML and combined into a single
self-contained **`index.html`** dashboard:

1. Daily alerts time series
2. Top regions bar chart
3. Region × Month heatmap (alert-minutes)
4. **Hour-of-day × weekday heatmap** (local time) — "when do sirens sound"
5. (M)STL decomposition (4 panels)
6. ACF / PACF
7. Rolling mean with ±2σ band
8. **Anomalous-days** chart (threshold + flagged spikes)
9. **SARIMAX forecast** with 95% interval + backtest metrics
10. Duration distribution (histogram + violin)

## How to Run

```bash
# 1. Setup
cd "Time Series Analysis of air raid alerts in Ukraine"
source .venv/bin/activate

# 2. Download data
python scripts/download_data.py

# 3. Run full pipeline (writes charts + index.html + report.txt)
python main.py

# 4. Run tests
python -m pytest tests/ -q

# 5. Lint
ruff check src/ main.py scripts/ tests/
```

### Useful flags
```bash
python main.py --download                 # fetch dataset first
python main.py --rolling-window 14        # 14-day rolling window
python main.py --stl-period 7 --top-n 15
```

All knobs (timezone, forecast horizon, anomaly σ, annual period…) are also
overridable via `ARA_`-prefixed environment variables — see `config.py`.
