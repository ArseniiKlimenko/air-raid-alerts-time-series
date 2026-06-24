"""Data preprocessing: cleaning, overlap splitting, aggregation.

Key design decisions (per user requirements)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
1. **No median imputation** for missing ``finished_at``.
   Active alerts get ``finished_at = dataset_ceiling`` (the max observed
   ``finished_at`` across the whole dataset) and are explicitly flagged.

2. **Proportional overlap splitting**: a single alert crossing midnight
   is split into day-level segments so heatmaps are accurate.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd

from air_raid_analysis.config import settings
from air_raid_analysis.models import AlertStatus, ValidatedDataset

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1.  Active-alert ceiling
# ═══════════════════════════════════════════════════════════════════════════

def cap_active_alerts(dataset: ValidatedDataset) -> pd.DataFrame:
    """Convert validated records to a DataFrame and cap active alerts.

    For records where ``is_active=True`` the ``finished_at`` is set to
    ``dataset_ceiling`` — the maximum ``finished_at`` among *completed*
    records.  This keeps the data honest: we never invent a duration,
    we just say "at least this long for the purpose of current analysis".

    An ``is_active`` boolean column is preserved so downstream code can
    filter or annotate these rows on charts.

    Returns
    -------
    pd.DataFrame
        Columns: region, started_at, finished_at, duration_minutes,
                 status, is_active, is_capped.
    """
    if not dataset.records:
        logger.warning("Empty dataset — nothing to process.")
        return pd.DataFrame()

    rows = [
        {
            "region": r.region,
            "started_at": r.started_at,
            "finished_at": r.finished_at,
            "duration_minutes": r.duration_minutes,
            "status": r.status.value,
            "is_active": r.is_active,
        }
        for r in dataset.records
    ]
    df = pd.DataFrame(rows)
    df["started_at"] = pd.to_datetime(df["started_at"], utc=True)
    df["finished_at"] = pd.to_datetime(df["finished_at"], utc=True)

    # Compute ceiling from completed alerts only
    completed_mask = df["status"] == AlertStatus.COMPLETED.value
    if completed_mask.any():
        dataset_ceiling = df.loc[completed_mask, "finished_at"].max()
    else:
        dataset_ceiling = pd.Timestamp.now(tz="UTC")

    logger.info("Dataset ceiling for active alerts: %s", dataset_ceiling)

    # Cap active alerts
    active_mask = df["is_active"]
    df["is_capped"] = False
    if active_mask.any():
        n_active = active_mask.sum()
        logger.info(
            "Capping %d active alert(s) with finished_at = %s",
            n_active,
            dataset_ceiling,
        )
        capped_finished = df.loc[active_mask, "started_at"].clip(lower=dataset_ceiling)
        df.loc[active_mask, "finished_at"] = capped_finished
        df.loc[active_mask, "is_capped"] = True
        # Recompute duration for capped rows
        df.loc[active_mask, "duration_minutes"] = (
            (df.loc[active_mask, "finished_at"] - df.loc[active_mask, "started_at"])
            .dt.total_seconds()
            / 60.0
        )

    return df


# ═══════════════════════════════════════════════════════════════════════════
# 2.  Overlap splitting — proportional day-level distribution
# ═══════════════════════════════════════════════════════════════════════════

def _split_single_alert_by_day(
    region: str,
    started_at: datetime,
    finished_at: datetime,
    is_active: bool,
    is_capped: bool,
) -> list[dict]:
    """Split one (already local-time) alert into local-day segments.

    ``started_at`` / ``finished_at`` are expected to be timezone-aware
    datetimes expressed in the *analysis* timezone (Europe/Kyiv). Day
    boundaries are therefore *local* midnights, so a night-time alert is
    attributed to the correct Ukrainian calendar day.

    An alert 23:45 Mon → 02:15 Tue (Kyiv) becomes:
    * Mon segment: 15 min
    * Tue segment: 135 min (2h 15min)

    Returns one dict per local calendar day the alert spans.
    """
    segments: list[dict] = []
    current = started_at

    while current < finished_at:
        # End of the current *local* calendar day (next local midnight).
        day_end = (current + timedelta(days=1)).replace(
            hour=0, minute=0, second=0, microsecond=0,
        )
        segment_end = min(day_end, finished_at)
        duration_min = (segment_end - current).total_seconds() / 60.0

        if duration_min > 0:
            segments.append(
                {
                    "region": region,
                    "date": current.date(),
                    "duration_minutes": round(duration_min, 2),
                    "is_active": is_active,
                    "is_capped": is_capped,
                    "alert_start": started_at,
                    "alert_end": finished_at,
                }
            )
        current = segment_end

    return segments


def split_alerts_by_day(df: pd.DataFrame, tz: str | None = None) -> pd.DataFrame:
    """Split all alerts proportionally across *local* calendar days.

    This solves two issues at once:

    * **Overlap bug** — an alert crossing midnight is split so each day
      receives its fair share of alert-minutes (total is preserved).
    * **Timezone bug** — boundaries are local (Europe/Kyiv) midnights, not
      UTC, so night-time alerts land on the correct Ukrainian day.

    The common case (an alert that starts and ends on the same local day)
    is handled in a single vectorised pass; only multi-day alerts fall back
    to the per-row Python loop.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`cap_active_alerts`.
    tz : str, optional
        IANA timezone for day boundaries. Defaults to
        ``settings.analysis_timezone``.

    Returns
    -------
    pd.DataFrame
        One row per (region, date) segment. Columns: region, date,
        duration_minutes, is_active, is_capped, alert_start, alert_end.
    """
    if df.empty:
        return pd.DataFrame()

    tzinfo = ZoneInfo(tz or settings.analysis_timezone)

    work = df.copy()
    valid_mask = work["finished_at"].notna()
    skipped = int((~valid_mask).sum())
    work = work[valid_mask]
    if work.empty:
        if skipped:
            logger.warning("Skipped %d rows with NaT finished_at during split.", skipped)
        return pd.DataFrame()

    # Express both endpoints in local wall-clock time.
    start_local = work["started_at"].dt.tz_convert(tzinfo)
    end_local = work["finished_at"].dt.tz_convert(tzinfo)
    start_day = start_local.dt.normalize()
    end_day = (end_local - pd.Timedelta(nanoseconds=1)).dt.normalize()

    single_day = start_day.eq(end_day)

    # ── Fast path: alerts that live entirely within one local day ────────
    fast = pd.DataFrame(
        {
            "region": work["region"].to_numpy(),
            "date": start_day.dt.tz_localize(None).to_numpy(),
            "duration_minutes": (
                (end_local - start_local).dt.total_seconds() / 60.0
            ).round(2).to_numpy(),
            "is_active": work["is_active"].to_numpy(),
            "is_capped": work.get("is_capped", pd.Series(False, index=work.index)).to_numpy(),
            "alert_start": start_local.to_numpy(),
            "alert_end": end_local.to_numpy(),
        }
    )[single_day.to_numpy()]

    segments: list[dict] = [fast] if not fast.empty else []

    # ── Slow path: only the alerts that cross a local midnight ───────────
    multi = work[~single_day.to_numpy()]
    if not multi.empty:
        multi_start = start_local[~single_day]
        multi_end = end_local[~single_day]
        slow_rows: list[dict] = []
        for region, s, e, active, capped in zip(
            multi["region"],
            multi_start,
            multi_end,
            multi["is_active"],
            multi.get("is_capped", pd.Series(False, index=multi.index)),
        ):
            slow_rows.extend(
                _split_single_alert_by_day(
                    region=region,
                    started_at=s.to_pydatetime(),
                    finished_at=e.to_pydatetime(),
                    is_active=bool(active),
                    is_capped=bool(capped),
                )
            )
        if slow_rows:
            segments.append(pd.DataFrame(slow_rows))

    if skipped:
        logger.warning("Skipped %d rows with NaT finished_at during split.", skipped)

    if not segments:
        return pd.DataFrame()

    result = pd.concat(segments, ignore_index=True)
    result["date"] = pd.to_datetime(result["date"])
    return result


# ═══════════════════════════════════════════════════════════════════════════
# 3.  Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════

def aggregate_daily_alerts(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split segments into daily counts + total minutes per region.

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`split_alerts_by_day`.

    Returns
    -------
    pd.DataFrame
        Columns: date, region, alert_count, total_minutes, avg_duration_minutes,
        has_active_alerts.
    """
    if df.empty:
        return pd.DataFrame()

    grouped = (
        df.groupby(["date", "region"])
        .agg(
            alert_count=("duration_minutes", "count"),
            total_minutes=("duration_minutes", "sum"),
            avg_duration_minutes=("duration_minutes", "mean"),
            has_active_alerts=("is_active", "any"),
        )
        .reset_index()
    )
    grouped["total_minutes"] = grouped["total_minutes"].round(2)
    grouped["avg_duration_minutes"] = grouped["avg_duration_minutes"].round(2)
    return grouped


def aggregate_daily_national(split_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate split segments into national-level daily time series.

    Parameters
    ----------
    split_df : pd.DataFrame
        Output of :func:`split_alerts_by_day`.

    Returns
    -------
    pd.DataFrame
        Columns: date, alert_count, total_minutes, has_active_alerts.
        Indexed by date for easy time-series operations.
    """
    if split_df.empty:
        return pd.DataFrame()

    grouped = (
        split_df.groupby("date")
        .agg(
            alert_count=("duration_minutes", "count"),
            total_minutes=("duration_minutes", "sum"),
            has_active_alerts=("is_active", "any"),
        )
        .reset_index()
    )
    grouped["total_minutes"] = grouped["total_minutes"].round(2)

    # Fill missing dates with zeros for continuous time series
    full_range = pd.date_range(
        start=grouped["date"].min(),
        end=grouped["date"].max(),
        freq="D",
    )
    grouped = grouped.set_index("date").reindex(full_range)
    grouped.index.name = "date"
    grouped["alert_count"] = grouped["alert_count"].fillna(0).astype(int)
    grouped["total_minutes"] = grouped["total_minutes"].fillna(0.0)
    # Compare to True elementwise: NaN/False → False, True → True (no fillna
    # downcast warning, clean bool dtype).
    grouped["has_active_alerts"] = grouped["has_active_alerts"] == True  # noqa: E712
    grouped = grouped.reset_index()
    grouped = grouped.rename(columns={"index": "date"})

    return grouped


def build_region_month_pivot(daily_df: pd.DataFrame) -> pd.DataFrame:
    """Build a region × month pivot table for heatmap visualisation.

    Parameters
    ----------
    daily_df : pd.DataFrame
        Output of :func:`aggregate_daily_alerts`.

    Returns
    -------
    pd.DataFrame
        Pivot with regions as rows, year-month as columns, values =
        total alert minutes.
    """
    if daily_df.empty:
        return pd.DataFrame()

    df = daily_df.copy()
    df["year_month"] = pd.to_datetime(df["date"]).dt.to_period("M").astype(str)

    pivot = df.pivot_table(
        index="region",
        columns="year_month",
        values="total_minutes",
        aggfunc="sum",
        fill_value=0,
    )
    # Sort columns chronologically
    pivot = pivot[sorted(pivot.columns)]
    return pivot


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate alert rows (same region + start + end).

    Parameters
    ----------
    df : pd.DataFrame
        Output of :func:`cap_active_alerts`.

    Returns
    -------
    pd.DataFrame
        De-duplicated DataFrame.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["region", "started_at", "finished_at"])
    after = len(df)
    if before != after:
        logger.info("Removed %d duplicate alert(s).", before - after)
    return df.reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════════════
# 4.  Full preprocessing pipeline
# ═══════════════════════════════════════════════════════════════════════════

def preprocess(dataset: ValidatedDataset) -> dict[str, pd.DataFrame]:
    """Run the complete preprocessing pipeline.

    Returns a dict with the following DataFrames:

    * ``alerts``      — cleaned & capped alerts (one row per alert)
    * ``split``       — day-level segments (overlap-split)
    * ``daily``       — aggregated daily counts by region
    * ``national``    — aggregated national-level daily series
    * ``heatmap``     — region × month pivot

    Parameters
    ----------
    dataset : ValidatedDataset
        Output from the data loader.
    """
    logger.info("Starting preprocessing pipeline …")

    # Step 1: Convert + cap active alerts
    alerts_df = cap_active_alerts(dataset)
    alerts_df = remove_duplicates(alerts_df)
    logger.info("Alerts after dedup: %d", len(alerts_df))

    # Step 2: Split overlapping alerts proportionally by day
    split_df = split_alerts_by_day(alerts_df)
    logger.info("Day-level segments: %d", len(split_df))

    # Step 3: Aggregate
    daily_df = aggregate_daily_alerts(split_df)
    national_df = aggregate_daily_national(split_df)
    heatmap_df = build_region_month_pivot(daily_df)

    logger.info("Preprocessing complete.")
    return {
        "alerts": alerts_df,
        "split": split_df,
        "daily": daily_df,
        "national": national_df,
        "heatmap": heatmap_df,
    }
