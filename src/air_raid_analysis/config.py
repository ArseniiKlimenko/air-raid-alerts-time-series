"""Centralized project configuration.

Uses Pydantic BaseSettings so every parameter can be overridden via
environment variables (prefix: ARA_) or a .env file.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings


def _project_root() -> Path:
    """Return the project root (two levels up from this file)."""
    return Path(__file__).resolve().parent.parent.parent


def _default_raw_data_dir() -> Path:
    return _project_root() / "data" / "raw"


def _default_output_dir() -> Path:
    return _project_root() / "output"


class AnalysisSettings(BaseSettings):
    """All tuneable knobs for the analysis pipeline."""

    model_config = {"env_prefix": "ARA_"}

    # ── Paths ────────────────────────────────────────────────────────────
    project_root: Path = Field(default_factory=_project_root)
    raw_data_dir: Path = Field(default_factory=_default_raw_data_dir)
    output_dir: Path = Field(default_factory=_default_output_dir)

    # ── Data source ──────────────────────────────────────────────────────
    dataset_url: str = (
        "https://raw.githubusercontent.com/Vadimkin/"
        "ukrainian-air-raid-sirens-dataset/main/datasets/"
        "official_data_en.csv"
    )
    cache_max_age_hours: int = Field(default=24, ge=1)

    # ── Analysis parameters ──────────────────────────────────────────────
    rolling_window_days: int = Field(default=7, ge=1)
    top_n_regions: int = Field(default=10, ge=1)
    stl_period: int = Field(default=7, ge=2, description="STL seasonal period in days")
    max_alert_duration_hours: float = Field(
        default=72.0,
        ge=1.0,
        description="Alerts longer than this (hours) are flagged as anomalies",
    )

    # Calendar/local-time handling. Alerts are split across *local* calendar
    # days so that night-time alerts (the majority) are attributed to the
    # correct Ukrainian day rather than the UTC day (which starts at 02:00/
    # 03:00 Kyiv time).
    analysis_timezone: str = "Europe/Kyiv"

    # Multi-seasonal decomposition: weekly + annual periodicity.
    enable_multiseasonal: bool = True
    annual_period: int = Field(default=365, ge=2, description="Annual seasonal period (days)")

    # Statistical anomaly detection on the national daily series.
    anomaly_sigma: float = Field(
        default=3.0, ge=1.0, description="Rolling-band width (σ) for day-level anomaly flagging"
    )

    # Forecasting (SARIMAX).
    forecast_horizon_days: int = Field(default=30, ge=1, description="Days to forecast ahead")
    forecast_backtest_folds: int = Field(
        default=3, ge=1, description="Rolling-origin backtest folds for forecast evaluation"
    )

    # ── LLM analyst (optional) ───────────────────────────────────────────
    # Activated when a token is present. Two providers are auto-detected:
    #   * OPENAI_API_KEY  → OpenAI directly (base_url stays default).
    #   * GITHUB_TOKEN    → GitHub Models (free, token-based) for an MVP.
    # Override the endpoint/model via ARA_LLM_BASE_URL / ARA_LLM_MODEL.
    llm_model: str = "gpt-4o-mini"
    llm_base_url: str | None = None
    github_models_base_url: str = "https://models.github.ai/inference"
    llm_max_tokens: int = Field(default=700, ge=64, description="Max tokens for the AI insight")

    # ── Validation boundaries ────────────────────────────────────────────
    war_start_date: str = "2022-02-24"

    @property
    def raw_csv_path(self) -> Path:
        return self.raw_data_dir / "sirens.csv"


# Singleton — import and use directly.
settings = AnalysisSettings()

