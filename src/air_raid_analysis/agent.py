"""Optional LLM analyst — turns the numeric report into a narrative insight.

This is the "agentic" layer of the pipeline. It is **strictly optional and
gracefully degradable**:

* It runs **only** when a token is available, auto-detecting the provider:
    - ``OPENAI_API_KEY``  → OpenAI directly.
    - ``GITHUB_TOKEN`` (or ``GITHUB_MODELS_TOKEN``) → **GitHub Models**, a free
      token-based, OpenAI-compatible inference service — ideal for an MVP.
  With no token it logs a notice and returns ``None`` — so local runs and CI
  stay green and never call out to the network.
* The ``openai`` SDK is imported **lazily**, so it stays an optional extra
  (``pip install -e ".[agent]"``); the base install never needs it.
* Any API/SDK failure is caught and degraded to ``None`` — a flaky LLM call
  must never break the analytics pipeline.

The model is instructed to reason **only** over the numbers it is given, so it
cannot fabricate figures that aren't in the report.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

from air_raid_analysis.analysis import BasicStats, ForecastResult
from air_raid_analysis.config import settings

logger = logging.getLogger(__name__)


@dataclass
class _LLMConfig:
    """Resolved provider settings for a single analyst call."""

    api_key: str
    base_url: str | None
    model: str
    provider: str


def _resolve_llm_config() -> _LLMConfig | None:
    """Pick the LLM provider from the environment, or ``None`` if unavailable.

    Priority: an explicit ``OPENAI_API_KEY`` wins; otherwise fall back to a
    GitHub token and route through GitHub Models (the free MVP path).
    """
    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return _LLMConfig(
            api_key=openai_key,
            base_url=settings.llm_base_url,  # None → OpenAI default
            model=settings.llm_model,
            provider="openai",
        )

    gh_token = os.environ.get("GITHUB_MODELS_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if gh_token:
        # GitHub Models namespaces models as e.g. "openai/gpt-4o-mini".
        model = settings.llm_model if "/" in settings.llm_model else f"openai/{settings.llm_model}"
        return _LLMConfig(
            api_key=gh_token,
            base_url=settings.llm_base_url or settings.github_models_base_url,
            model=model,
            provider="github-models",
        )

    return None

_SYSTEM_PROMPT = (
    "You are a defense-analytics assistant summarising an automated time-series "
    "report on air-raid alerts in Ukraine. Write a concise, sober briefing for an "
    "analyst audience. Use ONLY the figures provided — never invent numbers, "
    "regions, or dates. If the forecast backtest shows MASE >= 1, state plainly "
    "that the model does not beat a seasonal-naive baseline. Output GitHub-flavoured "
    "Markdown: a short '## AI Analyst Briefing' heading, 3-5 tight bullet points on "
    "trend, seasonality, anomalies and forecast reliability, then one '> ' caveat line."
)


def _build_user_prompt(
    report_text: str,
    stats: BasicStats,
    forecast: ForecastResult | None,
) -> str:
    """Assemble a grounded prompt from the already-computed numbers."""
    lines = [
        "Here is the automated numeric report. Summarise it faithfully.",
        "",
        "=== REPORT ===",
        report_text.strip(),
    ]
    if forecast is not None and forecast.metrics:
        m = forecast.metrics
        lines += [
            "",
            "=== FORECAST BACKTEST (out-of-sample) ===",
            f"horizon_days={len(forecast.forecast_dates)}, "
            f"MAE={m.get('mae')}, RMSE={m.get('rmse')}, MASE={m.get('mase')}",
        ]
    return "\n".join(lines)


def generate_ai_insights(
    report_text: str,
    stats: BasicStats,
    forecast: ForecastResult | None = None,
) -> str | None:
    """Generate a Markdown insight briefing, or ``None`` if unavailable.

    Returns ``None`` (never raises) when no token is available, the SDK is not
    installed, or the API call fails — the caller treats insights as optional.
    """
    cfg = _resolve_llm_config()
    if cfg is None:
        logger.info(
            "No LLM token (OPENAI_API_KEY / GITHUB_TOKEN) — skipping AI analyst step."
        )
        return None

    try:
        from openai import OpenAI
    except ImportError:
        logger.warning(
            "openai SDK not installed — run `pip install -e \".[agent]\"`. "
            "Skipping AI analyst step."
        )
        return None

    try:
        client = OpenAI(api_key=cfg.api_key, base_url=cfg.base_url)
        response = client.chat.completions.create(
            model=cfg.model,
            temperature=0.3,
            max_tokens=settings.llm_max_tokens,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(report_text, stats, forecast)},
            ],
        )
        text = (response.choices[0].message.content or "").strip()
        if not text:
            logger.warning("AI analyst returned empty content — skipping.")
            return None
        logger.info(
            "AI analyst produced a %d-char briefing (provider=%s, model=%s).",
            len(text), cfg.provider, cfg.model,
        )
        return text
    except Exception as exc:  # noqa: BLE001 — the pipeline must survive a flaky LLM call
        logger.warning("AI analyst call failed (%s) — continuing without insights.", exc)
        return None
