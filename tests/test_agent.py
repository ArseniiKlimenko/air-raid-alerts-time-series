"""Tests for the optional LLM analyst layer (graceful degradation)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from air_raid_analysis.agent import _resolve_llm_config, generate_ai_insights
from air_raid_analysis.analysis import BasicStats
from air_raid_analysis.visualization import _render_markdown

_TOKEN_VARS = ("OPENAI_API_KEY", "GITHUB_MODELS_TOKEN", "GITHUB_TOKEN")


def test_insights_skipped_without_any_token(monkeypatch):
    """No token at all → returns None without raising or calling out."""
    for var in _TOKEN_VARS:
        monkeypatch.delenv(var, raising=False)
    assert _resolve_llm_config() is None
    assert generate_ai_insights("dummy report", BasicStats(), None) is None


def test_openai_key_takes_priority(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    monkeypatch.setenv("GITHUB_TOKEN", "ghs-test")
    cfg = _resolve_llm_config()
    assert cfg is not None
    assert cfg.provider == "openai"
    assert cfg.base_url is None  # OpenAI default endpoint


def test_github_token_routes_to_github_models(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GITHUB_MODELS_TOKEN", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghs-test")
    cfg = _resolve_llm_config()
    assert cfg is not None
    assert cfg.provider == "github-models"
    assert cfg.base_url.startswith("https://models.github.ai")
    assert cfg.model == "openai/gpt-4o-mini"  # namespaced for GitHub Models


def test_render_markdown_escapes_and_formats():
    md = "## Title\n- one **bold**\n- two\n> caveat <script>"
    html = _render_markdown(md)
    assert "<h2>Title</h2>" in html
    assert "<ul>" in html and "<li>one <strong>bold</strong></li>" in html
    assert "<blockquote>caveat" in html
    # Raw HTML must be escaped, never injected.
    assert "<script>" not in html
    assert "&lt;script&gt;" in html
