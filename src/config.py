"""Centralized configuration loaded from environment variables."""

import os


def _require_env(key: str) -> str:
    """Return an env var's value or raise if missing."""
    value = os.getenv(key)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {key}")
    return value


OPENAI_API_KEY: str = _require_env("OPENAI_API_KEY")
OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

SYSTEM_PROMPT: str = os.getenv(
    "BRIGHTBOT_SYSTEM_PROMPT",
    "You are BrightBot, an assistant for the Brightly team. "
    "Help with migration tools, Superset dashboards, and internal processes.",
)

MAX_HISTORY: int = int(os.getenv("MAX_HISTORY", "50"))
MAX_SESSION_TOKENS: int = int(os.getenv("MAX_SESSION_TOKENS", "100000"))
