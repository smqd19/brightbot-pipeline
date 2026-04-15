"""Per-session token usage tracking and budget enforcement."""

from dataclasses import dataclass, field


@dataclass
class SessionUsage:
    """Accumulated token usage for a single session."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    request_count: int = 0

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON responses."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "request_count": self.request_count,
        }


class CostTracker:
    """Tracks token usage per session and enforces a configurable budget.

    The budget is checked *before* each API call. If the session's
    cumulative total_tokens exceeds max_session_tokens, the request
    is rejected with HTTP 429.
    """

    def __init__(self, max_session_tokens: int = 100_000) -> None:
        self._usage: dict[str, SessionUsage] = {}
        self._max_session_tokens = max_session_tokens

    def _get_usage(self, session_id: str) -> SessionUsage:
        if session_id not in self._usage:
            self._usage[session_id] = SessionUsage()
        return self._usage[session_id]

    def check_budget(self, session_id: str) -> bool:
        """Return True if the session is still within its token budget."""
        usage = self._get_usage(session_id)
        return usage.total_tokens < self._max_session_tokens

    def record_usage(self, session_id: str, usage: dict) -> SessionUsage:
        """Accumulate tokens from an OpenAI response.usage object.

        Args:
            session_id: The session to record against.
            usage: Dict with prompt_tokens, completion_tokens, total_tokens.

        Returns:
            The updated SessionUsage for the session.
        """
        session_usage = self._get_usage(session_id)
        session_usage.prompt_tokens += usage.get("prompt_tokens", 0)
        session_usage.completion_tokens += usage.get("completion_tokens", 0)
        session_usage.total_tokens += usage.get("total_tokens", 0)
        session_usage.request_count += 1
        return session_usage

    def get_usage(self, session_id: str) -> SessionUsage:
        """Return current usage stats for a session."""
        return self._get_usage(session_id)

    def clear(self, session_id: str) -> None:
        """Reset usage for a session."""
        self._usage.pop(session_id, None)
