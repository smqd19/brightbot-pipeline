"""In-memory session and conversation history management."""

import uuid


class SessionStore:
    """Stores conversation history per session using an in-memory dict.

    Session IDs are passed via the request body (not cookies).
    Unknown session IDs are silently treated as new sessions.
    """

    def __init__(self, max_history: int = 50) -> None:
        self._conversations: dict[str, list[dict]] = {}
        self._max_history = max_history

    def get_or_create_session(self, session_id: str | None) -> str:
        """Return the given session_id or generate a new UUID hex string."""
        if session_id and isinstance(session_id, str) and session_id.strip():
            return session_id.strip()
        return uuid.uuid4().hex

    def get_history(self, session_id: str) -> list[dict]:
        """Return the full conversation history for a session."""
        if session_id not in self._conversations:
            self._conversations[session_id] = []
        return self._conversations[session_id]

    def append_message(self, session_id: str, role: str, content: str) -> None:
        """Append a message and auto-trim to max_history."""
        history = self.get_history(session_id)
        history.append({"role": role, "content": content})
        # Trim oldest messages if over limit
        if len(history) > self._max_history:
            self._conversations[session_id] = history[-self._max_history :]

    def get_trimmed_history(self, session_id: str) -> list[dict]:
        """Return the last max_history messages for building the API call."""
        history = self.get_history(session_id)
        return history[-self._max_history :]

    def clear(self, session_id: str) -> None:
        """Remove all history for a session."""
        self._conversations.pop(session_id, None)
