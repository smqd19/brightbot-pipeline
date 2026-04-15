"""Unit tests for SessionStore."""

from src.session_store import SessionStore


def test_new_session_creates_uuid():
    """get_or_create_session(None) returns a 32-char hex UUID."""
    store = SessionStore()
    sid = store.get_or_create_session(None)
    assert isinstance(sid, str)
    assert len(sid) == 32
    int(sid, 16)  # Should be valid hex


def test_existing_session_returned():
    """Passing an existing session_id returns it unchanged."""
    store = SessionStore()
    assert store.get_or_create_session("my-id") == "my-id"


def test_empty_string_generates_new():
    """Empty or whitespace-only session_id generates a new one."""
    store = SessionStore()
    sid = store.get_or_create_session("   ")
    assert len(sid) == 32


def test_append_and_retrieve():
    """Messages round-trip through append and get_history."""
    store = SessionStore()
    store.append_message("s1", "user", "hello")
    store.append_message("s1", "assistant", "hi there")

    history = store.get_history("s1")
    assert len(history) == 2
    assert history[0] == {"role": "user", "content": "hello"}
    assert history[1] == {"role": "assistant", "content": "hi there"}


def test_trimming():
    """History beyond max_history is trimmed (oldest dropped)."""
    store = SessionStore(max_history=3)
    for i in range(5):
        store.append_message("s1", "user", f"msg-{i}")

    history = store.get_history("s1")
    assert len(history) == 3
    assert history[0]["content"] == "msg-2"
    assert history[2]["content"] == "msg-4"


def test_clear():
    """clear() removes all history for a session."""
    store = SessionStore()
    store.append_message("s1", "user", "hello")
    store.clear("s1")
    assert store.get_history("s1") == []
