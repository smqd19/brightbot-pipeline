"""Integration tests for the POST /chat endpoint."""

import json
from unittest.mock import MagicMock


# -- Happy path --

def test_chat_success(client, mock_openai):
    """POST /chat with a valid message returns 200 with response and session_id."""
    resp = client.post("/chat", json={"message": "Hi"})
    assert resp.status_code == 200

    data = resp.get_json()
    assert "response" in data
    assert "session_id" in data
    assert "usage" in data
    assert data["response"] == "Hello from BrightBot!"


def test_chat_generates_session_id(client):
    """When no session_id is provided, a new one is generated."""
    resp = client.post("/chat", json={"message": "Hi"})
    data = resp.get_json()
    assert isinstance(data["session_id"], str)
    assert len(data["session_id"]) == 32  # uuid4 hex


def test_chat_reuses_session_id(client):
    """When session_id is provided, the same one is returned."""
    resp = client.post("/chat", json={"message": "Hi", "session_id": "my-session-123"})
    data = resp.get_json()
    assert data["session_id"] == "my-session-123"


def test_chat_maintains_history(client, app, mock_openai):
    """Second message in the same session includes the first exchange in context."""
    # First message
    resp1 = client.post("/chat", json={"message": "Hello", "session_id": "sess-1"})
    assert resp1.status_code == 200

    # Second message
    resp2 = client.post("/chat", json={"message": "Follow up", "session_id": "sess-1"})
    assert resp2.status_code == 200

    # Check that the second OpenAI call included the history
    second_call_args = mock_openai.chat.completions.create.call_args_list[1]
    messages = second_call_args.kwargs.get("messages") or second_call_args[1].get("messages")
    # Should have: system + user("Hello") + assistant(reply) + user("Follow up")
    roles = [m["role"] for m in messages]
    assert roles == ["system", "user", "assistant", "user"]


def test_chat_returns_usage(client):
    """Response includes usage stats from OpenAI."""
    resp = client.post("/chat", json={"message": "Hi"})
    data = resp.get_json()
    assert data["usage"]["prompt_tokens"] == 25
    assert data["usage"]["completion_tokens"] == 10
    assert data["usage"]["total_tokens"] == 35


# -- Error cases --

def test_chat_missing_message(client):
    """POST /chat without message field returns 400."""
    resp = client.post("/chat", json={"session_id": "abc"})
    assert resp.status_code == 400
    assert "message" in resp.get_json()["error"].lower()


def test_chat_empty_message(client):
    """POST /chat with whitespace-only message returns 400."""
    resp = client.post("/chat", json={"message": "   "})
    assert resp.status_code == 400


def test_chat_no_json(client):
    """POST /chat with non-JSON body returns 400."""
    resp = client.post("/chat", data="not json", content_type="text/plain")
    assert resp.status_code == 400


def test_chat_openai_failure(client, mock_openai):
    """When OpenAI raises an exception, return 502."""
    mock_openai.chat.completions.create.side_effect = Exception("API down")
    resp = client.post("/chat", json={"message": "Hi"})
    assert resp.status_code == 502
    assert "OpenAI API error" in resp.get_json()["error"]


def test_chat_token_limit_exceeded(client, app):
    """When session exceeds token budget, return 429."""
    # Manually exhaust the budget for a session
    app.cost_tracker.record_usage("budget-test", {
        "prompt_tokens": 100,
        "completion_tokens": 100,
        "total_tokens": 200,
    })

    resp = client.post("/chat", json={"message": "Hi", "session_id": "budget-test"})
    assert resp.status_code == 429
    data = resp.get_json()
    assert "token limit" in data["error"].lower()
    assert data["session_id"] == "budget-test"
    assert "usage" in data


def test_chat_history_trimming(client, app, mock_openai):
    """Only MAX_HISTORY messages are sent to OpenAI (MAX_HISTORY=5 in test config)."""
    sid = "trim-test"
    # Send 7 messages (MAX_HISTORY is 5 in test fixture)
    for i in range(7):
        client.post("/chat", json={"message": f"msg-{i}", "session_id": sid})

    # The last call should have system + at most 5 history messages + 1 new user message
    last_call = mock_openai.chat.completions.create.call_args_list[-1]
    messages = last_call.kwargs.get("messages") or last_call[1].get("messages")
    # system(1) + trimmed history(<=5) + new user(1) = at most 7
    non_system = [m for m in messages if m["role"] != "system"]
    assert len(non_system) <= 6  # 5 history + 1 new user
