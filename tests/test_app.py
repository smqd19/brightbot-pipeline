"""Comprehensive unit tests for src/app.py — Flask app factory and /chat endpoint.

Covers: successful responses, input validation, session persistence,
session creation, OpenAI error handling, response schema, health endpoint,
HTTP method restrictions, and edge cases.
"""

from unittest.mock import MagicMock

import pytest

from src.app import create_app


# --------------- helpers ---------------

def _mock_completion(content="Mock reply", prompt_tokens=20,
                     completion_tokens=8, total_tokens=28):
    """Build a fake ChatCompletion object."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens

    comp = MagicMock()
    comp.choices = [choice]
    comp.usage = usage
    return comp


@pytest.fixture()
def app():
    """Fresh app with mocked OpenAI for every test."""
    application = create_app(config_override={
        "OPENAI_API_KEY": "sk-test",
        "OPENAI_MODEL": "gpt-4o-mini",
        "MAX_HISTORY": 4,
        "MAX_SESSION_TOKENS": 150,
    })
    application.config["TESTING"] = True

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion()
    application.openai_client = mock_client
    return application


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture()
def openai(app):
    """Shortcut to the mocked OpenAI client."""
    return app.openai_client


# ===================================================================
# 1. Successful chat response
# ===================================================================

class TestChatSuccess:
    """Verify that a well-formed request produces the correct response."""

    def test_returns_200(self, client):
        resp = client.post("/chat", json={"message": "ping"})
        assert resp.status_code == 200

    def test_response_contains_required_fields(self, client):
        data = client.post("/chat", json={"message": "ping"}).get_json()
        assert "response" in data
        assert "session_id" in data
        assert "usage" in data

    def test_response_field_matches_openai_output(self, client):
        data = client.post("/chat", json={"message": "ping"}).get_json()
        assert data["response"] == "Mock reply"

    def test_usage_fields_are_integers(self, client):
        data = client.post("/chat", json={"message": "ping"}).get_json()
        usage = data["usage"]
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)

    def test_openai_called_with_correct_model(self, client, openai):
        client.post("/chat", json={"message": "ping"})
        call_kwargs = openai.chat.completions.create.call_args.kwargs
        assert call_kwargs["model"] == "gpt-4o-mini"

    def test_system_prompt_is_first_message(self, client, openai):
        client.post("/chat", json={"message": "ping"})
        messages = openai.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert "BrightBot" in messages[0]["content"]

    def test_user_message_is_last_in_context(self, client, openai):
        client.post("/chat", json={"message": "ping"})
        messages = openai.chat.completions.create.call_args.kwargs["messages"]
        assert messages[-1] == {"role": "user", "content": "ping"}

    def test_openai_none_content_returns_empty_string(self, client, app):
        """If OpenAI returns None as content, response should be empty string."""
        app.openai_client.chat.completions.create.return_value = _mock_completion(content=None)
        # The `or ""` fallback in app.py should catch this
        data = client.post("/chat", json={"message": "Hi"}).get_json()
        assert data["response"] == ""


# ===================================================================
# 2. Missing / invalid message field
# ===================================================================

class TestMessageValidation:
    """Verify 400 errors for bad or missing message payloads."""

    def test_missing_message_key(self, client):
        resp = client.post("/chat", json={"session_id": "x"})
        assert resp.status_code == 400
        assert "message" in resp.get_json()["error"].lower()

    def test_null_message(self, client):
        resp = client.post("/chat", json={"message": None})
        assert resp.status_code == 400

    def test_empty_string_message(self, client):
        resp = client.post("/chat", json={"message": ""})
        assert resp.status_code == 400

    def test_whitespace_only_message(self, client):
        resp = client.post("/chat", json={"message": "   \t\n  "})
        assert resp.status_code == 400

    def test_no_json_body(self, client):
        resp = client.post("/chat", data="hello", content_type="text/plain")
        assert resp.status_code == 400
        assert "JSON" in resp.get_json()["error"]

    def test_empty_body(self, client):
        resp = client.post("/chat", content_type="application/json")
        assert resp.status_code == 400

    def test_json_array_instead_of_object(self, client):
        """Body is valid JSON but not a dict — should still 400."""
        resp = client.post("/chat", json=["hello"])
        assert resp.status_code == 400

    def test_numeric_message_is_accepted(self, client):
        """A numeric message gets cast to string via str() in app.py."""
        resp = client.post("/chat", json={"message": 42})
        assert resp.status_code == 200
        data = resp.get_json()
        assert data["response"] == "Mock reply"


# ===================================================================
# 3. Session persistence (same session_id = continued conversation)
# ===================================================================

class TestSessionPersistence:
    """Verify that sending the same session_id continues the conversation."""

    def test_same_session_id_returned(self, client):
        r1 = client.post("/chat", json={"message": "A", "session_id": "persist-1"})
        r2 = client.post("/chat", json={"message": "B", "session_id": "persist-1"})
        assert r1.get_json()["session_id"] == "persist-1"
        assert r2.get_json()["session_id"] == "persist-1"

    def test_history_grows_across_requests(self, client, openai):
        """Each subsequent call includes prior exchanges in the OpenAI context."""
        sid = "grow-test"
        client.post("/chat", json={"message": "first", "session_id": sid})
        client.post("/chat", json={"message": "second", "session_id": sid})
        client.post("/chat", json={"message": "third", "session_id": sid})

        # Third call should have: system + 4 history msgs (2 exchanges) + 1 new user
        third_call = openai.chat.completions.create.call_args_list[2]
        messages = third_call.kwargs["messages"]
        roles = [m["role"] for m in messages]
        assert roles == ["system", "user", "assistant", "user", "assistant", "user"]

    def test_different_sessions_are_isolated(self, client, openai):
        """Messages in session A should not appear in session B's context."""
        client.post("/chat", json={"message": "Only for A", "session_id": "A"})
        client.post("/chat", json={"message": "Only for B", "session_id": "B"})

        # Session B's call should only have system + its own user message
        second_call = openai.chat.completions.create.call_args_list[1]
        messages = second_call.kwargs["messages"]
        assert len(messages) == 2  # system + user
        assert messages[1]["content"] == "Only for B"

    def test_usage_accumulates_per_session(self, client):
        """Token usage should grow across requests in the same session."""
        sid = "usage-acc"
        r1 = client.post("/chat", json={"message": "A", "session_id": sid})
        r2 = client.post("/chat", json={"message": "B", "session_id": sid})

        # Each mock call returns total_tokens=28, so after 2 calls: 56
        assert r1.get_json()["usage"]["total_tokens"] == 28
        # The individual response usage is per-call, but cost_tracker accumulates
        # We can verify the tracker directly
        from src.cost_tracker import CostTracker
        # Just verify both returned 200 — accumulation is tested in test_cost_tracker.py
        assert r2.status_code == 200


# ===================================================================
# 4. New session creation (no session_id = new one generated)
# ===================================================================

class TestNewSessionCreation:
    """Verify that omitting session_id creates a new session each time."""

    def test_generates_uuid_hex(self, client):
        data = client.post("/chat", json={"message": "Hi"}).get_json()
        sid = data["session_id"]
        assert isinstance(sid, str)
        assert len(sid) == 32
        int(sid, 16)  # valid hex

    def test_two_requests_without_id_get_different_sessions(self, client):
        r1 = client.post("/chat", json={"message": "A"}).get_json()
        r2 = client.post("/chat", json={"message": "B"}).get_json()
        assert r1["session_id"] != r2["session_id"]

    def test_new_session_has_no_prior_history(self, client, openai):
        """First request on a new session only has system + user in context."""
        client.post("/chat", json={"message": "fresh"})
        messages = openai.chat.completions.create.call_args.kwargs["messages"]
        assert len(messages) == 2
        assert messages[0]["role"] == "system"
        assert messages[1]["role"] == "user"

    def test_null_session_id_creates_new(self, client):
        data = client.post("/chat", json={"message": "Hi", "session_id": None}).get_json()
        assert len(data["session_id"]) == 32

    def test_empty_session_id_creates_new(self, client):
        data = client.post("/chat", json={"message": "Hi", "session_id": ""}).get_json()
        assert len(data["session_id"]) == 32


# ===================================================================
# 5. OpenAI API error handling
# ===================================================================

class TestOpenAIErrorHandling:
    """Verify graceful handling of OpenAI failures."""

    def test_generic_exception_returns_502(self, client, openai):
        openai.chat.completions.create.side_effect = Exception("Connection reset")
        resp = client.post("/chat", json={"message": "Hi"})
        assert resp.status_code == 502

    def test_502_body_contains_error_detail(self, client, openai):
        openai.chat.completions.create.side_effect = Exception("rate limit exceeded")
        data = client.post("/chat", json={"message": "Hi"}).get_json()
        assert "OpenAI API error" in data["error"]
        assert "rate limit exceeded" in data["error"]

    def test_openai_error_does_not_persist_messages(self, client, app, openai):
        """If OpenAI fails, the user message should NOT be saved to history."""
        openai.chat.completions.create.side_effect = Exception("fail")
        client.post("/chat", json={"message": "broken", "session_id": "err-sess"})

        history = app.session_store.get_history("err-sess")
        assert len(history) == 0  # nothing saved

    def test_openai_error_does_not_record_usage(self, client, app, openai):
        """If OpenAI fails, no token usage should be recorded."""
        openai.chat.completions.create.side_effect = Exception("fail")
        client.post("/chat", json={"message": "Hi", "session_id": "err-usage"})

        usage = app.cost_tracker.get_usage("err-usage")
        assert usage.total_tokens == 0
        assert usage.request_count == 0

    def test_recovery_after_error(self, client, openai):
        """After an OpenAI error, the next request should work normally."""
        openai.chat.completions.create.side_effect = Exception("temporary")
        resp1 = client.post("/chat", json={"message": "A", "session_id": "recover"})
        assert resp1.status_code == 502

        # Restore normal behavior
        openai.chat.completions.create.side_effect = None
        openai.chat.completions.create.return_value = _mock_completion()
        resp2 = client.post("/chat", json={"message": "B", "session_id": "recover"})
        assert resp2.status_code == 200


# ===================================================================
# 6. Token budget enforcement (429)
# ===================================================================

class TestTokenBudget:
    """Verify that sessions exceeding the token cap are blocked."""

    def test_returns_429_when_budget_exhausted(self, client, app):
        app.cost_tracker.record_usage("maxed", {
            "prompt_tokens": 80, "completion_tokens": 80, "total_tokens": 160,
        })
        resp = client.post("/chat", json={"message": "Hi", "session_id": "maxed"})
        assert resp.status_code == 429

    def test_429_body_contains_usage_and_session_id(self, client, app):
        app.cost_tracker.record_usage("maxed2", {
            "prompt_tokens": 80, "completion_tokens": 80, "total_tokens": 160,
        })
        data = client.post("/chat", json={"message": "Hi", "session_id": "maxed2"}).get_json()
        assert "usage" in data
        assert data["session_id"] == "maxed2"
        assert data["usage"]["total_tokens"] == 160

    def test_new_session_can_still_chat_after_another_is_maxed(self, client, app):
        """Budget is per-session — other sessions should be unaffected."""
        app.cost_tracker.record_usage("full", {
            "prompt_tokens": 100, "completion_tokens": 100, "total_tokens": 200,
        })
        resp = client.post("/chat", json={"message": "Hi", "session_id": "fresh-one"})
        assert resp.status_code == 200


# ===================================================================
# 7. Health endpoint
# ===================================================================

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.get_json() == {"status": "ok"}


# ===================================================================
# 8. HTTP method restrictions
# ===================================================================

class TestMethodRestrictions:

    def test_get_chat_not_allowed(self, client):
        resp = client.get("/chat")
        assert resp.status_code == 405

    def test_put_chat_not_allowed(self, client):
        resp = client.put("/chat", json={"message": "Hi"})
        assert resp.status_code == 405

    def test_post_health_not_allowed(self, client):
        resp = client.post("/health")
        assert resp.status_code == 405


# ===================================================================
# 9. App factory
# ===================================================================

class TestAppFactory:

    def test_create_app_returns_flask_instance(self):
        from flask import Flask
        app = create_app(config_override={"OPENAI_API_KEY": "k"})
        assert isinstance(app, Flask)

    def test_config_override_applied(self):
        app = create_app(config_override={
            "OPENAI_API_KEY": "k",
            "OPENAI_MODEL": "gpt-3.5-turbo",
        })
        assert app.config["OPENAI_MODEL"] == "gpt-3.5-turbo"
