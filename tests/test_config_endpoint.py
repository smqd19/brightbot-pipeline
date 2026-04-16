"""Tests for issue #2: configurable SYSTEM_PROMPT and GET /config endpoint.

Covers: GET /config happy path, custom system prompt via config_override,
default system prompt, system prompt used in /chat calls, SYSTEM_PROMPT env
var loading, HTTP method restrictions, and edge cases.
"""

import os
from unittest.mock import MagicMock, patch

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


def _make_app(config_override=None):
    """Create a test app with optional config overrides and mocked OpenAI."""
    base = {"OPENAI_API_KEY": "test-key", "OPENAI_MODEL": "gpt-4o-mini"}
    if config_override:
        base.update(config_override)
    app = create_app(config_override=base)
    app.config["TESTING"] = True

    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _mock_completion()
    app.openai_client = mock_client
    return app


# ===================================================================
# 1. GET /config — happy path
# ===================================================================

class TestConfigEndpointHappyPath:
    """Verify the GET /config endpoint returns the system prompt."""

    @pytest.fixture()
    def client(self):
        return _make_app().test_client()

    def test_returns_200(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200

    def test_response_is_json(self, client):
        resp = client.get("/config")
        assert resp.content_type.startswith("application/json")

    def test_response_contains_system_prompt_key(self, client):
        data = client.get("/config").get_json()
        assert "system_prompt" in data

    def test_default_system_prompt_value(self, client):
        data = client.get("/config").get_json()
        assert data["system_prompt"] == "You are BrightBot, a helpful assistant for the Brightly team."

    def test_response_has_only_system_prompt(self, client):
        """GET /config must NOT expose secrets like API keys."""
        data = client.get("/config").get_json()
        assert list(data.keys()) == ["system_prompt"]


# ===================================================================
# 2. GET /config — custom system prompt
# ===================================================================

class TestConfigEndpointCustomPrompt:
    """Verify that a custom SYSTEM_PROMPT is reflected in /config."""

    def test_custom_prompt_from_config_override(self):
        custom = "You are a pirate assistant. Arrr!"
        app = _make_app({"SYSTEM_PROMPT": custom})
        client = app.test_client()

        data = client.get("/config").get_json()
        assert data["system_prompt"] == custom

    def test_empty_string_prompt_is_allowed(self):
        app = _make_app({"SYSTEM_PROMPT": ""})
        client = app.test_client()

        data = client.get("/config").get_json()
        assert data["system_prompt"] == ""

    def test_long_prompt_is_preserved(self):
        long_prompt = "A" * 5000
        app = _make_app({"SYSTEM_PROMPT": long_prompt})
        client = app.test_client()

        data = client.get("/config").get_json()
        assert data["system_prompt"] == long_prompt

    def test_prompt_with_special_characters(self):
        special = 'Use "quotes" and <html> & symbols\nnewlines too'
        app = _make_app({"SYSTEM_PROMPT": special})
        client = app.test_client()

        data = client.get("/config").get_json()
        assert data["system_prompt"] == special


# ===================================================================
# 3. System prompt is used in /chat calls
# ===================================================================

class TestSystemPromptInChat:
    """Verify the configured system prompt is sent to OpenAI in /chat."""

    def test_default_prompt_sent_to_openai(self):
        app = _make_app()
        client = app.test_client()

        client.post("/chat", json={"message": "Hi"})
        messages = app.openai_client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == "You are BrightBot, a helpful assistant for the Brightly team."

    def test_custom_prompt_sent_to_openai(self):
        custom = "You are a helpful coding assistant."
        app = _make_app({"SYSTEM_PROMPT": custom})
        client = app.test_client()

        client.post("/chat", json={"message": "Hi"})
        messages = app.openai_client.chat.completions.create.call_args.kwargs["messages"]
        assert messages[0]["role"] == "system"
        assert messages[0]["content"] == custom

    def test_config_and_chat_use_same_prompt(self):
        """The prompt returned by /config should match what /chat sends to OpenAI."""
        custom = "Custom prompt for consistency check."
        app = _make_app({"SYSTEM_PROMPT": custom})
        client = app.test_client()

        config_prompt = client.get("/config").get_json()["system_prompt"]
        client.post("/chat", json={"message": "Hi"})
        chat_prompt = app.openai_client.chat.completions.create.call_args.kwargs["messages"][0]["content"]

        assert config_prompt == chat_prompt == custom


# ===================================================================
# 4. SYSTEM_PROMPT env var in config.py
# ===================================================================

class TestSystemPromptEnvVar:
    """Verify config.py reads from SYSTEM_PROMPT (not the old name)."""

    def test_env_var_is_read(self):
        custom = "Env-based prompt"
        with patch.dict(os.environ, {
            "OPENAI_API_KEY": "sk-test",
            "SYSTEM_PROMPT": custom,
        }):
            from importlib import reload
            import src.config
            reload(src.config)
            assert src.config.SYSTEM_PROMPT == custom

    def test_default_when_env_var_not_set(self):
        env = os.environ.copy()
        env.pop("SYSTEM_PROMPT", None)
        env["OPENAI_API_KEY"] = "sk-test"
        with patch.dict(os.environ, env, clear=True):
            from importlib import reload
            import src.config
            reload(src.config)
            assert src.config.SYSTEM_PROMPT == "You are BrightBot, a helpful assistant for the Brightly team."

    def test_old_env_var_name_is_not_used(self):
        """BRIGHTBOT_SYSTEM_PROMPT should NOT be read — only SYSTEM_PROMPT."""
        env = os.environ.copy()
        env.pop("SYSTEM_PROMPT", None)
        env["OPENAI_API_KEY"] = "sk-test"
        env["BRIGHTBOT_SYSTEM_PROMPT"] = "Old name prompt"
        with patch.dict(os.environ, env, clear=True):
            from importlib import reload
            import src.config
            reload(src.config)
            # Should use default, NOT the old env var
            assert src.config.SYSTEM_PROMPT == "You are BrightBot, a helpful assistant for the Brightly team."
            assert src.config.SYSTEM_PROMPT != "Old name prompt"


# ===================================================================
# 5. HTTP method restrictions on /config
# ===================================================================

class TestConfigMethodRestrictions:
    """Only GET should be allowed on /config."""

    @pytest.fixture()
    def client(self):
        return _make_app().test_client()

    def test_post_config_not_allowed(self, client):
        resp = client.post("/config")
        assert resp.status_code == 405

    def test_put_config_not_allowed(self, client):
        resp = client.put("/config")
        assert resp.status_code == 405

    def test_delete_config_not_allowed(self, client):
        resp = client.delete("/config")
        assert resp.status_code == 405

    def test_patch_config_not_allowed(self, client):
        resp = client.patch("/config")
        assert resp.status_code == 405


# ===================================================================
# 6. /config does not leak secrets
# ===================================================================

class TestConfigSecurityCheck:
    """Verify /config never exposes sensitive values."""

    def test_no_api_key_in_response(self):
        app = _make_app({"OPENAI_API_KEY": "sk-super-secret-key-123"})
        client = app.test_client()

        data = client.get("/config").get_json()
        response_str = str(data)
        assert "sk-super-secret" not in response_str
        assert "OPENAI_API_KEY" not in response_str

    def test_no_model_in_response(self):
        app = _make_app()
        client = app.test_client()

        data = client.get("/config").get_json()
        assert "model" not in data
        assert "OPENAI_MODEL" not in data
