"""Shared test fixtures."""

from unittest.mock import MagicMock, patch

import pytest

from src.app import create_app


def _make_mock_completion(content: str = "Hello from BrightBot!",
                          prompt_tokens: int = 25,
                          completion_tokens: int = 10,
                          total_tokens: int = 35):
    """Build a fake OpenAI ChatCompletion response."""
    choice = MagicMock()
    choice.message.content = content

    usage = MagicMock()
    usage.prompt_tokens = prompt_tokens
    usage.completion_tokens = completion_tokens
    usage.total_tokens = total_tokens

    completion = MagicMock()
    completion.choices = [choice]
    completion.usage = usage
    return completion


@pytest.fixture()
def app():
    """Create a test app with mocked OpenAI client."""
    test_config = {
        "OPENAI_API_KEY": "test-key",
        "OPENAI_MODEL": "gpt-4o-mini",
        "MAX_HISTORY": 5,
        "MAX_SESSION_TOKENS": 200,
    }
    application = create_app(config_override=test_config)
    application.config["TESTING"] = True

    # Patch the OpenAI client on the app
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = _make_mock_completion()
    application.openai_client = mock_client

    return application


@pytest.fixture()
def client(app):
    """Flask test client."""
    return app.test_client()


@pytest.fixture()
def mock_openai(app):
    """Provide direct access to the mocked OpenAI client for assertions."""
    return app.openai_client
