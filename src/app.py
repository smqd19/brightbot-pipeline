"""Flask application factory and route definitions."""

import logging

from flask import Flask, request, jsonify
from openai import OpenAI

logger = logging.getLogger(__name__)

from src.session_store import SessionStore
from src.cost_tracker import CostTracker


def create_app(config_override: dict | None = None) -> Flask:
    """Create and configure the Flask application.

    Args:
        config_override: Optional dict to override config values (useful for testing).

    Returns:
        Configured Flask app instance.
    """
    app = Flask(__name__)

    # Load config — allow overrides for testing
    if config_override:
        api_key = config_override.get("OPENAI_API_KEY", "test-key")
        model = config_override.get("OPENAI_MODEL", "gpt-4o-mini")
        system_prompt = config_override.get(
            "SYSTEM_PROMPT",
            "You are BrightBot, a helpful assistant for the Brightly team.",
        )
        max_history = config_override.get("MAX_HISTORY", 50)
        max_session_tokens = config_override.get("MAX_SESSION_TOKENS", 100_000)
    else:
        from src.config import (
            OPENAI_API_KEY as api_key,
            OPENAI_MODEL as model,
            SYSTEM_PROMPT as system_prompt,
            MAX_HISTORY as max_history,
            MAX_SESSION_TOKENS as max_session_tokens,
        )

    # Store config on the app for access in routes
    app.config["OPENAI_MODEL"] = model
    app.config["SYSTEM_PROMPT"] = system_prompt

    # Initialize dependencies
    app.session_store = SessionStore(max_history=max_history)
    app.cost_tracker = CostTracker(max_session_tokens=max_session_tokens)
    app.openai_client = OpenAI(api_key=api_key)

    _register_routes(app)
    return app


def _register_routes(app: Flask) -> None:
    """Register all route handlers on the app."""

    @app.route("/chat", methods=["POST"])
    def chat():
        """Handle a chat message.

        Accepts JSON: {"message": "...", "session_id": "..."}
        Returns JSON:  {"response": "...", "session_id": "...", "usage": {...}}
        """
        data = request.get_json(silent=True)
        if not data or not isinstance(data, dict):
            return jsonify({"error": "Request body must be valid JSON."}), 400

        message = data.get("message")
        if message is None or not str(message).strip():
            return jsonify({"error": "message is required."}), 400

        message = str(message).strip()

        # Resolve session
        session_id = app.session_store.get_or_create_session(
            data.get("session_id")
        )

        # Check token budget before calling OpenAI
        if not app.cost_tracker.check_budget(session_id):
            usage = app.cost_tracker.get_usage(session_id)
            return jsonify({
                "error": "Session token limit exceeded. Start a new session to continue.",
                "session_id": session_id,
                "usage": usage.to_dict(),
            }), 429

        # Build message list for OpenAI
        history = app.session_store.get_trimmed_history(session_id)
        messages = (
            [{"role": "system", "content": app.config["SYSTEM_PROMPT"]}]
            + history
            + [{"role": "user", "content": message}]
        )

        # Call OpenAI
        try:
            completion = app.openai_client.chat.completions.create(
                model=app.config["OPENAI_MODEL"],
                messages=messages,
                temperature=0.7,
            )
            assistant_content = completion.choices[0].message.content or ""
        except Exception as e:
            logger.exception("OpenAI API call failed for session %s", session_id)
            return jsonify({"error": "OpenAI API error. Please try again later."}), 502

        # Record token usage
        usage_dict = {
            "prompt_tokens": completion.usage.prompt_tokens,
            "completion_tokens": completion.usage.completion_tokens,
            "total_tokens": completion.usage.total_tokens,
        }
        app.cost_tracker.record_usage(session_id, usage_dict)

        # Persist messages in session history
        app.session_store.append_message(session_id, "user", message)
        app.session_store.append_message(session_id, "assistant", assistant_content)

        return jsonify({
            "response": assistant_content,
            "session_id": session_id,
            "usage": usage_dict,
        }), 200

    @app.route("/config", methods=["GET"])
    def config():
        """Return non-secret configuration values."""
        return jsonify({"system_prompt": app.config["SYSTEM_PROMPT"]}), 200

    @app.route("/health", methods=["GET"])
    def health():
        """Simple health check endpoint."""
        return jsonify({"status": "ok"}), 200
