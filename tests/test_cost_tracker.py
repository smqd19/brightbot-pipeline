"""Unit tests for CostTracker."""

from src.cost_tracker import CostTracker


def test_record_usage():
    """Tokens accumulate across multiple recordings."""
    tracker = CostTracker()
    tracker.record_usage("s1", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    tracker.record_usage("s1", {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30})

    usage = tracker.get_usage("s1")
    assert usage.prompt_tokens == 30
    assert usage.completion_tokens == 15
    assert usage.total_tokens == 45
    assert usage.request_count == 2


def test_check_budget_within_limit():
    """Returns True when session is under the token limit."""
    tracker = CostTracker(max_session_tokens=100)
    tracker.record_usage("s1", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    assert tracker.check_budget("s1") is True


def test_check_budget_exceeded():
    """Returns False when session has exceeded the token limit."""
    tracker = CostTracker(max_session_tokens=100)
    tracker.record_usage("s1", {"prompt_tokens": 60, "completion_tokens": 50, "total_tokens": 110})
    assert tracker.check_budget("s1") is False


def test_clear_resets():
    """clear() resets usage to zero for a session."""
    tracker = CostTracker()
    tracker.record_usage("s1", {"prompt_tokens": 50, "completion_tokens": 50, "total_tokens": 100})
    tracker.clear("s1")

    usage = tracker.get_usage("s1")
    assert usage.total_tokens == 0
    assert usage.request_count == 0


def test_to_dict():
    """SessionUsage.to_dict() returns a plain dict."""
    tracker = CostTracker()
    tracker.record_usage("s1", {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15})
    d = tracker.get_usage("s1").to_dict()
    assert d == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
        "request_count": 1,
    }
