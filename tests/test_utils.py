"""Tests for utility functions."""

import pytest
from lovli.utils import extract_chat_history


def test_extract_chat_history_empty():
    """Test extracting chat history from empty list."""
    result = extract_chat_history([])
    assert result == []


def test_extract_chat_history_single_message():
    """Test extracting chat history with single message."""
    messages = [{"role": "user", "content": "Test question"}]
    result = extract_chat_history(messages, exclude_current=True)
    assert result == []


def test_extract_chat_history_multiple_messages():
    """Test extracting chat history with multiple messages."""
    messages = [
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
        {"role": "user", "content": "Second question"},
    ]
    result = extract_chat_history(messages, window_size=4, exclude_current=True)
    assert len(result) == 2
    assert result[0]["role"] == "user"
    assert result[0]["content"] == "First question"
    assert result[1]["role"] == "assistant"
    assert result[1]["content"] == "First answer"


def test_extract_chat_history_filters_empty():
    """Test that empty messages are filtered out."""
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "assistant", "content": ""},  # Empty
        {"role": "user", "content": "Another question"},
    ]
    result = extract_chat_history(messages, exclude_current=True)
    # After excluding current (last) message, we have 2 messages, but 1 is empty
    # So we should get 1 message (the non-empty one)
    assert len(result) == 1
    assert result[0]["content"] == "Question"
    assert all(msg["content"] for msg in result)


def test_extract_chat_history_window_size():
    """Test that window size limits messages."""
    messages = [
        {"role": "user", "content": f"Question {i}"}
        for i in range(10)
    ]
    result = extract_chat_history(messages, window_size=3, exclude_current=True)
    assert len(result) == 3
    assert result[0]["content"] == "Question 6"
    assert result[-1]["content"] == "Question 8"


def test_extract_chat_history_filters_roles():
    """Test that only user/assistant messages are included."""
    messages = [
        {"role": "user", "content": "Question"},
        {"role": "system", "content": "System message"},  # Should be filtered
        {"role": "assistant", "content": "Answer"},
    ]
    result = extract_chat_history(messages, exclude_current=True)
    # After excluding current (assistant) message, we have 2 messages left
    # But system message is filtered out, so we get 1 message (user)
    assert len(result) == 1
    assert result[0]["role"] == "user"
    assert all(msg["role"] in ("user", "assistant") for msg in result)
