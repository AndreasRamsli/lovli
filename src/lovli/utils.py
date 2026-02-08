"""Utility functions for Lovli."""

from typing import Dict, Any


def extract_chat_history(
    messages: list[Dict[str, Any]], 
    window_size: int = 6,
    exclude_current: bool = True
) -> list[Dict[str, str]]:
    """
    Extract chat history for query rewriting.
    
    Args:
        messages: List of message dicts from session state
        window_size: Number of messages to include (default: 6)
        exclude_current: If True, exclude the last message (current question)
    
    Returns:
        List of message dicts with 'role' and 'content' keys, filtered to user/assistant only
    """
    if not messages:
        return []
    
    # Exclude current message if requested
    # If exclude_current=True, always exclude the last message (even if it's the only one)
    messages_to_process = messages[:-1] if exclude_current else messages
    
    # Filter and format messages
    chat_history = []
    for msg in messages_to_process[-window_size:]:
        role = msg.get("role", "").lower()
        if role in ("user", "assistant"):
            content = msg.get("content", "").strip()
            if content:  # Skip empty messages
                chat_history.append({
                    "role": role,
                    "content": content
                })
    
    return chat_history
