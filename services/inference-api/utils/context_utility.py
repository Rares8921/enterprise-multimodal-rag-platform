from typing import List, Dict, Any
import re

try:
    from ..config import Settings
except ImportError:  # Supports service-root imports used by tests and Docker.
    from config import Settings

settings = Settings() # used in main and here lol

def limit_context_size(context: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    total_chars = 0
    filtered = []

    for chunk in context:
        text = chunk.get('text', '')
        chunk_size = len(text)

        if total_chars + chunk_size > settings.max_context_chars:
            break

        filtered.append(chunk)
        total_chars += chunk_size

    return filtered

def sanitize_context(text: str) -> str:
    # TODO: add more obviously
    dangerous_patterns = [
        r"ign[o0]re\s+(all\s+)?previous\s+instructions?",
        r"disregard\s+previous",
        r"forget\s+everything",
        r"new\s+instructions?:",
        r"system\s*:",
        r"assistant\s*:",
        r"<\|im_start\|>",
        r"<\|im_end\|>",
        r"<\|system\|>",
        r"<\|user\|>",
        r"<\|assistant\|>",
        r"promptize",
        r"jailbreak",
        r"you\s+are\s+now",
        r"act\s+as\s+if",
        r"pretend\s+to\s+be",
        r"simulate\s+that",
        r"override\s+your",
        r"bypass\s+your"
    ]

    sanitized = text
    for pattern in dangerous_patterns:
        sanitized = re.sub(pattern, "[FILTERED]", sanitized, flags=re.IGNORECASE)

    # Remove control characters and excessive whitespace
    sanitized = ''.join(char for char in sanitized if ord(char) >= 32 or char in '\n\t')
    sanitized = re.sub(r'\s+', ' ', sanitized).strip()

    # Truncate extremely long single chunks
    if len(sanitized) > 8000:
        sanitized = sanitized[:8000] + "...[truncated]"

    return sanitized
