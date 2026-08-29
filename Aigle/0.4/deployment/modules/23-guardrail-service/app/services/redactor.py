"""Content redactor (GB-5).

Replaces only the spans matched by the given patterns with `[REDACTED]`, leaving
the surrounding text intact. Reuses the regex_detector's compiled-pattern cache.
"""
from __future__ import annotations

from app.services.regex_detector import DEFAULT_JAILBREAK_PATTERNS, _compiled

REDACTED = "[REDACTED]"


def redact(content: str, patterns: list[str] | None = None, placeholder: str = REDACTED) -> str:
    """Return `content` with every matched span replaced by `placeholder`.

    Only the matched substring is replaced (surrounding context preserved).
    `patterns` defaults to DEFAULT_JAILBREAK_PATTERNS when None/empty.
    """
    out = content or ""
    for pattern in (patterns or DEFAULT_JAILBREAK_PATTERNS):
        out = _compiled(pattern).sub(placeholder, out)
    return out
