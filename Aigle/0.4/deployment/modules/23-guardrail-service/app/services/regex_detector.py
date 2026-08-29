"""Regex/keyword detector (GB-5).

Fast, deterministic detection for known jailbreak / PII patterns — no LLM call.
Patterns come from a policy rule (`rule.patterns`); when a rule provides none we
fall back to the default jailbreak set shipped here. Patterns are compiled once
and cached (case-insensitive by default).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

# Shipped defaults — overridable per-rule via policy `patterns`.
DEFAULT_JAILBREAK_PATTERNS: list[str] = [
    r"ignore (?:all |previous )*instructions",   # catches "ignore [all] [previous] instructions"
    r"you are now .{0,50} without (any )?restrictions",
    r"pretend you (are|have) no (safety|ethical) (guidelines|constraints)",
    r"DAN mode",
    r"developer mode",
]


@lru_cache(maxsize=1024)
def _compiled(pattern: str) -> re.Pattern:
    """Compile-once, cached, case-insensitive."""
    return re.compile(pattern, re.IGNORECASE)


@dataclass
class RegexMatch:
    matched: bool
    matched_pattern: str | None = None


def detect(content: str, patterns: list[str] | None = None) -> RegexMatch:
    """Return the first matching pattern (or a no-match result).

    `patterns` defaults to DEFAULT_JAILBREAK_PATTERNS when None/empty.
    """
    for pattern in (patterns or DEFAULT_JAILBREAK_PATTERNS):
        if _compiled(pattern).search(content or ""):
            return RegexMatch(matched=True, matched_pattern=pattern)
    return RegexMatch(matched=False, matched_pattern=None)
