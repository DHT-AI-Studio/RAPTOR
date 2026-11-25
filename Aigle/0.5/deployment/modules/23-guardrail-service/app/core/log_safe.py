"""Log-safe description of screened text (VIE01-354).

Service logs must not carry the text users send any more than the audit
tables do: log its length and a short SHA-256 prefix instead. The prefix is
the start of the same hash `raptor_audit` stores in `audit_events.content_hash`,
so a log line can still be matched to its audit row.
"""
from __future__ import annotations

from raptor_audit import hash_value

_HASH_PREFIX = 12


def describe(text: str | None) -> str:
    """`len=<chars> sha=<first 12 hex of SHA-256>` — never the text itself."""
    if text is None:
        return "len=0 sha=-"
    return f"len={len(text)} sha={hash_value(text)[:_HASH_PREFIX]}"
