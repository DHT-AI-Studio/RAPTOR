"""Where guardrail decisions get audited — one seam for every policy-aware
check (app/services/policy_check_service.py and POST /guardrail/policy/decide).

Writes through app/services/audit_log.py: a CF-4 (VIE01-234) `audit_events`
row plus its linked `guardrail_violations` row, hash-only (content and
user_id are stored as SHA-256), fire-and-forget — never blocks or fails the
caller.
"""
from __future__ import annotations

from uuid import UUID

from app.services import audit_log


def record_decision(*, policy_id: UUID | None, module: str | None, direction: str, action_taken: str,
                    category: str | None, request_id: str | None, content: str | None = None,
                    policy_version: str | None = None, user_id: str | None = None,
                    session_id: str | None = None) -> None:
    """Schedule the audit write and return immediately."""
    audit_log.record_violation(
        policy_id=policy_id, module=module, direction=direction, category=category,
        action_taken=action_taken, request_id=request_id, content=content,
        user_id=user_id, session_id=session_id, policy_version=policy_version,
    )
