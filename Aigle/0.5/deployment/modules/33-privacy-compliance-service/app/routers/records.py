"""GD-9: Article 30 Record of Processing Activities (RoPA) endpoints.

GET /processing-records        → GDPR Article 30 machine-readable RoPA JSON
GET /processing-records/ccpa   → CCPA-formatted equivalent

Both routes require X-Internal-Token (enforced at the app level in main.py,
same as every other router in this service). The gdpr-admin role check is
Module 13's responsibility — this service only trusts the shared secret.

Controller and DPO contact are populated from GD_CONTROLLER_NAME and
GD_DPO_EMAIL env vars (see app/core/config.py).
"""
from __future__ import annotations

from datetime import date

from fastapi import APIRouter

from app.core.config import settings

router = APIRouter(prefix="/processing-records", tags=["records"])

# ── Article 30 RoPA activity catalogue ──────────────────────────────────────
#
# Each entry maps to one processing activity RAPTOR performs.
# Fields follow GDPR Article 30(1) exactly:
#   name, purposes, legal_basis, data_categories, data_subjects,
#   recipients, retention_period, technical_measures, third_country_transfers
#
# This is a static declaration of *what* is processed and *why*.
# Actual volumes and per-category retention windows live in the retention
# engine's policy table — GET /retention/policy for those.

_ACTIVITIES: list[dict] = [
    {
        "name": "User authentication and session management",
        "purposes": ["Authentication", "Session management", "Security"],
        "legal_basis": "Contract performance (Article 6(1)(b)); Legitimate interest (Article 6(1)(f))",
        "data_categories": ["Identity data", "Authentication credentials", "Session tokens"],
        "data_subjects": ["Registered users"],
        "recipients": ["Keycloak identity provider (internal)", "API gateway (internal)"],
        "retention_period": "Session tokens: until logout or expiry. Account data: until erasure request.",
        "technical_measures": ["TLS in transit", "Bcrypt password hashing", "JWT RS256 signing"],
        "third_country_transfers": None,
    },
    {
        "name": "AI/ML inference and chat completions",
        "purposes": ["Service delivery", "Natural language processing"],
        "legal_basis": "Contract performance (Article 6(1)(b))",
        "data_categories": ["Message content", "User identifiers", "Inference inputs and outputs"],
        "data_subjects": ["Registered users"],
        "recipients": ["LLM inference service (internal)", "Chat service (internal)"],
        "retention_period": "Governed by per-store retention policy (see GET /retention/policy).",
        "technical_measures": ["TLS in transit", "Per-user data isolation", "Guardrail content moderation"],
        "third_country_transfers": None,
    },
    {
        "name": "Personal knowledge base (documents, audio, images, video)",
        "purposes": ["Personal data storage", "Retrieval-augmented generation", "Search"],
        "legal_basis": (
            "Contract performance (Article 6(1)(b)); "
            "Consent for voice recordings (Article 6(1)(a))"
        ),
        "data_categories": [
            "Document content", "Audio recordings", "Images", "Video",
            "Extracted text and embeddings", "Metadata",
        ],
        "data_subjects": ["Registered users"],
        "recipients": [
            "Object storage / SeaweedFS (internal)",
            "Vector database / Qdrant (internal)",
            "Personal DB service / ArcadeDB (internal)",
        ],
        "retention_period": (
            "Until erasure request or consent withdrawal. "
            "Voice recordings: until voice_recording consent withdrawn."
        ),
        "technical_measures": [
            "TLS in transit",
            "Per-user bucket isolation",
            "Encrypted storage volumes",
        ],
        "third_country_transfers": None,
    },
    {
        # CM-5 / VIE01-408: automated flagging + admin review, distinct from the
        # automated guardrail checking already listed as a technical measure under
        # "AI/ML inference and chat completions" above -- this activity is a human
        # reviewer looking at the actual flagged asset, not an automated check.
        "name": "Content moderation review",
        "purposes": ["Trust and safety", "Content policy enforcement", "Illegal content prevention"],
        "legal_basis": "Legitimate interest (Article 6(1)(f)) -- platform safety and content policy enforcement",
        "data_categories": [
            "Document content", "Audio recordings", "Images", "Video",
            "Moderation verdicts", "Reviewer notes",
        ],
        "data_subjects": ["Registered users", "Moderation administrators (reviewers)"],
        "recipients": ["Guardrail service (internal)", "Moderation administrators (internal)"],
        "retention_period": (
            "Flag records: governed by per-store retention policy (see GET /retention/policy). "
            "Linked audit events: per GUARDRAIL_AUDIT_RETENTION_DAYS."
        ),
        "technical_measures": [
            "TLS in transit",
            "moderation-admin role gate (Keycloak)",
            "Every review access logged (GD-7 data access audit log)",
        ],
        "third_country_transfers": None,
    },
    {
        "name": "Memory service (session and long-term memory)",
        "purposes": ["Personalisation", "Conversation continuity"],
        "legal_basis": "Consent (Article 6(1)(a))",
        "data_categories": ["Conversation summaries", "User preferences", "Behavioural patterns"],
        "data_subjects": ["Registered users"],
        "recipients": ["Memory service (internal)", "Redis cache (internal)"],
        "retention_period": "Until memory consent withdrawn or erasure request.",
        "technical_measures": ["TLS in transit", "Per-user key namespacing in Redis"],
        "third_country_transfers": None,
    },
    {
        "name": "Benchmark and evaluation runs",
        "purposes": ["Quality assurance", "Model evaluation"],
        "legal_basis": "Consent (Article 6(1)(a))",
        "data_categories": [
            "Evaluation inputs", "Model outputs", "Scoring results", "User identifiers",
        ],
        "data_subjects": ["Registered users"],
        "recipients": ["Benchmark service (internal)", "MLflow tracking server (internal)"],
        "retention_period": "Until benchmark consent withdrawn or erasure request.",
        "technical_measures": ["TLS in transit", "Per-user run isolation"],
        "third_country_transfers": None,
    },
    {
        "name": "Data access audit logging",
        "purposes": ["GDPR compliance", "Security monitoring", "Accountability"],
        "legal_basis": "Legal obligation (Article 6(1)(c)) — GDPR Article 30",
        "data_categories": [
            "Access timestamps", "Endpoint paths",
            "Hashed accessor identifiers", "Hashed IP addresses",
        ],
        "data_subjects": ["Registered users", "Administrators"],
        "recipients": ["Privacy compliance service (internal)"],
        "retention_period": "365 days (governed by gdpr_access_log retention policy).",
        "technical_measures": [
            "TLS in transit",
            "IP address SHA-256 hashing before storage",
            "Append-only log table",
        ],
        "third_country_transfers": None,
    },
    {
        "name": "GDPR/CCPA rights fulfilment (erasure, export, DSAR, consent)",
        "purposes": ["Legal compliance", "Data subject rights fulfilment"],
        "legal_basis": "Legal obligation (Article 6(1)(c))",
        "data_categories": ["User identifiers", "Job status records", "Consent history"],
        "data_subjects": ["Registered users", "Former users (erasure job records)"],
        "recipients": [
            "Privacy compliance service (internal)",
            "DPO (email notification on partial failure)",
        ],
        "retention_period": (
            "Erasure job records: 3 years (legal hold). "
            "Consent history: 3 years."
        ),
        "technical_measures": [
            "TLS in transit",
            "Shared-secret service authentication (X-Internal-Token)",
        ],
        "third_country_transfers": None,
    },
]


def _build_ropa() -> dict:
    """Assemble the Article 30 RoPA document from static catalogue + env config."""
    return {
        "schema_version": "1.0",
        "generated_at": date.today().isoformat(),
        "controller": {
            "name": settings.controller_name,
            "dpo_contact": settings.dpo_email,
        },
        "record_of_processing_activities": _ACTIVITIES,
    }


def _build_ccpa() -> dict:
    """CCPA-formatted equivalent of the RoPA.

    Key differences from the GDPR version:
    - Top-level ``sells_data: false`` explicit declaration (CCPA § 1798.100)
    - ``shares_data_for_cross_context_behavioral_advertising: false``
    - ``legal_basis`` renamed to ``business_purpose``
    - ``data_subjects`` renamed to ``consumers``
    - ``third_country_transfers`` omitted (not a CCPA concept)
    """
    ccpa_activities = []
    for activity in _ACTIVITIES:
        ccpa_activities.append({
            "name": activity["name"],
            "business_purpose": activity["legal_basis"],
            "categories_of_personal_information": activity["data_categories"],
            "consumers": activity["data_subjects"],
            "service_providers": activity["recipients"],
            "retention_period": activity["retention_period"],
            "technical_measures": activity["technical_measures"],
        })

    return {
        "schema_version": "1.0",
        "generated_at": date.today().isoformat(),
        "framework": "CCPA",
        "sells_data": False,
        "shares_data_for_cross_context_behavioral_advertising": False,
        "business": {
            "name": settings.controller_name,
            "privacy_contact": settings.dpo_email,
        },
        "categories_of_personal_information_collected": ccpa_activities,
    }


@router.get(
    "",
    summary="Article 30 Record of Processing Activities (RoPA)",
    description=(
        "Machine-readable GDPR Article 30 RoPA. "
        "Controller and DPO contact are populated from GD_CONTROLLER_NAME "
        "and GD_DPO_EMAIL. Requires gdpr-admin role (enforced by Module 13 "
        "before proxying here)."
    ),
)
def get_processing_records() -> dict:
    return _build_ropa()


@router.get(
    "/ccpa",
    summary="CCPA-formatted Record of Processing Activities",
    description=(
        "CCPA equivalent of the Article 30 RoPA. "
        'Includes explicit ``"sells_data": false`` declaration per CCPA § 1798.100. '
        "Requires gdpr-admin role (enforced by Module 13)."
    ),
)
def get_ccpa_records() -> dict:
    return _build_ccpa()