"""GD-5 / VIE01-250: consent purposes and records.

The current state of each purpose lives in Keycloak user attributes
(gdpr_consent_<purpose> = "true"/"false", plus gdpr_consent_version);
every change is also appended to the immutable gdpr_consent_history table,
which is where granted_at comes from.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum

from pydantic import BaseModel


class ConsentPurposeId(str, Enum):
    MEMORY = "memory"
    ANALYTICS = "analytics"
    BENCHMARK = "benchmark"
    VOICE_RECORDING = "voice_recording"
    FACE_DETECTION = "face_detection"
    VOICE_CLONE = "voice_clone"


@dataclass(frozen=True)
class ConsentPurpose:
    id: ConsentPurposeId
    description: str
    default: bool
    # Store whose data a withdrawal deletes (same module_XX keys as the
    # erasure job's store_statuses). None = no cascade.
    withdrawal_store: str | None = None

    @property
    def attribute(self) -> str:
        return f"gdpr_consent_{self.id.value}"


# The ticket's purpose table, in display order.
PURPOSES: dict[ConsentPurposeId, ConsentPurpose] = {
    p.id: p for p in (
        ConsentPurpose(
            ConsentPurposeId.MEMORY,
            "Store and recall conversation history (Module 26 MemVID)",
            default=True, withdrawal_store="module_26",
        ),
        ConsentPurpose(
            ConsentPurposeId.ANALYTICS,
            "Aggregate usage analytics",
            default=False,  # no cascade -- the data is already aggregate
        ),
        ConsentPurpose(
            ConsentPurposeId.BENCHMARK,
            "Store my benchmark schemas and results (Module 22)",
            default=True, withdrawal_store="module_22",
        ),
        ConsentPurpose(
            ConsentPurposeId.VOICE_RECORDING,
            "Record voice sessions as WAV files -- biometric data "
            "(GDPR Art. 9 special category / CCPA CPRA)",
            default=False,  # explicit opt-in required for biometric data
            withdrawal_store="module_34",
        ),
        ConsentPurpose(
            ConsentPurposeId.FACE_DETECTION,
            "Find people in my videos by their face (People Search) -- biometric data "
            "(GDPR Art. 9 special category)",
            # MTR-24: People Search is denied unless granted, so explicit opt-in.
            # GD-11: also gates face detection (Module 11) and seed photos (Module 13);
            # withdrawing deletes the face layer of the user's Module 25 database.
            default=False, withdrawal_store="module_25",
        ),
        ConsentPurpose(
            ConsentPurposeId.VOICE_CLONE,
            "Clone my voice from a reference recording (Module 35) -- biometric data "
            "(GDPR Art. 9 special category)",
            # GD-11: Module 35 /voice/clone is denied unless granted, so explicit opt-in.
            default=False, withdrawal_store="module_35",
        ),
    )
}

CONSENT_VERSION_ATTRIBUTE = "gdpr_consent_version"


class ConsentAction(str, Enum):
    GRANT = "grant"
    WITHDRAW = "withdraw"


class ConsentSource(str, Enum):
    """How a gdpr_consent_history row came about."""
    USER = "user"
    REGISTRATION_DEFAULT = "registration_default"
    # Users created before GD-5 (or whose registration webhook call failed)
    # get the defaults written the first time their consent is read.
    FIRST_READ_DEFAULT = "first_read_default"
    # PRIV-X3 / VIE01-357: a GDPR Art. 21 objection -- a withdrawal that stops
    # future processing without GD-5's deletion cascade.
    OBJECTION = "objection"


class ConsentRecord(BaseModel):
    """One purpose's current state, as returned by GET /consent."""
    id: ConsentPurposeId
    granted: bool
    # Most recent grant; null while withdrawn, or when the grant predates
    # the history table (never happens for users created after GD-5).
    granted_at: datetime | None = None
    description: str


class ConsentResponse(BaseModel):
    purposes: list[ConsentRecord]


class ConsentUpdateRequest(BaseModel):
    granted: bool


class ConsentUpdateResponse(BaseModel):
    id: ConsentPurposeId
    granted: bool
    # False when the purpose already had this value -- nothing recorded.
    changed: bool
    # Set when a withdrawal enqueued a partial erasure job.
    erasure_job_id: str | None = None


class NewUserRequest(BaseModel):
    user_id: str
