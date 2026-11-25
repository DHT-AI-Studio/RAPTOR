"""GD-5 / VIE01-250: registration webhook to Module 33 (privacy compliance).

Called once a new user is fully created -- by POST /SSO/register and
POST /admin/keycloak/user, the only two places users are created -- so
Module 33 sets the default consent attributes and policy version.

Best-effort by design: a failure is logged, never raised. Rolling back a
freshly created account over consent defaults would be worse than the miss,
and Module 33 writes the same defaults the first time the user's consent is
read anyway.
"""
import logging

import requests

from app.core.config import settings

logger = logging.getLogger(__name__)


def notify_new_user(user_id: str) -> bool:
    if not settings.GD_INTERNAL_TOKEN:
        logger.warning("GD_INTERNAL_TOKEN not set -- consent defaults for new user %s not sent to Module 33", user_id)
        return False
    try:
        resp = requests.post(
            f"{settings.PRIVACY_COMPLIANCE_URL}/internal/gdpr/new-user",
            headers={"X-Internal-Token": settings.GD_INTERNAL_TOKEN},
            json={"user_id": user_id},
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except requests.RequestException as e:
        logger.error("Module 33 new-user webhook failed for user %s: %s", user_id, e)
        return False
