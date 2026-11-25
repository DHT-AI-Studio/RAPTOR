"""Weekly password rotation for designated accounts.

Generates a strong random password for each account in the account list,
pushes it to Keycloak via the admin API, and emails the new password to
the account's contact address.

Runs as a long-lived process (see docker-compose.yml's password-rotator
entrypoint): computes seconds until the next occurrence of
ROTATE_WEEKDAY / ROTATE_HOUR, sleeps, rotates every configured account,
and repeats — forever. No state is persisted anywhere: "next Monday
09:00" is recomputed fresh from the current wall-clock time on every
iteration, so restarting the sidecar (e.g. to pick up an edited account
list) just resumes the same weekly schedule — it never fires an extra
run and never skips one.

The account list is read fresh every run from ROTATE_ACCOUNTS_FILE (a
bind-mounted file, edited on the host — no container restart needed) if
set, otherwise from the ROTATE_ACCOUNTS env var.

For onboarding a brand-new account without waiting for the next
scheduled day, run a one-off pass manually (all accounts):
    docker exec -e ROTATE_RUN_ONCE=1 raptor-password-rotator \
        python -m app.scripts.rotate_passwords
or just a single account (leaves everyone else untouched):
    docker exec -e ROTATE_RUN_ONCE=1 -e ROTATE_ONLY=someuser \
        raptor-password-rotator python -m app.scripts.rotate_passwords

Account list format (one per line in the file, or comma-separated in
the env var): "username1:email1,username2:email2,..."
"""
import datetime
import logging
import os
import secrets
import string
import sys
import time

from app.keycloak.master import KeycloakMaster
from app.keycloak.user import KeycloakUser
from app.routers.email_system import _send

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("rotate_passwords")

REALM = os.getenv("ROTATE_REALM", "dhtsolution")
PASSWORD_LENGTH = int(os.getenv("ROTATE_PASSWORD_LENGTH", "20"))
BCC_EMAIL = os.getenv("ROTATE_BCC_EMAIL", "").strip() or None
_ALPHABET = string.ascii_letters + string.digits + "!@#%^&*"

_WEEKDAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def _generate_password(length: int = PASSWORD_LENGTH) -> str:
    return "".join(secrets.choice(_ALPHABET) for _ in range(length))


def _parse_accounts(raw: str):
    """Comment/blank filtering happens per line first (so a comment line's
    own punctuation, e.g. a comma, can never split it into a bogus entry),
    then each remaining line is comma-split to support both the one-per-line
    file format and the single-line comma-separated env var format."""
    accounts = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        for entry in line.split(","):
            entry = entry.strip()
            if not entry:
                continue
            username, _, email = entry.partition(":")
            username, email = username.strip(), email.strip()
            if not username or not email:
                logger.warning("Skipping malformed account entry: %r", entry)
                continue
            accounts.append((username, email))
    return accounts


def _load_accounts():
    """ROTATE_ACCOUNTS_FILE (re-read every run, no restart needed to edit)
    takes precedence over the ROTATE_ACCOUNTS env var (baked in at
    container creation)."""
    file_path = os.getenv("ROTATE_ACCOUNTS_FILE", "").strip()
    if file_path and os.path.isfile(file_path):
        with open(file_path, "r", encoding="utf-8") as fh:
            return _parse_accounts(fh.read())
    return _parse_accounts(os.getenv("ROTATE_ACCOUNTS", ""))


def _group_by_username(accounts):
    """Collapse repeated 'username:email' lines for the same username into
    one entry with a list of emails — so a shared account with several
    recipients gets exactly one password reset, sent to everyone, instead
    of one reset per line (which would silently invalidate every password
    but the last one generated)."""
    grouped = {}
    for username, email in accounts:
        grouped.setdefault(username, [])
        if email not in grouped[username]:
            grouped[username].append(email)
    return grouped


def rotate_all() -> int:
    """Rotate every configured account (or just ROTATE_ONLY, if set — one
    username, handy for a manual one-off run that shouldn't touch anyone
    else in the list). Returns the number of failures."""
    accounts = _load_accounts()
    if not accounts:
        logger.warning("No accounts configured (ROTATE_ACCOUNTS_FILE / ROTATE_ACCOUNTS) — nothing to rotate")
        return 0

    only_user = os.getenv("ROTATE_ONLY", "").strip()
    if only_user:
        accounts = [a for a in accounts if a[0] == only_user]
        if not accounts:
            logger.warning("ROTATE_ONLY=%r not found in the configured account list — nothing to rotate", only_user)
            return 0

    token = KeycloakMaster.MasterToken()
    failures = 0
    for username, emails in _group_by_username(accounts).items():
        try:
            user_id = KeycloakUser.get_user_id_by_username(username, REALM, token)
            new_password = _generate_password()
            resp = KeycloakUser.set_user_password(user_id, new_password, REALM, token)
            if resp.status_code != 204:
                raise RuntimeError(f"reset-password failed: {resp.status_code} {resp.text}")
            body = (
                f"您好，\n\n"
                f"帳號 {username} 的密碼已依照每週輪替政策更新，新密碼如下：\n\n"
                f"{new_password}\n\n"
                f"請妥善保管，切勿轉發。\n"
            )
            for i, email in enumerate(emails):
                # Only BCC once per account, not once per recipient — the
                # admin only needs to see the password one time.
                _send(
                    to=email,
                    subject="RAPTOR 帳號密碼更新通知",
                    body=body,
                    bcc=BCC_EMAIL if i == 0 else None,
                )
            logger.info("Rotated password for '%s' and emailed %s", username, ", ".join(emails))
        except Exception as exc:  # noqa: BLE001 -- spans Keycloak requests calls, RuntimeError, and smtplib send; batch loop logs and continues
            failures += 1
            logger.error("Failed to rotate '%s': %s", username, exc)
    return failures


def _seconds_until_next_run(target_weekday: int, target_hour: int, target_minute: int = 0) -> float:
    """target_weekday: Monday=0 ... Sunday=6 (matches datetime.weekday())."""
    now = datetime.datetime.now()
    days_ahead = (target_weekday - now.weekday()) % 7
    candidate = (now + datetime.timedelta(days=days_ahead)).replace(
        hour=target_hour, minute=target_minute, second=0, microsecond=0
    )
    if candidate <= now:
        candidate += datetime.timedelta(days=7)
    return (candidate - now).total_seconds()


def run_forever():
    weekday = int(os.getenv("ROTATE_WEEKDAY", "0"))  # 0=Monday
    hour = int(os.getenv("ROTATE_HOUR", "9"))
    minute = int(os.getenv("ROTATE_MINUTE", "0"))
    while True:
        wait_s = _seconds_until_next_run(weekday, hour, minute)
        logger.info(
            "Next rotation in %.0fs (%s %02d:%02d)", wait_s, _WEEKDAY_NAMES[weekday], hour, minute
        )
        time.sleep(wait_s)
        rotate_all()


if __name__ == "__main__":
    if os.getenv("ROTATE_RUN_ONCE"):
        sys.exit(1 if rotate_all() else 0)
    run_forever()
