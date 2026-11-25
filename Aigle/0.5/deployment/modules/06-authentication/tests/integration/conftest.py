import os

from app.core.config import settings


# Module 06's own .env carries docker-internal hostnames (raptor-keycloak:8080)
# and placeholder credentials -- none of that is valid from a host-side test
# process. The real credentials, and the ports docker-compose actually
# forwards to the host, live in the unified deployment/modules/.env that
# build.py passes to every container's environment. Parsed once here instead
# of hardcoding magic port numbers or re-deriving credentials per test file
# (same pattern as 04-object-storage/tests/integration/conftest.py).
def _load_unified_env():
    path = os.path.join(os.path.dirname(__file__), "..", "..", "..", ".env")
    env = {}
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.split("#", 1)[0].strip()  # strip inline comments
                if "=" in line:
                    k, v = line.split("=", 1)
                    env[k.strip()] = v.strip()
    # Resolve ${VAR}-style self-references, which docker-compose expands at
    # container-start time but a plain line parser doesn't.
    for k, v in env.items():
        if v.startswith("${") and v.endswith("}"):
            env[k] = env.get(v[2:-1], v)
    return env


_unified = _load_unified_env()

_host = _unified.get("HOST_IP", "localhost")

# Real credentials -- Module 06's own settings are reused for everything
# else (KEYCLOAK_REALM/KEYCLOAK_CLIENT_ID), only the host-visible pieces are
# overridden here.
settings.KEYCLOAK_URL = f"http://{_host}:{_unified.get('PORT_KEYCLOAK', '8080')}"
if "PERMANENT_MASTER_ADMIN_USER" in _unified:
    settings.PERMANENT_MASTER_ADMIN_USER = _unified["PERMANENT_MASTER_ADMIN_USER"]
if "PERMANENT_ADMIN_PASSWORD" in _unified:
    settings.PERMANENT_ADMIN_PASSWORD = _unified["PERMANENT_ADMIN_PASSWORD"]

# Module 06's own FastAPI service -- real HTTP calls against it need the
# same host-visible port, not the docker-internal one.
MODULE06_BASE_URL = f"http://{_host}:{_unified.get('PORT_KEYCLOAK_API', '8800')}"
KC_BASE_URL = settings.KEYCLOAK_URL

ADMIN_USER = settings.PERMANENT_MASTER_ADMIN_USER
ADMIN_PASS = settings.PERMANENT_ADMIN_PASSWORD
ADMIN_REALM = "master"
ADMIN_CLIENT = "admin-cli"

TEST_REALM = settings.KEYCLOAK_REALM
TEST_CLIENT = settings.KEYCLOAK_CLIENT_ID

# Self-register test flow doesn't need a deliverable inbox -- it only
# checks the API response shape and simulates verification via a direct
# Keycloak admin call, never actually reads an email. Overridable via the
# unified .env for anyone who does want to eyeball a real delivery.
REG_EMAIL = _unified.get("TEST_REG_EMAIL", "pytest-selfreg@example.com")
