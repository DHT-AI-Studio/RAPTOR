import os

# app.core.config.Settings() is constructed eagerly at import time and
# requires these even though nothing in tests/unit/
# touches live infra -- set harmless placeholders before any test module
# imports the package, so unit tests run standalone in CI with no .env.
# setdefault() never overrides a real value already in the environment
# (e.g. from tests/test_erasure_e2e.py's own live-deployment env vars).
for key, default in {
    "GD_INTERNAL_TOKEN": "unused",
    "GD_POSTGRES_PASSWORD": "unused",
    "GD_KEYCLOAK_ADMIN_USERNAME": "unused",
    "GD_KEYCLOAK_ADMIN_PASSWORD": "unused",
}.items():
    os.environ.setdefault(key, default)
