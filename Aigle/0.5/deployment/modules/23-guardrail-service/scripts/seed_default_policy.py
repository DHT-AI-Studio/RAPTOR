"""One-time migration: POST the default 13-category Standard Policy to a
running guardrail-service and activate it.

This is the operator-runnable counterpart to app/main.py's lifespan()
fallback (which, at startup, activates the default policy whenever no
policy is active — creating it first if it has never existed). Run this
script any time you want to (re-)seed the default policy explicitly — e.g.
to restore its built-in content after it was hand-edited, or to switch back
to it from another active policy without waiting for a restart. Note that
re-running it overwrites the default policy's raw_content with the built-in
13 categories. Both this script and the automatic fallback read the identical
DEFAULT_STANDARD_POLICY constant (app/services/default_policy.py), so they
can never drift apart.

Usage:
    python scripts/seed_default_policy.py [--url http://localhost:8023]

Safe to re-run: POSTing the same name+version again updates that policy's
content in place (app/db/repo.py's upsert_policy_content) rather than
creating a duplicate.
"""
from __future__ import annotations

import argparse
import json
import sys

import httpx

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])   # allow `from app...` when run directly

from app.services.default_policy import DEFAULT_STANDARD_POLICY

_POLICY_NAME = "default-standard-policy"
_POLICY_VERSION = "1.0"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8023", help="Base URL of the running guardrail-service")
    parser.add_argument("--name", default=_POLICY_NAME)
    parser.add_argument("--version", default=_POLICY_VERSION)
    args = parser.parse_args()

    with httpx.Client(timeout=30.0) as client:
        resp = client.post(
            f"{args.url}/guardrail/policies",
            params={"name": args.name, "version": args.version, "target": "original"},
            content=json.dumps(DEFAULT_STANDARD_POLICY),
            headers={"Content-Type": "application/json"},
        )
        resp.raise_for_status()
        policy_id = resp.json()["id"]
        print(f"created/updated policy {args.name} v{args.version} (id={policy_id})")

        resp = client.put(f"{args.url}/guardrail/policies/{policy_id}/activate")
        resp.raise_for_status()
        print(f"activated policy {args.name} v{args.version}")


if __name__ == "__main__":
    main()
