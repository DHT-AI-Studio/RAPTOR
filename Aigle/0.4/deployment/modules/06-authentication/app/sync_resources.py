"""Sync Keycloak authorization resource URIs via Admin REST API.

Usage:
    python sync_resources.py                     # dry-run, shows current vs desired
    python sync_resources.py --apply             # apply changes
    python sync_resources.py --apply --verbose   # apply + show full diff

Environment variables (all have defaults matching .env.example):
    KEYCLOAK_URL              default: http://localhost:8080
    KEYCLOAK_ADMIN_USERNAME   default: admin
    KEYCLOAK_ADMIN_PASSWORD   (required)
    KEYCLOAK_REALM            default: dhtsolution
    KEYCLOAK_CLIENT_ID        default: raptor
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Any, Dict, List

import requests

from app.core.config import settings

# ---------------------------------------------------------------------------
# Config from environment
# ---------------------------------------------------------------------------
KC_URL    = settings.KEYCLOAK_URL.rstrip("/")
KC_ADMIN  = settings.PERMANENT_MASTER_ADMIN_USER
KC_PASS   = settings.PERMANENT_ADMIN_PASSWORD
KC_REALM  = settings.KEYCLOAK_REALM
KC_CLIENT = settings.KEYCLOAK_CLIENT_ID


def _uris_from_env(env_var: str, fallback: List[str]) -> List[str]:
    raw = os.getenv(env_var, "").strip()
    if raw:
        return [u.strip() for u in raw.split(",") if u.strip()]
    return fallback


# Only used if KEYCLOAK_INFERENCE_URIS / KEYCLOAK_FINETUNING_URIS are unset —
# normal deployments set both from .env.example, where the paths are built
# from the shared API_VERSION var instead of being repeated here.
_API_VERSION = os.getenv("API_VERSION", "0.4")

DESIRED_RESOURCES: Dict[str, List[str]] = {
    "raptor:inference:resource": _uris_from_env(
        "KEYCLOAK_INFERENCE_URIS",
        [
            f"/api/{_API_VERSION}/search/*",
            f"/api/{_API_VERSION}/asset/*",
            f"/api/{_API_VERSION}/processing/*",
            f"/api/{_API_VERSION}/chat/*",
            f"/api/{_API_VERSION}/document/*",
            f"/api/{_API_VERSION}/sync/*",
            f"/api/{_API_VERSION}/a2a/*",
            f"/api/{_API_VERSION}/memory/*",
            f"/api/{_API_VERSION}/personal-db/*",
            f"/api/{_API_VERSION}/mcp/*",
            f"/api/{_API_VERSION}/benchmark/*",
            "/API/inference/*",
        ],
    ),
    "raptor:fine-tuning:resource": _uris_from_env(
        "KEYCLOAK_FINETUNING_URIS",
        [
            "/API/finetune/*",
            f"/api/{_API_VERSION}/aiml/*",
            f"/api/{_API_VERSION}/training/*",
        ],
    ),
}


def wait_for_keycloak(retries: int, delay: int) -> None:
    url = f"{KC_URL}/realms/master"
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, timeout=5)
            if resp.status_code < 500:
                return
        except requests.RequestException:
            pass
        print(f"  Waiting for Keycloak ({attempt}/{retries})...")
        time.sleep(delay)
    raise RuntimeError(f"Keycloak not ready after {retries * delay}s")


def get_admin_token() -> str:
    resp = requests.post(
        f"{KC_URL}/realms/master/protocol/openid-connect/token",
        data={
            "grant_type": "password",
            "client_id": "admin-cli",
            "username": KC_ADMIN,
            "password": KC_PASS,
        },
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def get_client_uuid(token: str) -> str:
    resp = requests.get(
        f"{KC_URL}/admin/realms/{KC_REALM}/clients",
        headers={"Authorization": f"Bearer {token}"},
        params={"clientId": KC_CLIENT},
        timeout=10,
    )
    resp.raise_for_status()
    clients = resp.json()
    if not clients:
        raise RuntimeError(f"Client '{KC_CLIENT}' not found in realm '{KC_REALM}'")
    return clients[0]["id"]


def list_resources(token: str, client_uuid: str) -> List[Dict[str, Any]]:
    resp = requests.get(
        f"{KC_URL}/admin/realms/{KC_REALM}/clients/{client_uuid}/authz/resource-server/resource",
        headers={"Authorization": f"Bearer {token}"},
        params={"max": 100},
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def update_resource(token: str, client_uuid: str, resource: Dict[str, Any]) -> None:
    resp = requests.put(
        f"{KC_URL}/admin/realms/{KC_REALM}/clients/{client_uuid}"
        f"/authz/resource-server/resource/{resource['_id']}",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=resource,
        timeout=10,
    )
    resp.raise_for_status()


def sync(apply: bool, verbose: bool, retries: int, retry_delay: int) -> None:
    if not KC_PASS:
        print("ERROR: KEYCLOAK_ADMIN_PASSWORD is not set", file=sys.stderr)
        sys.exit(1)

    print(f"Connecting to {KC_URL} (realm={KC_REALM}, client={KC_CLIENT})")
    wait_for_keycloak(retries, retry_delay)
    token = get_admin_token()
    client_uuid = get_client_uuid(token)
    resources = list_resources(token, client_uuid)

    resource_by_name = {r["name"]: r for r in resources}
    changes_needed = False

    for resource_name, desired_uris in DESIRED_RESOURCES.items():
        if resource_name not in resource_by_name:
            print(f"  SKIP  {resource_name} — not found in Keycloak")
            continue

        resource = resource_by_name[resource_name]
        current_uris = set(resource.get("uris", []))
        desired_set  = set(desired_uris)

        added   = desired_set - current_uris
        removed = current_uris - desired_set

        if not added and not removed:
            print(f"  OK    {resource_name}")
            if verbose:
                for uri in sorted(desired_set):
                    print(f"          {uri}")
            continue

        changes_needed = True
        print(f"  DIFF  {resource_name}")
        for uri in sorted(added):
            print(f"          + {uri}")
        for uri in sorted(removed):
            print(f"          - {uri}")

        if apply:
            resource["uris"] = sorted(desired_uris)
            update_resource(token, client_uuid, resource)
            print(f"          → applied")

    if not changes_needed:
        print("All resources are up to date.")
    elif not apply:
        print("\nDry-run complete. Run with --apply to apply changes.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sync Keycloak resource URIs")
    parser.add_argument("--apply",       action="store_true", help="Apply changes (default: dry-run)")
    parser.add_argument("--verbose",     action="store_true", help="Show all URIs even when unchanged")
    parser.add_argument("--retries",     type=int, default=12, help="Max attempts waiting for Keycloak (default: 12)")
    parser.add_argument("--retry-delay", type=int, default=10, help="Seconds between retries (default: 10)")
    args = parser.parse_args()
    sync(apply=args.apply, verbose=args.verbose, retries=args.retries, retry_delay=args.retry_delay)
