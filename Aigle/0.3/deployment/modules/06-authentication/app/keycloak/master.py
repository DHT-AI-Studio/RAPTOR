from pathlib import Path
import requests
import json

from app.core.config import settings

PROJECT_ROOT_DIR = Path(__file__).parent.parent.parent


class KeycloakMaster:
    @staticmethod
    def MasterToken():
        resp = requests.post(
            f"{settings.KEYCLOAK_URL}/realms/master/protocol/openid-connect/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "username": settings.PERMANENT_MASTER_ADMIN_USER,
                "password": settings.PERMANENT_ADMIN_PASSWORD,
                "grant_type": "password",
                "client_id": "admin-cli",
            },
        )
        if resp.status_code != 200:
            raise RuntimeError(f"Failed to retrieve access token from Keycloak: {resp.text}")
        return resp.json()["access_token"]

    @staticmethod
    def get_realm():
        realms_url = f"{settings.KEYCLOAK_URL}/admin/realms"
        headers = {"Authorization": f"Bearer {KeycloakMaster.MasterToken()}"}
        realms_response = requests.get(realms_url, headers=headers)
        realms_response.raise_for_status()
        return realms_response

    @staticmethod
    def get_realm_json():
        realms_url = f"{settings.KEYCLOAK_URL}/admin/realms"
        headers = {"Authorization": f"Bearer {KeycloakMaster.MasterToken()}"}
        realms_response = requests.get(realms_url, headers=headers)
        realms_response.raise_for_status()
        realms_list = {r["realm"]: r for r in realms_response.json()}
        json_path = PROJECT_ROOT_DIR / "config" / "realms.json"
        with open(json_path, "w") as f:
            json.dump(realms_list, f, indent=4)
        return realms_list

    @staticmethod
    def get_realm_key(realm: str):
        realms_url = f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/keys"
        headers = {"Authorization": f"Bearer {KeycloakMaster.MasterToken()}"}
        return requests.get(realms_url, headers=headers)
