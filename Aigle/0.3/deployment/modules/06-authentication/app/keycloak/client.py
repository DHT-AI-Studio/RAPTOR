import requests

from app.core.config import settings


class KeycloakClient:
    @staticmethod
    def get_client_info(realm_name, admin_token, client_name):
        """Get client information. To get UUID: resp.json()[0]["id"]"""
        url = f"{settings.KEYCLOAK_URL}/admin/realms/{realm_name}/clients"
        headers = {"Authorization": f"Bearer {admin_token}"}
        params = {"clientId": client_name}
        return requests.get(url, headers=headers, params=params)

    @staticmethod
    def get_client_secret(realm_name, admin_token, client_name):
        """Get client secret. To get value: .json()["value"]"""
        client_uuid = KeycloakClient.get_client_info(realm_name, admin_token, client_name).json()[0]["id"]
        url = f"{settings.KEYCLOAK_URL}/admin/realms/{realm_name}/clients/{client_uuid}/client-secret"
        headers = {"Authorization": f"Bearer {admin_token}"}
        return requests.get(url, headers=headers)
