import requests

from app.core.config import settings

UMA_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:uma-ticket"


class KeycloakAuthentication:
    @staticmethod
    def check_permission_by_uri(realm_name: str, user_token: str, resource_uri: str, client_id: str):
        """UMA：以 URI 查詢權限，回傳原始 response 供呼叫端判斷狀態碼。"""
        url = f"{settings.KEYCLOAK_URL}/realms/{realm_name}/protocol/openid-connect/token"
        data = {
            "grant_type": UMA_GRANT_TYPE,
            "audience": client_id,
            "permission": resource_uri,
            "permission_resource_format": "uri",
            "permission_resource_matching_uri": "true",
        }
        headers = {
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        return requests.post(url, data=data, headers=headers)

    @staticmethod
    def check_permission(realm_name: str, user_token: str, resource_server_client_id: str):
        """UMA：不指定資源，只確認 user 對該 client 是否有任何權限，回傳原始 response。"""
        data = {
            "grant_type": UMA_GRANT_TYPE,
            "audience": resource_server_client_id,
        }
        headers = {
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        return requests.post(
            f"{settings.KEYCLOAK_URL}/realms/{realm_name}/protocol/openid-connect/token",
            data=data,
            headers=headers,
            timeout=5,
        )

    @staticmethod
    def check_permission_by_resource_name(
        realm_name: str,
        user_token: str,
        resource_server_client_id: str,
        resource_name: str,
    ) -> bool:
        """UMA：以 Keycloak 資源名稱（非 URI）查詢權限，回傳 True/False。"""
        data = {
            "grant_type": UMA_GRANT_TYPE,
            "audience": resource_server_client_id,
            "permission": resource_name,
        }
        headers = {
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        resp = requests.post(
            f"{settings.KEYCLOAK_URL}/realms/{realm_name}/protocol/openid-connect/token",
            data=data,
            headers=headers,
            timeout=5,
        )
        if resp.status_code == 200:
            return True
        if resp.status_code in (401, 403):
            return False
        raise RuntimeError(f"Keycloak UMA error {resp.status_code}: {resp.text}")
