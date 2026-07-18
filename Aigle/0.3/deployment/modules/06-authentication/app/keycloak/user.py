import requests

from app.keycloak.master import KeycloakMaster
from app.keycloak.client import KeycloakClient
from app.core.config import settings
from app.schemas.users import UserCreateRequest, UserRepresentation, UserCredentials, KeycloakPayload


class KeycloakUser:
    @staticmethod
    def user_create_api_basic(user_data: UserCreateRequest, realm: str, token: str, credentials: UserCredentials = None, email_verified: bool = True):
        """使用傳入的 admin token 建立使用者（Pydantic schema 版）。
        email_verified=True（預設）：user 直接可登入。
        email_verified=False（SMTP 已設定時）：user 需驗證 email 並設定新密碼後才能登入。"""
        if credentials is None:
            credentials = UserCredentials()
        user_repr = UserRepresentation(
            **user_data.model_dump(exclude={"groups", "realmRoles", "clientRoles"}),
            emailVerified=email_verified,
        )
        payload = KeycloakPayload.user_payload(user=user_repr, credentials=credentials)
        if not email_verified:
            payload["requiredActions"] = ["VERIFY_EMAIL", "UPDATE_PASSWORD"]
        return requests.post(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/users",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=payload,
        )

    @staticmethod
    def get_user_id_by_username(username: str, realm: str, token: str) -> str:
        """建立後取得使用者 UUID（使用 exact=true 避免前綴誤判）。"""
        response = requests.get(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/users",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            params={"username": username, "exact": "true"},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to look up user '{username}': {response.status_code}")
        users = response.json()
        if not users:
            raise ValueError(f"User '{username}' not found after creation")
        return users[0]["id"]

    @staticmethod
    def get_user_id_from_token(token: str, realm: str) -> str:
        """呼叫 Keycloak /userinfo 驗證 token 簽章並回傳使用者 UUID（sub claim）。"""
        response = requests.get(
            f"{settings.KEYCLOAK_URL}/realms/{realm}/protocol/openid-connect/userinfo",
            headers={"Authorization": f"Bearer {token}"},
        )
        if response.status_code != 200:
            raise ValueError(f"Invalid or expired token: {response.status_code}")
        return response.json()["sub"]

    @staticmethod
    def assign_user_to_group(user_id: str, group_id: str, realm: str, token: str):
        """PUT /admin/realms/{realm}/users/{userId}/groups/{groupId} — 回傳 204 表示成功。"""
        return requests.put(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/users/{user_id}/groups/{group_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )

    @staticmethod
    def get_realm_role(role_name: str, realm: str, token: str) -> dict:
        """取得 realm role representation（含 id 與 name）。找不到時 raise ValueError。"""
        response = requests.get(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/roles/{role_name}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        if response.status_code == 404:
            raise ValueError(f"Realm role '{role_name}' not found")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch role '{role_name}': {response.status_code}")
        return response.json()

    @staticmethod
    def assign_realm_roles_to_user(user_id: str, roles: list, realm: str, token: str):
        """POST /admin/realms/{realm}/users/{userId}/role-mappings/realm。roles 為 [{"id":"...","name":"..."}]。"""
        return requests.post(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/users/{user_id}/role-mappings/realm",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=roles,
        )

    @staticmethod
    def get_client_uuid(client_id: str, realm: str, token: str) -> str:
        """透過 clientId 名稱取得 Keycloak client UUID。找不到時 raise ValueError。"""
        response = requests.get(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/clients",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            params={"clientId": client_id},
        )
        if response.status_code != 200:
            raise RuntimeError(f"Failed to query client '{client_id}': {response.status_code}")
        clients = response.json()
        exact_match = next((c for c in clients if c.get("clientId") == client_id), None)
        if not exact_match:
            raise ValueError(f"Client '{client_id}' not found in realm '{realm}'")
        return exact_match["id"]

    @staticmethod
    def get_client_role(role_name: str, client_uuid: str, realm: str, token: str) -> dict:
        """取得 client role representation（含 id, name）。找不到時 raise ValueError。"""
        response = requests.get(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/clients/{client_uuid}/roles/{role_name}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
        if response.status_code == 404:
            raise ValueError(f"Client role '{role_name}' not found in client '{client_uuid}'")
        if response.status_code != 200:
            raise RuntimeError(f"Failed to fetch client role '{role_name}': {response.status_code}")
        return response.json()

    @staticmethod
    def assign_client_roles_to_user(user_id: str, client_uuid: str, roles: list, realm: str, token: str):
        """POST /admin/realms/{realm}/users/{userId}/role-mappings/clients/{clientUUID}。
        roles 為 [{"id":"...","name":"..."}]，回傳 204 表示成功。"""
        return requests.post(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/users/{user_id}/role-mappings/clients/{client_uuid}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=roles,
        )

    @staticmethod
    def user_get_basic_api_username(username: str, realm: str, token: str):
        """依 username 查詢使用者基本資料。"""
        return requests.get(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/users",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            params={"username": username},
        )

    @staticmethod
    def user_get_basic_api_email(email: str, realm: str, token: str):
        """依 email 查詢使用者基本資料。"""
        return requests.get(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/users",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            params={"email": email},
        )

    @staticmethod
    def user_delete_api(user_id: str, realm: str, token: str):
        """依 user_id 刪除使用者。"""
        return requests.delete(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/users/{user_id}",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )

    @staticmethod
    def set_user_password(user_id: str, new_password: str, realm: str, token: str):
        """設定使用者密碼（非暫時）。"""
        return requests.put(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/users/{user_id}/reset-password",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json={"type": "password", "value": new_password, "temporary": False},
        )

    @staticmethod
    def send_verify_email_api(user_id: str, realm: str, token: str):
        """PUT /admin/realms/{realm}/users/{userId}/execute-actions-email — 送出驗證 email。"""
        return requests.put(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm}/users/{user_id}/execute-actions-email",
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
            json=["VERIFY_EMAIL"],
        )

    @staticmethod
    def user_login_secret_client(username, password, realm_name, client_id):
        data = {
            "username": username,
            "password": password,
            "grant_type": "password",
            "client_id": client_id,
            "scope": "openid profile email",
        }
        try:
            secret_resp = KeycloakClient.get_client_secret(
                realm_name, KeycloakMaster.MasterToken(), client_id
            )
            if secret_resp.status_code == 200:
                secret = secret_resp.json().get("value")
                if secret:
                    data["client_secret"] = secret
        except Exception:
            pass
        return requests.post(
            f"{settings.KEYCLOAK_URL}/realms/{realm_name}/protocol/openid-connect/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=data,
        )

    @staticmethod
    def user_logout(user_token, realm_name):
        if not isinstance(user_token, str):
            user_token = user_token["access_token"] if isinstance(user_token, dict) else user_token.json()["access_token"]

        user_id = KeycloakUser.get_user_id_from_token(user_token, realm_name)
        master_token = KeycloakMaster.MasterToken()

        return requests.post(
            f"{settings.KEYCLOAK_URL}/admin/realms/{realm_name}/users/{user_id}/logout",
            headers={"Authorization": f"Bearer {master_token}"},
        )
