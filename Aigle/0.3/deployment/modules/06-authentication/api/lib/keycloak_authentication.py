# keycloak/scripts/keycloak_master.py

from pathlib import Path
from dotenv import load_dotenv
import os
from jose import jwt
import requests
import json

# from keycloak.scripts.keycloak_user import KeycloakUser
# from keycloak.scripts.keycloak_client import KeycloakClient
# from keycloak.scripts.keycloak_master import KeycloakMaster

from lib.keycloak_user import KeycloakUser
from lib.keycloak_client import KeycloakClient
from lib.keycloak_master import KeycloakMaster

# PATH
PROJECT_ROOT_DIR = Path(__file__).parent.parent
THIS_FILE_DIR = Path(__file__).parent
DOTENV_PATH = PROJECT_ROOT_DIR / ".env"

# Get environment variables
load_dotenv(dotenv_path=DOTENV_PATH) # Specify the .env file path
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL")



UMA_GRANT_TYPE = "urn:ietf:params:oauth:grant-type:uma-ticket"

class KeycloakAuthentication:
    def is_permission_by_uri(realm_name: str, user_token: str, resource_uri: str, client_id: str) -> bool:
        """
        使用 Keycloak UMA Ticket 透過 Resource 的 URIs 驗證權限
        """
        url = f"{KEYCLOAK_URL}/realms/{realm_name}/protocol/openid-connect/token"

        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:uma-ticket",
            "audience": client_id,
            "permission": f"{resource_uri}",  # 使用 Resource 的 URI
            "permission_resource_format": "uri",
            "permission_resource_matching_uri": "true" # permission_resource_matching_uri=ture -> 子路徑也可以
                                                        # permission_resource_matching_uri=false(default) -> 路徑必須完全相同
        }

        headers = {
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        resp = requests.post(url, data=data, headers=headers)

        if resp.status_code == 200:
            return True
        elif resp.status_code in (401, 403):
            return False
        
        raise RuntimeError(f"Keycloak UMA error {resp.status_code}: {resp.text}")
    
    def check_permission_by_uri(realm_name: str, user_token: str, resource_uri: str, client_id: str):
        """
        使用 Keycloak UMA Ticket 透過 Resource 的 URIs 驗證權限
        """
        url = f"{KEYCLOAK_URL}/realms/{realm_name}/protocol/openid-connect/token"

        data = {
            "grant_type": "urn:ietf:params:oauth:grant-type:uma-ticket",
            "audience": client_id,
            "permission": f"{resource_uri}",  # 使用 Resource 的 URI
            "permission_resource_format": "uri",
            "permission_resource_matching_uri": "true" # permission_resource_matching_uri=ture -> 子路徑也可以
                                                        # permission_resource_matching_uri=false(default) -> 路徑必須完全相同
        }

        headers = {
            "Authorization": f"Bearer {user_token}",
            "Content-Type": "application/x-www-form-urlencoded"
        }

        resp = requests.post(url, data=data, headers=headers)
        return resp
        if resp.status_code != 200:
            raise RuntimeError(f"Keycloak UMA error {resp.status_code}: {resp.text}")
        return resp.json()
        

def check_permission(
    realm_name: str,
    user_access_token: str,
    resource_server_client_id: str,
    #resource_name: str,
    #scope: str,
):
    """
    使用 UMA Authorization 向 Keycloak 檢查是否有存取權限
    """

    data = {
        "grant_type": UMA_GRANT_TYPE,
        "audience": resource_server_client_id,
        #"permission": f"{resource_name}#{scope}",
        #"permission": f"{resource_name}",
    }

    headers = {
        "Authorization": f"Bearer {user_access_token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    resp = requests.post(
        f"{KEYCLOAK_URL}/realms/{realm_name}/protocol/openid-connect/token",
        data=data,
        headers=headers,
        timeout=5,
    )
    return resp
    # 200 = 允許
    if resp.status_code == 200:
        #print(json.dumps(resp.json(), indent=2))
        return resp
        return True

    # 403 / 401 = 拒絕
    if resp.status_code in (401, 403):
        #print(json.dumps(resp.json(), indent=2))
        return False

    # 其他錯誤
    raise RuntimeError(
        f"Keycloak UMA error {resp.status_code}: {resp.text}"
    )

def check_permission_by_resource_name(
    realm_name: str,
    user_access_token: str,
    resource_server_client_id: str,
    resource_name: str,
    #scope: str,
) -> bool:
    """
    使用 UMA Authorization 向 Keycloak 檢查是否有存取權限
    """

    data = {
        "grant_type": UMA_GRANT_TYPE,
        "audience": resource_server_client_id,
        #"permission": f"{resource_name}#{scope}",
        "permission": f"{resource_name}",
    }

    headers = {
        "Authorization": f"Bearer {user_access_token}",
        "Content-Type": "application/x-www-form-urlencoded",
    }

    resp = requests.post(
        f"{KEYCLOAK_URL}/realms/{realm_name}/protocol/openid-connect/token",
        data=data,
        headers=headers,
        timeout=5,
    )

    # 200 = 允許
    if resp.status_code == 200:
        #print(json.dumps(resp.json(), indent=2))
        return True

    # 403 / 401 = 拒絕
    if resp.status_code in (401, 403):
        #print(json.dumps(resp.json(), indent=2))
        return False

    # 其他錯誤
    raise RuntimeError(
        f"Keycloak UMA error {resp.status_code}: {resp.text}"
    )


def check_permission_by_uri(realm_name: str, user_token: str, resource_uri: str, client_id: str):
    """
    使用 Keycloak UMA Ticket 透過 Resource 的 URIs 驗證權限
    """
    url = f"{KEYCLOAK_URL}/realms/{realm_name}/protocol/openid-connect/token"

    data = {
        "grant_type": "urn:ietf:params:oauth:grant-type:uma-ticket",
        "audience": client_id,
        "permission": f"{resource_uri}",  # 使用 Resource 的 URI
        "permission_resource_format": "uri",
        "permission_resource_matching_uri": "true" # permission_resource_matching_uri=ture -> 子路徑也可以
                                                    # permission_resource_matching_uri=false(default) -> 路徑必須完全相同
    }

    headers = {
        "Authorization": f"Bearer {user_token}",
        "Content-Type": "application/x-www-form-urlencoded"
    }

    resp = requests.post(url, data=data, headers=headers)
    if resp.status_code != 200:
        raise RuntimeError(f"Keycloak UMA error {resp.status_code}: {resp.text}")
    return resp.json()

'''
if __name__ == "__main__":
    print('-'*10, "standard user", '-'*10)
    resp = KeycloakUser.user_login_secret_client("test_standarduser", "12345", "dhtsolution", "raptor")
    permission = KeycloakAuthentication.is_permission_by_uri("dhtsolution", resp.json()["access_token"], "/API/finetune/", "raptor")
    print(f"fintrune: {permission}")
    permission = KeycloakAuthentication.is_permission_by_uri("dhtsolution", resp.json()["access_token"], "/API/inference/", "raptor")
    print(f"inference: {permission}")
    KeycloakUser.user_logout(resp.json()["access_token"], "dhtsolution")

    print('-'*10, "basic user", '-'*10)
    resp = KeycloakUser.user_login_secret_client("test_basicuser", "12345", "dhtsolution", "raptor")
    permission = KeycloakAuthentication.is_permission_by_uri("dhtsolution", resp.json()["access_token"], "/API/finetune/", "raptor")
    print(f"fintrune: {permission}")
    permission = KeycloakAuthentication.is_permission_by_uri("dhtsolution", resp.json()["access_token"], "/API/inference/", "raptor")
    print(f"inference: {permission}")
    KeycloakUser.user_logout(resp.json()["access_token"], "dhtsolution")
'''