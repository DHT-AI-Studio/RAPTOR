# keycloak/scripts/keycloak_master.py

from pathlib import Path
from dotenv import load_dotenv
import os
from jose import jwt
import requests
import json


# PATH
PROJECT_ROOT_DIR = Path(__file__).parent.parent
THIS_FILE_DIR = Path(__file__).parent
DOTENV_PATH = PROJECT_ROOT_DIR / ".env"

# Get environment variables
load_dotenv(dotenv_path=DOTENV_PATH) # Specify the .env file path
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL")
PERMANENT_MASTER_ADMIN_USER = os.getenv("PERMANENT_MASTER_ADMIN_USER")
PERMANENT_ADMIN_PASSWORD = os.getenv("PERMANENT_ADMIN_PASSWORD")
PERMANENT_MASTER_ADMIN_USER = os.getenv("PERMANENT_MASTER_ADMIN_USER")
PERMANENT_ADMIN_PASSWORD = os.getenv("PERMANENT_ADMIN_PASSWORD")


class KeycloakMaster:
    @staticmethod
    def MasterToken():

        token_url = f"{KEYCLOAK_URL}/realms/master/protocol/openid-connect/token"
        resp = requests.post(
            token_url,
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "username": PERMANENT_MASTER_ADMIN_USER,
                "password": PERMANENT_ADMIN_PASSWORD,
                "grant_type": "password",
                "client_id": "admin-cli"
            }
        )
        if resp.status_code != 200:
            raise RuntimeError(f"❌ Failed to retrieve access token from Keycloak: {resp.text}")
        return resp.json()["access_token"]
    
    @staticmethod
    def get_realm():
        realms_url = f"{KEYCLOAK_URL}/admin/realms"
        headers = {"Authorization": f"Bearer {KeycloakMaster.MasterToken()}"}

        realms_response = requests.get(realms_url, headers=headers)

        return realms_response
        realms_response.raise_for_status()

        realms = realms_response.json()
        print("All Keycloak Realms:")
        for realm in realms:
            print(f"- {realm['realm']} (id: {realm['id']})")

    @staticmethod
    def get_realm_json():
        realms_url = f"{KEYCLOAK_URL}/admin/realms"
        headers = {"Authorization": f"Bearer {KeycloakMaster.MasterToken()}"}

        realms_response = requests.get(realms_url, headers=headers)
        realms_response.raise_for_status()

        realms = realms_response.json()
        # 只保留 realm 名稱與 id
        #realms_list = [{"realm": r["realm"], "id": r["id"]} for r in realms]
        #realms_list = [{r["realm"]: r} for r in realms]
        #realms_dict = {r["realm"]: r["id"] for r in realms} #錯誤寫法
        realms_list = {r["realm"]: r for r in realms}

        # 輸出成 JSON 
        json_path = PROJECT_ROOT_DIR / "config" / "realms.json"
        with open(json_path, "w") as f:
            json.dump(realms_list, f, indent=4)

        print("Realms have been saved to realms.json")
        return realms_list
    
    @staticmethod
    def get_realm_key(realm: str):
        realms_url = f"{KEYCLOAK_URL}/admin/realms/{realm}/keys"
        headers = {"Authorization": f"Bearer {KeycloakMaster.MasterToken()}"}

        realms_response = requests.get(realms_url, headers=headers)
        
        return realms_response
    

'''
if __name__ == "__main__":
    realm_key = KeycloakMaster.get_realm_key("dhtsolution").json()
    #print(json.dumps(realm_key.json(), indent=2))

    public_key = realm_key["keys"][1]["publicKey"]  # 取 RS256 key
    public_key_pem = f"-----BEGIN PUBLIC KEY-----\n{public_key}\n-----END PUBLIC KEY-----"
'''
