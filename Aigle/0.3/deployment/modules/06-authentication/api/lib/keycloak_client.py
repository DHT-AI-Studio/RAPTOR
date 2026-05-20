# keycloak/scripts/keyclak_cient.py

from pathlib import Path
from dotenv import load_dotenv
import os
from jose import jwt
import requests
import json

# from keycloak.scripts.keycloak_master import KeycloakMaster

from lib.keycloak_master import KeycloakMaster

# PATH
PROJECT_ROOT_DIR = Path(__file__).parent.parent
THIS_FILE_DIR = Path(__file__).parent
DOTENV_PATH = PROJECT_ROOT_DIR / ".env"

# Get environment variables
load_dotenv(dotenv_path=DOTENV_PATH) # Specify the .env file path
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL")
PERMANENT_MASTER_ADMIN_USER = os.getenv("PERMANENT_MASTER_ADMIN_USER")
PERMANENT_ADMIN_PASSWORD = os.getenv("PERMANENT_ADMIN_PASSWORD")

class KeycloakClient:
    @staticmethod
    def get_client_info(realm_name, admin_token, client_name):
        '''
        gettint client information
        if want to get "client id" -> resp.json()[0]["id"]
        '''
        url = f"{KEYCLOAK_URL}/admin/realms/{realm_name}/clients"
        headers = {"Authorization": f"Bearer {admin_token}"}
        params = {"clientId": client_name}

        resp = requests.get(url, headers=headers, params=params)
        return resp
    
    @staticmethod
    def get_client_secret(realm_name, admin_token, client_name):
        '''
        responce body
        {
            "type": "secret",
            "value": "0123456789abcdefghijklmnopqrstuv"
        }
        If want to get client secret value -> KeycloakClient.get_client_secret(realm_name, admin_token, client_name).json()["value"]
        '''
        client_uuid = KeycloakClient.get_client_info(realm_name, admin_token, client_name).json()[0]["id"]
        url = f"{KEYCLOAK_URL}/admin/realms/{realm_name}/clients/{client_uuid}/client-secret"
        headers = {"Authorization": f"Bearer {admin_token}"}

        resp = requests.get(url, headers=headers)
        # data = resp.json()
        
        return resp
        
        
'''    
if __name__ == "__main__":
    client_info = KeycloakClient.get_client_info("dhtsolution", KeycloakMaster.MasterToken(), "raptor")
    client_id = client_info.json()[0]["id"]

    client_secret = KeycloakClient.get_client_secret("dhtsolution", KeycloakMaster.MasterToken(), "raptor")
    print(client_secret.json()["value"])
'''

