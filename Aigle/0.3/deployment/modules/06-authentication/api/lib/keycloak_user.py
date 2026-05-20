# keycloak/scripts/keyclak_user.py

from pathlib import Path
from dotenv import load_dotenv
import os
from jose import jwt
import requests
import json

# from keycloak.scripts.keycloak_master import KeycloakMaster
# from keycloak.scripts.keycloak_client import KeycloakClient

from lib.keycloak_master import KeycloakMaster
from lib.keycloak_client import KeycloakClient

# PATH
PROJECT_ROOT_DIR = Path(__file__).parent.parent
THIS_FILE_DIR = Path(__file__).parent
# DOTENV_PATH = PROJECT_ROOT_DIR / ".env"
DOTENV_PATH = Path(__file__).parent / ".env"

# Get environment variables
load_dotenv(dotenv_path=DOTENV_PATH) # Specify the .env file path
KEYCLOAK_URL = os.getenv("KEYCLOAK_URL")
PERMANENT_MASTER_ADMIN_USER = os.getenv("PERMANENT_MASTER_ADMIN_USER")
PERMANENT_ADMIN_PASSWORD = os.getenv("PERMANENT_ADMIN_PASSWORD")

class KeycloakUser:   
    @staticmethod
    def create_user(realm_name: str, admin_token: str, user_data: dict):
        """
        Create a Keycloak user using JSON input.
        Password becomes temporary and email verification is required.
        
        Expected JSON format:
        {
            "username": "john",
            "password": "TempPass123!",
            "email": "john@example.com",
            "firstName": "John",
            "lastName": "Doe"
        }
        """
        # input preprocessing
        # if not isinstance(user_data, dict):
        #     user_data = json.loads(user_data)

        print(f"🛠️👤 Creating user: {user_data.get('username')}")

        payload = {
            "username": user_data.get("username"),
            "email": user_data.get("email"),
            "enabled": True,
            "emailVerified": False,   # The user must verify their email.
            "firstName": user_data.get("firstName"),
            "lastName": user_data.get("lastName"),
            "credentials": [
                {
                    "type": "password",
                    "value": user_data.get("password"),
                    "temporary": True  # ⬅️ Temporary password
                }
            ],
            "requiredActions": [
                "VERIFY_EMAIL",      # ⬅️ Require email verification
                "UPDATE_PASSWORD"    # ⬅️ User must change password after login
            ]
        }

        # 1️⃣ creating user
        response = requests.post(
            f"{KEYCLOAK_URL}/admin/realms/{realm_name}/users",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Content-Type": "application/json"
            },
            json=payload
        )

        if response.status_code == 201:
            print("✅ Create user successful")
        elif response.status_code == 409:
            print("⚠️ User already exists")
        else:
            print(f"❌ User creation failed: {response.status_code}")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(response.text)
        
        # 2️⃣ getting user id by email
        user_id = KeycloakUser.get_user_info_from_email(realm_name, admin_token, user_data.get("email")).json()[0]["id"]

        # 3️⃣ send verify email to user
        KeycloakUser.send_verify_email(realm_name, admin_token, user_id)

        return response
    
    @staticmethod
    def create_user_str(realm_name: str, admin_token: str, username: str, password: str, email: str, firstName: str, lastName: str):
        """
        Create a Keycloak user using JSON input.
        Password becomes temporary and email verification is required.
        
        Expected JSON format:
        {
            "username": "john",
            "password": "TempPass123!",
            "email": "john@example.com",
            "firstName": "John",
            "lastName": "Doe"
        }
        """
        # input preprocessing
        # if not isinstance(user_data, dict):
        #     user_data = json.loads(user_data)

        print(f"🛠️👤 Creating user: {username}")
        
        payload = {
            "username": username,
            "email": email,
            "enabled": True,
            "emailVerified": False,   # The user must verify their email.
            "firstName": firstName,
            "lastName": lastName,
            "credentials": [
                {
                    "type": "password",
                    "value": password,
                    "temporary": True  # ⬅️ Temporary password
                }
            ],
            "requiredActions": [
                "VERIFY_EMAIL",      # ⬅️ Require email verification
                "UPDATE_PASSWORD"    # ⬅️ User must change password after login
            ]
        }

        # 1️⃣ creating user
        response = requests.post(
            f"{KEYCLOAK_URL}/admin/realms/{realm_name}/users",
            headers={
                "Authorization": f"Bearer {admin_token}",
                "Content-Type": "application/json"
            },
            json=payload
        )

        if response.status_code == 201:
            print("✅ Create user successful")
        elif response.status_code == 409:
            print("⚠️ User already exists")
        else:
            print(f"❌ User creation failed: {response.status_code}")
            try:
                print(json.dumps(response.json(), indent=2))
            except:
                print(response.text)
        
        # 2️⃣ getting user id by email
        user_id = KeycloakUser.get_user_info_from_email(realm_name, admin_token, email).json()[0]["id"]

        # 3️⃣ send verify email to user
        KeycloakUser.send_verify_email(realm_name, admin_token, user_id)

        return response
    
    def get_user_info_from_email(realm_name: str, admin_token: str, user_email: str):
        '''
        getting user information by email
        If want to get user id: get_user_info_from_email(realm_name, admin_token, user_email).json()[0]["id"]
        '''
        url = f"{KEYCLOAK_URL}/admin/realms/{realm_name}/users"
        headers={"Authorization": f"Bearer {admin_token}"}
        params={"email": user_email}

        user_info = requests.get(url, headers=headers, params=params)

        if user_info.status_code == 200:
            print("✅ Get user information successful")
        elif user_info.status_code == 403:
            print("⚠️ Forbidden")
        else:
            print(f"❌ User creation failed: {user_info.status_code}")
            try:
                print(json.dumps(user_info.json(), indent=2))
            except:
                print(user_info.text)

        return user_info
    
    def send_verify_email(realm_name: str, admin_token: str, user_id: str):
        '''
        send verify email to user
        '''
        verify = requests.put(
            f"{KEYCLOAK_URL}/admin/realms/{realm_name}/users/{user_id}/execute-actions-email",
            headers={"Authorization": f"Bearer {admin_token}"}
        )

        '''
        if verify.status_code == 204:
            print("✅ verif email is sended")
        elif verify.status_code == 400:
            print("⚠️ Forbidden")
        elif verify.status_code == 403:
            print("⚠️ Forbidden")
        else:
            print(f"❌ User creation failed: {verify.status_code}")
            try:
                print(json.dumps(verify.json(), indent=2))
            except:
                print(verify.text)
        '''

        return verify
    
    @staticmethod
    def user_login(username, password, realm_name, client_id):
        print("🧑‍💻 User logging")
        resp = requests.post(
            f"{KEYCLOAK_URL}/realms/{realm_name}/protocol/openid-connect/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "username": username,
                "password": password,
                "grant_type": "password",
                "client_id": client_id
            }
        )         
        
        if resp.status_code == 200:
            print("✅🧑‍💻 User logging successful")
            return resp
        
        return resp

    @staticmethod
    def user_login_secret_client(username, password, realm_name, client_id):
        print(f"user {username} is logging to secret client...")
        resp = requests.post(
            f"{KEYCLOAK_URL}/realms/{realm_name}/protocol/openid-connect/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data={
                "username": username,
                "password": password,
                "grant_type": "password",
                "client_id": client_id,
                "client_secret": KeycloakClient.get_client_secret(realm_name, KeycloakMaster.MasterToken(), client_id).json()["value"]
            }
        )         
        
        if resp.status_code == 200:
            print(f"✅ user {username} logging to {client_id} successful")
            return resp
        
        return resp
   
    @staticmethod
    def user_logout(user_token, realm_name):
        master_token = KeycloakMaster.MasterToken()
        realm_key = KeycloakMaster.get_realm_key(realm_name).json()
        public_key = realm_key["keys"][1]["publicKey"]  # 取 RS256 key
        public_key_pem = f"-----BEGIN PUBLIC KEY-----\n{public_key}\n-----END PUBLIC KEY-----"

        if type(user_token) != str:
            if type(user_token) == dict:
                user_token = user_token["access_token"]
            else:
                user_token = user_token.json()["access_token"]

        jwt_decoded = jwt.decode(
            user_token,
            public_key_pem,
            algorithms=["RS256"],
            options={"verify_aud": False}  # 不驗證 audience
            #audience=decode_payload(user_token)["aud"]   # 或你的 client_id
        )
        #print(json.dumps(jwt_decoded, indent=2))
        user_id = KeycloakUser.get_user_info_from_email(realm_name, master_token, jwt_decoded["email"]).json()[0]["id"]
        
        resp = requests.post(
            f"{KEYCLOAK_URL}/admin/realms/{realm_name}/users/{user_id}/logout",
            headers={"Authorization": f"Bearer {master_token}"}
        )
        if resp.status_code == 204:
            print("✅ User logout successful. All sessions invalidated!")
        else:
            print(f"❌ Logout failed: {resp.status_code}")
            print(resp.text)
        return resp
        

import base64
def decode_payload(token):
        payload_b64 = token.split('.')[1]
        payload_bytes = base64.urlsafe_b64decode(payload_b64 + '==')
        return json.loads(payload_bytes)
'''
if __name__ == "__main__":
    user_data = {
            "username": "minnie_test",
            "password": "password",
            "email": "minnie@dhtsolution.group",
            "firstName": "Minnie",
            "lastName": "Wei"
        }
    #KeycloakUser.create_user("dhtsolution", KeycloakMaster.MasterToken(), user_data)

    loging = KeycloakUser.user_login("minnie_test", "12345", "dhtsolution", "raptor")
   
    #print(json.dumps(loging.json(), indent=2))
    

      
    #logout = KeycloakUser.user_logout(loging.json()["access_token"], "dhtsolution")

    logout = KeycloakUser.user_logout_new(loging, "dhtsolution")
    print(logout)
'''


'''
if __name__ == "__main__":
    #KeycloakUser.create_user_str("dhtsolution", KeycloakMaster.MasterToken(), "minnie", "password", "minnie@dhtsolution.group", "minnie", "wei")
    resp = KeycloakUser.user_login_secret_client("test_standarduser", "12345", "dhtsolution", "raptor")
    print(resp)
    print(json.dumps(resp.json(), indent=2))


    KeycloakUser.user_logout(resp, "dhtsolution")
'''

