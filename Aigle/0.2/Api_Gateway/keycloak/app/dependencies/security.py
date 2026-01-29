# app/dependencies/request_context.py
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
from keycloak.scripts.keycloak_authentication import KeycloakAuthentication
#from app.routers.login import router

bearer_scheme = HTTPBearer(auto_error=True)  # 這裡 auto_error=True，沒有 token 就 401

class APIsecurity:
    @staticmethod
    async def authenticate(
        request: Request,
        token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
        ):
        """
        所有 API 都會執行這裡
        """
        # 基本請求資訊
        request_path = request.url.path
        user_token = token.credentials
        # request_method = request.method
        # full_url = str(request.url)


        # ✅ 讀取 cookies
        request.state.cookies = request.cookies

        # 單獨取某個 cookie
        realm_name = request.cookies.get("realm_name")
        client_id = request.cookies.get("client_id")

        # print(f"request_path: {request_path}")
        # print(f"realm_name: {realm_name} \nclient_id: {client_id}")
        # print(f"user_token: {user_token}")

        if not realm_name:
            raise HTTPException(status_code=400, detail=f"The realm_name does not exist. Please check your login status.")
        if not client_id:
            raise HTTPException(status_code=400, detail=f"The client_id does not exist. Please check your login status.")
            
        permission = KeycloakAuthentication.check_permission_by_uri(
            realm_name=realm_name,
            user_token = user_token,
            resource_uri = request_path,
            client_id = client_id
            )
        
        if permission.status_code in (401, 403):
            raise HTTPException(status_code=permission.status_code, detail=f"Authentication error {permission.status_code}: {permission.text}")