# app/routers/login_user.py

from fastapi import APIRouter, HTTPException, Form, Query, Response, Cookie, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from keycloak.scripts.keycloak_user import KeycloakUser


router = APIRouter()

bearer_scheme = HTTPBearer(auto_error=True)  # 這裡 auto_error=True，沒有 token 就 401

@router.post("/login", tags=["Singo Sign-On"])
def login(
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    realm_name: str = Form("dhtsolution"),
    client_id: str = Query("raptor", enum=["raptor"])
):
    """
    使用者登入，透過 Keycloak 密碼模式取得 JWT access token。
    The user authenticates via Keycloak using the password grant flow to obtain a JWT access token.
    """
    try:
        resp = KeycloakUser.user_login_secret_client(
            username=username,
            password=password,
            realm_name=realm_name,
            client_id=client_id
        )

        if resp.status_code != 200:
            error_desc = resp.json().get("error_description", resp.text)
            raise HTTPException(status_code=resp.status_code, detail=error_desc)
        
        # 設定輔助 cookie (不可被 JS 讀取)
        response.set_cookie(
            key="realm_name",
            value=realm_name,
            httponly=True,
            secure=False,  # HTTPS時改成 True
            samesite="Lax"
        )
        response.set_cookie(
            key="client_id",
            value=client_id,
            httponly=True,
            secure=False,
            samesite="Lax"
        )

        # 儲存 token 到 HTTPOnly Cookie（前端看不到、不能修改）
        response.set_cookie(
            key="access_token",
            value=resp.json()["access_token"],
            httponly=True,
            secure=True,
            samesite="Lax",
            max_age=1800  # 秒數，30 分鐘
        )

        # Return the JWT access token directly.
        return resp.json()["access_token"]

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logout", tags=["Singo Sign-On"])
def logout(
    request: Request,
    response: Response,
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    realm_name: str = Cookie(None)  # 從 cookie 取得 realm_name
):
    """
    使用 JWT token 登出 Keycloak。
    token 必須透過 Authorization: Bearer <token> 傳入。
    同時刪除所有 cookie。
    """
    if not realm_name:
        raise HTTPException(status_code=400, detail="Missing realm_name cookie")

    try:
        # 呼叫 Keycloak 登出
        resp = KeycloakUser.user_logout(token.credentials, realm_name)
        if resp.status_code != 204:
            raise HTTPException(status_code=resp.status_code, detail=f"Logout failed: {resp.text}")

        # 刪除所有 cookie
        for cookie_name in request.cookies.keys():
            response.delete_cookie(cookie_name)

        return {"message": "Logout successful, all cookies cleared"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))