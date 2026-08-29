from fastapi import APIRouter, HTTPException, Form, Response, Cookie, Depends, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.keycloak.user import KeycloakUser

router = APIRouter()

bearer_scheme = HTTPBearer(auto_error=True)


@router.post("/login", tags=["Public"])
def login(
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    realm_name: str = Form("dhtsolution"),
    client_id: str = Form("raptor")
):
    """
    使用者登入，透過 Keycloak 密碼模式取得 JWT access token。
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

        response.set_cookie(key="realm_name", value=realm_name, httponly=True, secure=False, samesite="Lax")
        response.set_cookie(key="client_id", value=client_id, httponly=True, secure=False, samesite="Lax")
        response.set_cookie(
            key="access_token", value=resp.json()["access_token"],
            httponly=True, secure=False, samesite="Lax", max_age=1800
        )

        return resp.json()["access_token"]

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/logout", tags=["User"])
def logout(
    request: Request,
    response: Response,
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    realm_name: str = Cookie(None)
):
    """
    使用 JWT token 登出 Keycloak。token 必須透過 Authorization: Bearer <token> 傳入。
    同時刪除所有 cookie。
    """
    if not realm_name:
        raise HTTPException(status_code=400, detail="Missing realm_name cookie")

    try:
        resp = KeycloakUser.user_logout(token.credentials, realm_name)
        if resp.status_code != 204:
            raise HTTPException(status_code=resp.status_code, detail=f"Logout failed: {resp.text}")

        for cookie_name in request.cookies.keys():
            response.delete_cookie(cookie_name)

        return {"message": "Logout successful, all cookies cleared"}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
