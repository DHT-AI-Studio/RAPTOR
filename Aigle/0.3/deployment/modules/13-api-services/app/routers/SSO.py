from fastapi import (
    APIRouter,
    HTTPException,
    Form,
    Query,
    Response,
    Cookie,
    Depends,
    Request
)
from fastapi.security import OAuth2PasswordBearer
import httpx

from app.core.config import get_settings

router = APIRouter()

# OAuth2 Password Flow for Swagger UI - points to SSO login endpoint
bearer_scheme = OAuth2PasswordBearer(tokenUrl=f"/api/{get_settings().api_version}/sso/login", auto_error=True)

# ==============================
# Downstream Auth Service
# ==============================

TIMEOUT = 60


# ==============================
# LOGIN (Proxy)
# ==============================
@router.post("/login", tags=["Single Sign-On"])
async def login(
    response: Response,
    username: str = Form(...),
    password: str = Form(...),
    realm_name: str = Form("dhtsolution"),
    client_id: str = Query("raptor"),
):
    """
    Proxy login request to Auth Service.
    """
    settings = get_settings()
    auth_url = f"{settings.auth_service_url}/login"

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                auth_url,
                data={
                    "username": username,
                    "password": password,
                    "realm_name": realm_name,
                },
                params={
                    "client_id": client_id
                }
            )

        if resp.status_code != 200:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise HTTPException(status_code=resp.status_code, detail=detail)

        # 取得 token，統一成 OAuth2 標準格式讓 Swagger UI 能正確提取
        raw = resp.json()
        if isinstance(raw, str):
            token_response = {"access_token": raw, "token_type": "bearer"}
        elif isinstance(raw, dict) and "access_token" not in raw:
            token_response = {"access_token": str(raw), "token_type": "bearer"}
        else:
            token_response = raw
        access_token = token_response["access_token"]

        # 設定 cookie（跟 1.py 一樣）
        response.set_cookie(
            key="realm_name",
            value=realm_name,
            httponly=True,
            secure=False,
            samesite="Lax",
        )

        response.set_cookie(
            key="client_id",
            value=client_id,
            httponly=True,
            secure=False,
            samesite="Lax",
        )

        response.set_cookie(
            key="access_token",
            value=access_token,
            httponly=True,
            secure=True,
            samesite="Lax",
            max_age=1800,
        )

        return token_response

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Auth service unreachable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ==============================
# LOGOUT (Proxy)
# ==============================
@router.post("/logout", tags=["Single Sign-On"])
async def logout(
    request: Request,
    response: Response,
    token: str = Depends(bearer_scheme),
    realm_name: str = Cookie(None),
):
    """
    Proxy logout to Auth Service.
    """
    settings = get_settings()
    auth_url = f"{settings.auth_service_url}/logout"

    if not realm_name:
        raise HTTPException(status_code=400, detail="Missing realm_name cookie")

    try:
        async with httpx.AsyncClient(timeout=TIMEOUT) as client:
            resp = await client.post(
                auth_url,
                headers={
                    "Authorization": f"Bearer {token}"
                },
                cookies={
                    "realm_name": realm_name
                }
            )

        if resp.status_code not in (200, 204):
            raise HTTPException(
                status_code=resp.status_code,
                detail=f"Logout failed: {resp.text}"
            )

        # 刪除所有 cookie
        for cookie_name in request.cookies.keys():
            response.delete_cookie(cookie_name)

        return {"message": "Logout successful, all cookies cleared"}

    except httpx.RequestError as e:
        raise HTTPException(status_code=502, detail=f"Auth service unreachable: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
