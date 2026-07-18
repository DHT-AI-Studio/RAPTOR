from fastapi import APIRouter, HTTPException, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from typing import Optional

from app.keycloak.group import KeycloakGroup
from app.keycloak.master import KeycloakMaster
from app.keycloak.user import KeycloakUser

router = APIRouter()

bearer_scheme = HTTPBearer(auto_error=True)


def _keycloak_error(response) -> str:
    try:
        body = response.json()
        msg = body.get("errorMessage") or body.get("error_description") or body.get("error") or str(body)
    except Exception:
        msg = response.text or f"HTTP {response.status_code}"
    if response.status_code == 403:
        return f"{msg} — insufficient permissions (admin token required)"
    return msg


@router.post("/group", tags=["Admin"])
def create_group(
    group_name: str,
    realm_name: str = Query("dhtsolution", enum=["dhtsolution"]),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """建立 Group。"""
    try:
        response = KeycloakGroup.creat_groups_api(
            group_name=group_name,
            realm=realm_name,
            token=token.credentials,
        )
        if response.status_code == 201:
            return JSONResponse(
                status_code=201,
                content={"message": f"Group {group_name} is created."},
            )
        raise HTTPException(status_code=response.status_code, detail=_keycloak_error(response))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/group", tags=["Admin"])
def get_groups(
    realm_name: str = Query("dhtsolution", enum=["dhtsolution"]),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """取得 Realm 內的所有 Group 清單。"""
    try:
        response = KeycloakGroup.get_groups_api(realm=realm_name, token=token.credentials)
        if response.status_code == 200:
            return JSONResponse(status_code=200, content=response.json())
        raise HTTPException(status_code=response.status_code, detail=_keycloak_error(response))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-groups", tags=["User"])
def get_user_groups(
    realm_name: str = Query("dhtsolution", enum=["dhtsolution"]),
    user_token: Optional[str] = Query(None, description="指定使用者的 access token，省略則使用 Bearer token"),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """取得使用者所屬的所有 Group 資訊。user_token 有值時查詢該 token 對應的使用者，否則查詢 Bearer token 使用者。"""
    try:
        target_token = user_token if user_token else token.credentials
        try:
            user_id = KeycloakUser.get_user_id_from_token(target_token, realm_name)
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))
        response = KeycloakGroup.get_user_groups_api(
            user_id=user_id,
            realm=realm_name,
            token=KeycloakMaster.MasterToken(),
        )
        if response.status_code == 200:
            return JSONResponse(status_code=200, content=response.json())
        raise HTTPException(status_code=response.status_code, detail=_keycloak_error(response))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user-group-id", tags=["User"])
def get_user_group_id(
    realm_name: str = Query("dhtsolution", enum=["dhtsolution"]),
    user_token: Optional[str] = Query(None, description="指定使用者的 access token，省略則使用 Bearer token"),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """取得使用者所屬所有 Group 的 ID 與路徑。user_token 有值時查詢該 token 對應的使用者，否則查詢 Bearer token 使用者。"""
    try:
        target_token = user_token if user_token else token.credentials
        try:
            user_id = KeycloakUser.get_user_id_from_token(target_token, realm_name)
        except ValueError as e:
            raise HTTPException(status_code=401, detail=str(e))
        response = KeycloakGroup.get_user_groups_api(
            user_id=user_id,
            realm=realm_name,
            token=KeycloakMaster.MasterToken(),
        )
        if response.status_code == 200:
            groups = response.json()
            return JSONResponse(
                status_code=200,
                content=[{"id": g.get("id"), "path": g.get("path")} for g in groups],
            )
        raise HTTPException(status_code=response.status_code, detail=_keycloak_error(response))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
