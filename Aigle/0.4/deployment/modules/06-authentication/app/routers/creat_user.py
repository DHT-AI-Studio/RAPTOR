from fastapi import APIRouter, HTTPException, Body, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from typing import Annotated

from app.keycloak.user import KeycloakUser
from app.keycloak.group import KeycloakGroup
from app.keycloak.master import KeycloakMaster
from app.schemas.users import UserCreateRequest, UserCredentials
from app.core.config import settings

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


def _rollback_and_raise(user_id: str, realm: str, failed_at: str, detail: str):
    """刪除已建立的 user（使用 master token 確保有刪除權限）並拋出 HTTPException。"""
    try:
        master_token = KeycloakMaster.MasterToken()
        KeycloakUser.user_delete_api(user_id=user_id, realm=realm, token=master_token)
    except Exception:
        pass
    raise HTTPException(
        status_code=500,
        detail={
            "error": "User creation rolled back: user deleted due to assignment failure",
            "failedAt": failed_at,
            "detail": detail,
        },
    )


@router.post("/user", tags=["Admin"])
def create_user_api(
    user_data: Annotated[
        UserCreateRequest,
        Body(
            examples=[
                {
                    "username": "john",
                    "email": "john@example.com",
                    "firstName": "John",
                    "lastName": "Doe",
                    "groups": ["/default_group"],
                    "realmRoles": ["Developer"],
                    "clientRoles": {"raptor": ["user_basic", "user_standard"]},
                }
            ]
        ),
    ],
    realm_name: str = Query("dhtsolution", enum=["dhtsolution"]),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """建立使用者（需要 admin token）。回傳暫時密碼供 admin 轉交使用者。
    SMTP 已設定時自動寄送驗證信；未設定則 emailVerified=true 直接可登入。
    任一指派失敗會自動刪除 user（rollback）。"""
    try:
        # 1. 建立使用者
        credentials = UserCredentials()
        smtp_configured = bool(settings.SMTP_HOST)
        response = KeycloakUser.user_create_api_basic(
            user_data=user_data,
            realm=realm_name,
            token=token.credentials,
            credentials=credentials,
            email_verified=not smtp_configured,
        )
        if response.status_code != 201:
            raise HTTPException(status_code=response.status_code, detail=_keycloak_error(response))

        assigned_groups = []
        assigned_realm_roles = []
        assigned_client_roles = {}

        # 2. 取得使用者 UUID
        try:
            user_id = KeycloakUser.get_user_id_by_username(
                username=user_data.username, realm=realm_name, token=token.credentials
            )
        except (ValueError, RuntimeError) as e:
            raise HTTPException(status_code=500,
                detail=f"User created but cannot retrieve user ID: {e}")

        # 3. 送出驗證 email（僅在 SMTP 已設定時執行）
        email_sent = False
        if settings.SMTP_HOST:
            try:
                email_resp = KeycloakUser.send_verify_email_api(
                    user_id=user_id, realm=realm_name, token=KeycloakMaster.MasterToken()
                )
                email_sent = email_resp.status_code == 204
            except Exception:
                pass

        if user_data.groups or user_data.realmRoles or user_data.clientRoles:
            # 4. 指派 groups
            for group_path in (user_data.groups or []):
                try:
                    group_id = KeycloakGroup.resolve_group_id(
                        realm_name=realm_name, group_path=group_path, token=token.credentials
                    )
                except (ValueError, RuntimeError) as e:
                    _rollback_and_raise(user_id, realm_name, f"group assignment: '{group_path}'", str(e))

                grp_resp = KeycloakUser.assign_user_to_group(
                    user_id=user_id, group_id=group_id, realm=realm_name, token=token.credentials
                )
                if grp_resp.status_code != 204:
                    _rollback_and_raise(user_id, realm_name,
                        f"group assignment: '{group_path}'",
                        f"Keycloak returned {grp_resp.status_code}: {grp_resp.text}")
                assigned_groups.append(group_path)

            # 5. 指派 realm roles
            if user_data.realmRoles:
                role_reprs = []
                for role_name in user_data.realmRoles:
                    try:
                        r = KeycloakUser.get_realm_role(
                            role_name=role_name, realm=realm_name, token=token.credentials
                        )
                    except (ValueError, RuntimeError) as e:
                        _rollback_and_raise(user_id, realm_name, f"realmRole: '{role_name}'", str(e))
                    role_reprs.append({"id": r["id"], "name": r["name"]})

                roles_resp = KeycloakUser.assign_realm_roles_to_user(
                    user_id=user_id, roles=role_reprs,
                    realm=realm_name, token=token.credentials
                )
                if roles_resp.status_code != 204:
                    _rollback_and_raise(user_id, realm_name, "realmRoles assignment",
                        f"Keycloak returned {roles_resp.status_code}: {roles_resp.text}")
                assigned_realm_roles = [r["name"] for r in role_reprs]

            # 6. 指派 client roles
            for client_id, role_names in (user_data.clientRoles or {}).items():
                try:
                    client_uuid = KeycloakUser.get_client_uuid(
                        client_id=client_id, realm=realm_name, token=token.credentials
                    )
                except (ValueError, RuntimeError) as e:
                    _rollback_and_raise(user_id, realm_name, f"clientRole client lookup: '{client_id}'", str(e))

                client_role_reprs = []
                for role_name in role_names:
                    try:
                        r = KeycloakUser.get_client_role(
                            role_name=role_name, client_uuid=client_uuid,
                            realm=realm_name, token=token.credentials
                        )
                    except (ValueError, RuntimeError) as e:
                        _rollback_and_raise(user_id, realm_name, f"clientRole: '{client_id}/{role_name}'", str(e))
                    client_role_reprs.append({"id": r["id"], "name": r["name"]})

                cr_resp = KeycloakUser.assign_client_roles_to_user(
                    user_id=user_id, client_uuid=client_uuid,
                    roles=client_role_reprs, realm=realm_name, token=token.credentials
                )
                if cr_resp.status_code != 204:
                    _rollback_and_raise(user_id, realm_name, f"clientRole assignment: '{client_id}'",
                        f"Keycloak returned {cr_resp.status_code}: {cr_resp.text}")
                assigned_client_roles[client_id] = role_names

        return {
            "message": "User created successfully.",
            "username": user_data.username,
            "email": user_data.email,
            "temporaryPassword": credentials.password,
            "emailSent": email_sent,
            "assignedGroups": assigned_groups,
            "assignedRealmRoles": assigned_realm_roles,
            "assignedClientRoles": assigned_client_roles,
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/user", tags=["Admin"])
def get_user_api(
    keyword: str,
    query: str = Query("username", enum=["username", "email"]),
    realm_name: str = Query("dhtsolution", enum=["dhtsolution"]),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """依 username 或 email 查詢使用者基本資料。"""
    try:
        if query == "username":
            response = KeycloakUser.user_get_basic_api_username(
                username=keyword, realm=realm_name, token=token.credentials
            )
        else:
            response = KeycloakUser.user_get_basic_api_email(
                email=keyword, realm=realm_name, token=token.credentials
            )

        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, detail=_keycloak_error(response))

        return_data = [
            {
                "id": u.get("id"),
                "username": u.get("username"),
                "email": u.get("email"),
                "firstname": u.get("firstName"),
                "lastname": u.get("lastName"),
            }
            for u in response.json()
        ]
        return JSONResponse(status_code=200, content=return_data)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/user", tags=["Admin"])
def delete_user_api(
    user_id: str,
    realm_name: str = Query("dhtsolution", enum=["dhtsolution"]),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """依 user_id 刪除使用者。user_id 可透過 GET /user 取得。"""
    try:
        response = KeycloakUser.user_delete_api(
            user_id=user_id, realm=realm_name, token=token.credentials
        )
        if response.status_code == 204:
            return JSONResponse(status_code=200, content={"message": "The account has been deleted."})
        raise HTTPException(status_code=response.status_code, detail=_keycloak_error(response))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
