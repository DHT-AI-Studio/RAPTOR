from fastapi import APIRouter, HTTPException, Form, Depends, Query
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

from app.keycloak.user import KeycloakUser
from app.keycloak.master import KeycloakMaster
from app.schemas.users import UserCreateRequest, UserCredentials
from app.core.config import settings

router = APIRouter()

bearer_scheme = HTTPBearer(auto_error=True)


@router.post("/register", tags=["Public"])
def register(
    username: str = Form(...),
    email: str = Form(...),
    firstName: str = Form(""),
    lastName: str = Form(""),
    realm_name: str = Form("dhtsolution"),
):
    """
    使用者自助註冊。由 server 使用 admin 權限建立帳號並回傳暫時密碼。
    SMTP 已設定時會寄送驗證信，使用者需驗證 email 並設定新密碼後才能登入。
    """
    user_data = UserCreateRequest(
        username=username,
        email=email,
        firstName=firstName,
        lastName=lastName,
    )
    credentials = UserCredentials()
    smtp_configured = bool(settings.SMTP_HOST)

    try:
        master_token = KeycloakMaster.MasterToken()
        response = KeycloakUser.user_create_api_basic(
            user_data=user_data,
            realm=realm_name,
            token=master_token,
            credentials=credentials,
            email_verified=not smtp_configured,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if response.status_code == 201:
        user_id = None
        try:
            user_id = KeycloakUser.get_user_id_by_username(
                username=username, realm=realm_name, token=master_token
            )
            client_uuid = KeycloakUser.get_client_uuid(
                client_id="raptor", realm=realm_name, token=master_token
            )
            role = KeycloakUser.get_client_role(
                role_name="user_basic", client_uuid=client_uuid,
                realm=realm_name, token=master_token
            )
            KeycloakUser.assign_client_roles_to_user(
                user_id=user_id, client_uuid=client_uuid,
                roles=[{"id": role["id"], "name": role["name"]}],
                realm=realm_name, token=master_token,
            )
        except Exception as e:
            try:
                if user_id:
                    rollback_token = KeycloakMaster.MasterToken()
                    KeycloakUser.user_delete_api(user_id=user_id, realm=realm_name, token=rollback_token)
            except Exception:
                pass
            raise HTTPException(
                status_code=500,
                detail={
                    "error": "Registration rolled back: user deleted due to role assignment failure",
                    "detail": str(e),
                },
            )

        if smtp_configured:
            try:
                KeycloakUser.send_verify_email_api(
                    user_id=user_id, realm=realm_name, token=master_token
                )
            except Exception:
                pass

        result = {
            "message": "Registration successful.",
            "username": username,
            "email": email,
        }
        if not smtp_configured:
            result["temporaryPassword"] = credentials.password
        return result

    try:
        detail = response.json().get("errorMessage") or response.json().get("error") or response.text
    except Exception:
        detail = response.text
    raise HTTPException(status_code=response.status_code, detail=detail)


@router.post("/change-password", tags=["User"])
def change_password(
    new_password: str = Form(...),
    realm_name: str = Form("dhtsolution"),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """
    使用者修改自己的密碼。需要有效的 Bearer token。
    """
    try:
        user_id = KeycloakUser.get_user_id_from_token(token.credentials, realm_name)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        master_token = KeycloakMaster.MasterToken()
        resp = KeycloakUser.set_user_password(
            user_id=user_id,
            new_password=new_password,
            realm=realm_name,
            token=master_token,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if resp.status_code == 204:
        return {"message": "Password updated successfully."}
    raise HTTPException(status_code=resp.status_code, detail=resp.text)


@router.delete("/account", tags=["User"])
def delete_account(
    realm_name: str = Query("dhtsolution"),
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """
    使用者刪除自己的帳號。需要有效的 Bearer token。
    """
    try:
        user_id = KeycloakUser.get_user_id_from_token(token.credentials, realm_name)
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))

    try:
        master_token = KeycloakMaster.MasterToken()
        resp = KeycloakUser.user_delete_api(
            user_id=user_id,
            realm=realm_name,
            token=master_token,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    if resp.status_code == 204:
        return {"message": "Account deleted successfully."}
    raise HTTPException(status_code=resp.status_code, detail=resp.text)
