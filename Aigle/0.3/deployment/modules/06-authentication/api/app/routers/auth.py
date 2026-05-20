from fastapi import APIRouter, Request, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from lib.keycloak_authentication import KeycloakAuthentication

router = APIRouter(prefix="/auth", tags=["auth"])

bearer_scheme = HTTPBearer(auto_error=True)


@router.get("/permission")
async def permission(
    request_path: str,
    request: Request,
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """
    檢查使用者 token 是否有權限
    """

    # request_path = request.url.path
    user_token = token.credentials

    # cookies
    realm_name = request.cookies.get("realm_name")
    client_id = request.cookies.get("client_id")

    if not realm_name:
        raise HTTPException(
            status_code=400,
            detail="The realm_name does not exist. Please check your login status."
        )

    if not client_id:
        raise HTTPException(
            status_code=400,
            detail="The client_id does not exist. Please check your login status."
        )

    permission = KeycloakAuthentication.check_permission_by_uri(
        realm_name=realm_name,
        user_token=user_token,
        resource_uri=request_path,
        client_id=client_id
    )

    if permission.status_code in (401, 403):
        raise HTTPException(
            status_code=permission.status_code,
            detail=f"Authentication error {permission.status_code}: {permission.text}"
        )

    return {"message": "Permission granted"}