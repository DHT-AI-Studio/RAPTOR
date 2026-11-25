import jwt
from fastapi import APIRouter, Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
from app.keycloak.authentication import KeycloakAuthentication

router = APIRouter(tags=["System"])

bearer_scheme = HTTPBearer(auto_error=True)


@router.get("/permission")
async def permission(
    request_path: str,
    token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
):
    """
    檢查使用者 token 是否有存取指定路徑的權限。
    realm 從 token 的 iss 解析；權限一律對 resource server（KEYCLOAK_CLIENT_ID，預設 raptor）檢查，
    不用 token 的 azp——raptor-mcp 等沒有權限資源的 client 簽出的 token 也能正確判斷。
    """
    user_token = token.credentials

    try:
        claims = jwt.decode(user_token, options={"verify_signature": False})
        iss = claims.get("iss", "")
        realm_name = iss.split("/realms/")[-1] if "/realms/" in iss else None
    except jwt.exceptions.PyJWTError:
        raise HTTPException(status_code=401, detail="Invalid token format")

    if not realm_name:
        raise HTTPException(status_code=401, detail="Cannot determine realm from token issuer")

    result = KeycloakAuthentication.check_permission_by_uri(
        realm_name=realm_name,
        user_token=user_token,
        resource_uri=request_path,
        # The resource server holding the API's UMA resources, not the token's azp: a token from
        # the public raptor-mcp client (MCP OAuth login) has azp=raptor-mcp, which has no
        # authorization services, and Keycloak answers that UMA request with a 500 -> 403 here.
        client_id=settings.KEYCLOAK_CLIENT_ID,
    )

    if result.status_code == 200:
        return {"message": "Permission granted"}

    try:
        body = result.json()
        detail = body.get("error_description") or body.get("error") or result.text
    except ValueError:
        detail = result.text or f"HTTP {result.status_code}"

    raise HTTPException(status_code=403, detail=detail)
