from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt

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
    realm 與 client_id 從 token 的 iss / azp claim 自動解析，支援 server-to-server 呼叫。
    """
    user_token = token.credentials

    try:
        claims = jwt.decode(user_token, options={"verify_signature": False})
        iss = claims.get("iss", "")
        realm_name = iss.split("/realms/")[-1] if "/realms/" in iss else None
        client_id = claims.get("azp")
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token format")

    if not realm_name:
        raise HTTPException(status_code=401, detail="Cannot determine realm from token issuer")
    if not client_id:
        raise HTTPException(status_code=401, detail="Cannot determine client_id from token")

    result = KeycloakAuthentication.check_permission_by_uri(
        realm_name=realm_name,
        user_token=user_token,
        resource_uri=request_path,
        client_id=client_id,
    )

    if result.status_code == 200:
        return {"message": "Permission granted"}

    try:
        body = result.json()
        detail = body.get("error_description") or body.get("error") or result.text
    except Exception:
        detail = result.text or f"HTTP {result.status_code}"

    raise HTTPException(status_code=403, detail=detail)
