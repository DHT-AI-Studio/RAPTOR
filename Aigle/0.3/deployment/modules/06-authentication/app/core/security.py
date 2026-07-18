from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi import Depends
import jwt

from app.keycloak.authentication import KeycloakAuthentication

bearer_scheme = HTTPBearer(auto_error=True)


class APIsecurity:
    @staticmethod
    async def authenticate(
        request: Request,
        token: HTTPAuthorizationCredentials = Depends(bearer_scheme),
    ):
        request_path = request.url.path
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

        permission = KeycloakAuthentication.check_permission_by_uri(
            realm_name=realm_name,
            user_token=user_token,
            resource_uri=request_path,
            client_id=client_id,
        )

        if permission.status_code in (401, 403):
            raise HTTPException(
                status_code=permission.status_code,
                detail=f"Authentication error {permission.status_code}: {permission.text}",
            )
