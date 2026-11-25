import jwt
from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.core.config import settings
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
        except jwt.exceptions.PyJWTError:
            raise HTTPException(status_code=401, detail="Invalid token format")

        if not realm_name:
            raise HTTPException(status_code=401, detail="Cannot determine realm from token issuer")

        permission = KeycloakAuthentication.check_permission_by_uri(
            realm_name=realm_name,
            user_token=user_token,
            resource_uri=request_path,
            # The resource server holding the API's UMA resources, not the token's azp: a token
            # from the public raptor-mcp client (MCP OAuth login) has azp=raptor-mcp, which has no
            # authorization services, and Keycloak answers that UMA request with a 500 -> 403 here.
            client_id=settings.KEYCLOAK_CLIENT_ID,
        )

        if permission.status_code in (401, 403):
            raise HTTPException(
                status_code=permission.status_code,
                detail=f"Authentication error {permission.status_code}: {permission.text}",
            )
