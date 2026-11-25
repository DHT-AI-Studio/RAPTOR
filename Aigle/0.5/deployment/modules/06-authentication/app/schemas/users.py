import secrets

from pydantic import BaseModel, EmailStr, Field


class UserBasic(BaseModel):
    username: str
    email: EmailStr
    firstName: str | None = "firstName"
    lastName: str | None = "lastName"


class UserCreateRequest(BaseModel):
    username: str
    email: EmailStr
    firstName: str | None = "firstName"
    lastName: str | None = "lastName"
    groups: list[str] | None = None
    realmRoles: list[str] | None = None
    clientRoles: dict[str, list[str]] | None = None
    # clientRoles format: {"raptor": ["role1", "role2"], "other-client": ["role3"]}
    skip_verification: bool = False
    # Admin-create only (POST /admin/keycloak/user) -- register() builds its own
    # UserCreateRequest from a fixed set of Form fields and never sets this, so
    # self-service registration is unaffected either way. Default False keeps
    # today's behaviour (emailVerified tied to whether SMTP is configured on
    # this deployment) unchanged for existing callers. True forces
    # emailVerified=true and skips sending the verification email outright,
    # for an admin who already controls/knows the account and doesn't need
    # Keycloak's VERIFY_EMAIL/UPDATE_PASSWORD required actions in the way --
    # those make sense for a stranger self-registering, not for an admin
    # provisioning an account on purpose.


class UserCredentials(BaseModel):
    password: str = Field(default_factory=lambda: secrets.token_urlsafe(16))
    temporary: bool = False


class UserRepresentation(UserBasic):
    emailVerified: bool = False
    enabled: bool = True


class KeycloakPayload:
    @staticmethod
    def user_payload(user: UserRepresentation, credentials: UserCredentials):
        payload = user.model_dump(exclude_none=True)
        payload["credentials"] = [
            {
                "type": "password",
                "value": credentials.password,
                "temporary": credentials.temporary,
            }
        ]
        return payload
