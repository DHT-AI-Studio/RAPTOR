import secrets
from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List, Dict


class UserBasic(BaseModel):
    username: str
    email: EmailStr
    firstName: Optional[str] = "firstName"
    lastName: Optional[str] = "lastName"


class UserCreateRequest(BaseModel):
    username: str
    email: EmailStr
    firstName: Optional[str] = "firstName"
    lastName: Optional[str] = "lastName"
    groups: Optional[List[str]] = None
    realmRoles: Optional[List[str]] = None
    clientRoles: Optional[Dict[str, List[str]]] = None
    # clientRoles format: {"raptor": ["role1", "role2"], "other-client": ["role3"]}


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
