"""
Authentication service for handling user login and token generation.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import jwt

from app.core.config import Settings
from passlib.context import CryptContext

from app.services.user_service import UserService

_logger = logging.getLogger(__name__)


_pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")


class AuthService:
    def __init__(self, settings: Settings, user_service: UserService):
        self.settings = settings
        self.user_service = user_service

    def hash_password(self, password: str) -> str:
        """Securely hash a plaintext password."""
        # bcrypt has a maximum password length of 72 bytes.
        # We truncate the password to prevent a ValueError from passlib.
        return _pwd_context.hash(password.encode("utf-8")[:72])

    def verify_password(self, plain_password: str, stored_password: str) -> bool:
        """Verify a plaintext password against a stored hash."""
        try:
            return _pwd_context.verify(plain_password, stored_password)
        except ValueError:  # pragma: no cover - hash format errors are unexpected
            _logger.error("Stored password hash is invalid", extra={"hash": stored_password})
            return False

    async def get_user(self, username: str) -> dict | None:
        """Retrieves a user record from the user store."""
        stored_password = await self.user_service.get_user_password(username)
        if not stored_password:
            return None
        return {"username": username, "hashed_password": stored_password}

    def create_access_token(self, data: dict, expires_delta: timedelta | None = None) -> str:
        """Creates a new JWT access token."""
        to_encode = data.copy()
        expire = datetime.now(timezone.utc) + (expires_delta or timedelta(minutes=15))
        to_encode.update({"exp": expire})
        return jwt.encode(
            to_encode,
            self.settings.jwt_secret_key,
            algorithm=self.settings.jwt_algorithm,
        )

    async def authenticate_user(self, username: str, password: str) -> dict | None:
        """Authenticates a gateway user using the configured credential store."""
        user = await self.get_user(username)
        if not user:
            _logger.warning("Authentication failed: user '%s' not found", username)
            return None
        if not self.verify_password(password, user["hashed_password"]):
            _logger.warning("Authentication failed: invalid password for user '%s'", username)
            return None
        return user

    async def register_user(self, username: str, password: str) -> dict:
        """Register a new gateway user and return basic information."""
        hashed_password = self.hash_password(password)
        await self.user_service.create_user(username=username, hashed_password=hashed_password)
        return {"username": username}
