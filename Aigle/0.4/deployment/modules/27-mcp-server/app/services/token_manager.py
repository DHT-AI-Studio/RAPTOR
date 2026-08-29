from __future__ import annotations

import json
import secrets
import time
from typing import Optional

import httpx
import redis.asyncio as aioredis
from cryptography.fernet import Fernet, InvalidToken

from app.core.config import Settings

_AGENT_PREFIX = "mcp-agent-"       # opaque agent_token prefix
_CREDS_KEY = "mcp:creds:{token}"   # Redis key holding the agent's credential blob


class TokenManager:
    """Issues and refreshes tokens for two callers:

    * the server's own service account — password grant via the gateway
      (``get_server_token``), used when a client forwards no JWT;
    * autonomous agents — OAuth2 **client_credentials** grant straight from
      Keycloak (``register_agent`` / ``get_valid_keycloak_token``), so a
      long-running agent never fails on JWT expiry (MC-7).
    """

    def __init__(self, settings: Settings):
        self._s = settings
        self._redis: Optional[aioredis.Redis] = None
        # In-process fallback when Redis is unavailable (server token only)
        self._mem_token: Optional[str] = None
        self._mem_expires: float = 0.0
        self._fernet: Optional[Fernet] = None

    def _get_fernet(self) -> Fernet:
        """Lazy — most deployments never call register_agent, so don't fail
        startup over a missing key that isn't needed yet."""
        if self._fernet is None:
            if not self._s.secret_encryption_key:
                raise RuntimeError(
                    "MCP_SECRET_ENCRYPTION_KEY is not set — required to store an "
                    "agent's client_secret. Generate one with: python -c "
                    "\"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
                )
            self._fernet = Fernet(self._s.secret_encryption_key.encode())
        return self._fernet

    def _encrypt_secret(self, client_secret: str) -> str:
        return self._get_fernet().encrypt(client_secret.encode()).decode()

    def _decrypt_secret(self, encrypted: str) -> str:
        try:
            return self._get_fernet().decrypt(encrypted.encode()).decode()
        except InvalidToken as exc:
            raise RuntimeError(
                "Stored agent client_secret could not be decrypted — "
                "MCP_SECRET_ENCRYPTION_KEY may have changed since it was registered"
            ) from exc

    async def _get_redis(self) -> Optional[aioredis.Redis]:
        if self._redis is not None:
            return self._redis
        try:
            self._redis = aioredis.Redis.from_url(
                self._s.redis_url, decode_responses=True
            )
            await self._redis.ping()
        except Exception:
            self._redis = None
        return self._redis

    # ── server service account (password grant via gateway) ──────────────────

    async def _fetch_token(self, username: str, password: str) -> tuple[str, float]:
        """Log in via Module 13's ``/sso/login`` proxy rather than hitting
        Keycloak's token endpoint directly — ``raptor`` is a confidential
        client, so a bare password grant would be rejected without the
        client secret that only Module 13 (via Module 06) holds.
        """
        url = f"{self._s.api_gateway_url.rstrip('/')}/sso/login"
        async with httpx.AsyncClient(timeout=self._s.timeout_default) as client:
            r = await client.post(url, data={
                "username": username,
                "password": password,
                "realm_name": self._s.realm_name,
                "client_id": self._s.client_id,
            })
            r.raise_for_status()
            token = r.json()["access_token"]
        return token, time.time() + self._s.token_ttl_seconds

    async def get_server_token(self) -> str:
        """Return a valid token for the server's own service account."""
        redis = await self._get_redis()

        if redis:
            cached = await redis.get("mcp:token:server")
            if cached:
                data = json.loads(cached)
                if data["expires_at"] - time.time() > self._s.refresh_margin_seconds:
                    return data["token"]
        elif self._mem_token and self._mem_expires - time.time() > self._s.refresh_margin_seconds:
            return self._mem_token

        token, expires_at = await self._fetch_token(
            self._s.keycloak_username, self._s.keycloak_password
        )

        if redis:
            await redis.set(
                "mcp:token:server",
                json.dumps({"token": token, "expires_at": expires_at}),
                ex=self._s.token_ttl_seconds,
            )
        else:
            self._mem_token, self._mem_expires = token, expires_at
        return token

    # ── autonomous agents (client_credentials grant via Keycloak) ────────────

    async def _fetch_client_credentials_token(
        self, client_id: str, client_secret: str, scope: Optional[str],
    ) -> tuple[str, float]:
        """Mint a Keycloak access token with the client_credentials grant."""
        url = (
            f"{self._s.keycloak_url.rstrip('/')}"
            f"/realms/{self._s.realm_name}/protocol/openid-connect/token"
        )
        form = {
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        }
        if scope:
            form["scope"] = scope
        async with httpx.AsyncClient(timeout=self._s.timeout_default) as client:
            r = await client.post(url, data=form)
            r.raise_for_status()
            body = r.json()
        token = body["access_token"]
        expires_at = time.time() + int(body.get("expires_in", self._s.token_ttl_seconds))
        return token, expires_at

    @staticmethod
    def _creds_key(agent_token: str) -> str:
        return _CREDS_KEY.format(token=agent_token)

    async def register_agent(
        self, client_id: str, client_secret: str, scope: Optional[str] = None,
    ) -> str:
        """Validate client credentials, mint an opaque agent_token, and cache
        the credential blob (+ first Keycloak token) in Redis for 24 h.

        Raises on invalid credentials (surfaced by the router as 401) or when
        Redis is unavailable (503).
        """
        redis = await self._get_redis()
        if not redis:
            raise RuntimeError("Redis is required for agent credential storage")
        self._get_fernet()  # fail fast on missing config, before any network I/O

        # Validate the credentials up front and prime the cache.
        keycloak_token, expires_at = await self._fetch_client_credentials_token(
            client_id, client_secret, scope
        )

        agent_token = _AGENT_PREFIX + secrets.token_urlsafe(24)
        await redis.set(
            self._creds_key(agent_token),
            json.dumps({
                "client_id": client_id,
                "client_secret_enc": self._encrypt_secret(client_secret),
                "scope": scope,
                "keycloak_access_token": keycloak_token,
                "expires_at": expires_at,
            }),
            ex=self._s.agent_token_ttl_seconds,
        )
        return agent_token

    async def get_valid_keycloak_token(self, agent_token: str) -> Optional[str]:
        """Resolve an agent_token to a valid Keycloak access token.

        Returns the cached token while it has > 60 s left; otherwise refreshes
        via client_credentials, updates Redis, and extends the 24 h idle TTL
        (agent_token is refreshed on use). Returns ``None`` if the agent_token
        is unknown or revoked.
        """
        redis = await self._get_redis()
        if not redis:
            return None

        key = self._creds_key(agent_token)
        raw = await redis.get(key)
        if not raw:
            return None
        creds = json.loads(raw)

        if creds.get("expires_at", 0) - time.time() > self._s.refresh_margin_seconds:
            # Still valid — extend idle TTL (refreshed-on-use) and return cached.
            await redis.expire(key, self._s.agent_token_ttl_seconds)
            return creds["keycloak_access_token"]

        client_secret = self._decrypt_secret(creds["client_secret_enc"])
        token, expires_at = await self._fetch_client_credentials_token(
            creds["client_id"], client_secret, creds.get("scope")
        )
        creds["keycloak_access_token"] = token
        creds["expires_at"] = expires_at
        await redis.set(key, json.dumps(creds), ex=self._s.agent_token_ttl_seconds)
        return token

    async def revoke_agent(self, agent_token: str) -> bool:
        """Revoke an agent_token (delete its Redis key). Returns True if removed."""
        redis = await self._get_redis()
        if not redis:
            return False
        return bool(await redis.delete(self._creds_key(agent_token)))
