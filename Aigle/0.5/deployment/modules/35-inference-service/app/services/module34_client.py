"""HTTP client for Module 34 (realtime-audio)'s one-shot STT/TTS endpoints.

VIE01-316/317 (INF-10/11): rather than this service loading a second copy of the
Whisper/TTS models Module 34 already has resident for live audio sessions, these two
tasks forward to Module 34's ``POST /audio/transcribe`` / ``POST /audio/synthesize``
and reuse its models and admission-controlled GPU slots. Module 34 publishes directly
to clients (no gateway route of its own -- see docs/_internal/security/threat-model.md B2), so
both endpoints there are gated behind ``X-Internal-Token``, the same shared
GD_INTERNAL_TOKEN Module 33 already uses for its own internal calls -- not a new
secret.

Known tradeoff, not solved here: a batch call from this client competes for the same
GPU slots as Module 34's live sessions (its own semaphore, unchanged by this client).
A high rate of batch calls can add latency to concurrent live audio. Acceptable for
now (VIE01-373's own test traffic, and no other caller yet); a separate admission
pool is a reasonable follow-up if batch volume grows.
"""
from __future__ import annotations

import logging

import httpx

from app.errors import ServiceError

logger = logging.getLogger(__name__)


class Module34Client:
    def __init__(self, client: httpx.AsyncClient, api_url: str, internal_token: str, timeout_s: float) -> None:
        self._client = client
        self._api = api_url.rstrip("/")
        self._token = internal_token
        self._timeout = timeout_s

    async def transcribe(self, wav_bytes: bytes, language: str | None, user_id: str) -> dict:
        """``wav_bytes``: 16 kHz mono PCM16 WAV (the normalized form ``read_audio()`` /
        ``Audio.wav()`` already produce). ``user_id``: forwarded as ``X-User-Id`` so Module 34
        can charge this call against the caller's own per-user slot quota (shared with their
        live WS sessions), not an unlimited pool."""
        data = {"language": language} if language else {}
        try:
            resp = await self._client.post(
                f"{self._api}/audio/transcribe",
                files={"audio": ("audio.wav", wav_bytes, "audio/wav")},
                data=data,
                headers={"X-Internal-Token": self._token, "X-User-Id": user_id},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise ServiceError(504, "upstream_timeout", "Module 34 did not respond in time") from exc
        except httpx.HTTPError as exc:
            raise ServiceError(503, "module34_unavailable", str(exc)) from exc
        if resp.status_code >= 400:
            raise _map_error(resp)
        return resp.json()

    async def synthesize(self, text: str, language: str | None, user_id: str) -> bytes:
        data = {"text": text, **({"language": language} if language else {})}
        try:
            resp = await self._client.post(
                f"{self._api}/audio/synthesize",
                data=data,
                headers={"X-Internal-Token": self._token, "X-User-Id": user_id},
                timeout=self._timeout,
            )
        except httpx.TimeoutException as exc:
            raise ServiceError(504, "upstream_timeout", "Module 34 did not respond in time") from exc
        except httpx.HTTPError as exc:
            raise ServiceError(503, "module34_unavailable", str(exc)) from exc
        if resp.status_code >= 400:
            raise _map_error(resp)
        return resp.content


def _map_error(resp: httpx.Response) -> ServiceError:
    try:
        detail = resp.json().get("detail", "")
    except ValueError:
        detail = resp.text[:300]
    if resp.status_code == 503:
        return ServiceError(503, "module34_unavailable", str(detail))
    if resp.status_code == 429:
        return ServiceError(429, str(detail) or "too_many_requests")
    if resp.status_code in (400, 422):
        return ServiceError(resp.status_code, "invalid_audio", str(detail))
    if resp.status_code == 401:
        # Only a deployment/config bug (mismatched GD_INTERNAL_TOKEN) can produce this --
        # never a caller-supplied value, so it's an upstream error, not 4xx-to-the-caller.
        return ServiceError(502, "module34_auth_misconfigured", str(detail))
    return ServiceError(502, "upstream_error", str(detail), upstream_status=resp.status_code)
