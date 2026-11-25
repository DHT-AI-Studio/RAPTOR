"""HTTP client for Module 07 (AI lifecycle API) and the Ollama daemon it fronts.

All text generation goes through Module 07 ``POST /inference/infer`` so the
07 guardrail hook, option normalisation, usage counting and model limits stay
in one place (see 07-ai-ml-services/docs/INFERENCE_TEXT_CONTRACT.md).  This
module maps 07's error codes onto Module 35's own contract.
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.errors import ServiceError

logger = logging.getLogger(__name__)

# 07 detail.code values that mean "the model / version the caller asked for does not exist".
_MODEL_MISSING_CODES = {"model_not_found", "model_version_not_found", "model_alias_not_found"}


class UpstreamUnavailable(Exception):
    """Registry/Ollama listing could not be fetched (connect error, timeout, non-2xx)."""


def _detail(resp: httpx.Response) -> dict[str, Any]:
    try:
        body = resp.json()
    except ValueError:
        return {"message": resp.text[:500]}
    detail = body.get("detail", body) if isinstance(body, dict) else body
    if isinstance(detail, dict):
        # /v1-style envelope: {"error": {"message", "type", "code"}}
        if isinstance(detail.get("error"), dict):
            return detail["error"]
        return detail
    return {"message": str(detail)}


def map_module07_error(resp: httpx.Response, model: str) -> ServiceError:
    """Translate a non-2xx Module 07 response into Module 35's error contract."""
    detail = _detail(resp)
    code = detail.get("code") or detail.get("error_type") or "upstream_error"
    message = detail.get("message") or ""
    status = resp.status_code

    if code == "max_tokens_exceeded":
        return ServiceError(
            422, "max_tokens_exceeded", message,
            model=model, model_limit=detail.get("model_limit"), requested=detail.get("requested"),
        )
    if status == 404 and code in _MODEL_MISSING_CODES:
        # Caller (resolver) turns this into the 422 model_not_found body with the available list.
        return ServiceError(422, "model_not_found", message, model=model, upstream_code=code)
    if status == 504 or code == "inference_timeout":
        return ServiceError(504, "upstream_timeout", message or "Module 07 inference timed out", model=model)
    if status == 503:
        headers = None
        extra: dict[str, Any] = {}
        if detail.get("retry_after") is not None:
            headers = {"Retry-After": str(detail["retry_after"])}
            extra["retry_after"] = detail["retry_after"]
        return ServiceError(503, code, message, headers=headers, model=model, **extra)
    if status == 400:
        extra = {k: detail[k] for k in ("parameter", "parameters", "category", "direction") if k in detail}
        return ServiceError(400, code, message, model=model, **extra)
    return ServiceError(502, "upstream_error", message, model=model, upstream_status=status, upstream_code=code)


class Module07Client:
    def __init__(
        self,
        client: httpx.AsyncClient,
        api_url: str,
        ollama_host: str,
        inference_timeout_s: float,
        registry_timeout_s: float,
    ) -> None:
        self._client = client
        self._api = api_url.rstrip("/")
        self._ollama = ollama_host.rstrip("/")
        self._infer_timeout = inference_timeout_s
        self._registry_timeout = registry_timeout_s

    # ----- discovery (INF-2) -----

    async def _get_json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        try:
            resp = await self._client.get(url, params=params, timeout=self._registry_timeout)
        except httpx.HTTPError as exc:
            raise UpstreamUnavailable(f"{url}: {exc.__class__.__name__}: {exc}") from exc
        if resp.status_code >= 400:
            raise UpstreamUnavailable(f"{url}: HTTP {resp.status_code}")
        try:
            return resp.json()
        except ValueError as exc:
            raise UpstreamUnavailable(f"{url}: invalid JSON") from exc

    async def list_registered_models(self) -> list[dict[str, Any]]:
        data = await self._get_json(f"{self._api}/models/registered_in_mlflow", {"show_all": "true"})
        if not isinstance(data, list):
            raise UpstreamUnavailable("registered_in_mlflow: unexpected payload")
        return data

    async def loaded_models(self) -> dict[str, Any]:
        data = await self._get_json(f"{self._api}/inference/loaded-models")
        return data if isinstance(data, dict) else {}

    async def ollama_tags(self) -> list[dict[str, Any]]:
        data = await self._get_json(f"{self._ollama}/api/tags")
        return list(data.get("models") or []) if isinstance(data, dict) else []

    async def ollama_ps(self) -> list[dict[str, Any]]:
        data = await self._get_json(f"{self._ollama}/api/ps")
        return list(data.get("models") or []) if isinstance(data, dict) else []

    # ----- inference (INF-5..8) -----

    def _selector(self, name: str, version: str | None, engine: str | None) -> dict[str, Any]:
        sel: dict[str, Any] = {"model_name": name}
        if version:
            sel["model_version"] = version
        if engine:
            sel["engine"] = engine
        return sel

    async def model_info(self, name: str, version: str | None, engine: str | None) -> dict[str, Any]:
        params = self._selector(name, version, engine)
        try:
            resp = await self._client.get(
                f"{self._api}/inference/model-info", params=params, timeout=self._registry_timeout
            )
        except httpx.TimeoutException as exc:
            raise ServiceError(504, "upstream_timeout", "Module 07 model-info timed out", model=name) from exc
        except httpx.HTTPError as exc:
            raise ServiceError(503, "module07_unavailable", str(exc), model=name) from exc
        if resp.status_code >= 400:
            raise map_module07_error(resp, name)
        return resp.json()

    async def infer(
        self,
        name: str,
        version: str | None,
        engine: str | None,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ) -> dict[str, Any]:
        payload = {
            **self._selector(name, version, engine),
            "task": "text-generation",
            "data": {"messages": messages},
            "options": options,
        }
        try:
            resp = await self._client.post(
                f"{self._api}/inference/infer", json=payload, timeout=self._infer_timeout
            )
        except httpx.TimeoutException as exc:
            raise ServiceError(504, "upstream_timeout", "Module 07 did not respond in time", model=name) from exc
        except httpx.HTTPError as exc:
            raise ServiceError(503, "module07_unavailable", str(exc), model=name) from exc
        if resp.status_code >= 400:
            raise map_module07_error(resp, name)
        body = resp.json()
        if not body.get("success", True):
            raise ServiceError(502, "upstream_error", str(body.get("error") or "inference failed"), model=name)
        return body
        
    async def infer_stream(
        self,
        name: str,
        version: str | None,
        engine: str | None,
        messages: list[dict[str, str]],
        options: dict[str, Any],
    ):
        """POST /v1/chat/completions with stream=true; yields OpenAI SSE chunks.
    
        Establishes the HTTP connection before returning the response so that
        connection errors surface as ServiceError *before* StreamingResponse
        starts — the HTTP layer can still return a proper error status code.
        """
        payload = {
            **self._selector(name, version, engine),
            "model": name,
            "messages": messages,
            "stream": True,
            "max_tokens": options.get("max_tokens"),
            "temperature": options.get("temperature"),
        }
        # strip None values — upstream rejects unknown nulls
        payload = {k: v for k, v in payload.items() if v is not None}
        if options.get("stop_sequences"):
            payload["stop"] = options["stop_sequences"]
    
        try:
            req = self._client.build_request(
                "POST",
                f"{self._api}/v1/chat/completions",
                json=payload,
                extensions={"timeout": {"connect": 10.0, "read": self._infer_timeout, "write": 10.0, "pool": 10.0}},
            )
            response = await self._client.send(req, stream=True)
        except httpx.TimeoutException as exc:
            raise ServiceError(504, "upstream_timeout", "Module 07 did not respond in time", model=name) from exc
        except httpx.HTTPError as exc:
            raise ServiceError(503, "module07_unavailable", str(exc), model=name) from exc
    
        if response.status_code >= 400:
            body = await response.aread()
            await response.aclose()
            fake = httpx.Response(response.status_code, content=body, headers=response.headers)
            raise map_module07_error(fake, name)
    
        return response  # caller owns aclose()