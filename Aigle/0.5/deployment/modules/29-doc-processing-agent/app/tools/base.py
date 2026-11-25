"""Shared base for all Module-07-backed DocAgent tools."""
from __future__ import annotations

import logging
from typing import Any

import httpx

from app.core.config import DocAgentSettings, get_settings

logger = logging.getLogger(__name__)

import asyncio
import os

_SEM: asyncio.Semaphore | None = None


def _get_sem() -> asyncio.Semaphore:
    """Lazy-create so it binds to the running event loop."""
    global _SEM
    if _SEM is None:
        _SEM = asyncio.Semaphore(int(os.getenv("DA_MAX_CONCURRENT_REQUESTS", "2")))
    return _SEM

class LLMToolError(Exception):
    """Raised when a tool call to Module 07 fails."""
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class BaseLLMTool:
    """Calls Module 07 POST /inference/infer with a text-generation task."""

    task_name: str = "unnamed"

    def __init__(
        self,
        settings: DocAgentSettings | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._cfg = settings or get_settings()
        self._client = http_client

    async def _infer(self, prompt: str) -> str:
        """Send prompt to Module 07, return the response string."""
        url = self._cfg.module07_url.rstrip("/") + "/inference/infer"
        body: dict[str, Any] = {
            "task": "text-generation",
            "model_name": self._cfg.llm_model,
            # Module 07 ignores this once model_name is MLflow-registered; until
            # then it's the only way to reach its Ollama-daemon-tag fallback, and
            # llm_model (LLM_MODEL platform-wide) is an Ollama tag by convention
            # everywhere else that consumes it (OLLAMA_MODEL, ollama/${LLM_MODEL}).
            "engine": "ollama",
            "data": {"inputs": prompt},
            "options": {
                "max_tokens": self._cfg.llm_max_tokens,
                "temperature": 0.2,
            },
        }
        own_client = self._client is None
        client = self._client or httpx.AsyncClient(
            timeout=httpx.Timeout(self._cfg.llm_timeout)
        )
        try:
            async with _get_sem():
                resp = await client.post(
                    url, json=body,
                    timeout=httpx.Timeout(self._cfg.llm_timeout),  # 明確套用
                )
            if resp.status_code == 503:
                raise LLMToolError(
                    f"Module 07 unavailable (503) for task={self.task_name}",
                    status_code=503,
                )
            resp.raise_for_status()
            result = resp.json().get("result") or {}
            text = result.get("response") or result.get("text") or ""
            return text.strip()
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            raise LLMToolError(
                f"Module 07 returned {code} for task={self.task_name}: "
                f"{exc.response.text[:200]}",
                status_code=code if 400 <= code < 500 else 502,
            ) from exc
        except httpx.TimeoutException as exc:
            raise LLMToolError(
                f"Module 07 timeout after {self._cfg.llm_timeout}s "
                f"for task={self.task_name}",
                status_code=504,
            ) from exc
        except httpx.ConnectError as exc:
            raise LLMToolError(
                f"Module 07 unreachable at {self._cfg.module07_url}: {exc}",
                status_code=503,
            ) from exc
        finally:
            if own_client:
                await client.aclose()

    async def run(self, **kwargs) -> dict[str, Any]:
        raise NotImplementedError
