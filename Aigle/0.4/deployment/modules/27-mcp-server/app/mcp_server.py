from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from mcp.server.fastmcp import FastMCP

from app.core.config import get_settings
from app.services.token_manager import TokenManager
from app.tools import search, chat, asset, graph, a2a, processing, pipeline, memory
from app.resources import raptor_resources
from app.prompts import raptor_prompts

_logger = logging.getLogger(__name__)


@asynccontextmanager
async def _lifespan(server: FastMCP) -> AsyncIterator[dict]:
    settings = get_settings()
    logging.basicConfig(level=getattr(logging, settings.log_level.upper(), logging.INFO))

    token_manager = TokenManager(settings)

    if settings.keycloak_username and settings.keycloak_password:
        try:
            await token_manager.get_server_token()
            _logger.info("Server token obtained successfully")
        except Exception as exc:
            _logger.warning("Failed to obtain server token on startup: %s", exc)
    else:
        _logger.info("No server credentials — client must forward Bearer JWT")

    yield {"settings": settings, "token_manager": token_manager}

    _logger.info("MCP lifespan shutdown")


mcp = FastMCP(get_settings().server_name, lifespan=_lifespan)

search.register(mcp)
chat.register(mcp)
asset.register(mcp)
graph.register(mcp)
a2a.register(mcp)
processing.register(mcp)
pipeline.register(mcp)
memory.register(mcp)

raptor_resources.register(mcp)
raptor_prompts.register(mcp)
