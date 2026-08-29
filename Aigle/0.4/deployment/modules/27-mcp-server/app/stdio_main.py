from __future__ import annotations

import asyncio
import logging

from app.core.config import get_settings
from app.mcp_server import mcp


def main() -> None:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        handlers=[logging.FileHandler(settings.stdio_log_path)],  # avoid polluting stdout
    )

    # mcp's own lifespan (app/mcp_server.py::_lifespan) obtains the server
    # token and builds settings/token_manager — run_stdio_async() triggers it.
    asyncio.run(mcp.run_stdio_async())


if __name__ == "__main__":
    main()
