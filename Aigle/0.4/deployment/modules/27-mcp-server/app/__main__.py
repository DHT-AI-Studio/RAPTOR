from __future__ import annotations

from app.core.config import get_settings

_settings = get_settings()

if _settings.transport.lower() == "stdio":
    from app.stdio_main import main
    main()
else:
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=_settings.port,
    )
