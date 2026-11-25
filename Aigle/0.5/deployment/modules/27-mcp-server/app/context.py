from contextvars import ContextVar

current_bearer: ContextVar[str | None] = ContextVar("current_bearer", default=None)
