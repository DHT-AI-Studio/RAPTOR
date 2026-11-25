"""Per-call state shared across interceptors via contextvars.

Each incoming RPC runs in its own asyncio Task, so a ContextVar set early in
the chain (by RequestIdInterceptor) is safely visible to interceptors that
run later for the same call (LoggingInterceptor, MetricsInterceptor) without
needing to thread it through method signatures.
"""
from __future__ import annotations

from contextvars import ContextVar

import grpc

request_id_var: ContextVar[str] = ContextVar("request_id", default="")

# Set by AuthenticationInterceptor when credentials are missing. Deliberately
# not enforced there — Auth is the outermost interceptor, so aborting on the
# spot would skip RequestId/Logging/Metrics entirely for every rejected call.
# ExceptionMappingInterceptor (innermost) performs the actual abort instead,
# so the layers in between still observe rejected calls.
auth_error_var: ContextVar[tuple[grpc.StatusCode, str] | None] = ContextVar("auth_error", default=None)

# Set by IdentityInterceptor from the Bearer token's `sub` claim (decoded
# without signature verification — a stub, same posture as
# AuthenticationInterceptor). Falls back to a fixed dev identity when absent
# or undecodable, e.g. when the call authenticated via x-api-key instead.
user_var: ContextVar[dict[str, str]] = ContextVar(
    "user", default={"user_id": "dev-user", "branch_id": "dev-user"}  # noqa: B039 -- never mutated in place, only .get()/.set(new dict)
)
