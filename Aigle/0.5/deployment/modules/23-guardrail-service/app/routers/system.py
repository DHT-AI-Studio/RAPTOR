"""Guardrail enable/disable switches.

Mounted under /guardrail → POST /guardrail/system/enable, POST
/guardrail/system/disable, POST /guardrail/system/policy/enable, POST
/guardrail/system/policy/disable, GET /guardrail/system/status. Each switch
is backed by its own Redis key with no TTL, so a state change takes effect
on the very next request — no restart required.

The policy switch (policy/enable, policy/disable) only gates
/policy/check/llm/* and /debug/policy/check/llm/* (see
app/services/state.py's is_policy_check_enabled()) — it has no effect on
/guard/check/* or the GB-4 proxy checker.
"""
from __future__ import annotations

from fastapi import APIRouter

from app.models.system import PolicyEnabledResponse, SystemEnabledResponse, SystemStatus
from app.services import state

router = APIRouter(tags=["Guardrail (system)"])


@router.post("/system/enable", response_model=SystemEnabledResponse, summary="Enable guardrail checking globally")
async def enable() -> SystemEnabledResponse:
    await state.set_enabled(True)
    return SystemEnabledResponse(enabled=True)


@router.post("/system/disable", response_model=SystemEnabledResponse, summary="Disable guardrail checking globally")
async def disable() -> SystemEnabledResponse:
    await state.set_enabled(False)
    return SystemEnabledResponse(enabled=False)


@router.post("/system/policy/enable", response_model=PolicyEnabledResponse,
             summary="Enable policy-based LLM checking (/policy/check/llm/*, /debug/policy/check/llm/*)")
async def enable_policy() -> PolicyEnabledResponse:
    await state.set_policy_enabled(True)
    return PolicyEnabledResponse(policy_enabled=True)


@router.post("/system/policy/disable", response_model=PolicyEnabledResponse,
             summary="Disable policy-based LLM checking (/policy/check/llm/*, /debug/policy/check/llm/*)")
async def disable_policy() -> PolicyEnabledResponse:
    await state.set_policy_enabled(False)
    return PolicyEnabledResponse(policy_enabled=False)


@router.get("/system/status", response_model=SystemStatus, summary="Current guardrail switch state + active policy")
async def status() -> SystemStatus:
    return await state.get_status()
