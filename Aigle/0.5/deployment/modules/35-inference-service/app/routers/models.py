"""GET /models, GET /models/{name} (INF-2) and GET/POST /models/default (INF-3)."""
from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response

from app.errors import ServiceError
from app.schemas import SetDefaultRequest
from app.services.model_defaults import ModelResolver
from app.services.model_registry import ModelRegistry

router = APIRouter(prefix="/models", tags=["models"])


def get_registry(request: Request) -> ModelRegistry:
    return request.app.state.registry


def get_resolver(request: Request) -> ModelResolver:
    return request.app.state.resolver


@router.get("", summary="List models from Module 07 MLflow + Ollama (cached)")
async def list_models(registry: Annotated[ModelRegistry, Depends(get_registry)]):
    snap = await registry.snapshot()
    return {
        "models": [m.public() for m in snap.models],
        "count": len(snap.models),
        "sources": snap.sources,
    }


# Declared before /{name} so "default" is not captured as a model name.
@router.get("/default", summary="Current default model per task")
async def get_defaults(resolver: Annotated[ModelResolver, Depends(get_resolver)]):
    return await resolver.defaults.get_all()


@router.post("/default", status_code=204, summary="Set the default model for a task (persisted in Redis)")
async def set_default(body: SetDefaultRequest, resolver: Annotated[ModelResolver, Depends(get_resolver)]):
    await resolver.validate(body.task, body.model)
    await resolver.defaults.set(body.task, body.model)
    return Response(status_code=204)


@router.get("/{name:path}", summary="One model's registry entry")
async def get_model(name: str, registry: Annotated[ModelRegistry, Depends(get_registry)]):
    entry = await registry.get(name)
    if entry is None:
        raise ServiceError(404, "model_not_found", model=name)
    return entry.public()
