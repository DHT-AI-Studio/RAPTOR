"""Schema management REST API (BM-3).

Two upload paths, both usable from Swagger UI:
- POST /schemas         — JSON body (typed as BenchmarkSchema → Swagger shows a
  schema-aware editor + example, with automatic field-level validation).
- POST /schemas/upload  — upload a YAML or JSON *file* (Swagger shows a file picker).

Invalid schemas return 422 with field-level errors.
"""
from __future__ import annotations

import json
from typing import Any, Dict, List

import yaml
from fastapi import APIRouter, File, HTTPException, Query, Response, UploadFile, status
from pydantic import ValidationError

from app.models.schema import BenchmarkSchema
from app.services import schema_store

router = APIRouter(prefix="/benchmark", tags=["Schemas"])


def _validate_or_422(data: Any) -> BenchmarkSchema:
    """Validate a parsed mapping into a BenchmarkSchema, or raise a clean 422."""
    if not isinstance(data, dict):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail="Schema body must be a mapping/object")
    try:
        return BenchmarkSchema.model_validate(data)
    except ValidationError as exc:
        # exc.json() renders a fully JSON-safe error list (exc.errors() can embed
        # raw exception objects in `ctx`).
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=json.loads(exc.json()))


@router.post("/schemas", status_code=status.HTTP_201_CREATED,
             summary="Upload a marking schema (JSON body)")
async def upload_schema(schema: BenchmarkSchema) -> Dict[str, Any]:
    return await schema_store.create_schema(schema)


@router.post("/schemas/upload", status_code=status.HTTP_201_CREATED,
             summary="Upload a marking schema from a YAML or JSON file")
async def upload_schema_file(
    file: UploadFile = File(..., description="A .yaml / .yml / .json schema file"),
) -> Dict[str, Any]:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail="Empty file")
    try:
        data = yaml.safe_load(raw)  # YAML is a JSON superset — handles both formats
    except yaml.YAMLError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"Malformed file: {exc}")
    return await schema_store.create_schema(_validate_or_422(data))


@router.get("/schemas", summary="List marking schemas (paginated)")
async def list_schemas(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    return await schema_store.list_schemas(limit=limit, offset=offset)


@router.get("/schemas/{schema_id}", summary="Get full schema definition")
async def get_schema(schema_id: str) -> Dict[str, Any]:
    schema = await schema_store.get_schema(schema_id)
    if schema is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schema not found")
    return schema


@router.delete("/schemas/{schema_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete a schema")
async def delete_schema(schema_id: str) -> Response:
    deleted = await schema_store.delete_schema(schema_id)
    if not deleted:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Schema not found")
    return Response(status_code=status.HTTP_204_NO_CONTENT)
