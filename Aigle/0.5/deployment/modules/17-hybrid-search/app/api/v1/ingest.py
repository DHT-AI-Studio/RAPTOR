import logging

from app.core.dependencies import (
    get_embedding_service,
    get_ingest_service,
    get_task_manager,
)
from app.core.embedding import EmbeddingManager
from app.core.exceptions import IngestError
from app.core.task_manager import TaskManager
from app.schemas import DocumentItem
from app.services.ingest import IngestService
from fastapi import APIRouter, BackgroundTasks, Depends, File, HTTPException, UploadFile

logger = logging.getLogger(__name__)
router = APIRouter()

@router.post("/json")
async def ingest_json(
    items: list[DocumentItem],
    embedder: EmbeddingManager = Depends(get_embedding_service),
    ingest_service: IngestService = Depends(get_ingest_service),
    payload_schema: str = "contextual"
):
    """Ingest JSON body with optional payload schema."""
    try:
        logger.info(f"Starting to ingest JSON body, total items: {len(items)}")
        total_count = await ingest_service.ingest_docs(items, embedder, batch_size=32, payload_schema=payload_schema)
        logger.info(f"Finished ingesting JSON body, total items: {total_count}")
        return {"status": "success", "total_count": total_count}
    except Exception as e:  # noqa: BLE001 -- endpoint handler, any failure surfaces as typed 500
        raise IngestError(f"Failed to ingest JSON body: {e!s}")

@router.post("/file")
async def ingest_file(
    file: UploadFile = File(...),
    embedder: EmbeddingManager = Depends(get_embedding_service),
    ingest_service: IngestService = Depends(get_ingest_service),
    payload_schema: str = "contextual"
):
    """Ingest file with optional payload schema."""
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files allowed")

    try:
        total_count = await ingest_service.ingest_file(file.file, embedder, batch_size=128, payload_schema=payload_schema)
        logger.info(f"Finished ingesting file: {file.filename}, total items: {total_count}")
        return {"status": "success", "total_count": total_count}
    except Exception as e:  # noqa: BLE001 -- endpoint handler, any failure surfaces as typed 500
        raise IngestError(f"Failed to ingest file: {e!s}")

@router.post("/file-background")
async def ingest_file_background(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    embedder: EmbeddingManager = Depends(get_embedding_service),
    ingest_service: IngestService = Depends(get_ingest_service),
    task_manager: TaskManager = Depends(get_task_manager),
    payload_schema: str = "contextual"
):
    """Ingest file in the background with optional payload schema   ."""
    if not file.filename.endswith(".json"):
        raise HTTPException(status_code=400, detail="Only .json files allowed")

    task_id = task_manager.create_task(file.filename)
    background_tasks.add_task(
        ingest_service.ingest_file_task,
        file.file,
        embedder,
        task_manager,
        task_id,
        128,
        payload_schema
    )

    return {"status": "started", "task_id": task_id}

@router.get("/task/{task_id}")
async def get_task_status(
    task_id: str,
    task_manager: TaskManager = Depends(get_task_manager)
):
    """Get the status of a background task."""
    status = task_manager.get_status(task_id)
    if not status:
        raise HTTPException(status_code=404, detail="Task not found")
    return status
