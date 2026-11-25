"""Router for POST /infer/Llama-Guard-3-11B-Vision."""
from __future__ import annotations

import asyncio
import logging
import os
import re

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.moderation import ImageModality
from app.services import moderation
from app.services.llama_guard_vision_inference import run_llama_guard_vision_inference
from app.services.model_resource_manager import (
    InsufficientResourcesError,
    ModelBusyTimeoutError,
)

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Image Guard"])


def _parse_llama_guard(raw: str) -> dict:
    lines = raw.strip().splitlines()
    label = lines[0].strip().lower() if lines else "unknown"
    categories = re.findall(r"\bS\d+\b", raw)
    return {"label": label, "categories": categories, "raw": raw.strip()}


@router.post(
    "/infer/Llama-Guard-3-11B-Vision",
    summary="Run Llama Guard 3 11B Vision safety classification on an uploaded image",
    description=(
        "Classifies the uploaded image using `meta-llama/Llama-Guard-3-11B-Vision`.\n\n"
        "The model path is read from the `LLAMA_GUARD_3_MODEL_PATH` environment variable.\n\n"
        "`prompt` is the user text that accompanied the image — pass it to assess text+image "
        "together; omit for image-only checks.\n\n"
        "Returns 503 when the shared GPU lacks free memory for the model or the "
        "request timed out waiting for the GPU; 502 for other load/inference failures.\n\n"
        "Returns `label` (\"safe\" or \"unsafe\"), a list of violated `categories` "
        "(S1–S13, empty when safe), and the `raw` decoded model output.\n\n"
        "**`modality`** (optional, CM-1): `image` or `video_frame`. When set, the response also "
        "has a `moderation` verdict: the reported S-codes mapped onto the moderation categories "
        "(nsfw, violence, hate, self_harm) with that modality's decisions, from the active "
        "policy's `moderation` block or the built-in default. Llama Guard gives no score, so "
        "thresholds don't apply here. Without `modality` the response is unchanged."
    ),
)
async def infer_llama_guard_vision(
    image: UploadFile = File(..., description="Image file to classify"),
    prompt: str | None = Form(None, description="User text that accompanied the image (optional)"),
    modality: ImageModality | None = Form(
        None, description="`image` or `video_frame`: also return a policy-aware `moderation` verdict (CM-1)"),
) -> dict:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="uploaded file 'image' is empty")

    model_path = os.environ.get(
        "LLAMA_GUARD_3_MODEL_PATH",
        "/opt/nfsc/Download/DATA/git/meta-llama/Llama-Guard-3-11B-Vision",
    )

    try:
        raw = await asyncio.to_thread(
            run_llama_guard_vision_inference, model_path, image_bytes, prompt
        )
    except (InsufficientResourcesError, ModelBusyTimeoutError) as exc:
        # Retryable: the GPU is shared and currently too full / busy.  Never a verdict.
        logger.warning("llama-guard-vision unavailable: %s", exc)
        raise HTTPException(
            status_code=503, detail=f"llama-guard-vision unavailable: {exc}"
        ) from exc
    except Exception as exc:
        logger.exception("llama-guard-vision inference failed")
        raise HTTPException(
            status_code=502, detail=f"llama-guard-vision inference failed: {exc}"
        ) from exc

    result = _parse_llama_guard(raw)
    if modality is not None:
        result["moderation"] = moderation.moderate_codes(
            result["categories"], modality, await moderation.resolve_moderation())
    return result
