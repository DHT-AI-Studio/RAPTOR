"""Router for POST /infer/google-shieldgemma-2."""
from __future__ import annotations

import asyncio
import logging
import os

from fastapi import APIRouter, File, Form, HTTPException, UploadFile

from app.models.moderation import ImageModality
from app.services import moderation
from app.services.model_resource_manager import (
    InsufficientResourcesError,
    ModelBusyTimeoutError,
)
from app.services.shieldgemma2_inference import run_shieldgemma2_inference

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Image Guard"])


@router.post(
    "/infer/google-shieldgemma-2",
    summary="Run Google ShieldGemma 2 safety classification on an uploaded image",
    description=(
        "Classifies the uploaded image using `google/shieldgemma-2-4b-it` against its "
        "3 built-in safety policies (dangerous content, sexually explicit content, "
        "violence/gore).\n\n"
        "The model path is read from the `SHIELDGEMMA2_MODEL_PATH` environment variable.\n\n"
        "Returns 503 when the shared GPU lacks free memory for the model or the "
        "request timed out waiting for the GPU; 502 for other load/inference failures.\n\n"
        "Returns an `overall_verdict` (\"safe\"/\"unsafe\") and a per-policy breakdown "
        "with `unsafe_probability`, `safe_probability`, `verdict`, and `policy_definition`.\n\n"
        "**`modality`** (optional, CM-1): `image` or `video_frame`. When set, the response also "
        "has a `moderation` verdict: the scores mapped onto the moderation categories (nsfw, "
        "violence, hate, self_harm) against that modality's thresholds, from the active "
        "policy's `moderation` block or the built-in default. `not_covered` lists categories "
        "ShieldGemma 2 can't judge (hate). Without `modality` the response is unchanged."
    ),
)
async def infer_shieldgemma2(
    image: UploadFile = File(..., description="Image file to classify"),
    modality: ImageModality | None = Form(
        None, description="`image` or `video_frame`: also return a policy-aware `moderation` verdict (CM-1)"),
) -> dict:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="uploaded file 'image' is empty")

    model_path = os.environ.get(
        "SHIELDGEMMA2_MODEL_PATH",
        "/opt/nfsc/Download/DATA/git/google/shieldgemma-2-4b-it",
    )

    try:
        result = await asyncio.to_thread(run_shieldgemma2_inference, model_path, image_bytes)
    except (InsufficientResourcesError, ModelBusyTimeoutError) as exc:
        # Retryable: the GPU is shared and currently too full / busy.  Never a verdict.
        logger.warning("shieldgemma-2 unavailable: %s", exc)
        raise HTTPException(
            status_code=503, detail=f"shieldgemma-2 unavailable: {exc}"
        ) from exc
    except Exception as exc:
        logger.exception("shieldgemma-2 inference failed")
        raise HTTPException(
            status_code=502, detail=f"shieldgemma-2 inference failed: {exc}"
        ) from exc

    if modality is not None:
        scores = {name: p["unsafe_probability"] for name, p in result["policies"].items()}
        result["moderation"] = moderation.moderate_scores(scores, modality, await moderation.resolve_moderation())
    return result
