"""Module 16 training client (AUTOTUNE Phase B4).

Wraps Module 16's ``/training/submit`` + ``/training/status`` + ``/training/cancel``
into a single blocking ``train_and_wait`` used by the orchestrator loop:

    model_path = await train_and_wait(plan, knobs, minutes_per_experiment)

Guarantees the loop can rely on:
- returns the fine-tuned checkpoint path on success;
- raises ``TrainingFailed`` if the job fails/cancels;
- cancels the job and raises ``TrainingTimeout`` if it exceeds the per-experiment
  budget (so one slow candidate can't eat the whole optimization).
"""
from __future__ import annotations

import asyncio
import copy
import logging
import time
from typing import Any, Awaitable, Callable, Dict, Mapping, Optional

import httpx

from app.core.config import get_settings
from app.models.experiment import Plan

logger = logging.getLogger(__name__)


class TrainingError(Exception):
    """Base class for training-client failures."""


class TrainingFailed(TrainingError):
    """The training job reported failed/cancelled."""


class TrainingTimeout(TrainingError):
    """The training job exceeded the per-experiment budget and was cancelled."""


# Friendly search-space knob names → nested TrainerConfig path. Anything not
# listed (and without a dotted name) is treated as a top-level TrainerConfig key.
_KNOB_ALIASES: Dict[str, tuple] = {
    "lora_r": ("lora_config", "r"),
    "lora_alpha": ("lora_config", "lora_alpha"),
    "lora_dropout": ("lora_config", "lora_dropout"),
    "target_modules": ("lora_config", "target_modules"),
}

# The search space proposes a hashable preset NAME for target_modules; expand it
# to the real module list only here, at the point we build the Module 16 request.
_TARGET_MODULE_PRESETS: Dict[str, list] = {
    "attn": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "attn_mlp": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
}


def _assign(config: Dict[str, Any], path: tuple, value: Any) -> None:
    """Set a (possibly nested) key inside config, creating dicts as needed."""
    node = config
    for key in path[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):  # a scalar was where we needed a dict
            raise ValueError(f"cannot nest under non-dict key {key!r}")
    node[path[-1]] = value


def build_training_config(base: Mapping[str, Any], knobs: Mapping[str, Any]) -> Dict[str, Any]:
    """Deep-merge the fixed base config with this iteration's searched knobs.

    Knob names may be: a known alias (``lora_r`` → ``lora_config.r``), a dotted
    path (``lora_config.lora_alpha``), or a plain top-level field.
    """
    merged = copy.deepcopy(dict(base))
    for name, value in knobs.items():
        if name == "lora_variant":  # expand preset into the two PEFT LoraConfig flags
            _assign(merged, ("lora_config", "use_rslora"), value == "rslora")
            _assign(merged, ("lora_config", "use_dora"), value == "dora")
            continue
        if name == "target_modules" and isinstance(value, str):
            value = _TARGET_MODULE_PRESETS.get(value, _TARGET_MODULE_PRESETS["attn"])
        if name in _KNOB_ALIASES:
            _assign(merged, _KNOB_ALIASES[name], value)
        elif "." in name:
            _assign(merged, tuple(name.split(".")), value)
        else:
            merged[name] = value
    return merged


def build_submission(plan: Plan, knobs: Mapping[str, Any]) -> Dict[str, Any]:
    """Assemble the Module 16 TrainingJobSubmission body for one candidate."""
    return {
        "task_type": plan.task_type,
        "select_multiple_gpus": plan.select_multiple_gpus,
        "vram_budget_gb": plan.vram_budget_gb,
        "training_config": build_training_config(plan.base_training_config, knobs),
        "dataset_config": plan.dataset_config,
    }


def _model_path(status: Mapping[str, Any]) -> str | None:
    metrics = status.get("metrics") or {}
    return status.get("model_path") or metrics.get("final_model_path")


async def train_and_wait(plan: Plan, knobs: Mapping[str, Any],
                         minutes_per_experiment: float,
                         on_job_id: Optional[Callable[[str], Awaitable[None]]] = None) -> str:
    """Submit one training job, block until it finishes, return the model path.

    ``on_job_id`` (if given) is awaited with the job id right after submit, so the
    caller can record which job is in flight (e.g. for a live progress UI).

    Raises TrainingFailed / TrainingTimeout so the caller can record the iteration
    as failed and continue (never aborting the whole experiment).
    """
    s = get_settings()
    submission = build_submission(plan, knobs)
    submit_url = s.training_url.rstrip("/") + s.training_submit_path

    async with httpx.AsyncClient(timeout=s.training_submit_timeout) as client:
        resp = await client.post(submit_url, json=submission)
        resp.raise_for_status()
        job = resp.json()
    job_id = job.get("job_id")
    if not job_id:
        raise TrainingFailed(f"submit returned no job_id: {job}")
    logger.info("Training job %s submitted (knobs=%s)", job_id, dict(knobs))
    if on_job_id is not None:
        try:
            await on_job_id(job_id)
        except Exception:  # noqa: BLE001 — display-only, never break training
            pass

    status_url = f"{s.training_url.rstrip('/')}{s.training_status_path}/{job_id}"
    deadline = time.monotonic() + minutes_per_experiment * 60.0

    async with httpx.AsyncClient(timeout=s.training_submit_timeout) as client:
        while time.monotonic() < deadline:
            await asyncio.sleep(s.training_poll_seconds)
            try:
                r = await client.get(status_url)
                r.raise_for_status()
                st = r.json()
            except Exception as exc:  # noqa: BLE001 — transient poll error, retry
                logger.warning("Training %s status poll failed: %s", job_id, exc)
                continue

            state = st.get("status")
            if state == "completed":
                path = _model_path(st)
                if not path:
                    raise TrainingFailed(f"job {job_id} completed but returned no model_path")
                logger.info("Training job %s completed → %s", job_id, path)
                return path
            if state in ("failed", "cancelled"):
                err = (st.get("metrics") or {}).get("error", state)
                raise TrainingFailed(f"job {job_id} {state}: {err}")

    # Budget exceeded — cancel and WAIT for the job to actually terminate before
    # giving up. Module 16 cancellation is graceful (checked per training step,
    # ~3s throttle, then Lightning stops at the batch boundary) but the GPU is
    # only freed after teardown/checkpoint-save. If we returned immediately, the
    # loop's next candidate would submit while this job is still winding down.
    # (Module 16's scheduler VRAM-gates new jobs so this can't OOM, but waiting
    # keeps the budget honest and avoids piling up terminating jobs.)
    await _cancel_and_wait(job_id)
    raise TrainingTimeout(f"job {job_id} exceeded {minutes_per_experiment} min budget")


async def _cancel_and_wait(job_id: str) -> None:
    s = get_settings()
    cancel_url = f"{s.training_url.rstrip('/')}{s.training_cancel_path}/{job_id}"
    status_url = f"{s.training_url.rstrip('/')}{s.training_status_path}/{job_id}"
    try:
        async with httpx.AsyncClient(timeout=s.training_submit_timeout) as client:
            await client.post(cancel_url)
            logger.info("Requested cancel of over-budget job %s; waiting for GPU release", job_id)
            grace_deadline = time.monotonic() + s.training_cancel_grace_seconds
            while time.monotonic() < grace_deadline:
                await asyncio.sleep(s.training_poll_seconds)
                try:
                    r = await client.get(status_url)
                    r.raise_for_status()
                    state = r.json().get("status")
                except Exception:  # noqa: BLE001 — transient poll error, retry
                    continue
                if state in ("cancelled", "failed", "completed"):
                    logger.info("Job %s reached terminal state '%s' (GPU freed)", job_id, state)
                    return
            logger.warning("Job %s did not terminate within %ss grace; proceeding "
                           "(scheduler VRAM-gates the next job)", job_id, s.training_cancel_grace_seconds)
    except Exception as exc:  # noqa: BLE001 — best-effort cleanup
        logger.warning("cancel-and-wait for job %s failed: %s", job_id, exc)
