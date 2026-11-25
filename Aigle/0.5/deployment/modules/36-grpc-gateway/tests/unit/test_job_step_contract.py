"""Contract between the media orchestrators and WatchJob/GetJob's step -> state table.

WatchJob only knows a job's state through the `step` string the orchestrators
(modules 09-12) write to `{media}_orchestrator:{job_id}`. A step the gateway has no
entry for reads as UNSPECIFIED, which is how failed jobs once looked "stuck forever"
(`error` was missing). This test reads the orchestrators' own source for every `step`
they write and fails when one is not mapped, so a new or renamed step in 09-12 fails
the first PR that introduces it, not a user's WatchJob.

Only each module's `*_orchestrator_service` is read, plus module 13's KafkaService
(which writes the first `queued`): sub-services such as video_analysis_service keep
their own state under their own Redis prefix, which the gateway never reads.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest
from raptor.v1 import raptor_pb2
from services.job_service import _STATE_BY_STEP, _TERMINAL_STATES

MODULES_DIR = Path(__file__).resolve().parents[3]

_STEP_WRITE = re.compile(
    r"""(?:["']step["']\s*:\s*["']([^"'\s]+)["']|\[["']step["']\]\s*=\s*["']([^"'\s]+)["'])"""
)
_ORCHESTRATOR_MODULES = {
    "09": "audio",
    "10": "image",
    "11": "video",
    "12": "document",
}


def _steps_written_by(files) -> set[str]:
    steps: set[str] = set()
    for f in files:
        for match in _STEP_WRITE.finditer(f.read_text(errors="ignore")):
            steps.add(match.group(1) or match.group(2))
    return steps


def _orchestrator_steps() -> dict[str, set[str]]:
    found: dict[str, set[str]] = {}
    for module_id, media in _ORCHESTRATOR_MODULES.items():
        module_dir = next(MODULES_DIR.glob(f"{module_id}-*"), None)
        files = list(module_dir.glob("worker/services/*_orchestrator_service/**/*.py")) if module_dir else []
        found[media] = _steps_written_by(files)
    found["kafka_service"] = _steps_written_by([MODULES_DIR / "13-api-services" / "app" / "services" / "kafka_service.py"])
    return found


STEPS = _orchestrator_steps()


@pytest.mark.parametrize("media", list(_ORCHESTRATOR_MODULES.values()))
def test_the_scan_found_each_orchestrators_steps(media):
    # Guards the scan itself: if an orchestrator moves or stops using this literal
    # form, the contract test below would silently check nothing for it.
    assert {"complete", "error", "indexing"} <= STEPS[media], (
        f"found {sorted(STEPS[media])} for the {media} orchestrator; the scan in this file "
        "needs updating to match how it now writes its step"
    )


def test_kafka_service_writes_the_initial_queued_step():
    assert "queued" in STEPS["kafka_service"]


@pytest.mark.parametrize(
    "media,step",
    sorted((media, step) for media, steps in STEPS.items() for step in steps),
)
def test_every_step_an_orchestrator_writes_maps_to_a_known_state(media, step):
    assert _STATE_BY_STEP.get(step, raptor_pb2.JOB_STATE_UNSPECIFIED) != raptor_pb2.JOB_STATE_UNSPECIFIED, (
        f"{media} writes step {step!r}, which job_service._STATE_BY_STEP does not map: "
        "WatchJob/GetJob would report UNSPECIFIED for it. Add it (RUNNING for an intermediate "
        "step) and update docs/_internal/grpc-job-domain.md."
    )


def test_success_and_failure_steps_are_terminal_and_nothing_else_is():
    assert _STATE_BY_STEP["complete"] == raptor_pb2.JOB_STATE_COMPLETE
    assert _STATE_BY_STEP["error"] == raptor_pb2.JOB_STATE_FAILED
    assert {_STATE_BY_STEP["complete"], _STATE_BY_STEP["error"]} <= _TERMINAL_STATES
    non_terminal_steps = {s for s in set().union(*STEPS.values()) if s not in {"complete", "error"}}
    assert all(_STATE_BY_STEP[s] not in _TERMINAL_STATES for s in non_terminal_steps), (
        "an intermediate step must not end the WatchJob stream"
    )
