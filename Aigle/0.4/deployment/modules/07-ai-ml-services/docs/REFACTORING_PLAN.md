# 07-ai-ml-services Refactoring Plan (v3 cleanup)

Tracks what's still unresolved after the v2 → v3 (spec-driven `InferenceService`) refactor, and the follow-up actions. This document is a **to-do list + decision log**, not a design doc.

- Refactor starting point: commit 4482ae1 (`Refactor inference architecture: ...`)
- Written: 2026-05-26
- File root involved: `deployment/modules/07-ai-ml-services/`

---

## Priority

- **P0** — Broken on startup or examples don't run at all, must fix
- **P1** — Refactor debt, semantic confusion, next PR
- **P2** — Design-level, discuss later
- **P3** — Documentation cleanup

---

## P0 — Must fix (this PR's scope)

### P0-1. Remove stale / broken APIs and engines
- `src/api/tts_api.py` still does `from ..inference.manager import inference_manager` — that module was replaced by `service.py`; `main.py` still does `include_router(tts_api.router)`, so **starting uvicorn raises an ImportError immediately**.
- `src/inference/engines/vibevoice_engine.py` does `from .base import BaseEngine` — `BaseEngine` was deleted, `engines/base.py` no longer exists; not currently imported anywhere, but it's a landmine left in place.
- `src/api/datasets.py` — empty file.
- `src/api/training_api.py` — already moved to module 16, already commented out in `main.py`; remove the file.

**Action**: delete these 4 files (including the whole `engines/` folder), and clean up the corresponding imports and router registrations in `main.py`.

> If TTS functionality is worth keeping, it should be re-added as "register VibeVoice as an HF model + `custom_handler`", going through the unified `/inference/infer`; for this round, just delete it and decide later whether to rebuild it.

### P0-2. Fill in `spec.py:_TASK_ALIAS`
`inference.yaml.task_to_models`'s keys use underscores (`text_generation`, `audio_classification`, `document_analysis`, `image_captioning`, `video_summary`, `audio_transcription`, `scene_detection`), but `spec.py` only aliases the `-ollama` / `-hf` / hyphenated legacy task names → the registration path calls `canonicalize_task` and throws a `ValidationError`.

**Action**: fill in the aliases so `_TASK_ALIAS` covers every task name that ever appeared in the yaml, including its underscore-style legacy form.

> Not handled in P0: the `vad-hf → asr` semantic bug (VAD ≠ ASR) — leave it for P1 to split out.

### P0-3. Relax the task/engine check in `register_lakefs_to_mlflow`
`model_manager.py:919` still requires `task in inference_task_rules and engine in inference_engines`, but `task_to_models` in the yaml is a stale config, and the new service doesn't consume `discovery/strategy/filter` at all. This makes README workflows B/C (`task: "text-generation"`) fail outright.

**Action**: change the whitelist check to:
1. `task` is accepted by `canonicalize_task` (no longer depends on the yaml).
2. `engine` is one of `{"transformers", "huggingface", "hf", "ollama"}` (i.e. the keys of `_ENGINE_TO_RUNTIME`).

### P0-4. Fix `main.py`'s version string, health key, `/api/info`
- `version="2.0.0"` → `3.0.0` (to match `inference/__init__.py`).
- `/health`'s services key `"inference_manager"` → `"inference_service"`.
- `/api/info` — remove the two now-nonexistent endpoint categories `inference_engine` and `training` (while at it, also remove mentions of stale endpoints under the `inference` section, e.g. `/inference/batch_infer`, `/select_model`, `/available_models/{task}`, `/estimate_vram`, `/gpu_status`, `/supported_tasks` — replace with the actual current endpoints).

### P0-5. Verification
- `python -m compileall src` reports no errors.
- `python -c "from src.main import app"` (inside the container, or locally with `PYTHONPATH` set) imports cleanly, no more ImportError.

---

## P1 — Next PR (not this round)

1. Clean up `inference.yaml`: remove `task_to_models`, `engines.vllm`, `engines.custom`, `fixed_inference`, `memory_manager`; keep only `engines.ollama` and `engines.transformers`.
2. Delete `manage_inference_priority` and its associated tags (200+ lines of orphaned logic).
3. Fix the `vad-hf` alias bug, unify task naming (recommend all-kebab-case).
4. Consolidate the MLflow tag schema: newly-registered models write only the v3 schema; `to_mlflow_tags` no longer writes back `inference_engine` / `inference_task`.
5. When `service.infer()`'s task doesn't match, throw a ValidationError instead of silently rewriting it.

## P2 — Design-level (P2-1/2/3/5 done; P2-4 deferred)

1. ✅ Releasing the lock during an Adapter LRU cache miss caused duplicate loads — added a per-key load lock with double-checked locking.
2. ✅ `_on_unload` popped dict fields too early — changed to "never pop proactively", rely on GC + `empty_cache` to release.
3. ✅ HF adapter's mutually-exclusive handling of quantization and `torch_dtype`; when BnB isn't available, throw `ModelLoadError` instead.
4. ⏸ Splitting up `model_manager.py`'s 1426 lines — pure relocation, no bug fixes, no test coverage yet; defer until after P3.
5. ✅ `download_model`'s `NameError` in the branch where `destination_path` is not None — moved `model_name_replace` outside the branch.

## P3 — Docs / cleanup

1. Delete stale markdown: `src/api/API_ANALYSIS.md`, `src/api/API_UPDATE_RECOMMENDATIONS.md`, the `*_FIX*.md`/`*_COMMIT_MESSAGE.md`/`MODEL_HANDLER_FIX.md` implementation-artifact files under `docs/`, etc.
2. Add a proper `.env.example` (the README lists required variables but the repo ships no template).

---

## Impact of the change

- External API behavior:
  - `/inference/tts` **will disappear** (no other Raptor module calls this endpoint: grepping `module/15`, `module/09`, `module/13` shows no references — safe to remove this round).
  - `/inference/infer`, `/models/*`, `/datasets/*`, `/gpu/*`, `/config/*` behavior unchanged.
- MLflow Registry: tag schema untouched this round; existing models need no migration.
- `inference.yaml`: untouched this round; cleanup happens in P1.

---

## Acceptance criteria (this PR)

- [ ] `from src.main import app` raises no ImportError.
- [ ] `python -m compileall src` — 0 syntax warnings.
- [ ] README workflow A (Ollama) runs end to end.
- [ ] README workflow B (HF text-generation, `task: "text-generation"`) registers and infers successfully.
- [ ] `/health` returns `services.inference_service: true`.
- [ ] `/` and `/api/info` no longer list `tts` or `inference_engine` / `training`.
