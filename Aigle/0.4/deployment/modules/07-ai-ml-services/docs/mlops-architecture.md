# Raptor MLOps Architecture Design: A Continuous-Learning Loop Centered on the AI Model Lifecycle

> Status: design proposal (2026-07)
> Scope: integration between module 07 (ai-ml-services / AIML API) and module 16 (training-service),
> with the goal of letting models keep getting smarter from the data accumulated through real usage (a continuous learning flywheel).

---

## Contents

1. [Current-state inventory](#1-current-state-inventory)
2. [Gap analysis: what's missing from the loop](#2-gap-analysis-whats-missing-from-the-loop)
3. [Target architecture: the MLOps continuous-learning loop](#3-target-architecture-the-mlops-continuous-learning-loop)
4. [Detailed design per stage](#4-detailed-design-per-stage)
5. [Inter-module integration contract](#5-inter-module-integration-contract)
6. [Infrastructure unification plan](#6-infrastructure-unification-plan)
7. [Phased rollout roadmap](#7-phased-rollout-roadmap)
8. [Risks & mitigations](#8-risks--mitigations)
9. [Fine-tuning feasibility for multimodal & non-LLM models](#9-fine-tuning-feasibility-for-multimodal--non-llm-models)
10. [A layered strategy for "gets smarter with use": RAG × memory × fine-tuning](#10-a-layered-strategy-for-gets-smarter-with-use-rag--memory--fine-tuning)

---

## 1. Current-state inventory

### 1.1 Module 07 (ai-ml-services, port 8010) — already implemented

Module 07 is already the hub for "where models come from, how they're versioned, how they go live":

| Capability | Implementation |
|---|---|
| Model acquisition | `POST /models/download` (HuggingFace), `/models/batch_download`, local Ollama management |
| Version control | LakeFS git-like atomic commits (`/models/upload_to_lakefs`); every MLflow version points to an immutable commit via a `lakefs_commit_id` tag |
| Model registration | `POST /models/register_from_lakefs`, `/models/register_ollama` — writes the spec tag at registration (`runtime` / `model_class` / `processor_class` / `pipeline_task` / `custom_handler` / `torch_dtype`) |
| Lifecycle | `POST /models/transition_stage` (production / staging / archived / none), can auto-archive old versions |
| Dataset management | LakeFS upload/download, network download (`/datasets/download_from_network`), MLflow registration, search |
| Unified inference gateway | `POST /inference/infer`, spec-driven dispatch to the Ollama / HF Transformers adapters; task families cover text-gen / VLM / ASR / OCR / audio-cls / video / document / TTS / embedding / rerank |
| OpenAI-compatible API | `/v1/models`, `/v1/chat/completions` (with genuine token-by-token streaming), `/v1/completions`, `/v1/audio/transcriptions`, `/v1/audio/speech`, `/v1/embeddings`, `/v1/rerank` |
| Resource management | adapter LRU cache, automatic idle unload (`idle_timeout` / `keep_alive`), GPU status API, VRAM estimation |

### 1.2 Module 16 (training-service, port 8009) — already implemented

Module 16 is a fully-featured **fine-tuning execution engine**:

| Capability | Implementation |
|---|---|
| Training job API | `POST /api/v1/training/submit`, `GET /status/{job_id}`, `GET /list`, `POST /cancel/{job_id}`, `DELETE /delete/{job_id}` |
| Fine-tuning methods | LoRA (PEFT), QLoRA (bitsandbytes 4-bit nf4 + double quant), 8-bit AdamW, gradient checkpointing, bf16, Flash Attention 2 (optional) |
| Distributed training | DeepSpeed ZeRO Stage 3 (param/optimizer CPU offload), multi-GPU |
| GPU scheduling | NVML monitoring + a Worst-Fit VRAM allocation algorithm (single-GPU: picks the one with the most free VRAM; multi-GPU: accumulates GPUs until `vram_budget_gb` is met), reserving overhead |
| Job lifecycle | Job state persisted in Redis (queued / running / completed / failed / cancelled), subprocess CUDA-isolated execution, graceful cancel with checkpoint saving |
| Progress tracking | MLflow experiment logging (loss / lr / progress / TTC estimate), `ttc_progress_callback` |
| Dataset loading | two task types, text and instruction; flexible `column_mapping` (messages/input/output/context/reasoning + automatic fallback, role-alias normalization) |
| Training framework | wrapped in a PyTorch Lightning module; after training, the best/last checkpoint is consolidated into HF format (`save_hf_model` → `final_model_path`) |
| Ops | an NVIDIA persistence watchdog (cron + healthcheck, auto-restarts containers with a stale `/dev/nvidia*`) |

### 1.3 How they interact today

Module 16's README already defines a manual flow — "download the model/dataset via the AIML API first, then submit training with the local paths":

```
Client → 07 /models/download            → model_path
Client → 07 /datasets/download_from_network → dataset_path
Client → 16 /api/v1/training/submit     → job_id
Client → 16 /api/v1/training/status     → poll until completed
```

**The flow stops dead at `final_model_path`** — the training artifact sits in a directory on NFS and never makes it back into any registry.

---

## 2. Gap analysis: what's missing from the loop

Breaking "models get smarter from use" apart, you need a complete loop:
**serve → collect usage data → curate into training data → fine-tune → evaluate → register → promote to production → serve (new version)**.

Against the current state:

| # | Gap | Description |
|---|---|---|
| G1 | **Training output never flows back** | Module 16 only leaves a `final_model_path` after training — nothing gets uploaded to LakeFS, registered in the MLflow Registry, or given a spec tag → module 07 can't serve it |
| G2 | **Two separate MLflow / infrastructure setups** | Module 16 points at its own compose's `http://mlflow:5555`; module 07 has its own `raptor-mlflow` — experiment logs and model registrations live in two separate worlds |
| G3 | **No usage-data collection** | Module 07's inference gateway doesn't log requests/responses, and there's no channel for user feedback (thumbs up/down, corrections, acceptance rate) → no raw material for "learning from use" |
| G4 | **No dataset curation** | Raw interaction logs ≠ training data; missing de-identification, quality filtering, format conversion (→ ChatML messages), and a split against a golden set |
| G5 | **No evaluation gate** | Whether a freshly-trained model is actually good has no objective judgment — promotion to production is entirely manual |
| G6 | **No pipeline orchestrator** | The whole pipeline above has no state machine tying it together, and no trigger mechanism (scheduled / feedback-volume / drift detection) |
| G7 | **A gap in serving LoRA adapters** | Module 16 produces a LoRA adapter (or a fully-merged model); module 07's HF adapter today is built mainly around full models, and the spec has no defined way to load a `base_model + adapter` combination |
| G8 | **Training/inference GPU contention** | Module 16 and module 07 each have their own GPU manager, unaware of each other; on a shared host, a training job can push inference into OOM |

---

## 3. Target architecture: the MLOps continuous-learning loop

### 3.1 Overview

With **07 as the control plane (the single registry, version control, service entry point, and orchestration hub)** and
**16 as the training execution plane (a pure fine-tuning execution engine)**. Reasoning for this split:

- 07 already has the MLflow Registry + LakeFS + spec system — there should be exactly one single source of truth for "what is this model, what stage is it in."
- 16's value is in GPU scheduling and training execution; it shouldn't reinvent a registry — it just needs to "take the job, train, report back."

```mermaid
graph TB
    subgraph Serving["① Serving layer (07)"]
        GW[Inference gateway<br/>/inference/infer + /v1/*]
        REG[MLflow Registry<br/>stage: production/staging]
    end

    subgraph Flywheel["② Data flywheel (new in 07)"]
        LOG[Interaction log<br/>request/response/latency]
        FB[Feedback collection API<br/>rating/correction/acceptance]
        CUR[Dataset Curator<br/>filter→de-identify→convert to ChatML]
    end

    subgraph Training["③ Training layer (16)"]
        TS[Training Service<br/>LoRA/QLoRA + DeepSpeed]
    end

    subgraph Gate["④ Evaluation & promotion (new in 07)"]
        EVAL[Evaluation Runner<br/>golden set + LLM-judge]
        PROMO[Promotion gate<br/>transition_stage]
    end

    subgraph Infra["Shared infrastructure"]
        MLF[(Single MLflow<br/>Tracking+Registry)]
        LFS[(LakeFS<br/>model+data versioning)]
        RDS[(Redis)]
        NFS[(NFS shared storage)]
    end

    ORCH[Lifecycle Orchestrator<br/>new in 07: pipeline state machine]

    GW -->|every inference| LOG
    FB --> CUR
    LOG --> CUR
    CUR -->|versioned dataset| LFS
    ORCH -->|1. trigger| CUR
    ORCH -->|2. submit job| TS
    TS -->|metrics| MLF
    TS -->|final_model_path (NFS)| ORCH
    ORCH -->|3. upload+register staging| LFS
    ORCH --> REG
    ORCH -->|4. evaluate| EVAL
    EVAL -->|pass| PROMO
    PROMO -->|production| REG
    REG -->|resolve spec| GW
```

### 3.2 The full story of one loop

1. A user goes through Module 07's `/v1/chat/completions` or `/inference/infer` to use the production model.
2. The gateway asynchronously writes every interaction (prompt, response, model version, latency) to the interaction log; the frontend/upstream service writes user feedback back (thumbs up/down, a human-corrected draft, whether it was accepted) through the new `/feedback` API, linked to the interaction ID.
3. The Orchestrator kicks off the pipeline based on a trigger condition (N accumulated high-quality feedback items / a periodic cron / manual):
   - The Curator turns "interactions with a positive signal" and "interactions with a human correction" into an instruction-format dataset, splits it into train/val, and holds back a **golden eval set** that never enters training; uploads to LakeFS as a new commit, registers it with MLflow datasets.
4. The Orchestrator calls 16's `POST /api/v1/training/submit` (both model and dataset paths live on shared NFS), and polls `/status`.
5. Once training finishes, the Orchestrator runs **post-training flow-back**:
   - Uploads `final_model_path` (or the LoRA adapter) to LakeFS → a new commit;
   - Registers it via `register_from_lakefs` as a new model version, inheriting the base model's spec tags, with added lineage tags (`base_model_version`, `dataset_commit_id`, `training_job_id`, `mlflow_run_id`);
   - Sets stage to `staging`.
6. The Evaluation Runner runs the golden set against the staging version (through Module 07's own inference gateway — a staging model can be called by name directly), comparing it against the current production version: task metrics + LLM-as-judge scoring + regression checks (must not regress on any prior golden case).
7. Passes the gate → `transition_stage` promotes it to production (the old version auto-archives, and can be rolled back at any time); fails → marked rejected, with the full experiment record kept for analysis.
8. The gateway resolves to the new production version on its next call, closing the loop — back to step 1.

Every part of every loop (dataset commit, training run, model version, evaluation report) is traceable: **any production model can answer "which dataset trained you, from which base, with what hyperparameters, and how you scored on evaluation."**

---

## 4. Detailed design per stage

### 4.1 ② Data flywheel: interaction logging & feedback collection (new in 07)

**Interaction logging (inference logging)**

- After `InferenceService.run()` completes, write asynchronously to storage (without blocking the inference response), recording:
  `interaction_id`, `timestamp`, `model_name`, `model_version`, `task`, an input summary (or full text, depending on the privacy setting), the full output, latency, token count, `client_id`.
- Storage recommendation: write to a Redis Stream first (low latency, natural queue semantics), with a background worker batching it out to a partitioned JSONL file on NFS (`/aiml/data/interactions/{model}/{date}.jsonl`). Switch to PostgreSQL/ClickHouse once volume grows — the interface stays the same.
- Add a switch: `INFERENCE_LOGGING=off|metadata|full` — large multimodal payloads only log metadata + a file pointer.

**Feedback API (new router: `src/api/feedback_api.py`)**

| Endpoint | Purpose |
|---|---|
| `POST /feedback` | `{interaction_id, rating: up/down, correction?: str, tags?: [..]}` — an upstream app writes back a user signal |
| `GET /feedback/stats` | per-model feedback volume, positive/negative ratio, count awaiting curation (used by the Orchestrator as a trigger condition) |

`correction` (a human-corrected draft) is the highest-value training signal: it directly forms an `(input, corrected_output)` pair.

### 4.2 ② Data flywheel: Dataset Curator (new in 07)

`src/core/dataset_curator.py` + `POST /datasets/curate`:

1. **Extraction**: pull "raw pairs with rating=up" and "corrected pairs with a correction" (weighted higher) for a given model within a time window, from the interaction log.
2. **Cleaning**: dedup (near-duplicate detection via embeddings — calling our own `/v1/embeddings` directly), length/language filtering, PII de-identification, dropping refusals/error responses.
3. **Formatting**: convert to the ChatML `messages` format already supported by module 16 — deliberately aligned with 16's `column_mapping: {messages: messages}`, so module 16 needs no code changes.
4. **Splitting**: train / val / **golden** (golden is evaluation-only, never enters training, and accumulates across loops to form a growing regression test set).
5. **Versioning**: upload to the LakeFS dataset repo (a new commit), `POST /datasets/register` to register with MLflow, with metadata recording the source time window, filter rules, and sample count.

### 4.3 ③ Training layer: keep 16 thin

Module 16 needs **almost no changes**. The only two suggested tweaks:

1. Point `MLFLOW_TRACKING_URI` at 07's shared MLflow (see §6), so training curves and the Registry live in the same UI.
2. (Optional) add `callback_url` support to the `/submit` response and Redis job record: POST a notification to the Orchestrator on completion, saving a poll loop. Polling is fine for P0.

Training parameter templates (LoRA r/alpha, lr, epochs, quantization, DeepSpeed config) are managed on 07's side as **training profiles** (e.g. `configs/training_profiles/{small-lora,qlora-24gb,multi-gpu-zero3}.yaml`); the Orchestrator expands them into 16's submission payload at submit time. This keeps 16 a stateless executor, with strategy concentrated in the control plane.

### 4.4 Post-training flow-back (the Orchestrator's core step in 07, closes G1)

Once training completes:

```
final_model_path (NFS)
  → 07 /models/upload_to_lakefs        # repo: {registered_name}, a new commit
  → 07 /models/register_from_lakefs    # a new version, stage=staging
       spec tags: inherited from the base model (runtime=transformers, model_class, torch_dtype…)
       lineage tags:
         base_model        = Qwen/Qwen2.5-1.5B-Instruct
         base_model_version= models:/qwen2.5-1.5b/3
         adapter_type      = lora | merged
         dataset_commit_id = lakefs://datasets/{repo}/{commit}
         training_job_id   = {16's job_id}
         training_run_id   = {MLflow run_id}
```

**Two serving paths for a LoRA adapter** (closes G7):

| Path | Approach | Trade-off |
|---|---|---|
| A. Merge, then upload (**recommended for P0**) | Run `merge_and_unload()` before flowing back, producing a full HF model to upload | Module 07's existing HF adapter serves it with zero changes; the cost is each version taking a full model's worth of storage |
| B. Load the adapter separately (P2) | Add `adapter_path` + `base_model` to the spec; the HF adapter loads the base then applies the PEFT adapter | Saves storage (an adapter is only tens of MB); multiple adapters on the same base can share weights; requires changing 07's `hf_transformers.py` loading logic |

### 4.5 ④ Evaluation gate (new in 07, closes G5)

`src/core/evaluation_runner.py` + `POST /evaluation/run`:

- Inputs: candidate (the staging version), baseline (the current production version), the golden dataset commit.
- Both models answer every question through Module 07's own inference gateway (which incidentally also verifies "this model can really be served by this platform" — deployment validation and quality evaluation done in one pass).
- Three metric tiers:
  1. **Hard metrics**: can it load, can it infer, is latency/throughput within budget;
  2. **Automated quality**: task-relevant metrics (exact match / BLEU / embedding similarity — reusing `/v1/embeddings`);
  3. **LLM-as-judge**: a stronger already-registered model (via `/v1/chat/completions`) does a blind pairwise comparison of candidate vs. baseline, producing a win rate.
- Produces an evaluation report stored as an MLflow run (attached under the candidate version), returning `pass/fail + a recommendation`.
- Promotion strategy is configurable: `auto` (auto-promote when the win rate ≥ a threshold with no regression) or `manual` (wait for a human to call `transition_stage` after seeing the report). **Manual is recommended for the initial rollout**.

### 4.6 Lifecycle Orchestrator (new in 07, closes G6)

`src/core/lifecycle_orchestrator.py` + `src/api/lifecycle_api.py`. A lightweight Redis-backed state machine (following the same pattern as 16's job manager) — **no Airflow/Kubeflow** — at single-machine/small-cluster scale, the operational cost of building a state machine yourself is far lower than adopting a whole workflow platform.

Pipeline states:

```
TRIGGERED → CURATING → DATASET_READY → TRAINING → TRAINED
          → REGISTERING → STAGED → EVALUATING → EVALUATED
          → (auto/manual) PROMOTED | REJECTED
Any step failing → FAILED (with the step name and error; leftover resources can be cleaned up)
```

API:

| Endpoint | Purpose |
|---|---|
| `POST /lifecycle/pipelines` | Start one loop: `{model_name, training_profile, dataset_source: feedback\|explicit, trigger: manual\|scheduled\|feedback_threshold}` |
| `GET /lifecycle/pipelines/{id}` | Query pipeline status (including each step's downstream ID: dataset commit, job_id, model version, eval run) |
| `GET /lifecycle/pipelines` | History listing |
| `POST /lifecycle/pipelines/{id}/approve` | Approve promotion in manual mode |
| `POST /lifecycle/pipelines/{id}/abort` | Abort (cascades to cancelling 16's training job) |

Triggers (phase 3): periodic cron, `/feedback/stats` hitting a threshold, (future) drift detection.

### 4.7 GPU coordination (closes G8)

A pragmatic tiered approach:

- **P0 (separate machines/GPUs)**: at deployment time, just use `CUDA_VISIBLE_DEVICES` to keep training and inference apart (16 already supports this) — a documentation convention is enough.
- **P1 (sharing a GPU on the same machine)**: before 16 submits a job, the Orchestrator first checks 07's `/gpu` status and loaded models; if needed, it calls 07's `/inference/unload-model` to free an idle model before training starts. 07's idle-unload already naturally reduces contention.
- **P2**: extract a shared GPU-lease service (Redis records per-GPU VRAM leases; both 07's and 16's gpu_manager acquire a lease before occupying a GPU). Not worth doing before that.

---

## 5. Inter-module integration contract

### 5.1 Service call matrix

| Caller → callee | Interface | Purpose |
|---|---|---|
| Orchestrator(07) → 16 | `POST /api/v1/training/submit`, `GET /status/{id}`, `POST /cancel/{id}` | submit/track/abort training |
| 16 → shared MLflow | tracking API | training metrics (unchanged behavior, just a different URI) |
| Orchestrator(07) → 07 itself | internal calls to model_manager / dataset_manager | upload to LakeFS, register, transition stage |
| Evaluation(07) → 07's inference gateway | `POST /inference/infer` | staging vs. production head-to-head |
| Upstream app → 07 | `POST /feedback` | feedback write-back |

### 5.2 Shared storage path convention (NFS)

Both modules must mount the same NFS (**note: the correct endpoint on this machine is `.165:2050`, not the old `.123:2049` in the .env**; module 16's `.env` currently still has `192.168.157.123` — this needs fixing as part of the integration):

```
/aiml/tmp/models/{name}          # a base model downloaded by 07 → 16's model_name_or_path
/aiml/data/datasets/{name}       # a curated dataset → 16's dataset_name_or_path
/aiml/data/checkpoints/{job_id}/ # 16's training checkpoints and final_model_path
/aiml/data/interactions/         # the interaction-log JSONL
```

Principle: **paths are only ever passed within the same NFS namespace** — both modules' containers must mount at the same point (both as `/aiml`), to avoid "a path 16 returns that 07 can't read."

### 5.3 Model naming & versioning convention

- Fine-tuned artifact registered name: `{base_short_name}-ft-{scope}` (e.g. `qwen2.5-1.5b-ft-taiwanchat`); iterations increment the MLflow version — **don't** stuff a date into `registered_name`.
- At most one `production` and one `staging` per registered_name at a time (07's transition_stage already auto-archives).
- Rollback = transition the previous archived version back to production — there's no "delete a version" in the closed loop.

---

## 6. Infrastructure unification plan (closes G2)

| Component | Current state | Target |
|---|---|---|
| MLflow | 16 has its own (`:5555`), 07 has its own (`raptor-mlflow`) | **Keep only 07's** `raptor-mlflow`; point 16's `MLFLOW_TRACKING_URI` at it. Remove the mlflow service from 16's compose |
| Redis | 16 uses `raptor-redis-standalone` | Share the same Redis, isolated by key prefix (16 already uses its own prefix; the Orchestrator uses `lifecycle:*`) |
| LakeFS | Only 07 has it | Keep it that way — only 07 talks to LakeFS directly; 16 remains completely unaware LakeFS exists (decoupled via NFS paths) |
| NFS | Each has its own .env config, with a stale endpoint | Unify on `.165:2050`, with a unified mount point of `/aiml` |
| Network | Each compose forms its own network | Join the same external docker network (e.g. `raptor-net`); the Orchestrator calls `raptor-training-service:8009` by service name |

Migration note: after switching the MLflow URI, module 16's old experiment records stay on the old instance — that's one-time historical data, not worth migrating; just keep the old volume around for reference.

---

## 7. Phased rollout roadmap

### Phase 0 — Wire up the flow-back (minimal loop, manually triggered)
> Once done: a single API call chain can go "take a production model + a specified dataset → fine-tune → automatically appear in staging, ready for inference."

- [ ] Infrastructure unification: shared MLflow / Redis / NFS (including fixing 16's NFS endpoint), shared docker network
- [ ] Add the Orchestrator skeleton to 07 + `POST /lifecycle/pipelines` (dataset_source initially only supports explicitly specifying an existing dataset)
- [ ] Post-training flow-back: merge LoRA → upload_to_lakefs → register (staging) + lineage tags
- [ ] Training profiles (start with 2: `qlora-single-gpu`, `lora-multi-gpu`)
- [ ] Verification: run `qwen2.5-1.5b` + a TaiwanChat subset through the whole chain; the staging model can be called by name via `/v1/chat/completions`

### Phase 1 — Evaluation gate
- [ ] Evaluation Runner (hard metrics + LLM-as-judge head-to-head) + golden set management
- [ ] Wire `approve` / auto-promote into `transition_stage`
- [ ] Evaluation reports stored in MLflow

### Phase 2 — Data flywheel
- [ ] Inference-gateway interaction logging (Redis Stream → NFS JSONL)
- [ ] `POST /feedback` API + stats
- [ ] Dataset Curator (feedback → ChatML → versioned in LakeFS)
- [ ] The pipeline's `dataset_source: feedback` mode
- [ ] (Optional) load LoRA adapters separately (spec `adapter_path`), saving storage

### Phase 3 — Automation & guardrails
- [ ] Triggers: cron / feedback threshold
- [ ] GPU coordination P1 (unload idle inference models before training starts)
- [ ] Regression golden set accumulates across loops; auto-block on regression
- [ ] (Optional) canary: split traffic between staging/production by ratio on the OpenAI API, comparing live metrics before promoting

---

## 8. Risks & mitigations

| Risk | Description | Mitigation |
|---|---|---|
| **Catastrophic forgetting** | Repeatedly fine-tuning only on user-interaction data degrades general ability | The golden set must include general-ability regression questions; curated data mixes in a fixed proportion of original instruct data (replay); LoRA's own low-rank updates already limit the blast radius |
| **Feedback bias / data poisoning** | Thumbs up/down signals are noisy; malicious input could pollute the training set | Correction-type samples get priority; rating-type samples must pass quality filtering; curation reports are available for spot-checking; the golden set blocks regressions |
| **Privacy** | Interaction logs contain user content | Tiered `INFERENCE_LOGGING` switch, PII de-identification at the curation stage, tightened permissions on log directories |
| **Evaluation gaming itself** | LLM-judge favoring format over correctness | Judge uses blind pairwise comparison + re-tested with order swapped; hard metrics and the regression set have veto power |
| **GPU contention** | Training crowding out inference | The tiered plan in §4.7; P0 relies on deployment-level isolation |
| **Storage bloat** | Every merge-round uploads a full model to LakeFS | Phase 2 switches to separate adapters; a periodic cleanup policy for archived versions (keep the most recent N) |
| **Orchestrator single point of failure** | Orchestrator going down stalls the pipeline | All state lives in Redis, recoverable on process restart; every step is idempotent (safe to rerun) |

---

## 9. Fine-tuning feasibility for multimodal & non-LLM models

> Cross-referenced against the model inventory in [all_raptor_models.md](all_raptor_models.md), family by family.
> The conclusion up front: **"technically fine-tunable" and "worth adding to the loop" are two different questions.** The
> real high-ROI items are LLM LoRA (already designed for) and **embedding / reranker fine-tuning** (cheap, and makes the
> entire RAG stack smarter); VLM LoRA is next (needs 16 extended); ASR after that; TTS / diarization / PaddleOCR are not
> recommended for inclusion.

### 9.1 Per-model-family assessment

| Model family | Fine-tunable? | Method | What needs to change in 16 | Source of usage data | Recommended priority |
|---|---|---|---|---|---|
| **Ollama LLM** (qwen3.5:9b, qwen2.5:7b, qwen3.5:0.8b) | ⚠️ Indirectly | The GGUF-quantized format **can't be fine-tuned directly**. Correct path: fine-tune the HF weights of the same model → serve the artifact via 07's transformers runtime, or convert back to GGUF with llama.cpp and `ollama create` it | No changes needed (this is the existing text-LoRA flow) | Conversation and feedback from 13/15/20/21 | **P0** (already designed for — just watch the GGUF↔HF conversion) |
| **VLM** (InternVL3_5-1B/4B, Qwen2.5-VL) | ✅ | LoRA fine-tune the LLM part, freeze the ViT (both vendors have an official finetune recipe; the 1B~4B tier fits QLoRA on a single 24GB GPU) | A new `task_type: vlm_instruction`: a multimodal dataset loader (image + conversation pairs), model-specific collate/chat template; the lightning module mounts PEFT onto the language-model submodule | Image captioning, frame description, and document-analysis output from 10/11/12 + human corrections | **P2**: once enough corrections have accumulated |
| **ASR** (the transformers version of Whisper large-v3) | ✅ | HF seq2seq fine-tuning (can use LoRA); notably effective for domain vocabulary/accent adaptation | A new audio dataset loader (audio + corrected transcript), `task_type: asr` | Transcriptions from 09 + human-corrected transcripts (a naturally high-value pair) | **P2~P3**: only worthwhile once there's a stable source of corrections |
| **Embedding** (BAAI/bge-m3) | ✅✅ | Contrastive-learning fine-tuning (the official FlagEmbedding scripts; query + positive + hard-negative triples); the model is small and cheap to train (a few hours on one GPU) | A different training paradigm (not causal LM): a new `task_type: embedding_contrastive`, with the trainer going through the sentence-transformers/FlagEmbedding route (can initially skip 16 entirely — an independent training-profile script the Orchestrator calls directly) | **Retrieval clicks/acceptance signals from 17/18**: a document the user ultimately accepted = a positive; one ranked high but skipped = a hard negative — the most natural usage data of any fine-tuning target here | **P1~P2, the highest ROI**: makes the entire RAG retrieval layer smarter, benefiting every downstream module |
| **Reranker** (bge-reranker-v2-m3, ms-marco cross-encoder) | ✅✅ | Cross-encoder pairwise fine-tuning; the same data requirement as embedding (can share the same batch of triples) | Same as above, sharing the contrastive data pipeline | Same as above | **P1~P2**, bundled together with embedding |
| **LightGBM ranker** (18 query-orchestrator's pipeline-fusion ranking) | ✅✅ | Not a deep model — it's **learning-to-rank** (18 already has a ranker-adjustment plan); retraining cost is measured in minutes | Doesn't go through 16 at all — CPU training is enough; the Orchestrator triggers 18's training script directly; the model file can still be version-controlled in MLflow | 18's pipeline-selection results + whether the final answer was accepted (a naturally-occurring relevance label) | **P1, the cheapest "gets smarter with use" win**: first teach the decision layer to pick the right pipeline |
| **TTS** (VibeVoice, SpeechT5, etc.) | ⚠️ Possible but not recommended | Voice cloning/timbre adaptation is fundamentally driven by a **reference voice** (VibeVoice is already reference-audio based) — the ROI on fine-tuning is very low | — | — | **Not included in the loop** |
| **Diarization / audio events** (pyannote, PANNs) | ⚠️ Possible within each framework | Needs manually-labeled speaker/event labels, which don't naturally arise from usage | Not worth integrating into 16 | Almost none | **Not included** |
| **PaddleOCR** (PP-OCRv5) | ⚠️ Possible within the Paddle ecosystem | Needs the Paddle training stack, entirely incompatible with 16's PyTorch stack | Don't integrate | Corrections are rare | **Not included**; for OCR quality issues, prefer filling the gap with VLM (InternVL) instead |

### 9.2 Summary of extensions needed for 16

Under this plan, the extensions 16 needs boil down to two things, both **additive, not restructuring**:

1. **Multimodal instruction fine-tuning** (P2): a `vlm_instruction` task type + an image-conversation dataset loader + a PEFT mount point pointed at the language-model submodule. GPU scheduling, job management, and MLflow tracking are all reused as-is.
2. **Contrastive-learning training** (P1~P2): the training paradigm for embedding/reranker is quite different from causal LM — **recommend not touching 16 for this at first** — run it as an independent script (the FlagEmbedding recipe) + an Orchestrator training profile; the artifact still goes through the same unified flow-back ("upload to LakeFS → register as staging → evaluate (using 17's retrieval eval set to compute NDCG/MRR) → promote"). Consider folding it into 16 once the process is stable.

The evaluation gate (§4.5) needs different metrics for non-generative models: embedding/reranker use retrieval-eval-set NDCG@k / MRR; ASR uses WER; VLM continues to use LLM-as-judge + task metrics.

---

## 10. A layered strategy for "gets smarter with use": RAG × memory × fine-tuning

### 10.1 Core idea: fine-tuning is only the last rung of the "learning ladder"

"Gets smarter with use" doesn't mean "keep retraining the model." Knowledge and behavior operate on different learning timescales, and should be layered:

```
L0 Knowledge layer (effective in seconds)  RAG knowledge write-back — no weight changes, effective the instant it's written to the retrieval store
L1 Experience layer (hours)                Dynamic few-shot / experience memory — good examples are stored and retrieved-and-injected at inference time
L2 Retrieval layer (days)                  Retraining embedding / reranker / the LightGBM ranker — makes "finding the data" more accurate
L3 Weight layer (weeks)                    LLM / VLM LoRA fine-tuning — internalizes behavior and reasoning
```

**Division of labor**:

| What's being learned | Which layer | Why |
|---|---|---|
| New facts, new documents, knowledge that changes | **L0 RAG** | Fine-tuning is inefficient at injecting knowledge and it goes stale; RAG is effective in seconds, traceable, and revocable |
| A good example of "this kind of question should be answered like this" | **L1 few-shot** | No training needed, effect is immediate, and the example library itself is version-controlled |
| "The user wants this document, not that one" | **L2 retrieval fine-tuning** | Click/acceptance signals exist naturally, and the models are small and cheap to train |
| Format, tone, domain reasoning patterns, tool-use habits | **L3 LoRA** | These don't fit in a prompt and can't be retrieved — only worth touching the weights for this |

The same **human correction** should be written twice: immediately into L0 (the very next similar question gets answered correctly), and in batch into the L3 training set (the next fine-tuning round internalizes it). This is "instant smart" layered on top of "lasting smart."

### 10.2 Raptor already has the RAG foundation — L0 is a wiring problem, not a construction problem

This is the most important practical fact behind this design: **all three pillars of RAG already exist in Raptor** — the only thing missing is the "feedback → knowledge base" write-back pipeline:

| Existing module | Capability provided | Role in L0 |
|---|---|---|
| **17 hybrid-search** | OpenSearch(BM25) + Qdrant(vector) + RRF fusion + bge-m3/reranker, supports custom payload schemas | The retrieval engine for both knowledge and experience |
| **19/20 graph-database/service** | Knowledge-graph storage and querying | Where structured facts and relations (entities, relationships) land |
| **18 query-orchestrator** | Query routing, LightGBM pipeline-fusion ranking | The decision layer for "where should this question be looked up" |
| **13/15/21** | gateway / chat / agent | RAG's consumers, and also where feedback signals originate |

**L0 knowledge write-back pipeline** (a new `knowledge_ingest` pipeline type added to the 07 Orchestrator):

```
/feedback receives a correction or a high-scoring interaction
  → Curator determines the type:
      Factual (can stand alone as a knowledge item) → extract into a QA/narrative entry
          → write to 17 (a dedicated payload_schema: "learned_knowledge", with source and timestamp)
          → entity/relation extraction → write to 20's knowledge graph
      Behavioral (format/style/process preference) → goes only into the L1 example library and the L3 training set
  → Next query: 18 routes → 17's retrieval hits the new knowledge → the LLM cites it in its answer
```

**L0 governance (preventing memory pollution)** — learned knowledge must be managed separately from original document knowledge:

- A dedicated payload schema / collection, with fields including `source_interaction_id`, `learned_at`, `confidence`, `ttl`;
- At retrieval time, learned-knowledge scores and document-knowledge scores are shown separately, and the prompt marks "the following is knowledge learned from past interactions";
- Conflict handling: when a new entry's embedding similarity to an existing one is too high, update it instead of adding a duplicate;
- One-click revocation (delete by interaction_id) — this is RAG's key advantage over fine-tuning, and it must be preserved.

### 10.3 L1 experience memory: dynamic few-shot

Store curated high-scoring interactions (especially post-correction versions) as an **example library** (just put it in 17, as an `exemplar` schema):

- At inference time (this can happen at the 13 gateway or 15 chat layer, or as an optional pre-processing step in 07's gateway): look up the top-k most similar cases from the example library for the user's question → inject them into the system prompt as few-shot examples;
- Every entry in the example library carries a `task_tag` and effectiveness stats (the score of the interaction it was injected into) — low-scoring examples are automatically down-weighted and retired, so the example library itself "gets smarter with use";
- Cost: zero training, one extra retrieval call (~10ms range); this is usually the **fastest-acting** of every approach here.

### 10.4 How L2 / L3 relate to the loop

L2 and L3 are just extensions of the existing pipeline in §3~§4 — the Orchestrator's pipeline type grows from one to four:

| pipeline type | Trigger cadence | Training execution | Evaluation metric |
|---|---|---|---|
| `knowledge_ingest` (L0) | Real-time/per-item | No training | Spot-checked entry quality |
| `retrieval_finetune` (L2) | Periodic, once click-signal volume hits a threshold | The FlagEmbedding script (embedding/reranker); 18's LTR script (LightGBM) | Retrieval-eval-set NDCG@k / MRR; 18's pipeline-selection accuracy |
| `llm_finetune` (L3) | Feedback-volume threshold/manual | 16 (the existing LoRA flow) | The §4.5 evaluation gate |
| `vlm_finetune` (L3) | Manual, once corrections have accumulated | 16 + the vlm_instruction extension | LLM-judge + task metrics |

All four pipeline types share the same flow-back skeleton (version in LakeFS → MLflow staging → evaluate → promote); the only difference is "what the training step actually executes."

### 10.5 Revised priority recommendation

Combining the §7 roadmap with this section, **reordered by "speed of payoff × cost"**:

1. **P0**: the minimal loop (§7 Phase 0, unchanged) + **the L1 example library** (very low cost, can run in parallel with Phase 0)
2. **P1**: the evaluation gate + **L0 knowledge write-back** (the RAG foundation already exists — it's just wiring) + **retraining 18's LightGBM** (the cheapest learning loop)
3. **P2**: completing the data flywheel + **L2 embedding/reranker fine-tuning** (starts the retrieval layer evolving)
4. **P3**: automatic triggers + **VLM LoRA** (extending 16 with vlm_instruction) + ASR fine-tuning (depending on correction volume)

The logic behind this ordering: first make the system "immediately look smarter" (L0/L1, delivered in days), then make it "actually get smarter" (L2/L3, iterating over weeks) — and the data and scores accumulated by L0/L1 are exactly the training raw material for L2/L3, so the four layers feed each other rather than being parallel options.

---

## Appendix: relationship to existing documents

- This document is the implementation version of [design-strategy.md](design-strategy.md)'s (§10 Model Lifecycle vision): Model Repository → LakeFS+MLflow (already exists); Fine-Tuner → module 16 (already exists); this design fills in the gap between the two and the "usage-data flow-back" piece.
- See [all_raptor_models.md](all_raptor_models.md) for the model inventory and per-service migration guide.
- See [16-training-service/README.md](../../16-training-service/README.md) for module 16's training API details (submission schema, column_mapping, GPU scheduling).
