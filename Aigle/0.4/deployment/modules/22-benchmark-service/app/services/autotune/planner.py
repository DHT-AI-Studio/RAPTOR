"""Planner: natural language goal → a runnable optimization Plan (AUTOTUNE Phase A).

Turns a customer sentence ("fine-tune gemma-3-270m-it on dolly so it follows
instructions better") into a Plan the orchestrator can run: a base training
config, a search space, and an auto-generated eval schema.

Hybrid + guard-railed by design (not "let the LLM write everything"):
  * base_training_config comes from a SAFE TEMPLATE (the proven LoRA + 4-bit
    setup); the LLM only supplies the model / dataset / task.
  * search_space is LLM-proposed but passed through a code whitelist that clamps
    ranges and drops unknown/dangerous knobs (``sanitize_search_space``).
  * the eval schema (test cases + scoring) is LLM-generated — the weakest link —
    so it is ALWAYS returned for customer preview/confirm before any GPU burns,
    and the held-out split guards against overfitting it.

The LLM call is injected (``complete``) exactly like the proposer, so a
smolagents-backed planner can drop in later without touching this logic.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple

import httpx
import json_repair
from pydantic import BaseModel, Field, ValidationError

from app.core.config import get_settings
from app.models.experiment import Budget, Plan, SearchDimension
from app.models.schema import BenchmarkSchema, TargetPipeline
from app.services import judge, schema_store
from app.services.judge import ContentPolicyBlockedError

logger = logging.getLogger(__name__)

CompleteFn = Callable[[str], Awaitable[str]]


async def _fetch_catalog() -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Best-effort (local_models, local_datasets) from Module 07.

    Grounds the planner in what actually exists locally so it stops guessing /
    hallucinating HF ids. Returns empty lists if Module 07 is unreachable.
    """
    s = get_settings()
    base = s.inference_url.rstrip("/")
    models: List[Dict[str, Any]] = []
    datasets: List[Dict[str, Any]] = []
    try:
        async with httpx.AsyncClient(timeout=s.inference_timeout) as client:
            rm = await client.get(base + "/models/local")
            if rm.status_code == 200:
                data = rm.json()
                models = data if isinstance(data, list) else data.get("models", [])
            rd = await client.get(base + "/datasets/local")
            if rd.status_code == 200:
                datasets = rd.json().get("local_datasets", [])
    except Exception as exc:  # noqa: BLE001 — grounding is best-effort
        logger.warning("planner: could not fetch local catalog from Module 07: %s", exc)
    return models, datasets


def _resolve(value: str, local_items: List[Dict[str, Any]], kind: str,
             download_endpoint: str) -> Tuple[str, Optional[Dict[str, Any]]]:
    """Match an LLM-named model/dataset against the local catalog.

    Returns (resolved_ref, needs_download_entry_or_None). If found locally, the
    resolved ref is the local *path* (what Module 16 wants); otherwise the value
    is kept and a needs_download entry is returned.
    """
    norm = value.replace("/", "_")
    for it in local_items:
        name = it.get("name", "")
        path = it.get("path", "")
        if value in (name, path) or norm == name or (name and (norm.endswith(name) or name.endswith(norm))):
            return path or value, None
    return value, {"kind": kind, "id": value, "download_endpoint": download_endpoint}


def _hub_id(value: str) -> str:
    """Best-effort HuggingFace id: keep a slash id as-is, else turn a sanitized
    directory name (``org_name``) back into ``org/name`` (first underscore)."""
    return value if "/" in value else value.replace("_", "/", 1)


def _resolve_dataset(value: str, datasets: List[Dict[str, Any]],
                     download_endpoint: str) -> Tuple[str, Optional[str], Optional[Dict[str, Any]]]:
    """Resolve a dataset to (hub_id, cache_dir, needs_download).

    Unlike a model, Module 16 loads a dataset by HF *id* (with a local cache dir),
    not by directory path — so a locally-cached dataset keeps its Hub id and gets
    its cache_dir pointed at the local copy.
    """
    hub = _hub_id(value)
    norm = hub.replace("/", "_")
    for it in datasets:
        name = it.get("name", "")
        path = it.get("path", "")
        if value in (name, path) or norm == name:
            return hub, path or None, None            # local → id + cache_dir
    return hub, None, {"kind": "dataset", "id": hub, "download_endpoint": download_endpoint}


# ── Real-data eval: sample rows straight from the local dataset ──────
# input = the user turn, expected_answer = the gold answer. No LLM invention.
_COSINE_SCORING = {
    "dimensions": [{"name": "similarity", "weight": 1.0, "method": "cosine_similarity"}],
    "aggregate": "weighted_sum",
}
# HF `Dataset.to_json` writes JSON Lines; prefer a held-out split over train.
_EVAL_SPLIT_PREFERENCE = ("test.json", "validation.json", "val.json", "train.json")


def _first_str(*vals: Any) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
    return None


def _row_to_case(row: Dict[str, Any], case_id: str) -> Optional[Dict[str, Any]]:
    """Adapt one dataset row to a (input, expected_answer) test case, or None.

    Handles the common instruction/chat layouts: OpenAI-style ``messages``,
    ``instruction``/``output``, ``prompt``/``completion``, ``question``/``answer``.
    """
    if not isinstance(row, dict):
        return None
    inp = exp = None
    msgs = row.get("messages")
    if isinstance(msgs, list) and msgs:
        users = [m.get("content") for m in msgs
                 if isinstance(m, dict) and m.get("role") == "user"]
        bots = [m.get("content") for m in msgs
                if isinstance(m, dict) and m.get("role") == "assistant"]
        inp = _first_str(*reversed(users))
        exp = _first_str(*reversed(bots))
    else:
        instr = _first_str(row.get("instruction"))
        ctx = _first_str(row.get("input"), row.get("context"))
        inp = _first_str(instr and (f"{instr}\n{ctx}" if ctx else instr),
                         row.get("prompt"), row.get("question"), row.get("query"))
        exp = _first_str(row.get("output"), row.get("response"),
                         row.get("completion"), row.get("answer"))
    if not (inp and exp):
        return None
    return {"id": case_id, "input": {"inputs": inp}, "expected_answer": exp}


def _read_dataset_rows(dir_path: str, n: int, seed: int) -> List[Dict[str, Any]]:
    """Randomly sample up to n rows from a local dataset dir (best-effort).

    Reads the JSONL split file (preferring a held-out split over ``train``) and
    samples with a dedicated eval ``seed`` — distinct from training's seed so the
    tiny training slice is unlikely to overlap the eval rows. Runs in a thread
    (called via asyncio.to_thread) because it touches the NFS mount.
    """
    try:
        present = set(os.listdir(dir_path))
    except OSError:
        return []
    fname = next((f for f in _EVAL_SPLIT_PREFERENCE if f in present), None)
    if fname is None:
        return []
    path = os.path.join(dir_path, fname)
    rows: List[Dict[str, Any]] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line or line in ("[", "]"):
                    continue
                if line.endswith(","):
                    line = line[:-1]
                try:
                    obj = json.loads(line)
                except ValueError:
                    continue
                if isinstance(obj, dict):
                    rows.append(obj)
    except OSError:
        return []
    if not rows:
        return []
    rng = random.Random(seed)
    if len(rows) > n:
        return rng.sample(rows, n)
    rng.shuffle(rows)
    return rows


class PlannerError(Exception):
    """Raised when the planner cannot produce a valid plan."""


# ── Safety whitelist: only these knobs are tunable, within these bounds ──
# The LLM may propose a search space, but each dimension is clamped to this
# table and anything not listed is dropped. This is the code-side guard rail.
_KNOB_BOUNDS: Dict[str, Dict[str, Any]] = {
    "learning_rate":  {"type": "float", "min": 1e-6, "max": 2e-3},
    "lora_r":         {"type": "categorical", "choices": [4, 8, 16, 32, 64, 128]},
    "lora_alpha":     {"type": "categorical", "choices": [8, 16, 32, 64, 128]},
    "lora_dropout":   {"type": "float", "min": 0.0, "max": 0.3},
    "max_epochs":     {"type": "int", "min": 1, "max": 8},
    "warmup_ratio":   {"type": "float", "min": 0.0, "max": 0.2},
    "weight_decay":   {"type": "float", "min": 0.0, "max": 0.3},
    # Which projections LoRA adapts. Named presets (not raw module lists) keep the
    # value hashable/scalar so it flows through clamp/sample/dedup unchanged;
    # training_client expands the preset to the actual module list.
    "target_modules": {"type": "categorical", "choices": ["attn", "attn_mlp"]},
    # LR schedule shape (Module 16 maps these to transformers get_scheduler). A
    # plain top-level TrainerConfig field — no alias needed.
    "lr_scheduler_type": {"type": "categorical", "choices": ["linear", "cosine", "constant"]},
    # Optimizer family (top-level TrainerConfig field). adamw = adaptive (default),
    # lion = sign-based/memory-light (prefers a smaller LR), sgd = momentum.
    "optimizer_type": {"type": "categorical", "choices": ["adamw", "lion", "sgd"]},
    # LoRA variant — expanded by training_client into lora_config.use_rslora/use_dora.
    # plain = vanilla LoRA; rslora = rank-stabilized (steadier at high r); dora =
    # weight-decomposed (more capacity, slower).
    "lora_variant": {"type": "categorical", "choices": ["plain", "rslora", "dora"]},
    # Effective batch = batch_size × this (top-level field). Larger = steadier
    # gradients but fewer optimizer steps.
    "gradient_accumulation_steps": {"type": "categorical", "choices": [1, 2, 4, 8]},
}


class PlannerOutput(BaseModel):
    """Structured output extracted from the goal by the LLM."""

    model_config = {"protected_namespaces": ()}  # allow model_name_or_path

    model_name_or_path: str
    dataset_name_or_path: str
    task_type: str = "instruction"
    search_space: Dict[str, Dict[str, Any]] = Field(default_factory=dict)
    budget: Optional[Dict[str, Any]] = None
    eval: Optional[Dict[str, Any]] = None      # {dev_cases, holdout_cases} if the goal says so
    training: Optional[Dict[str, Any]] = None  # {train_size, val_size} if the goal says so
    notes: Optional[str] = None


def _base_config_template(model_path: str) -> Dict[str, Any]:
    """The safe, proven LoRA + 4-bit training scaffold. search_space tunes it."""
    return {
        "model_name_or_path": model_path,
        "use_bfloat16": True,
        "use_flash_attn": False,
        "max_epochs": 1,
        "batch_size": 1,
        "gradient_accumulation_steps": 2,
        "warmup_steps": 2,
        "logging_steps": 5,
        "val_check_interval": 1.0,
        "gradient_checkpointing": True,
        "max_grad_norm": 0.3,
        "use_8bit_adamw": True,
        "lora_config": {"r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                        "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
                        "bias": "none", "modules_to_save": None},
        "quantization_config": {"load_in_4bit": True, "bnb_4bit_quant_type": "nf4",
                                "bnb_4bit_use_double_quant": True,
                                "bnb_4bit_compute_dtype": "bfloat16"},
    }


def sanitize_search_space(proposed: Dict[str, Any]) -> Dict[str, SearchDimension]:
    """Clamp an LLM-proposed search space to the safe knob whitelist.

    Unknown knobs are dropped; numeric ranges are clamped into bounds;
    categorical choices are intersected with the allowed set.
    """
    clean: Dict[str, SearchDimension] = {}
    for name, spec in (proposed or {}).items():
        bound = _KNOB_BOUNDS.get(name)
        if bound is None or not isinstance(spec, dict):
            continue  # not a whitelisted knob → drop
        kind = bound["type"]
        try:
            if kind == "categorical":
                allowed = set(bound["choices"])
                choices = [c for c in spec.get("choices", []) if c in allowed]
                if not choices:
                    continue
                clean[name] = SearchDimension(type="categorical", choices=choices)
            else:
                lo, hi = float(bound["min"]), float(bound["max"])
                pmin = max(lo, float(spec.get("min", lo)))
                pmax = min(hi, float(spec.get("max", hi)))
                if pmin > pmax:
                    pmin, pmax = lo, hi
                log = bool(spec.get("log", kind == "float" and name == "learning_rate"))
                if log and pmin <= 0:
                    pmin = lo
                clean[name] = SearchDimension(type=kind, min=pmin, max=pmax, log=log)
        except (ValidationError, ValueError, TypeError):
            continue
    return clean


def _clamp_int(proposed: Dict[str, Any], key: str, lo: int, hi: int) -> Optional[int]:
    """proposed[key] clamped into [lo, hi] as int, or None if absent/unparseable."""
    v = proposed.get(key)
    if v is None:
        return None
    try:
        return max(lo, min(hi, int(v)))
    except (ValueError, TypeError):
        return None


def _clamp_float(proposed: Dict[str, Any], key: str, lo: float, hi: float) -> Optional[float]:
    """proposed[key] clamped into [lo, hi] as float, or None if absent/unparseable."""
    v = proposed.get(key)
    if v is None:
        return None
    try:
        return max(lo, min(hi, float(v)))
    except (ValueError, TypeError):
        return None


def sanitize_budget(proposed: Any) -> Optional[Budget]:
    """Clamp an LLM-proposed budget to the safe Budget bounds, or None.

    Only fields the goal actually mentioned are set (the rest keep Budget's
    defaults). Anything missing/invalid is dropped; None means the planner found
    no budget hint in the goal, so the request/default budget applies instead.
    """
    if not isinstance(proposed, dict):
        return None
    clean: Dict[str, Any] = {}
    me = _clamp_int(proposed, "max_experiments", 1, 200)
    if me is not None:
        clean["max_experiments"] = me
    mins = _clamp_float(proposed, "minutes_per_experiment", 0.0, 720.0)
    if mins is not None and mins > 0:  # a per-experiment cap must be positive
        clean["minutes_per_experiment"] = mins
    esp = _clamp_int(proposed, "early_stop_patience", 1, 100)
    if esp is not None:
        clean["early_stop_patience"] = esp
    if not clean:
        return None
    try:
        return Budget(**clean)
    except ValidationError:
        return None


def sanitize_eval(proposed: Any, default_dev: int, default_holdout: int) -> Tuple[int, int]:
    """Resolve (dev_cases, held-out_cases) from an LLM proposal, clamped.

    Only fields the goal mentioned override the config defaults; anything
    missing/invalid keeps its default. The real-data path samples these as two
    independent, disjoint sets (no ratio).
    """
    dev, holdout = default_dev, default_holdout
    if isinstance(proposed, dict):
        dev = _clamp_int(proposed, "dev_cases", 4, 200) or dev
        holdout = _clamp_int(proposed, "holdout_cases", 1, 100) or holdout
    return dev, holdout


def sanitize_training(proposed: Any, default_train: int, default_val: int) -> Tuple[int, int]:
    """Resolve (train_size, val_size) from an LLM proposal, clamped to safe bounds."""
    train, val = default_train, default_val
    if isinstance(proposed, dict):
        train = _clamp_int(proposed, "train_size", 4, 5000) or train
        val = _clamp_int(proposed, "val_size", 1, 1000) or val
    return train, val


def _build_prompt(goal: str, models: List[Dict[str, Any]], datasets: List[Dict[str, Any]]) -> str:
    model_names = [m.get("name") for m in models if m.get("name")]
    dataset_names = [d.get("name") for d in datasets if d.get("name")]
    grounding = (
        f"Locally available models (prefer these): {model_names or 'none'}\n"
        f"Locally available datasets (prefer these): {dataset_names or 'none'}\n"
        "Prefer an available one when it fits the goal. ALWAYS write model_name_or_path "
        "and dataset_name_or_path as canonical HuggingFace ids WITH A SLASH "
        "(e.g. google/gemma-3-270m-it, philschmid/dolly-15k-oai-style) — never the "
        "underscore directory name. If the goal needs something not listed, name its "
        "HF id anyway; it will be flagged for download.\n\n"
    )
    return (
        "You set up a LoRA fine-tuning experiment from a user's goal. "
        "Return ONLY a JSON object, no prose:\n"
        "{\n"
        '  "model_name_or_path": "<HF model id or /app/tmp/models/... path>",\n'
        '  "dataset_name_or_path": "<HF dataset id>",\n'
        '  "task_type": "instruction",\n'
        '  "search_space": {"learning_rate": {"type":"float","min":1e-5,"max":1e-3,"log":true},\n'
        '                    "lora_r": {"type":"categorical","choices":[8,16,32,64]},\n'
        '                    "lora_alpha": {"type":"categorical","choices":[16,32,64]},\n'
        '                    "warmup_ratio": {"type":"float","min":0.0,"max":0.1},\n'
        '                    "target_modules": {"type":"categorical","choices":["attn","attn_mlp"]},\n'
        '                    "lr_scheduler_type": {"type":"categorical","choices":["linear","cosine","constant"]},\n'
        '                    "optimizer_type": {"type":"categorical","choices":["adamw","lion","sgd"]},\n'
        '                    "lora_variant": {"type":"categorical","choices":["plain","rslora","dora"]},\n'
        '                    "gradient_accumulation_steps": {"type":"categorical","choices":[1,2,4,8]}},\n'
        '  "budget": {"max_experiments":5,"minutes_per_experiment":10},\n'
        '  "eval": {"dev_cases":30,"holdout_cases":15},\n'
        '  "training": {"train_size":30,"val_size":6}\n'
        "}\n"
        "Rules: only tune learning_rate, lora_r, lora_alpha, lora_dropout, max_epochs, "
        "warmup_ratio, weight_decay, target_modules, lr_scheduler_type, optimizer_type, "
        "lora_variant, gradient_accumulation_steps. Propose 3-9 of them so the agent has a "
        'richer action space to search. target_modules is categorical ["attn", "attn_mlp"] '
        "(attn = q/k/v/o only; attn_mlp also adapts the MLP projections — more capacity, "
        'slower). lr_scheduler_type is ["linear", "cosine", "constant"] (cosine often edges '
        'out linear). optimizer_type is ["adamw", "lion", "sgd"] — if you pick lion, prefer '
        "a smaller learning_rate (Lion likes ~3-10x lower LR than AdamW). lora_variant is "
        '["plain", "rslora", "dora"] (rslora steadier at high lora_r; dora = more capacity, '
        "slower). "
        "The eval is built from real dataset rows (you do NOT write test cases). "
        "Set \"budget\" ONLY if the goal states how many experiments/trials to run "
        "(max_experiments) or a per-experiment time limit in minutes "
        "(minutes_per_experiment); otherwise OMIT it. Set \"eval\" ONLY if the goal "
        "states how many questions to evaluate/validate with — dev_cases (tuning) and "
        "holdout_cases (final validation); convert a percentage to a count if given; "
        "otherwise OMIT it. Set \"training\" ONLY if the goal states how many rows to "
        "fine-tune on (train_size) or validate on (val_size); otherwise OMIT it.\n\n"
        f"{grounding}"
        f"User goal: {goal}"
    )


class Planner:
    def __init__(self, model: Optional[str] = None, complete: Optional[CompleteFn] = None) -> None:
        s = get_settings()
        self._model = model or s.proposer_model or s.judge_model
        self._complete: CompleteFn = complete or self._default_complete

    async def _default_complete(self, prompt: str) -> str:
        s = get_settings()
        return await judge.complete(prompt, model=self._model,
                                    max_length=s.planner_max_tokens,
                                    timeout=s.planner_timeout)

    async def plan(self, goal: str) -> Plan:
        """Produce a runnable Plan from a natural-language goal.

        Creates the auto-generated eval schema as a side effect (returns its id
        inside the Plan). Raises PlannerError on unusable LLM output.
        """
        models, datasets = await _fetch_catalog()  # ground against what's actually local
        try:
            raw = await self._complete(_build_prompt(goal, models, datasets))
        except ContentPolicyBlockedError as exc:
            # Without this, the raw httpx exception propagates past this
            # method entirely uncaught -- app/routers/optimize.py only
            # catches PlannerError, so it fell through to FastAPI's generic
            # 500 instead of the intended clean 422 "planner could not
            # build a plan" response.
            raise PlannerError(f"planner prompt blocked by guardrail policy: {exc}") from exc
        data = json_repair.loads(raw)
        if not isinstance(data, dict):
            raise PlannerError("planner LLM did not return a JSON object")
        try:
            out = PlannerOutput.model_validate(data)
        except ValidationError as exc:
            raise PlannerError(f"planner output invalid: {exc}") from exc

        search_space = sanitize_search_space(out.search_space)
        if not search_space:
            raise PlannerError("planner produced no valid tunable knobs")

        # Resolve model/dataset against the local catalog: use the local path if
        # present, else flag it for download (never silently trust an LLM-guessed id).
        s = get_settings()
        model_ref, model_dl = _resolve(out.model_name_or_path, models, "model",
                                       f"POST {s.inference_url}/models/download")
        dataset_id, dataset_cache, dataset_dl = _resolve_dataset(
            out.dataset_name_or_path, datasets, f"POST {s.inference_url}/datasets/download_from_network")
        needs_download = [x for x in (model_dl, dataset_dl) if x]

        # Prefer real dataset rows as the eval (input=user turn, expected_answer=
        # gold); fall back to the LLM's invented cases only when the dataset isn't
        # local yet (nothing to sample from — launch is blocked by needs_download).
        # Real-data path returns an independently-sampled held-out set too.
        eval_schema_id, holdout_schema_id = await self._build_eval(goal, out, dataset_cache)

        train_size, val_size = sanitize_training(out.training, s.planner_train_size, s.planner_val_size)
        base = _base_config_template(model_ref)
        base["seed"] = s.planner_seed  # pin the training seed so candidates are comparable (adjust here)
        dataset_config = {
            "dataset_name_or_path": dataset_id,   # HF Hub id (Module 16 loads by id)
            "default_system_prompt": None,
            "train_size": train_size, "val_size": val_size, "max_length": 512,
            "column_mapping": {"messages": "messages"},
        }
        if dataset_cache:  # dataset is local → load from the cached copy, no re-download
            dataset_config["cache_dir"] = dataset_cache
        return Plan(
            task_type=out.task_type or "instruction",
            select_multiple_gpus=False,
            vram_budget_gb=12,
            base_training_config=base,
            dataset_config=dataset_config,
            search_space=search_space,
            eval_schema_id=eval_schema_id,
            holdout_schema_id=holdout_schema_id,   # pre-sampled (real-data path); else None
            budget=sanitize_budget(out.budget),    # only set if the goal mentioned it
            needs_download=needs_download,
        )

    async def _build_eval(self, goal: str, out: PlannerOutput,
                          dataset_cache: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
        """Build (dev_schema_id, held-out_schema_id) from real dataset rows.

        The eval is grounded in real data — never LLM-invented. If the dataset
        isn't local yet, we DON'T fabricate an eval: return (None, None) so the
        plan surfaces needs_download (like a missing model) and the eval is built
        once the dataset is downloaded and the goal is re-planned.
        """
        if not dataset_cache:  # dataset needs downloading → defer, don't fabricate
            logger.info("planner: dataset not local — eval deferred until it is downloaded")
            return None, None

        s = get_settings()
        dev_n, hold_n = sanitize_eval(out.eval, s.planner_dev_cases, s.planner_holdout_cases)
        rows = await asyncio.to_thread(
            _read_dataset_rows, dataset_cache, dev_n + hold_n, s.planner_eval_seed)
        cases = []
        for i, row in enumerate(rows):
            case = _row_to_case(row, f"ds_{i}")
            if case:
                cases.append(case)
        if not cases:
            # Dataset is local but no rows fit a known layout — a real problem, not
            # something to paper over with invented cases.
            raise PlannerError("dataset is local but no rows could be adapted into eval cases "
                               "(unsupported column layout)")
        # Split the disjoint pool: hold out the last `hold_n` (leaving ≥1 dev), the
        # rest is dev. A partition of a random sample = two independent draws.
        n = len(cases)
        h = min(hold_n, max(0, n - 1)) if s.holdout_enabled else 0
        dev_cases, hold_cases = cases[: n - h], cases[n - h:]
        logger.info("planner: built eval from real rows — %d dev, %d held-out",
                    len(dev_cases), len(hold_cases))
        dev_id = await self._persist_eval(goal, dev_cases, _COSINE_SCORING, "dev")
        hold_id = (await self._persist_eval(goal, hold_cases, _COSINE_SCORING, "holdout")
                   if hold_cases else None)
        return dev_id, hold_id

    async def _persist_eval(self, goal: str, cases: List[Dict[str, Any]],
                            scoring_schema: Dict[str, Any], label: str = "") -> str:
        """Validate + persist a local_infer eval schema; return its id."""
        definition = {
            "name": f"AutoPlan Eval — {goal[:56]}" + (f" [{label}]" if label else ""),
            "version": "1.0",
            "target_pipeline": TargetPipeline.local_infer.value,
            "test_cases": cases,
            "scoring_schema": scoring_schema,
        }
        try:
            schema = BenchmarkSchema.model_validate(definition)
        except ValidationError as exc:
            raise PlannerError(f"planner eval schema invalid: {exc}") from exc
        created = await schema_store.create_schema(schema)
        return created["id"]
