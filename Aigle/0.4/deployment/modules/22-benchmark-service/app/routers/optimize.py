"""Auto-tuning REST API (AUTOTUNE §4).

    POST /optimize                 → create an experiment (plan supplied for now)
    GET  /optimize                 → list experiments
    GET  /optimize/{id}            → status + best + history
    GET  /optimize/{id}/plan       → plan preview + eval考卷 preview (the guardrail)
    POST /optimize/{id}/confirm    → launch the optimization loop
    POST /optimize/{id}/stop       → cooperative stop after current iteration
    DELETE /optimize/{id}          → delete experiment + its derived/generated schemas

The caller may supply an explicit ``plan``, or omit it and let the Planner derive
one from the natural-language ``goal`` (grounded against Module 07's local model/
dataset catalog; the eval schema is auto-generated for preview/confirm).
"""
from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, Query, Response, status

from app.core.config import get_settings
from app.models.experiment import Budget, ExperimentStatus, OptimizeRequest
from app.services import run_manager, schema_store
from app.services.autotune import experiment_store, orchestrator
from app.services.autotune.planner import Planner, PlannerError

router = APIRouter(prefix="/optimize", tags=["Auto-Tune"])


async def _running_other_than(exp_id: Optional[str]) -> Optional[str]:
    """Return the id of a currently-running experiment (other than ``exp_id``), if any.

    Only one experiment may run at a time: they share a single GPU, and letting two
    optimize loops overlap makes their evals thrash the inference slot (each infer
    evicts the other's model → constant reload, minutes-long iterations).
    """
    for r in await experiment_store.list_running():
        if r["experiment_id"] != exp_id:
            return r["experiment_id"]
    return None


@router.post("", status_code=status.HTTP_201_CREATED,
             summary="Create an auto-tuning experiment from a goal + plan")
async def create_optimize(payload: OptimizeRequest,
                          auto_confirm: bool = Query(False, description="Skip the confirm gate and start immediately")
                          ) -> Dict[str, Any]:
    s = get_settings()
    plan = payload.plan
    generated_schema_id = None

    # Planner path: no plan supplied → derive one from the natural-language goal.
    # The planner grounds against Module 07's local catalog and auto-generates the
    # eval schema (tracked as generated_schema_id so it can be cleaned up later).
    if plan is None:
        try:
            plan = await Planner().plan(payload.goal)
        except PlannerError as exc:
            raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                                detail=f"planner could not build a plan: {exc}")
        generated_schema_id = plan.eval_schema_id

    # eval_schema_id is None when the dataset still needs downloading — the eval is
    # built from real rows once it's local (re-POST after download). Such a plan can
    # only sit in awaiting_confirm (needs_download blocks launch), never run.
    if plan.eval_schema_id and not await schema_store.schema_exists(plan.eval_schema_id):
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                            detail=f"eval_schema_id {plan.eval_schema_id} not found")

    # Anti-overfit: the loop optimizes on a dev set; the best config is re-scored
    # once on an untouched held-out set, so a big dev↔held-out gap flags overfitting.
    #
    # Real-data path: the planner already sampled dev and held-out as two independent,
    # disjoint sets of real rows — use its held-out directly (no ratio). Otherwise
    # (an explicit plan with a single eval) partition it by ratio.
    holdout_schema_id = plan.holdout_schema_id
    if not plan.eval_schema_id:
        holdout_schema_id = None
    elif holdout_schema_id:
        # dev (eval_schema_id) and held-out are both planner-derived; delete_optimize
        # cleans both via the holdout branch, so don't also track eval as "generated".
        generated_schema_id = None
    elif s.holdout_enabled:
        schema = await schema_store.get_schema(plan.eval_schema_id)
        n_cases = len((schema.get("definition") or {}).get("test_cases", [])) if schema else 0
        if n_cases >= s.holdout_min_cases:
            split = await schema_store.split_for_holdout(
                plan.eval_schema_id, s.holdout_ratio, s.holdout_seed)
            if split:
                dev_id, holdout_schema_id = split
                # If we split the planner's auto-eval, it is now redundant (dev +
                # held-out hold all its cases) → drop it so nothing leaks.
                if generated_schema_id == plan.eval_schema_id:
                    await schema_store.delete_schema(plan.eval_schema_id)
                    generated_schema_id = None
                plan.eval_schema_id = dev_id  # the loop optimizes on dev only

    # Budget precedence: an explicit budget on the request wins; otherwise use the
    # one the planner derived from the goal (e.g. "跑 5 次"); otherwise system defaults.
    budget = payload.budget or plan.budget or Budget()

    # Don't auto-launch if the plan references resources that aren't local yet —
    # training would otherwise try to pull them from the Hub (slow, may fill disk,
    # may 401 on gated repos). Leave it awaiting_confirm so the caller resolves it.
    can_launch = auto_confirm and not plan.needs_download and bool(plan.eval_schema_id)
    # One experiment at a time (shared GPU). If another is already running, create
    # the plan but leave it awaiting_confirm rather than auto-launching into a clash.
    busy = await _running_other_than(None) if can_launch else None
    if busy:
        can_launch = False
    initial = ExperimentStatus.running if can_launch else ExperimentStatus.awaiting_confirm
    exp_id = await experiment_store.create_experiment(
        payload.goal, budget, plan, initial, holdout_schema_id, generated_schema_id)

    if can_launch:
        orchestrator.launch(exp_id)
    return {"experiment_id": exp_id, "status": initial.value,
            "holdout_enabled": holdout_schema_id is not None,
            "needs_download": plan.needs_download,
            "blocked_by_running": busy}


@router.get("", summary="List experiments (newest first)")
async def list_optimize(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> List[Dict[str, Any]]:
    return await experiment_store.list_experiments(limit=limit, offset=offset)


@router.get("/{experiment_id}", summary="Experiment status, best result, and history")
async def get_optimize(experiment_id: str) -> Dict[str, Any]:
    exp = await experiment_store.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")

    history: List[Dict[str, Any]] = []
    if exp.get("eval_schema_id"):
        board = await run_manager.leaderboard(exp["eval_schema_id"], limit=20)
        history = []
        for b in board:
            cfg = b.get("config_override") or {}
            history.append({
                "run_id": b["run_id"],
                "aggregate_score": b["aggregate_score"],
                "config": {k: v for k, v in cfg.items() if k not in ("model_path", "_reason")},
                "reason": cfg.get("_reason"),  # why the proposer picked this config
                "created_at": b["created_at"],
            })

    return {
        "experiment_id": exp["experiment_id"],
        "goal": exp["goal"],
        "status": exp["status"],
        "budget": exp["budget"],
        "eval_schema_id": exp.get("eval_schema_id"),
        "iterations_done": exp["iterations_done"],
        "best": {
            "run_id": exp.get("best_run_id"),
            "aggregate_score": exp.get("best_score"),          # on the dev split
            "config": exp.get("best_config"),
        },
        "holdout": {
            "schema_id": exp.get("holdout_schema_id"),
            "score": exp.get("holdout_score"),                 # best config on unseen cases
        },
        # The candidate currently training (config + LLM reason), before it's scored.
        "current_candidate": (await orchestrator.get_current_candidate(experiment_id)
                              if exp["status"] == ExperimentStatus.running.value else None),
        "history": history,
        "error": exp.get("error"),
        "created_at": exp.get("created_at"),
        "completed_at": exp.get("completed_at"),
    }


@router.get("/{experiment_id}/plan",
            summary="Plan preview + eval考卷 preview (customer confirms before GPU burns)")
async def get_plan(experiment_id: str) -> Dict[str, Any]:
    exp = await experiment_store.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    plan = exp.get("plan")
    if plan is None:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Plan not ready")

    eval_preview: Dict[str, Any] = {}
    if exp.get("eval_schema_id"):
        schema = await schema_store.get_schema(exp["eval_schema_id"])
        if schema:
            definition = schema.get("definition") or {}
            eval_preview = {
                "schema_name": schema.get("name"),
                "test_cases": definition.get("test_cases", []),
                "scoring_schema": definition.get("scoring_schema"),
            }

    return {
        "experiment_id": exp["experiment_id"],
        "status": exp["status"],
        "base_training_config": plan.get("base_training_config"),
        "search_space": plan.get("search_space"),
        "dataset_config": plan.get("dataset_config"),
        "budget": exp.get("budget"),  # effective budget (goal-derived or explicit)
        "eval_schema_id": exp.get("eval_schema_id"),
        "eval_preview": eval_preview,
        # Resources the plan references that aren't local yet — the customer
        # should approve/trigger these downloads before confirming.
        "needs_download": plan.get("needs_download", []),
    }


@router.post("/{experiment_id}/confirm",
             summary="Confirm the plan and start the optimization loop")
async def confirm_optimize(experiment_id: str) -> Dict[str, Any]:
    exp = await experiment_store.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    if exp["status"] != ExperimentStatus.awaiting_confirm.value:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Experiment is '{exp['status']}', can only confirm 'awaiting_confirm'",
        )
    needs = (exp.get("plan") or {}).get("needs_download") or []
    if needs:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "Download these resources before confirming, then retry.",
                    "needs_download": needs},
        )
    busy = await _running_other_than(experiment_id)
    if busy:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail={"message": "Another experiment is already running (they share one GPU) — "
                               "stop it or wait for it to finish, then confirm.",
                    "running_experiment_id": busy},
        )
    await experiment_store.update_status(experiment_id, ExperimentStatus.running)
    orchestrator.launch(experiment_id)
    return {"experiment_id": experiment_id, "status": "running"}


@router.post("/{experiment_id}/stop",
             summary="Stop the experiment after the current iteration")
async def stop_optimize(experiment_id: str) -> Dict[str, Any]:
    stopped = await orchestrator.request_stop(experiment_id)
    if not stopped:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Experiment not found or not in a stoppable state",
        )
    return {"experiment_id": experiment_id, "status": "stopped"}


@router.delete("/{experiment_id}", status_code=status.HTTP_204_NO_CONTENT,
               summary="Delete an experiment and its auto-generated dev/held-out schemas")
async def delete_optimize(experiment_id: str) -> Response:
    exp = await experiment_store.get_experiment(experiment_id)
    if exp is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Experiment not found")
    if exp["status"] == ExperimentStatus.running.value:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT,
                            detail="Experiment is running — stop it first")

    # Clean up the derived schemas this experiment created (their runs cascade).
    # A held-out schema id means a split happened, so eval_schema_id is the
    # derived *dev* schema (safe to delete). Without a split, eval_schema_id is
    # the caller's original schema — leave it alone.
    holdout_id = exp.get("holdout_schema_id")
    if holdout_id:
        await schema_store.delete_schema(exp["eval_schema_id"])  # derived dev
        await schema_store.delete_schema(holdout_id)             # derived held-out
    generated_id = exp.get("generated_schema_id")
    if generated_id:  # planner auto-eval that was NOT split (kept as the eval)
        await schema_store.delete_schema(generated_id)

    await experiment_store.delete_experiment(experiment_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
