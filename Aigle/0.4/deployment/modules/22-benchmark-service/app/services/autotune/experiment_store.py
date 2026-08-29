"""PostgreSQL CRUD for auto-tuning experiments (AUTOTUNE §7).

One row per optimization goal in the ``experiments`` table. The per-iteration
(config → score) history is *not* stored here — it lives in ``benchmark_runs``
(config_override + aggregate_score) and is read back via run_manager.leaderboard.
That is what makes crash recovery cheap (AUTOTUNE §B7): all durable state is in
PostgreSQL, none in memory.
"""
from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from app.core.db import db
from app.models.experiment import Budget, ExperimentStatus, Plan


def _is_uuid(value: str) -> bool:
    try:
        uuid.UUID(str(value))
        return True
    except (ValueError, AttributeError, TypeError):
        return False


def _maybe_json(value: Any) -> Any:
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (ValueError, TypeError):
            return value
    return value


async def create_experiment(goal: str, budget: Budget, plan: Optional[Plan],
                            status: ExperimentStatus,
                            holdout_schema_id: Optional[str] = None,
                            generated_schema_id: Optional[str] = None) -> str:
    """Insert a new experiment; return its id."""
    eval_schema_id = plan.eval_schema_id if plan else None
    row = await db.pool.fetchrow(
        """
        INSERT INTO experiments
            (goal, status, plan, eval_schema_id, budget, holdout_schema_id, generated_schema_id)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        RETURNING id
        """,
        goal,
        status.value,
        json.dumps(plan.model_dump(mode="json")) if plan else None,
        eval_schema_id if (eval_schema_id and _is_uuid(eval_schema_id)) else None,
        json.dumps(budget.model_dump(mode="json")),
        holdout_schema_id if (holdout_schema_id and _is_uuid(holdout_schema_id)) else None,
        generated_schema_id if (generated_schema_id and _is_uuid(generated_schema_id)) else None,
    )
    return str(row["id"])


async def get_experiment(exp_id: str) -> Optional[Dict[str, Any]]:
    """Return the raw experiment row as a dict, or None."""
    if not _is_uuid(exp_id):
        return None
    row = await db.pool.fetchrow(
        """
        SELECT id, goal, status, plan, eval_schema_id, budget, iterations_done,
               best_run_id, best_score, best_config, holdout_schema_id, holdout_score,
               generated_schema_id, error, created_at, completed_at
        FROM experiments WHERE id = $1
        """,
        exp_id,
    )
    if row is None:
        return None
    return {
        "experiment_id": str(row["id"]),
        "goal": row["goal"],
        "status": row["status"],
        "plan": _maybe_json(row["plan"]),
        "eval_schema_id": str(row["eval_schema_id"]) if row["eval_schema_id"] else None,
        "budget": _maybe_json(row["budget"]) or {},
        "iterations_done": row["iterations_done"] or 0,
        "best_run_id": str(row["best_run_id"]) if row["best_run_id"] else None,
        "best_score": row["best_score"],
        "best_config": _maybe_json(row["best_config"]),
        "holdout_schema_id": str(row["holdout_schema_id"]) if row["holdout_schema_id"] else None,
        "holdout_score": row["holdout_score"],
        "generated_schema_id": str(row["generated_schema_id"]) if row["generated_schema_id"] else None,
        "error": row["error"],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
    }


async def update_status(exp_id: str, status: ExperimentStatus,
                        error: Optional[str] = None) -> None:
    """Set status; stamp completed_at for terminal states."""
    terminal = status in (ExperimentStatus.completed, ExperimentStatus.failed,
                          ExperimentStatus.stopped)
    completed_at = datetime.now(timezone.utc) if terminal else None
    await db.pool.execute(
        """
        UPDATE experiments
        SET status = $2,
            error = COALESCE($3, error),
            completed_at = CASE WHEN $4::timestamptz IS NOT NULL THEN $4 ELSE completed_at END
        WHERE id = $1
        """,
        exp_id, status.value, error, completed_at,
    )


async def set_iterations(exp_id: str, iterations_done: int) -> None:
    await db.pool.execute(
        "UPDATE experiments SET iterations_done = $2 WHERE id = $1",
        exp_id, iterations_done,
    )


async def set_holdout_score(exp_id: str, score: Optional[float]) -> None:
    await db.pool.execute(
        "UPDATE experiments SET holdout_score = $2 WHERE id = $1",
        exp_id, score,
    )


async def delete_experiment(exp_id: str) -> bool:
    """Delete an experiment row; return True if a row was removed."""
    if not _is_uuid(exp_id):
        return False
    result = await db.pool.execute("DELETE FROM experiments WHERE id = $1", exp_id)
    return result.rsplit(" ", 1)[-1] != "0"


async def update_best(exp_id: str, best_run_id: Optional[str], best_score: Optional[float],
                      best_config: Optional[Dict[str, Any]]) -> None:
    await db.pool.execute(
        """
        UPDATE experiments
        SET best_run_id = $2, best_score = $3, best_config = $4
        WHERE id = $1
        """,
        exp_id,
        best_run_id if (best_run_id and _is_uuid(best_run_id)) else None,
        best_score,
        json.dumps(best_config) if best_config is not None else None,
    )


async def list_experiments(limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
    rows = await db.pool.fetch(
        """
        SELECT id, goal, status, eval_schema_id, iterations_done,
               best_score, created_at
        FROM experiments
        ORDER BY created_at DESC
        LIMIT $1 OFFSET $2
        """,
        limit, offset,
    )
    return [
        {
            "experiment_id": str(r["id"]),
            "goal": r["goal"],
            "status": r["status"],
            "eval_schema_id": str(r["eval_schema_id"]) if r["eval_schema_id"] else None,
            "iterations_done": r["iterations_done"] or 0,
            "best_score": r["best_score"],
            "created_at": r["created_at"],
        }
        for r in rows
    ]


async def list_running() -> List[Dict[str, Any]]:
    """Experiments left in 'running' after a crash — to be resumed on startup."""
    rows = await db.pool.fetch(
        "SELECT id FROM experiments WHERE status = 'running' ORDER BY created_at ASC"
    )
    return [{"experiment_id": str(r["id"])} for r in rows]
