#!/usr/bin/env python3
"""Interactive AutoTune demo client.

Type a natural-language goal → see the planner's grounded plan → approve → watch
the optimization loop train + score each candidate, showing the LLM's tuning
strategy for every round, then the best config and its held-out score.

Talks to the Benchmark Service REST API (no extra deps — stdlib only).

    python demo.py                        # uses http://localhost:8022/api/v1
    python demo.py --url http://host:8022/api/v1
    python demo.py --max-experiments 2    # force a short run (else the goal decides)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import urllib.error
import urllib.request

DEFAULT_URL = os.environ.get("BM_DEMO_URL", "http://localhost:8022/api/v1")
# Module 16 training service (host-mapped port) — for live epoch/step progress.
DEFAULT_TRAINING_URL = os.environ.get("BM_TRAINING_URL", "http://localhost:8009/api/v1")

# Pre-filled at the prompt so you can just edit it live (or press Enter to accept).
DEFAULT_GOAL = "用 philschmid/dolly-15k-oai-style 微調 gemma-3-270m-it 讓它更會遵循指令,只跑 2 次實驗"

# ── tiny ANSI helpers (no-op if not a TTY) ───────────────────────────
_TTY = sys.stdout.isatty()


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _TTY else s


def bold(s): return _c("1", s)
def dim(s): return _c("2", s)
def green(s): return _c("32", s)
def yellow(s): return _c("33", s)
def cyan(s): return _c("36", s)
def red(s): return _c("31", s)


def input_with_prefill(prompt: str, default: str) -> str:
    """Prompt with ``default`` pre-typed on the line so it can be edited live.

    Uses readline when available (edit inline, press Enter to accept); otherwise
    falls back to showing the default and accepting Enter to keep it.
    """
    try:
        import readline

        def _hook():
            readline.insert_text(default)
            readline.redisplay()

        readline.set_pre_input_hook(_hook)
        try:
            return input(prompt)
        finally:
            readline.set_pre_input_hook()
    except Exception:
        raw = input(f"{prompt}\n  {dim('[Enter = default]')} {dim(default)}\n> ")
        return raw.strip() or default


def rule(title: str = "") -> None:
    line = "─" * 60
    if title:
        print(bold(cyan(f"\n── {title} " + "─" * max(0, 56 - len(title)))))
    else:
        print(dim(line))


# ── API ──────────────────────────────────────────────────────────────
class Api:
    def __init__(self, base: str):
        self.base = base.rstrip("/")

    def _req(self, method: str, path: str, body=None):
        url = self.base + path
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(url, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                raw = r.read()
                return json.loads(raw) if raw else {}
        except urllib.error.HTTPError as e:
            detail = e.read().decode(errors="replace")
            try:
                detail = json.loads(detail).get("detail", detail)
            except Exception:
                pass
            raise ApiError(e.code, detail)

    def post(self, path, body=None): return self._req("POST", path, body)
    def get(self, path): return self._req("GET", path)
    def delete(self, path): return self._req("DELETE", path)


class ApiError(Exception):
    def __init__(self, code, detail):
        super().__init__(f"HTTP {code}: {detail}")
        self.code, self.detail = code, detail


# ── formatting ────────────────────────────────────────────────────────
def show_plan(api: Api, exp_id: str) -> dict:
    plan = api.get(f"/optimize/{exp_id}/plan")
    base = plan.get("base_training_config") or {}
    ds = plan.get("dataset_config") or {}
    ev = plan.get("eval_preview") or {}
    budget = plan.get("budget") or {}
    needs = plan.get("needs_download") or []

    rule("Plan")
    model = base.get("model_name_or_path", "?")
    print(f"  {bold('Model')}    {model}  {green('(local)') if str(model).startswith('/') else yellow('(needs download)')}")
    dsid = ds.get("dataset_name_or_path", "?")
    cache = ds.get("cache_dir")
    print(f"  {bold('Dataset')}  {dsid}  {green('(local, cached)') if cache else yellow('(needs download)')}")
    if ds.get("train_size"):
        print(f"  {bold('Training')} {green(str(ds.get('train_size')))} rows "
              f"(+{ds.get('val_size', 0)} val)  {dim('shuffled from the dataset, disjoint from the eval')}")
    if budget:
        print(f"  {bold('Budget')}   {budget.get('max_experiments')} experiments "
              f"× ≤{budget.get('minutes_per_experiment')} min each  "
              f"(stop after {budget.get('early_stop_patience')} non-improving)")

    ss = plan.get("search_space") or {}
    print(f"  {bold('Search space')} (the agent's action space):")
    for name, d in ss.items():
        if d.get("type") == "categorical":
            print(f"      {name:<14} choices {d.get('choices')}")
        else:
            log = " log" if d.get("log") else ""
            print(f"      {name:<14} {d.get('type')} {d.get('min')} … {d.get('max')}{log}")

    cases = ev.get("test_cases") or []
    method = ((ev.get("scoring_schema") or {}).get("dimensions") or [{}])[0].get("method", "?")
    hold_n = _holdout_count(api, exp_id)
    print(f"  {bold('Eval')} (real dataset rows, not LLM-invented):")
    print(f"      dev: {green(str(len(cases)))} questions   "
          f"held-out: {green(str(hold_n) if hold_n is not None else '—')} questions   "
          f"scoring: {cyan(method)}")
    if cases:
        q = (cases[0].get("input") or {}).get("inputs", "")
        a = cases[0].get("expected_answer") or ""
        print(dim(f"      e.g. Q: {_trim(q, 70)}"))
        print(dim(f"           gold: {_trim(a, 70)}"))

    if needs:
        print(f"  {bold(red('Needs download'))}:")
        for n in needs:
            print(f"      {n.get('kind')}: {n.get('id')}   → {dim(n.get('download_endpoint',''))}")
    else:
        print(f"  {bold('Needs download')}  {green('none — ready to run')}")
    return plan


def _holdout_count(api: Api, exp_id: str):
    try:
        hid = (api.get(f"/optimize/{exp_id}").get("holdout") or {}).get("schema_id")
        if not hid:
            return None
        sch = api.get(f"/benchmark/schemas/{hid}")
        return len(((sch.get("definition") or sch).get("test_cases")) or [])
    except Exception:
        return None


def _trim(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:n] + "…"


def _cfg_str(cfg: dict) -> str:
    order = ["lora_r", "lora_alpha", "learning_rate", "lora_dropout", "max_epochs",
             "warmup_ratio", "weight_decay"]
    parts = []
    for k in order:
        if k in cfg:
            parts.append(f"{k}={cfg[k]}")
    for k, v in cfg.items():
        if k not in order:
            parts.append(f"{k}={v}")
    return ", ".join(parts)


# ── live training progress (Module 16) ───────────────────────────────
def training_progress(train_api: Api, job_id=None):
    """Progress of THIS experiment's training job (by ``job_id``), or None.

    Querying the specific job id avoids showing an unrelated job's progress. Falls
    back to "the running job" only when no id is known (older server without it).
    """
    try:
        if job_id:
            job = train_api.get(f"/training/status/{job_id}")
            if not job or job.get("status") != "running":
                return None
        else:
            jobs = train_api.get("/training/list?status=running")
            if not isinstance(jobs, list) or not jobs:
                return None
            job = jobs[-1]  # best-effort fallback
    except Exception:
        return None
    m = job.get("metrics") or {}
    cfg = ((job.get("config") or {}).get("trainer_config") or {})
    pct = m.get("progress_percentage")
    step, total = m.get("current_step"), m.get("total_steps")
    if pct is None and step is not None and total:
        pct = 100.0 * step / total
    return {
        "pct": pct,
        "epoch": m.get("current_epoch"),
        "max_epochs": cfg.get("max_epochs"),
        "step": step,
        "total_steps": total,
        "loss": m.get("step_loss") or m.get("train_loss") or m.get("loss"),
        "eta": m.get("estimated_time_remaining_seconds"),
    }


def _bar(pct, width=18):
    pct = max(0.0, min(100.0, float(pct)))
    fill = int(round(pct / 100 * width))
    return "█" * fill + "░" * (width - fill)


def _progress_line(cand: int, prog, elapsed: int, trained: bool = False) -> str:
    """One-line training status for the current candidate."""
    if not prog or prog.get("pct") is None:
        phase = (green("✓ trained") + dim(" — scoring on dev questions…")) if trained \
            else dim("preparing training…")
        return f"  candidate {cand}  {phase}   {dim(f'{elapsed}s')}"
    p = prog
    pct_txt = f"{p['pct']:.0f}%"
    seg = [f"{green(_bar(p['pct']))} {bold(pct_txt)}"]
    if p.get("epoch") is not None:
        ep = p["epoch"]
        ep_s = f"{ep:.0f}" if isinstance(ep, (int, float)) else str(ep)
        seg.append(f"epoch {ep_s}/{p.get('max_epochs','?')}")
    if p.get("step") is not None and p.get("total_steps"):
        seg.append(f"step {p['step']}/{p['total_steps']}")
    if p.get("loss") is not None:
        seg.append(f"loss {p['loss']:.3f}")
    return f"  candidate {cand}  " + "  ".join(seg) + f"   {dim(f'{elapsed}s')}"


# ── tracking loop ─────────────────────────────────────────────────────
SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


def _announce(iteration: int, config, reason) -> None:
    """Print the agent's decision for a candidate BEFORE it trains (full strategy)."""
    import textwrap

    _clear_line()
    print(f"\n  {bold(cyan(f'▶ candidate {iteration}'))}  will try: {_cfg_str(config or {})}")
    if reason:
        lines = textwrap.wrap(" ".join(str(reason).split()), width=88) or [str(reason)]
        print(f"       {dim('strategy:')} {cyan(lines[0])}")
        for ln in lines[1:]:
            print(f"                 {cyan(ln)}")


def track(api: Api, train_api: Api, exp_id: str, poll: float = 3.0, tick: float = 0.2) -> dict:
    """Track to completion. Polls the API every ``poll`` s, but refreshes the
    spinner/elapsed line every ``tick`` s so the counter ticks ~live."""
    rule("Tuning — the agent proposes a config, trains it, scores it")
    seen: set = set()
    announced: set = set()
    best = None
    t0 = time.time()
    spin_i = 0
    e = api.get(f"/optimize/{exp_id}")
    prog = training_progress(train_api, (e.get("current_candidate") or {}).get("job_id"))
    trained = False  # has the current candidate finished training (→ now scoring)?
    next_api = time.time() + poll
    while True:
        # refresh API state + training progress only every `poll` seconds
        if time.time() >= next_api:
            e = api.get(f"/optimize/{exp_id}")
            prog = training_progress(train_api, (e.get("current_candidate") or {}).get("job_id"))
            if prog and prog.get("pct") is not None:
                trained = True  # a training job is/was running for this candidate
            next_api = time.time() + poll

        status = e.get("status")

        # Announce the in-flight candidate (config + strategy) up front, before its
        # progress bar — so you see WHAT the agent decided and WHY, then watch it train.
        cur = e.get("current_candidate")
        if cur and cur.get("iteration") and cur["iteration"] not in announced:
            _announce(cur["iteration"], cur.get("config"), cur.get("reason"))
            announced.add(cur["iteration"])
            trained = False  # a fresh candidate is starting

        # Completed candidates → show the score (config/strategy already announced).
        hist = sorted(e.get("history") or [], key=lambda h: h.get("created_at") or "")
        for h in hist:
            rid = h.get("run_id")
            if rid in seen:
                continue
            seen.add(rid)
            n = len(seen)
            if n not in announced:  # a fast candidate we never caught in flight
                _announce(n, h.get("config"), h.get("reason"))
                announced.add(n)
            score = h.get("aggregate_score")
            star = ""
            if score is not None and (best is None or score > best):
                best = score
                star = green("  ★ new best")
            sc = f"{score:.4f}" if score is not None else "—"
            _clear_line()
            print(f"  {green('✓')} candidate {n}  dev score {bold(sc)}{star}")

        if status in ("completed", "failed", "stopped"):
            _clear_line()
            return e

        # animate: spinner + phase-aware status line (every `tick`s)
        if _TTY:
            it_done = e.get("iterations_done", 0)
            max_exp = (e.get("budget") or {}).get("max_experiments") or 0
            cur = e.get("current_candidate") or {}
            in_flight = cur.get("iteration") if cur.get("iteration", 0) > it_done else None
            el = int(time.time() - t0)
            spin_i = (spin_i + 1) % len(SPIN)
            if in_flight is not None:                       # a candidate is training/scoring
                line = _progress_line(in_flight, prog, el, trained)
            elif max_exp and it_done >= max_exp:            # loop done → final held-out check
                line = f"  {dim('validating the best config on the held-out set…')}   {dim(f'{el}s')}"
            else:                                            # between candidates (proposing next)
                line = f"  {dim('the agent is choosing the next candidate…')}   {dim(f'{el}s')}"
            _clear_line()
            sys.stdout.write(f"{yellow(SPIN[spin_i])}{line}")
            sys.stdout.flush()
        time.sleep(tick)


def _clear_line():
    if _TTY:
        sys.stdout.write("\r\033[K")
        sys.stdout.flush()


# ── main ──────────────────────────────────────────────────────────────
def main() -> int:
    ap = argparse.ArgumentParser(description="AutoTune demo client")
    ap.add_argument("--url", default=DEFAULT_URL, help=f"API base (default {DEFAULT_URL})")
    ap.add_argument("--training-url", default=DEFAULT_TRAINING_URL,
                    help=f"Module 16 training API for live progress (default {DEFAULT_TRAINING_URL})")
    ap.add_argument("--max-experiments", type=int, default=None,
                    help="force the iteration budget (else the goal/planner decides)")
    ap.add_argument("--goal", default=None,
                    help="pre-fill the prompt with this goal instead of the built-in default")
    ap.add_argument("--yes", action="store_true", help="use the goal as-is without prompting")
    args = ap.parse_args()
    api = Api(args.url)
    train_api = Api(args.training_url)

    print(bold(cyan("\n=== Raptor AutoTune Demo ===")))
    print(dim(f"    {args.url}"))
    try:
        api.get("/optimize?limit=1")
    except Exception as exc:
        print(red(f"\n✗ Cannot reach the Benchmark Service at {args.url}\n  {exc}"))
        return 1

    default_goal = args.goal or DEFAULT_GOAL
    if args.yes:
        goal = default_goal
        print(bold("\nGoal> ") + goal)
    else:
        goal = input_with_prefill(bold("\nGoal> "), default_goal).strip()
    if not goal:
        print(dim("no goal — bye"))
        return 0

    # 1) plan (grounded against local models/datasets)
    print(dim("\n[1/3] Planning — grounding against local models & datasets…"))
    body = {"goal": goal}
    if args.max_experiments:
        body["budget"] = {"max_experiments": args.max_experiments}
    try:
        created = api.post("/optimize", body)
    except ApiError as exc:
        print(red(f"✗ planning failed — {exc.detail}"))
        return 1
    exp_id = created["experiment_id"]
    print(green(f"  ✓ plan ready") + dim(f"   (experiment {exp_id})"))

    plan = show_plan(api, exp_id)

    if plan.get("needs_download"):
        print(yellow("\n⚠  Some resources aren't local yet. Download them (endpoints above), "
                     "then re-run with the same goal. Not training now."))
        return 0
    if not plan.get("eval_schema_id"):
        print(yellow("\n⚠  No eval was built (dataset not local). Nothing to train."))
        return 0

    # 2) human-in-the-loop approval
    ans = input(bold("\nApprove and start training? [y/N] ")).strip().lower()
    if ans not in ("y", "yes"):
        print(dim("not approved — cleaning up."))
        try:
            api.delete(f"/optimize/{exp_id}")
        except Exception:
            pass
        return 0

    print(dim("\n[2/3] Launching optimization loop…"))
    try:
        api.post(f"/optimize/{exp_id}/confirm")
    except ApiError as exc:
        print(red(f"✗ could not start — {exc.detail}"))
        return 1
    print(green("  ✓ running — the agent now trains + scores candidates on its own"))

    # 3) track to completion, showing each round's tuning strategy
    print(dim("\n[3/3] Tracking progress (Ctrl-C to stop watching; the run continues server-side)"))
    try:
        final = track(api, train_api, exp_id)
    except KeyboardInterrupt:
        print(dim("\n(stopped watching — run continues in the background)"))
        return 0

    rule("Result")
    if final.get("status") != "completed":
        print(red(f"  status: {final.get('status')}  error: {final.get('error')}"))
        return 1
    best = final.get("best") or {}
    hold = final.get("holdout") or {}
    dev = best.get("aggregate_score")
    hs = hold.get("score")
    dev_s = green(f"{dev:.4f}") if dev is not None else "—"
    hold_s = green(f"{hs:.4f}") if hs is not None else "—"
    print(f"  {bold('Best config')}   {_cfg_str(best.get('config') or {})}")
    print(f"  {bold('Dev score')}     {dev_s}   {dim('(the set it optimized on)')}")
    print(f"  {bold('Held-out')}      {hold_s}   "
          f"{dim('(unseen questions — the honest generalization score)')}")
    n = len(final.get("history") or [])
    print(dim(f"\n  {n} candidates trained & scored, fully autonomously, from one sentence.\n"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
