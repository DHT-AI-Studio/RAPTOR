"""Run the CM-6 moderation benchmark through Module 22 and write the report.

    python run_eval.py --bench http://localhost:8022/api/v1 \
        --schemas /data/cm6/schemas --out /data/cm6/report

Uploads each evaluation set built by build_datasets.py / translate_zh_tw.py
as a schema, runs every model on it (one Module 22 run per model), and
writes the moderation reports to --out as report.json and report.md. Only
numbers and case ids leave the sets: no case text or image is copied into
the report.

The models and the threshold per run are config_override values, so the
report records exactly what was measured.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import httpx

TEXT_MODELS = ["llama-guard3:8b", "granite4.1-guardian:8b", "gpt-oss-safeguard:20b"]
IMAGE_MODELS = ["shieldgemma-2", "llama-guard-3-vision"]
PLAN = {
    "moderation-text-en": [{"model": m} for m in TEXT_MODELS],
    "moderation-text-zh-tw": [{"model": m} for m in TEXT_MODELS],
    "moderation-image": [{"model": m} for m in IMAGE_MODELS],
}


def run_one(client: httpx.Client, schema_id: str, config: dict, poll: float) -> dict:
    run = client.post("/benchmark/runs", json={"schema_id": schema_id,
                                               "config_override": config}).json()
    run_id = run["run_id"]
    while True:
        state = client.get(f"/benchmark/runs/{run_id}").json()
        if state["status"] in ("completed", "failed"):
            break
        print(f"    {config} {state.get('progress_pct', 0):.0f}%", end="\r", flush=True)
        time.sleep(poll)
    if state["status"] == "failed":
        raise RuntimeError(f"run {run_id} failed: {state.get('scores_per_case')}")
    report = client.get(f"/benchmark/runs/{run_id}/moderation").json()
    report["mlflow_run_id"] = state.get("mlflow_run_id")
    latencies = sorted(c["latency_ms"] for c in state["scores_per_case"] if not c.get("error"))
    if latencies:
        report["latency_ms_p50"] = round(latencies[len(latencies) // 2], 1)
        report["latency_ms_p95"] = round(latencies[int(len(latencies) * 0.95) - 1], 1)
    return report


def _cell(m: dict) -> str:
    return f"{m['precision']:.2f} / {m['recall']:.2f} / **{m['f1']:.2f}**"


def markdown(reports: list[dict]) -> str:
    lines = ["# CM-6 moderation model evaluation", "",
             ("Cells are precision / recall / **F1**. A category is scored by the category "
              "the model names when it names one (`category`), else by whether it flags the "
              "case at all, over that category's cases plus the safe cases (`detection`). "
              "`errors` are cases with no verdict (not scored)."), ""]
    for name in dict.fromkeys(r["schema_name"] for r in reports):
        rows = [r for r in reports if r["schema_name"] == name]
        cats = sorted({c for r in rows for c in r["per_category"]})
        lines += [f"## {name}", "",
                  "| model | config | cases | errors | overall | " + " | ".join(cats)
                  + " | below target | p50 ms |",
                  "|---" * (6 + len(cats)) + "|---|"]
        for r in rows:
            per = [f"{_cell(r['per_category'][c][r['per_category'][c]['selected']])} "
                   f"({r['per_category'][c]['selected']})" if c in r["per_category"] else "-"
                   for c in cats]
            cfg = {k: v for k, v in (r.get("config_override") or {}).items() if k != "model"}
            lines.append(f"| {r['model']} | {json.dumps(cfg) if cfg else ''} | {r['cases']} | "
                         f"{r['errors']} | {_cell(r['overall'])} | " + " | ".join(per)
                         + f" | {', '.join(r['below_target']) or 'none'} | "
                         f"{r.get('latency_ms_p50', '-')} |")
        thresholds = [(r["model"], c, e["best_threshold"]) for r in rows
                      for c, e in r["per_category"].items() if e.get("best_threshold")]
        if thresholds:
            lines += ["", "Best threshold per category (score models):", "",
                      "| model | category | threshold | precision / recall / F1 |",
                      "|---|---|---|---|"]
            lines += [f"| {m} | {c} | {t['threshold']} | {_cell(t)} |" for m, c, t in thresholds]
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--bench", required=True, help="Module 22 base URL, e.g. http://localhost:8022/api/v1")
    ap.add_argument("--schemas", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--only", nargs="*", help="evaluation sets to run (default: all in the plan)")
    ap.add_argument("--models", nargs="*", help="models to run (default: all in the plan)")
    ap.add_argument("--extra", type=json.loads, default=[],
                    help='extra runs as JSON, e.g. \'[{"set": "moderation-image", '
                         '"config": {"model": "shieldgemma-2", "threshold": 0.3}}]\'')
    ap.add_argument("--poll", type=float, default=5.0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    plan = [(s, c) for s, configs in PLAN.items() for c in configs
            if (not args.only or s in args.only) and (not args.models or c["model"] in args.models)]
    plan += [(e["set"], e["config"]) for e in args.extra]

    report_path = args.out / "report.json"
    reports = json.loads(report_path.read_text()) if report_path.exists() else []
    schema_ids: dict[str, str] = {}
    with httpx.Client(base_url=args.bench, timeout=120) as client:
        for set_name, config in plan:
            path = args.schemas / f"{set_name}.json"
            if not path.exists():
                print(f"skip {set_name}: {path} not built")
                continue
            if set_name not in schema_ids:
                resp = client.post("/benchmark/schemas", content=path.read_bytes(),
                                   headers={"Content-Type": "application/json"})
                resp.raise_for_status()
                schema_ids[set_name] = resp.json()["id"]
            print(f"{set_name} {config}")
            report = run_one(client, schema_ids[set_name], config, args.poll)
            reports = [r for r in reports if not (r["schema_name"] == set_name
                                                  and r.get("config_override") == config)]
            reports.append(report)
            print(f"    overall F1 {report['overall']['f1']:.3f}, errors {report['errors']}, "
                  f"below target: {report['below_target'] or 'none'}")
            report_path.write_text(json.dumps(reports, ensure_ascii=False, indent=1))
            (args.out / "report.md").write_text(markdown(reports))


if __name__ == "__main__":
    main()
