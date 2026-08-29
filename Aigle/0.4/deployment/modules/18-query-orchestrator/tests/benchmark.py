"""
Benchmark: 1001 questions against query-orchestrator.
Reports: avg/min/max latency, accuracy (overall + per pipeline + per strategy).
"""
import json, time, statistics, httpx
from collections import defaultdict
from pathlib import Path

BASE = "http://localhost:8843"
QS_PATH = Path(__file__).parent / "benchmark_questions.json"

with open(QS_PATH, encoding="utf-8") as f:
    questions = json.load(f)


print(f"Running {len(questions)} questions against {BASE} …\n")

results = []
for i, item in enumerate(questions, 1):
    q, label = item["q"], item["label"]
    t0 = time.perf_counter()
    try:
        r = httpx.post(f"{BASE}/classify", json={"query": q, "k": 1}, timeout=60)
        r.raise_for_status()
        d = r.json()
        elapsed = time.perf_counter() - t0
        modes    = d.get("modes") or d.get("intent_type", [])
        conf     = d.get("confidence", 0.0)
        strategy = d.get("routing_strategy", "unknown")
        correct  = label in modes and len(modes) == 1
        results.append({
            "q": q, "label": label, "modes": modes,
            "strategy": strategy, "conf": conf,
            "elapsed": elapsed, "correct": correct, "error": None,
        })
    except Exception as e:
        elapsed = time.perf_counter() - t0
        results.append({
            "q": q, "label": label, "modes": [], "strategy": "error",
            "conf": 0.0, "elapsed": elapsed, "correct": False, "error": str(e),
        })

    if i % 100 == 0:
        done = [r for r in results if r["error"] is None]
        acc  = sum(r["correct"] for r in done) / len(done) * 100 if done else 0
        avg  = statistics.mean(r["elapsed"] for r in done) if done else 0
        print(f"  [{i:4d}/{len(questions)}]  acc={acc:.1f}%  avg={avg*1000:.0f}ms")

# ── Summary ──────────────────────────────────────────────────────────────────
ok      = [r for r in results if r["error"] is None]
latency = [r["elapsed"] for r in ok]

print("\n" + "="*62)
print("BENCHMARK RESULTS")
print("="*62)
print(f"Total questions : {len(results)}")
print(f"Errors          : {sum(1 for r in results if r['error'])}")

print(f"\n── Latency (ms) ──────────────────────────────────────────")
print(f"  Average : {statistics.mean(latency)*1000:7.1f} ms")
print(f"  Median  : {statistics.median(latency)*1000:7.1f} ms")
print(f"  Min     : {min(latency)*1000:7.1f} ms")
print(f"  Max     : {max(latency)*1000:7.1f} ms")
print(f"  P95     : {sorted(latency)[int(len(latency)*0.95)]*1000:7.1f} ms")
print(f"  P99     : {sorted(latency)[int(len(latency)*0.99)]*1000:7.1f} ms")

print(f"\n── Overall Accuracy ──────────────────────────────────────")
total_correct = sum(r["correct"] for r in ok)
print(f"  {total_correct}/{len(ok)} = {total_correct/len(ok)*100:.1f}%")

print(f"\n── Accuracy by Pipeline ──────────────────────────────────")
per_label = defaultdict(lambda: {"total": 0, "correct": 0, "times": []})
for r in ok:
    per_label[r["label"]]["total"] += 1
    per_label[r["label"]]["correct"] += r["correct"]
    per_label[r["label"]]["times"].append(r["elapsed"])
for label in sorted(per_label):
    d = per_label[label]
    acc = d["correct"] / d["total"] * 100
    avg = statistics.mean(d["times"]) * 1000
    print(f"  {label:<12} {d['correct']:3d}/{d['total']:3d} = {acc:5.1f}%   avg {avg:.0f}ms")

print(f"\n── Routing Strategy Breakdown ────────────────────────────")
per_strat = defaultdict(lambda: {"total": 0, "correct": 0, "times": []})
for r in ok:
    per_strat[r["strategy"]]["total"] += 1
    per_strat[r["strategy"]]["correct"] += r["correct"]
    per_strat[r["strategy"]]["times"].append(r["elapsed"])
for strat in sorted(per_strat, key=lambda s: -per_strat[s]["total"]):
    d  = per_strat[strat]
    acc = d["correct"] / d["total"] * 100 if d["total"] else 0
    avg = statistics.mean(d["times"]) * 1000 if d["times"] else 0
    pct = d["total"] / len(ok) * 100
    print(f"  {strat:<12} {d['total']:4d} q ({pct:4.1f}%)  acc={acc:5.1f}%  avg={avg:.0f}ms")

print(f"\n── Wrong Answers (first 20) ──────────────────────────────")
wrong = [r for r in ok if not r["correct"]]
for r in wrong[:20]:
    print(f"  [{r['strategy']:<10}] label={r['label']:<10} got={r['modes']}  {r['q'][:50]}")

print(f"\nTotal wrong: {len(wrong)}/{len(ok)}")

# Save full results
out_path = Path(__file__).parent / "benchmark_results.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)
print(f"\nFull results saved to: {out_path}")
