"""RTA-X1 / RTA-X12: MOS from the rating page's exports.

Each export is {"exported_at", "scale", "ratings": [{sample_id, clip_id, system, condition,
sentence_id, evaluator_id, evaluator_region, sample_rate, codec, score}]}.  Exports overlap (each
holds every rating so far), so one rating per evaluator and clip is kept, the latest export
winning.  Reports MOS per system and condition, and the RTA-X12 gate (telephony, >= 3 Taiwanese
evaluators per sentence, MOS >= 3.5) per system.

  python -m benchmarks.mos_results human_judge/*.json --output /out/mos-results.json
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path

from benchmarks.telephony import mos_gate


def merge(exports: list[dict]) -> list[dict]:
    kept: dict[tuple[str, str], dict] = {}
    for export in sorted(exports, key=lambda e: e.get("exported_at", "")):
        for rating in export["ratings"]:
            kept[(rating["evaluator_id"], rating["clip_id"])] = rating
    return list(kept.values())


def summarize(ratings: list[dict]) -> dict:
    groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in ratings:
        groups[(r["system"], r["condition"])].append(float(r["score"]))
    table = {f"{system}/{condition}": {"mos": round(statistics.mean(v), 2), "sd": round(statistics.pstdev(v), 2),
                                       "ratings": len(v)}
             for (system, condition), v in sorted(groups.items())}
    gates = {}
    for system in sorted({r["system"] for r in ratings}):
        telephony = [r for r in ratings if r["system"] == system and r["condition"] == "telephony"]
        if telephony:
            gate = mos_gate(telephony, {r["sample_id"] for r in telephony})
            gates[system] = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in gate.items()
                             if k != "per_sentence"}
            if "per_sentence" in gate:
                gates[system]["per_sentence"] = {k.split(":", 1)[1]: round(v, 2)
                                                 for k, v in sorted(gate["per_sentence"].items())}
    return {"evaluators": len({r["evaluator_id"] for r in ratings}),
            "regions": sorted({r["evaluator_region"] for r in ratings}), "ratings": len(ratings),
            "mos": table, "telephony_gate": gates}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("exports", type=Path, nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = summarize(merge([json.loads(p.read_text(encoding="utf-8")) for p in args.exports]))
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
