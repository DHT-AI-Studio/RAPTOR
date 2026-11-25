"""Moderation model evaluation (CM-6).

The guardrail / guardrail_image pipelines turn each Module 23 answer into one
verdict, stored as the case's output (JSON):

    {"model": "llama-guard3:8b", "unsafe": true, "categories": ["hate"],
     "codes": ["S10"], "scores": {}, "covers": ["hate", "sexual", "violence"]}

``categories`` are the evaluation's own names (sexual / violence / hate), so
models with different taxonomies can be compared on the same labels.
``covers`` is what the model can name at all: Llama Guard names all three,
ShieldGemma 2 has no hate policy, and Granite Guardian / gpt-oss-safeguard
answer only safe or unsafe.

``report`` turns a run's verdicts and the schema's gold labels into
per-category precision / recall / F1.
"""
from __future__ import annotations

import json
from typing import Any

CATEGORIES = ("hate", "sexual", "violence")

# Llama Guard 3 hazard codes (text and vision) -> evaluation category.
# S9 (indiscriminate weapons) and the other codes have no evaluation category:
# they still make the verdict unsafe, but name none of the three.
LLAMA_GUARD_CODES = {
    "S1": "violence",   # violent crimes
    "S3": "sexual",     # sex crimes
    "S4": "sexual",     # child sexual exploitation
    "S10": "hate",
    "S12": "sexual",    # sexual content
}

# ShieldGemma 2 policy -> evaluation category. "dangerous" has none.
SHIELDGEMMA_POLICIES = {"sexual": "sexual", "violence": "violence"}

IMAGE_ENDPOINTS = {
    "shieldgemma-2": "/infer/google-shieldgemma-2",
    "llama-guard-3-vision": "/infer/Llama-Guard-3-11B-Vision",
}

DEFAULT_THRESHOLD = 0.5


def _verdict(model: str, unsafe: bool, *, categories: list[str] | None = None,
             codes: list[str] | None = None, scores: dict[str, float] | None = None,
             covers: tuple[str, ...] = ()) -> dict[str, Any]:
    return {"model": model, "unsafe": unsafe, "categories": sorted(set(categories or [])),
            "codes": codes or [], "scores": scores or {}, "covers": sorted(covers)}


def _from_llama_guard_codes(model: str, unsafe: bool, codes: list[str]) -> dict[str, Any]:
    return _verdict(model, unsafe, codes=codes,
                    categories=[LLAMA_GUARD_CODES[c] for c in codes if c in LLAMA_GUARD_CODES],
                    covers=CATEGORIES)


def text_verdict(payload: dict[str, Any], model: str | None) -> dict[str, Any]:
    """One model's verdict from Module 23's /guard/check/input answer.

    With several guard models active, Module 23 answers with a combined
    verdict plus ``results`` per model; ``model`` picks one. Without
    ``model`` the combined verdict is scored.
    """
    entry: dict[str, Any] = payload
    results = payload.get("results")
    if model and results:
        matches = [r for r in results if r.get("model") == model]
        if not matches:
            raise ValueError(f"Module 23 gave no result for model {model!r} "
                             f"(active: {[r.get('model') for r in results]})")
        entry = matches[0]
    name = entry.get("model") or model or "combined"
    unsafe = not entry["safe"]
    codes = entry.get("categories") or []
    if "llama-guard" in name.lower():
        return _from_llama_guard_codes(name, unsafe, codes)
    return _verdict(name, unsafe, codes=codes)


def image_verdict(payload: dict[str, Any], model: str,
                  threshold: float = DEFAULT_THRESHOLD) -> dict[str, Any]:
    """One image guard's verdict from Module 23's /infer/<model> answer.

    ShieldGemma 2 gives a probability per policy; ``threshold`` decides which
    count. The scores are kept so the report can sweep thresholds.
    """
    if model == "shieldgemma-2":
        policies = payload["policies"]
        probs = {name: float(p["unsafe_probability"]) for name, p in policies.items()}
        scores = {SHIELDGEMMA_POLICIES[n]: v for n, v in probs.items() if n in SHIELDGEMMA_POLICIES}
        flagged = [n for n, v in probs.items() if v >= threshold]
        return _verdict(model, bool(flagged), codes=flagged, scores=scores,
                        categories=[SHIELDGEMMA_POLICIES[n] for n in flagged if n in SHIELDGEMMA_POLICIES],
                        covers=tuple(SHIELDGEMMA_POLICIES.values()))
    if model == "llama-guard-3-vision":
        return _from_llama_guard_codes(model, payload["label"] == "unsafe", payload.get("categories") or [])
    raise ValueError(f"unknown image guard model {model!r}; available: {sorted(IMAGE_ENDPOINTS)}")


def parse_output(output: str) -> dict[str, Any] | None:
    """A stored case output back to its verdict; None if the case had none."""
    try:
        verdict = json.loads(output)
    except (TypeError, ValueError):
        return None
    return verdict if isinstance(verdict, dict) and "unsafe" in verdict else None


# ── Report ───────────────────────────────────────────────────────────

def _prf(pairs: list[tuple[bool, bool]]) -> dict[str, Any]:
    """precision / recall / F1 over (gold, predicted) pairs."""
    tp = sum(1 for g, p in pairs if g and p)
    fp = sum(1 for g, p in pairs if not g and p)
    fn = sum(1 for g, p in pairs if g and not p)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
            "support": tp + fn, "tp": tp, "fp": fp, "fn": fn, "n": len(pairs)}


_GRID = [round(0.05 * i, 2) for i in range(1, 20)]


def _best_threshold(pairs: list[tuple[bool, float]]) -> dict[str, Any] | None:
    """The score threshold with the best F1 (ties: the higher threshold)."""
    if not any(g for g, _ in pairs):
        return None
    best: dict[str, Any] | None = None
    for t in _GRID:
        m = _prf([(g, s >= t) for g, s in pairs])
        if best is None or m["f1"] >= best["f1"]:
            best = {"threshold": t, **m}
    return best


def report(cases: list[tuple[list[str], dict[str, Any] | None]],
           target_f1: float) -> dict[str, Any]:
    """Per-category precision / recall / F1 for one run.

    ``cases`` pairs each case's gold labels ([] = safe) with its verdict
    (None when the call failed; those are counted, not scored).

    Two ways of scoring a category, both reported:

    * ``detection`` -- every model: over the cases labelled with the
      category plus the safe cases, is "unsafe" right? This is the only
      measure for a model that names no categories.
    * ``category`` -- a model that names the category: over all cases, does
      it name this one exactly? A hate post flagged as violence counts as a
      miss here but a hit under detection.

    ``selected`` is category where the model has it, detection otherwise;
    ``below_target`` lists the categories whose selected F1 is under
    ``target_f1``.
    """
    scored = [(labels, v) for labels, v in cases if v is not None]
    covers = set(scored[0][1].get("covers") or []) if scored else set()
    present = sorted({c for labels, _ in scored for c in labels})

    per_category: dict[str, Any] = {}
    for c in present:
        entry: dict[str, Any] = {
            "detection": _prf([(bool(labels), v["unsafe"]) for labels, v in scored
                               if c in labels or not labels]),
        }
        if c in covers:
            entry["category"] = _prf([(c in labels, c in v["categories"]) for labels, v in scored])
            score_pairs = [(c in labels, v["scores"][c]) for labels, v in scored if c in v.get("scores", {})]
            if score_pairs:
                entry["best_threshold"] = _best_threshold(score_pairs)
        entry["selected"] = "category" if "category" in entry else "detection"
        per_category[c] = entry

    below = [c for c, e in per_category.items() if e[e["selected"]]["f1"] < target_f1]
    return {
        "model": scored[0][1].get("model") if scored else None,
        "cases": len(cases),
        "errors": len(cases) - len(scored),
        "covers": sorted(covers),
        "overall": _prf([(bool(labels), v["unsafe"]) for labels, v in scored]),
        "per_category": per_category,
        "target_f1": target_f1,
        "below_target": below,
    }
