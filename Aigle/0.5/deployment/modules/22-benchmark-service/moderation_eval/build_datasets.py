"""Build the CM-6 moderation evaluation sets as Module 22 benchmark schemas.

    python build_datasets.py --src /data/cm6 --out /data/cm6/schemas

Reads the sources downloaded into --src (see README.md) and writes one schema
per set into --out:

    moderation-text-en.json     OpenAI moderation evaluation set (1,680 texts)
    moderation-image.json       UnsafeBench (images written to --out/images)
    manifest.json               per case: source, source id, sha256, labels

Nothing here is committed: the sets hold sexual, violent and hateful material.
The manifest is what makes a set reproducible from its public sources.

Labels are this evaluation's three categories, ``hate`` / ``sexual`` /
``violence`` ([] = safe). A case enters a set only when its labels are
certain:

* text: OpenAI's S / S3 -> sexual, H / H2 -> hate, V / V2 -> violence. A safe
  case must have every one of the eight OpenAI labels known and 0. A case
  flagged only for harassment or self-harm (HR / SH) is unsafe but in none
  of the three categories, so it is left out, as is a case whose S, H or V
  label is unknown.
* images: UnsafeBench's Sexual / Violence / Hate unsafe images from both
  splits, plus the test split's safe images. Its test split alone has only
  29 hateful images; no image model is trained here, so the train split is
  free to evaluate on. Unsafe images of its other eight categories
  (harassment, self-harm, shocking, illegal activity, ...) are left out.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

OPENAI_KEYS = ("S", "H", "V", "HR", "SH", "S3", "H2", "V2")
OPENAI_CATEGORY = {"S": "sexual", "S3": "sexual", "H": "hate", "H2": "hate",
                   "V": "violence", "V2": "violence"}
UNSAFEBENCH_CATEGORY = {"Sexual": "sexual", "Violence": "violence", "Hate": "hate"}


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def openai_labels(row: dict) -> list[str] | None:
    """This evaluation's labels for one OpenAI moderation row; None = leave out."""
    if any(row.get(k) is None for k in ("S", "H", "V")):
        return None
    labels = sorted({cat for key, cat in OPENAI_CATEGORY.items() if row.get(key) == 1})
    if labels:
        return labels
    if all(row.get(k) == 0 for k in OPENAI_KEYS):
        return []
    return None     # unsafe outside the three categories, or not fully labelled


def unsafebench_labels(row: dict) -> list[str] | None:
    if row["safety_label"] == "Safe":
        return []
    category = UNSAFEBENCH_CATEGORY.get(row["category"])
    return [category] if category else None


def _schema(name: str, pipeline: str, cases: list[dict]) -> dict:
    return {
        "name": name,
        "version": "1.0",
        "target_pipeline": pipeline,
        "test_cases": cases,
        "scoring_schema": {"dimensions": [{"name": "correct", "weight": 1.0,
                                           "method": "moderation_match"}]},
    }


def build_text_en(src: Path) -> tuple[dict, list[dict]]:
    path = src / "openai-moderation" / "samples-1680.jsonl.gz"
    cases, manifest = [], []
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        for i, line in enumerate(fh):
            row = json.loads(line)
            labels = openai_labels(row)
            if labels is None:
                continue
            case_id = f"oai-{i:04d}"
            cases.append({"id": case_id, "input": {"text": row["prompt"]}, "labels": labels})
            manifest.append({"id": case_id, "set": "moderation-text-en",
                             "source": "mmathys/openai-moderation-api-evaluation",
                             "source_id": i, "sha256": _sha256(row["prompt"].encode()),
                             "labels": labels})
    return _schema("moderation-text-en", "guardrail", cases), manifest


def build_images(src: Path, out: Path, image_root: str | None) -> tuple[dict, list[dict]]:
    import pyarrow.parquet as pq

    images = out / "images"
    images.mkdir(parents=True, exist_ok=True)
    cases, manifest = [], []
    for split in ("test", "train"):
        table = pq.read_table(src / "UnsafeBench" / "data" / f"{split}-00000-of-00001.parquet")
        for i, row in enumerate(table.to_pylist()):
            labels = unsafebench_labels(row)
            if labels is None or (split == "train" and not labels):
                continue
            data = row["image"]["bytes"]
            suffix = Path(row["image"].get("path") or "x.jpg").suffix or ".jpg"
            case_id = f"ub-{split}-{i:04d}"
            name = f"{case_id}{suffix}"
            (images / name).write_bytes(data)
            path = f"{image_root.rstrip('/')}/{name}" if image_root else str(images / name)
            cases.append({"id": case_id, "input": {"image_path": path}, "labels": labels})
            manifest.append({"id": case_id, "set": "moderation-image", "source": "yiting/UnsafeBench",
                             "source_id": f"{split}/{i}", "source_category": row["category"],
                             "source_origin": row["source"], "sha256": _sha256(data),
                             "labels": labels})
    return _schema("moderation-image", "guardrail_image", cases), manifest


def _counts(cases: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {"safe": 0}
    for c in cases:
        for label in c["labels"] or ["safe"]:
            counts[label] = counts.get(label, 0) + 1
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="directory the sources were downloaded to")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--image-root", help="image directory as the benchmark service sees it "
                                         "(default: --out/images as is)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    manifest = []
    for schema, entries in (build_text_en(args.src),
                            build_images(args.src, args.out, args.image_root)):
        (args.out / f"{schema['name']}.json").write_text(json.dumps(schema, ensure_ascii=False))
        manifest += entries
        print(schema["name"], len(schema["test_cases"]), _counts(schema["test_cases"]))
    (args.out / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
