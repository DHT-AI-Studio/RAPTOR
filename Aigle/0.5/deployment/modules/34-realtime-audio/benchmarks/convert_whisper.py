"""Convert an existing local Whisper snapshot to offline CTranslate2 float16."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("conversion destination already exists; existing weights are never overwritten")
    for name in ("config.json", "tokenizer.json", "preprocessor_config.json"):
        if not (args.source / name).is_file():
            raise FileNotFoundError(args.source / name)
    config = json.loads((args.source / "config.json").read_text())
    if config.get("model_type") != "whisper":
        raise ValueError("source must be a Whisper Transformers snapshot")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from ctranslate2.converters import TransformersConverter
    converter = TransformersConverter(
        str(args.source), copy_files=["tokenizer.json", "preprocessor_config.json"],
        load_as_float16=True, low_cpu_mem_usage=True, trust_remote_code=False,
    )
    print(f"Converting {args.source} to {args.output}", flush=True)
    converter.convert(str(args.output), quantization="float16", force=False)
    report = {"source": str(args.source), "quantization": "float16",
              "config_sha256": hashlib.sha256((args.source / "config.json").read_bytes()).hexdigest(),
              "ctranslate2": importlib.metadata.version("ctranslate2"),
              "transformers": importlib.metadata.version("transformers")}
    (args.output / "conversion.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
