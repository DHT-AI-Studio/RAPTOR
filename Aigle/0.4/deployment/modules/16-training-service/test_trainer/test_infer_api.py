"""Smoke-test the local inference endpoint (Module 16) — test_trainer style.

Mirrors train_with_raptor_api.py: a standalone `requests`-based script that
drives the REST API. Hits POST /api/v1/inference/infer, loading a local
HF-format model (here google/gemma-3-270m-it) and generating text.

Usage:
    python test_infer_api.py
    python test_infer_api.py --model-path /app/tmp/models/google_gemma-3-270m-it
"""

import argparse
import sys

import requests

TRAINING_BASE_URL = "http://localhost:8009"

# Container-visible path to the model dir (the tmp volume is mounted at /app/tmp).
DEFAULT_MODEL_PATH = "/app/tmp/models/google_gemma-3-270m-it"

PROMPTS = [
    "What is the capital of France? Answer in one sentence.",
    "用一句話解釋什麼是機器學習。",
]


def infer(model_path: str, prompt: str, max_new_tokens: int = 128, temperature: float = 0.7) -> dict:
    resp = requests.post(
        f"{TRAINING_BASE_URL}/api/v1/inference/infer",
        json={
            "model_path": model_path,
            "inputs": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
        },
        timeout=600,
    )
    resp.raise_for_status()
    return resp.json()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.7)
    args = parser.parse_args()

    print(f"Endpoint : {TRAINING_BASE_URL}/api/v1/inference/infer")
    print(f"Model    : {args.model_path}\n")

    ok = True
    for prompt in PROMPTS:
        print("=" * 60)
        print(f"PROMPT: {prompt}")
        try:
            result = infer(args.model_path, prompt, args.max_new_tokens, args.temperature)
            print(f"OUTPUT: {result['output']}")
            print(f"latency_ms: {result['latency_ms']:.1f}")
        except Exception as exc:  # noqa: BLE001
            ok = False
            print(f"FAILED: {exc}")
    print("=" * 60)
    print("RESULT:", "PASS ✅" if ok else "FAIL ❌")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
