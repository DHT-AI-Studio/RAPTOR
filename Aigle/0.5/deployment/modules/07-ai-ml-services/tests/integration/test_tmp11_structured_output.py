# tests/integration/test_tmp11_structured_output.py
"""
TMP-11 AC6 — Integration test: response_format with a TemporalFact schema
returns schema-valid JSON from Ollama.

Target model: gemma4:26b (AC6 spec). Falls back to OLLAMA_MODEL env var,
then any available gemma4:* variant if the specified model is not pulled.

Run:
    OLLAMA_BASE_URL=http://localhost:11434 \
    MODULE07_BASE_URL=http://localhost:8010 \
    OLLAMA_MODEL=gemma4:12b \
    pytest tests/integration/test_tmp11_structured_output.py -v
"""
from __future__ import annotations

import json
import os
from typing import Any

import pytest
import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OLLAMA_BASE_URL  = os.getenv("OLLAMA_BASE_URL",  "http://localhost:11434")
MODULE07_BASE_URL = os.getenv("MODULE07_BASE_URL", "http://localhost:8010")

_PREFERRED_MODEL = "gemma4:26b"
MODEL = os.getenv("OLLAMA_MODEL", _PREFERRED_MODEL)

TEMPORAL_FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "subject":        {"type": "string"},
        "predicate":      {"type": "string"},
        "object":         {"type": "string"},
        "start_year":     {"type": "integer", "minimum": 1, "maximum": 2100},
        "end_year":       {"type": "integer", "minimum": 1, "maximum": 2100},
        "confidence":     {"type": "number",  "minimum": 0.0, "maximum": 1.0},
        "source_snippet": {"type": "string"},
    },
    "required": ["subject", "predicate", "object", "confidence"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "TemporalFact",
        "schema": TEMPORAL_FACT_SCHEMA,
    },
}

PROMPT = (
    "Extract a single temporal fact from the following sentence as JSON.\n\n"
    "Sentence: 'Marie Curie won the Nobel Prize in Physics in 1903.'"
)

# ---------------------------------------------------------------------------
# Lazy helpers
# ---------------------------------------------------------------------------

def _pulled_models() -> list[str]:
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except requests.RequestException:
        pass
    return []


def _resolve_model() -> str:
    """Return MODEL if pulled; else first available gemma4:* variant; else MODEL."""
    pulled = _pulled_models()
    if any(MODEL == n or MODEL in n for n in pulled):
        return MODEL
    for name in pulled:
        if name.startswith("gemma4:"):
            return name
    return MODEL


# ---------------------------------------------------------------------------
# Inline schema validator
# ---------------------------------------------------------------------------

def _validate(value: Any, schema: dict, path: str = "$") -> list[str]:
    errors: list[str] = []

    def _is_type(v, t: str) -> bool:
        return {
            "string":  isinstance(v, str),
            "integer": isinstance(v, int) and not isinstance(v, bool),
            "number":  isinstance(v, (int, float)) and not isinstance(v, bool),
            "boolean": isinstance(v, bool),
            "object":  isinstance(v, dict),
            "array":   isinstance(v, list),
            "null":    v is None,
        }.get(t, False)

    def _check(v, s: dict, p: str) -> None:
        t = s.get("type")
        if t and not _is_type(v, t):
            errors.append(f"{p}: expected {t}, got {type(v).__name__}")
            return
        if isinstance(v, dict):
            for name in s.get("required") or []:
                if name not in v:
                    errors.append(f"{p}: missing required '{name}'")
            props = s.get("properties") or {}
            extra = s.get("additionalProperties", True)
            for k, item in v.items():
                if k in props:
                    _check(item, props[k], f"{p}.{k}")
                elif extra is False:
                    errors.append(f"{p}: unexpected property '{k}'")
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            if "minimum" in s and v < s["minimum"]:
                errors.append(f"{p}: {v} below minimum {s['minimum']}")
            if "maximum" in s and v > s["maximum"]:
                errors.append(f"{p}: {v} above maximum {s['maximum']}")

    _check(value, schema, path)
    return errors


# ---------------------------------------------------------------------------
# Fixture — lazy skip, runs at test time not collect time
# ---------------------------------------------------------------------------

@pytest.fixture(scope="class", autouse=True)
def require_services(request):
    pulled = _pulled_models()
    if not pulled:
        pytest.skip(f"Ollama daemon not reachable at {OLLAMA_BASE_URL}")

    resolved = _resolve_model()
    if not any(resolved == n or resolved in n for n in pulled):
        pytest.skip(
            f"Model '{resolved}' not pulled. Run: ollama pull {resolved}\n"
            f"Available: {', '.join(pulled[:8])}{'...' if len(pulled) > 8 else ''}"
        )

    try:
        r = requests.get(f"{MODULE07_BASE_URL}/inference/health", timeout=5)
        if r.status_code != 200:
            pytest.skip(f"Module 07 health check failed: {r.status_code}")
    except requests.RequestException as e:
        pytest.skip(f"Module 07 not reachable at {MODULE07_BASE_URL}: {e}")

    request.cls._model = resolved

    if resolved != _PREFERRED_MODEL:
        import warnings
        warnings.warn(
            f"\nAC6 specifies {_PREFERRED_MODEL} but it is not pulled; "
            f"running against {resolved}. "
            f"Pull {_PREFERRED_MODEL} for full AC6 compliance.",
            stacklevel=1,
        )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestTemporalFactStructuredOutput:
    """AC6: TemporalFact schema returns schema-valid JSON via Module 07 → Ollama."""

    _model: str  # injected by require_services

    def _chat(self, **extra) -> requests.Response:
        """POST /v1/chat/completions — uses engine='ollama' so MLflow lookup is bypassed."""
        payload = {
            "model":    self._model,
            "messages": [{"role": "user", "content": PROMPT}],
            "temperature": 0,
            "think":    False,
            "engine":   "ollama",   # bypass MLflow registry for daemon-only models
            **extra,
        }
        return requests.post(
            f"{MODULE07_BASE_URL}/v1/chat/completions",
            json=payload,
            timeout=120,
        )

    # --- AC6 core ---

    def test_response_format_returns_schema_valid_json(self):
        """/v1/chat/completions + TemporalFact schema → schema-valid JSON. (AC1, AC6)"""
        r = self._chat(response_format=RESPONSE_FORMAT)
        assert r.status_code == 200, f"unexpected {r.status_code}: {r.text[:400]}"

        content = r.json()["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            pytest.fail(f"content is not JSON: {e}\ncontent={content!r}")

        violations = _validate(parsed, TEMPORAL_FACT_SCHEMA)
        assert not violations, (
            "TemporalFact schema violations:\n"
            + "\n".join(f"  {v}" for v in violations)
            + f"\nparsed=\n{json.dumps(parsed, indent=2, ensure_ascii=False)}"
        )

        fact_str = json.dumps(parsed).lower()
        assert any(kw in fact_str for kw in ("curie", "marie", "nobel", "physics", "1903")), (
            f"extracted fact doesn't mention the source event: {parsed}"
        )

    def test_inference_infer_endpoint_also_works(self):
        """/inference/infer + options.format → result.parsed satisfies schema. (AC2)"""
        r = requests.post(
            f"{MODULE07_BASE_URL}/inference/infer",
            json={
                "model_name": self._model,
                "engine":     "ollama",
                "data": {"messages": [{"role": "user", "content": PROMPT}]},
                "options": {
                    "format":      RESPONSE_FORMAT,
                    "temperature": 0,
                    "think":       False,
                },
            },
            timeout=120,
        )
        assert r.status_code == 200, f"{r.status_code}: {r.text[:400]}"
        result = r.json()["result"]

        assert "parsed" in result, (
            f"result.parsed missing — structured output pipeline not applied.\n"
            f"result keys: {list(result.keys())}"
        )
        violations = _validate(result["parsed"], TEMPORAL_FACT_SCHEMA)
        assert not violations, (
            "TemporalFact schema violations:\n"
            + "\n".join(f"  {v}" for v in violations)
        )

    # --- AC1: json_object mode ---

    def test_json_object_mode_returns_any_json_object(self):
        """{type: json_object} → any valid JSON object, no schema constraint. (AC1)"""
        r = self._chat(response_format={"type": "json_object"})
        assert r.status_code == 200, f"{r.status_code}: {r.text[:400]}"
        content = r.json()["choices"][0]["message"]["content"]
        parsed = json.loads(content)
        assert isinstance(parsed, dict), f"expected object, got {type(parsed).__name__}"

    # --- AC3: unsupported capability ---

    def test_unsupported_engine_returns_400_unsupported_capability(self):
        """HF-backed model must return UNSUPPORTED_CAPABILITY. (AC3)"""
        hf_model = os.getenv("HF_TEXT_MODEL")
        if not hf_model:
            pytest.skip("Set HF_TEXT_MODEL to a registered HF text-generation model name")
        r = self._chat(model=hf_model, engine=None, response_format=RESPONSE_FORMAT)
        assert r.status_code == 400
        assert r.json()["detail"]["error"]["code"] == "unsupported_capability"

    # --- AC4: invalid output → 502 ---

    @pytest.mark.skip(reason="AC4 covered by unit/test_structured_ac4.py; impossible schema causes Ollama constrained-decoding hang")
    def test_output_violating_schema_is_caught_and_never_passed_on(self):
        """Schema with an impossible required field → 400 or 502, never 200. (AC4)"""
        trap_schema = {
            "type": "object",
            "properties": {
                "zzz_sentinel_field_that_model_will_never_produce": {
                    "type": "string",
                    "minLength": 500,
                },
            },
            "required": ["zzz_sentinel_field_that_model_will_never_produce"],
            "additionalProperties": False,
        }
        r = self._chat(response_format={
            "type": "json_schema",
            "json_schema": {"name": "Trap", "schema": trap_schema},
        })
        assert r.status_code in (400, 502), (
            f"expected 400 or 502, got {r.status_code} — "
            f"invalid output passed through undetected (AC4 regression).\n"
            f"body: {r.text[:400]}"
        )
