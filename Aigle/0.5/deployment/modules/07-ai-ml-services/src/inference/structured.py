# src/inference/structured.py
"""
結構化輸出（temporal plan §5.10 / P10）

呼叫端用 OpenAI 的 `response_format`（/v1/chat/completions）或 options 的 `format`
（/inference/infer）要求 JSON 輸出：

    {"type": "json_object"}                                       任何 JSON 物件
    {"type": "json_schema", "json_schema": {"name": ..., "schema": {...}}}
    "json" 或直接給一份 JSON Schema（Ollama 原生 format 的寫法）
    {"type": "text"} / 不給                                         一般文字（不檢查）

流程：
    1. output_contract() 驗證請求本身（schema 只能用 SUPPORTED_KEYWORDS，未知關鍵字回 400，
       不會默默忽略）
    2. 能照辦的 runtime（Ollama：/api/chat 的 `format`）才接受；其他回 400 unsupported_capability
    3. 模型輸出經 OutputContract.check()：寬鬆解析（<think>、```fences、前後雜訊、模型把
       schema 外殼一起回傳）只是有限度的第二道防線，解析結果一定要通過 schema 驗證，
       否則回 502 structured_output_error —— 不合格的輸出絕不當成合格的傳下去

schema 合法只代表格式對，不代表內容真實：呼叫端（例如 Module 25 的證據檢查）仍要自己驗證內容。
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .exceptions import InvalidParameterError, StructuredOutputError

# 驗證時真正檢查的關鍵字，以及只是說明用、不影響驗證的關鍵字
_CHECKED = {
    "type", "properties", "required", "additionalProperties", "items", "minItems", "maxItems",
    "enum", "const", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "minLength", "maxLength",
}
_ANNOTATIONS = {"$schema", "$id", "title", "description", "default", "examples", "format"}
SUPPORTED_KEYWORDS = frozenset(_CHECKED | _ANNOTATIONS)

_TYPES = {"object", "array", "string", "integer", "number", "boolean", "null"}
_MAX_DEPTH = 32
_MAX_ERRORS = 20


@dataclass(frozen=True)
class OutputContract:
    kind: str                           # "json_object" | "json_schema"
    schema: dict[str, Any] | None = None
    name: str | None = None

    def runtime_format(self) -> str | dict[str, Any]:
        """Ollama /api/chat 與 /api/generate 的 `format` 欄位值。"""
        return self.schema if self.schema is not None else "json"

    def check(self, raw: str) -> Any:
        """模型輸出 → 通過驗證的 JSON 值；不通過拋 StructuredOutputError。"""
        value = parse_output(raw)
        if self.schema is None:
            if not isinstance(value, dict):
                raise StructuredOutputError("output is not a JSON object", violations=["$: not an object"])
            return value
        errors = validate(value, self.schema)
        if not errors:
            return value
        unwrapped = self._unwrap(value)
        if unwrapped is not None:
            return unwrapped
        raise StructuredOutputError(
            f"output does not match the requested schema ({len(errors)} violation(s))", violations=errors)

    def _unwrap(self, value: Any) -> Any | None:
        """模型常見的兩種外殼：把答案填進 schema 的 "properties"，或包在 {name: ...} 裡。
        只在拆開後完全通過驗證時才接受。"""
        if not isinstance(value, dict):
            return None
        candidates = []
        if isinstance(value.get("properties"), dict):
            candidates.append(value["properties"])
        if self.name and len(value) == 1 and self.name in value:
            candidates.append(value[self.name])
        for c in candidates:
            if not validate(c, self.schema):
                return c
        return None


def output_contract(value: Any, parameter: str = "response_format") -> OutputContract | None:
    """請求的 response_format / format → OutputContract；要一般文字時回 None。格式不對拋 400。"""
    if value is None:
        return None
    if value == "json":
        return OutputContract("json_object")
    if not isinstance(value, dict):
        raise InvalidParameterError(
            f"{parameter} must be an object like {{\"type\": \"json_schema\", ...}}, \"json\" or a JSON Schema",
            parameter=parameter)
    kind = value.get("type")
    if kind == "text":
        return None
    if kind == "json_object":
        return OutputContract("json_object")
    if kind == "json_schema":
        spec = value.get("json_schema")
        if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
            raise InvalidParameterError(f"{parameter}.json_schema.schema must be a JSON Schema object",
                                        parameter=parameter)
        name = spec.get("name")
        check_schema(spec["schema"], parameter)
        return OutputContract("json_schema", spec["schema"], name if isinstance(name, str) else None)
    # Ollama 原生寫法：直接給 schema
    check_schema(value, parameter)
    return OutputContract("json_schema", value)


def check_schema(schema: Any, parameter: str = "response_format") -> None:
    """schema 只能用 SUPPORTED_KEYWORDS；用到其他關鍵字（$ref、anyOf、pattern ...）回 400。"""
    unsupported: set[str] = set()
    problems: list[str] = []

    def walk(node: Any, path: str, depth: int) -> None:
        if depth > _MAX_DEPTH:
            problems.append(f"{path}: nested deeper than {_MAX_DEPTH}")
            return
        if not isinstance(node, dict):
            problems.append(f"{path}: a schema must be an object")
            return
        unsupported.update(k for k in node if k not in SUPPORTED_KEYWORDS)
        t = node.get("type")
        types = t if isinstance(t, list) else [t] if t is not None else []
        if any(x not in _TYPES for x in types):
            problems.append(f"{path}.type: must be one of {sorted(_TYPES)}")
        props = node.get("properties")
        if props is not None:
            if not isinstance(props, dict):
                problems.append(f"{path}.properties: must be an object")
            else:
                for k, sub in props.items():
                    walk(sub, f"{path}.properties.{k}", depth + 1)
        req = node.get("required")
        if req is not None and not (isinstance(req, list) and all(isinstance(r, str) for r in req)):
            problems.append(f"{path}.required: must be a list of names")
        if "items" in node:
            walk(node["items"], f"{path}.items", depth + 1)
        extra = node.get("additionalProperties")
        if isinstance(extra, dict):
            walk(extra, f"{path}.additionalProperties", depth + 1)
        elif extra is not None and not isinstance(extra, bool):
            problems.append(f"{path}.additionalProperties: must be a boolean or a schema")
        if "enum" in node and not (isinstance(node["enum"], list) and node["enum"]):
            problems.append(f"{path}.enum: must be a non-empty list")

    walk(schema, "$", 0)
    if unsupported:
        raise InvalidParameterError(
            f"{parameter}: schema keywords not supported: {sorted(unsupported)} "
            f"(supported: {sorted(SUPPORTED_KEYWORDS)})", parameter=parameter, unsupported=sorted(unsupported))
    if problems:
        raise InvalidParameterError(f"{parameter}: invalid schema: {'; '.join(problems[:5])}",
                                    parameter=parameter)


def validate(value: Any, schema: dict[str, Any], path: str = "$") -> list[str]:
    """value 違反 schema 的地方（最多 _MAX_ERRORS 筆）；空 list = 通過。"""
    errors: list[str] = []
    _validate(value, schema, path, errors)
    return errors[:_MAX_ERRORS]


def _validate(v: Any, s: dict[str, Any], path: str, errors: list[str]) -> None:
    if len(errors) >= _MAX_ERRORS:
        return
    t = s.get("type")
    if t is not None:
        types = t if isinstance(t, list) else [t]
        if not any(_is_type(v, x) for x in types):
            errors.append(f"{path}: expected {' or '.join(types)}, got {_type_name(v)}")
            return
    if "const" in s and v != s["const"]:
        errors.append(f"{path}: must be {json.dumps(s['const'], ensure_ascii=False)}")
    if "enum" in s and v not in s["enum"]:
        errors.append(f"{path}: must be one of {json.dumps(s['enum'], ensure_ascii=False)}")
    if _is_type(v, "number"):
        for key, ok in (("minimum", lambda a, b: a >= b), ("maximum", lambda a, b: a <= b),
                        ("exclusiveMinimum", lambda a, b: a > b), ("exclusiveMaximum", lambda a, b: a < b)):
            if key in s and not ok(v, s[key]):
                errors.append(f"{path}: {key} {s[key]}, got {v}")
    if isinstance(v, str):
        if "minLength" in s and len(v) < s["minLength"]:
            errors.append(f"{path}: shorter than {s['minLength']}")
        if "maxLength" in s and len(v) > s["maxLength"]:
            errors.append(f"{path}: longer than {s['maxLength']}")
    if isinstance(v, list):
        if "minItems" in s and len(v) < s["minItems"]:
            errors.append(f"{path}: fewer than {s['minItems']} items")
        if "maxItems" in s and len(v) > s["maxItems"]:
            errors.append(f"{path}: more than {s['maxItems']} items")
        if isinstance(s.get("items"), dict):
            for i, item in enumerate(v):
                _validate(item, s["items"], f"{path}[{i}]", errors)
    if isinstance(v, dict):
        props = s.get("properties") or {}
        for name in s.get("required") or []:
            if name not in v:
                errors.append(f"{path}: missing required '{name}'")
        extra = s.get("additionalProperties", True)
        for k, item in v.items():
            if k in props:
                _validate(item, props[k], f"{path}.{k}", errors)
            elif extra is False:
                errors.append(f"{path}: unexpected property '{k}'")
            elif isinstance(extra, dict):
                _validate(item, extra, f"{path}.{k}", errors)


def _is_type(v: Any, t: str) -> bool:
    if t == "null":
        return v is None
    if t == "boolean":
        return isinstance(v, bool)
    if t == "integer":
        return (isinstance(v, int) and not isinstance(v, bool)) or (isinstance(v, float) and v.is_integer())
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    if t == "string":
        return isinstance(v, str)
    if t == "array":
        return isinstance(v, list)
    if t == "object":
        return isinstance(v, dict)
    return False


def _type_name(v: Any) -> str:
    for t in ("null", "boolean", "integer", "number", "string", "array", "object"):
        if _is_type(v, t):
            return t
    return type(v).__name__


_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def parse_output(raw: str) -> Any:
    """模型輸出 → JSON 值。依序嘗試：原文、```fence 內、最外層的 {...} / [...]。
    都不是 JSON 就拋 StructuredOutputError —— 不猜。"""
    text = _THINK.sub("", raw or "").strip()
    attempts = [text]
    fenced = _FENCE.search(text)
    if fenced:
        attempts.append(fenced.group(1).strip())
    for open_, close in (("{", "}"), ("[", "]")):
        start, end = text.find(open_), text.rfind(close)
        if 0 <= start < end:
            attempts.append(text[start:end + 1])
    for a in attempts:
        try:
            return json.loads(a, strict=False)
        except ValueError:
            continue
    raise StructuredOutputError("output is not JSON", violations=["$: not JSON"])
