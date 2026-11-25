# src/inference/registry.py
"""
MLflow 模型版本解析（推論與 model-info 共用）

resolve_model_version(client, name, version=None, alias=None) → {"version", "tags", "aliases"}

    - 未給 selector：沿用既有「最大版本號」政策（不默默改用 production）
    - model_version：精確版本；不存在 → ModelVersionNotFoundError（不回退）
    - model_alias  ：於本次呼叫解析為固定版本；未設定 → ModelAliasNotFoundError（不回退）
    - 模型不存在    → ModelNotFoundError
    - registry 連線／伺服器錯誤 → RegistryUnavailableError（不當作模型不存在）

client 只需要 MlflowClient 的 get_registered_model / get_model_version /
search_model_versions 三個方法，測試可注入 fake。
"""

from __future__ import annotations

from typing import Any

from .exceptions import (
    ConflictingParametersError,
    InvalidParameterError,
    ModelAliasNotFoundError,
    ModelNotFoundError,
    ModelVersionNotFoundError,
    RegistryUnavailableError,
)

# INVALID_PARAMETER_VALUE：MLflow 新版（如 3.16）在查詢時就驗證名稱，含 ':' 或 '/' 的名稱
# （例如 Ollama 原生 tag "gemma4:12b"）直接回此錯誤而非 RESOURCE_DOES_NOT_EXIST；
# 這種名稱不可能存在於 registry，視同不存在（engine="ollama" fallback 仍可用）。
# 版本號已先在 validate_selector 檢查過，不會因版本格式落到這裡。
_NOT_FOUND = ("RESOURCE_DOES_NOT_EXIST", "INVALID_PARAMETER_VALUE")


def validate_selector(model_version: str | None, model_alias: str | None) -> None:
    """model_version 與 model_alias 互斥；版本須為正整數字串。"""
    if model_version is not None and model_alias is not None:
        raise ConflictingParametersError(
            "model_version and model_alias are mutually exclusive; send only one of them",
            parameters=["model_version", "model_alias"],
        )
    if model_version is not None and not str(model_version).strip().isdigit():
        raise InvalidParameterError(
            f"model_version must be a numeric MLflow version, got {model_version!r}", parameter="model_version"
        )
    if model_alias is not None and not str(model_alias).strip():
        raise InvalidParameterError("model_alias must not be empty", parameter="model_alias")


def resolve_model_version(
    client: Any,
    model_name: str,
    model_version: str | None = None,
    model_alias: str | None = None,
) -> dict[str, Any]:
    validate_selector(model_version, model_alias)

    rm = _call(client.get_registered_model, model_name, not_found=ModelNotFoundError(
        f"model '{model_name}' not in MLflow registry"
    ))

    if model_alias is not None:
        alias = str(model_alias).strip()
        aliases = _aliases_of(rm)
        if alias not in aliases:
            raise ModelAliasNotFoundError(
                f"alias '{alias}' is not set on model '{model_name}'", model_name=model_name, model_alias=alias
            )
        version = str(aliases[alias])
    elif model_version is not None:
        version = str(model_version).strip()
    else:
        versions = _call(client.search_model_versions, f"name='{_quote(model_name)}'")
        numbers = [int(mv.version) for mv in versions if str(getattr(mv, "version", "")).isdigit()]
        if not numbers:
            raise ModelNotFoundError(f"model '{model_name}' has no versions in MLflow registry")
        version = str(max(numbers))

    mv = _call(client.get_model_version, model_name, version, not_found=ModelVersionNotFoundError(
        f"version '{version}' of model '{model_name}' not found", model_name=model_name, model_version=version
    ))
    return {
        "version": str(mv.version),
        "tags": dict(getattr(mv, "tags", None) or {}),
        "aliases": list(getattr(mv, "aliases", None) or []),
    }


def _call(fn, *args, not_found: Exception | None = None):
    try:
        return fn(*args)
    except Exception as e:
        if not_found is not None and getattr(e, "error_code", None) in _NOT_FOUND:
            raise not_found from e
        raise RegistryUnavailableError(f"MLflow registry request failed: {e}") from e


def _aliases_of(rm: Any) -> dict[str, str]:
    aliases = getattr(rm, "aliases", None) or {}
    if isinstance(aliases, dict):
        return {str(k): str(v) for k, v in aliases.items()}
    # 部分版本以 list[RegisteredModelAlias(alias, version)] 表示
    return {str(a.alias): str(a.version) for a in aliases}


def _quote(name: str) -> str:
    return name.replace("'", "\\'")
