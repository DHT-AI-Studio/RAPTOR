"""Canonical mapping between the four /policy/check/llm/* content targets:
the upload dropdown name (app/routers/guardrail_policies.py), the
guardrail_policies column that stores it (app/db/schema.py), and the
guard-adapter family it applies to (app/adapters/models/*.py's `.family`)
when check_policy_llm.py picks a model-specific policy prompt.

Adding a new guard model that needs its own policy prompt:
  1. add its adapter under app/adapters/models/ (see that package's docs)
  2. add one `content_<name> TEXT` column to guardrail_policies in app/db/schema.py
  3. add the matching field to PolicyDetail in app/engine/models.py
  4. add ONE TargetInfo entry below

Nothing else needs editing — the upload dropdown, the (name, version, target)
upsert, and check_policy_llm.py's per-model dispatch all derive from this
one table.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

PolicyTarget = Literal["original", "llama-guard", "granite-guardian", "gpt-oss-safeguard"]


@dataclass(frozen=True)
class TargetInfo:
    dropdown_name: str          # value offered in the POST /guardrail/policies `target` dropdown
    column: str                  # guardrail_policies column that stores this target's content
    adapter_family: str | None   # BaseGuardAdapter.family this applies to (None for "original")


TARGETS: tuple[TargetInfo, ...] = (
    TargetInfo(dropdown_name="original", column="raw_content", adapter_family=None),
    TargetInfo(dropdown_name="llama-guard", column="content_llama_guard", adapter_family="llama_guard3"),
    TargetInfo(dropdown_name="granite-guardian", column="content_granite_guardian", adapter_family="granite"),
    TargetInfo(dropdown_name="gpt-oss-safeguard", column="content_gpt_oss_safeguard", adapter_family="gpt_oss"),
)

COLUMN_BY_DROPDOWN_NAME: dict[str, str] = {t.dropdown_name: t.column for t in TARGETS}
COLUMN_BY_ADAPTER_FAMILY: dict[str, str] = {t.adapter_family: t.column for t in TARGETS if t.adapter_family is not None}
CONTENT_COLUMNS: tuple[str, ...] = tuple(t.column for t in TARGETS)              # all 4, incl. raw_content
