"""Unit tests for the structured GuardrailPolicy model (GB-2/GB-4)."""
import pytest

from app.models.policy import GuardrailPolicy, PolicyRule

_YAML = """
name: "Test Policy"
version: "1.0"
scope:
  modules: ["07", "15"]
input_guardrail:
  rules:
    - category: "S1"
      action: "block"
    - category: "jailbreak"
      action: "block"
      detector: "regex"
      patterns:
        - "ignore .* instructions"
output_guardrail:
  rules:
    - category: "S7"
      action: "redact"
"""


def test_parse_yaml_into_policy():
    p = GuardrailPolicy.from_raw(_YAML)
    assert p.name == "Test Policy"
    assert p.scope.modules == ["07", "15"]
    assert len(p.input_guardrail.rules) == 2
    # default detector is llama_guard; explicit regex rule keeps its detector
    assert p.input_guardrail.rules[0].detector == "llama_guard"
    assert p.input_guardrail.rules[1].detector == "regex"
    assert p.output_guardrail.rules[0].action == "redact"


def test_scope_applies_to():
    p = GuardrailPolicy.from_raw(_YAML)
    assert p.scope.applies_to("07") is True
    assert p.scope.applies_to("99") is False
    # wildcard scope matches everything
    assert GuardrailScope_wildcard().applies_to("anything") is True


def GuardrailScope_wildcard():
    from app.models.policy import GuardrailScope
    return GuardrailScope()          # defaults to ["*"]


def test_regex_rule_without_patterns_is_rejected():
    with pytest.raises(ValueError):
        PolicyRule(category="jailbreak", action="block", detector="regex")   # no patterns


def test_block_for_direction():
    p = GuardrailPolicy.from_raw(_YAML)
    assert p.block_for("input") is p.input_guardrail
    assert p.block_for("output") is p.output_guardrail
