"""Unit tests for app/adapters/standard_policy.py — the Standard Policy
Format parser. Must never raise: anything that isn't valid standard-format
JSON returns None so callers fall back to verbatim/legacy behavior."""
import json

from app.adapters.standard_policy import StandardPolicy, parse_standard_policies

_VALID_ARRAY = json.dumps([
    {
        "id": "M1", "name": "Medical Misinformation",
        "description": "Dangerous, unproven medical claims presented as fact.",
        "severity": "high", "decision": "block",
        "criteria": ["Claims a substance cures a disease without evidence."],
        "exceptions": ["Academic discussion of debunked claims."],
        "examples": {
            "violation": ["Drinking bleach cures viral infections."],
            "allowed": ["Bleach is dangerous if ingested."],
        },
    },
    {
        "id": "F2", "name": "Financial Fraud",
        "description": "Content promoting scams or fraudulent schemes.",
        "severity": "critical", "decision": "block",
        "criteria": ["Promotes a guaranteed-return investment scheme."],
    },
])


def test_parses_valid_standard_format_array():
    result = parse_standard_policies(_VALID_ARRAY)
    assert result is not None
    assert len(result) == 2
    assert result[0].id == "M1"
    assert result[0].examples.violation == ["Drinking bleach cures viral infections."]
    assert result[1].id == "F2"
    assert result[1].exceptions == []          # optional field defaults to empty
    assert result[1].examples.violation == []  # optional nested field defaults to empty


def test_returns_none_for_malformed_json():
    assert parse_standard_policies("{not valid json") is None


def test_returns_none_for_plain_text():
    assert parse_standard_policies("Block anything about weapons. Escalate suspicious intent.") is None


def test_returns_none_for_single_object_not_array():
    single = json.dumps({
        "id": "M1", "name": "n", "description": "d",
        "severity": "low", "decision": "allow", "criteria": ["c"],
    })
    assert parse_standard_policies(single) is None


def test_returns_none_for_empty_array():
    assert parse_standard_policies("[]") is None


def test_returns_none_when_required_field_missing():
    missing_criteria = json.dumps([{
        "id": "M1", "name": "n", "description": "d", "severity": "low", "decision": "allow",
    }])
    assert parse_standard_policies(missing_criteria) is None


def test_returns_none_for_invalid_enum_value():
    bad_severity = json.dumps([{
        "id": "M1", "name": "n", "description": "d",
        "severity": "extreme", "decision": "allow", "criteria": ["c"],
    }])
    assert parse_standard_policies(bad_severity) is None


def test_returns_none_for_yaml_that_is_not_json():
    # legacy policies are often YAML — must not accidentally validate as JSON
    assert parse_standard_policies("name: my-policy\nrules:\n  - id: R1\n") is None


def test_optional_fields_default_correctly():
    minimal = json.dumps([{
        "id": "H1", "name": "Hate", "description": "d",
        "severity": "medium", "decision": "warn", "criteria": ["c1", "c2"],
    }])
    result = parse_standard_policies(minimal)
    assert result == [StandardPolicy(
        id="H1", name="Hate", description="d", severity="medium", decision="warn",
        criteria=["c1", "c2"],
    )]
