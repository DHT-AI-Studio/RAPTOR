# tests/unit/test_structured_ac4.py
"""
AC4 unit test: OutputContract.check() 對違規輸出拋 StructuredOutputError。
不需要 Ollama、不需要 Module 07 服務。
"""
from __future__ import annotations

import json

import pytest
from src.inference.exceptions import StructuredOutputError
from src.inference.structured import OutputContract, output_contract

TEMPORAL_FACT_SCHEMA = {
    "type": "object",
    "properties": {
        "subject":    {"type": "string"},
        "predicate":  {"type": "string"},
        "object":     {"type": "string"},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
    },
    "required": ["subject", "predicate", "object", "confidence"],
    "additionalProperties": False,
}

RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {"name": "TemporalFact", "schema": TEMPORAL_FACT_SCHEMA},
}


class TestAC4OutputValidation:
    """AC4: 違規輸出絕不通過，一定拋 StructuredOutputError。"""

    def _contract(self) -> OutputContract:
        return output_contract(RESPONSE_FORMAT)

    def test_missing_required_field_raises(self):
        """缺少 required 欄位 → StructuredOutputError。"""
        bad_output = json.dumps({"subject": "Marie Curie", "predicate": "won"})
        # 缺 object 和 confidence
        contract = self._contract()
        with pytest.raises(StructuredOutputError) as exc_info:
            contract.check(bad_output)
        violations = " ".join(exc_info.value.details.get("violations", []))
        assert "object" in violations or "confidence" in violations

    def test_wrong_type_raises(self):
        """confidence 應為 number，給 string → StructuredOutputError。"""
        bad_output = json.dumps({
            "subject": "Marie Curie",
            "predicate": "won",
            "object": "Nobel Prize",
            "confidence": "high",   # 應為 number
        })
        contract = self._contract()
        with pytest.raises(StructuredOutputError):
            contract.check(bad_output)

    def test_additional_property_raises(self):
        """additionalProperties: false，多了 extra_field → StructuredOutputError。"""
        bad_output = json.dumps({
            "subject": "Marie Curie",
            "predicate": "won",
            "object": "Nobel Prize",
            "confidence": 0.99,
            "extra_field": "should not be here",
        })
        contract = self._contract()
        with pytest.raises(StructuredOutputError):
            contract.check(bad_output)

    def test_not_json_raises(self):
        """純文字（不是 JSON）→ StructuredOutputError。"""
        contract = self._contract()
        with pytest.raises(StructuredOutputError):
            contract.check("Marie Curie won the Nobel Prize in 1903.")

    def test_valid_output_passes(self):
        """正確輸出不應拋例外，sanity check。"""
        good_output = json.dumps({
            "subject": "Marie Curie",
            "predicate": "won",
            "object": "Nobel Prize in Physics",
            "confidence": 0.99,
        })
        contract = self._contract()
        result = contract.check(good_output)
        assert result["subject"] == "Marie Curie"
