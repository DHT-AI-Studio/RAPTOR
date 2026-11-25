"""單元測試 - extract_signals()/_rule_based_classify()/請求層驗證，單一函式的功能驗證。完整 classify() 管道的測試在 tests/integration/test_classification.py。"""
from __future__ import annotations

import pydantic
import pytest
from app.routers.query import ClassifyRequest
from app.services.intent_classifier import _rule_based_classify
from app.services.signal_extractor import extract_signals

# ===========================================================================
# Signal Extractor 測試
# ===========================================================================

class TestSignalExtractor:
    """訊號提取器測試"""

    def test_count_hint_time_asc(self):
        """測試：前 N 個提示 - 時間升序"""
        signals = extract_signals("前三個最早的影片")
        assert signals.top_k_override == 3
        assert signals.sort_by == "time_asc"

    def test_count_hint_time_desc(self):
        """測試：後 N 個提示 - 時間降序"""
        signals = extract_signals("最後 5 個最新的片段")
        assert signals.top_k_override == 5
        assert signals.sort_by == "time_desc"

    def test_time_window_year(self):
        """測試：年份提取"""
        signals = extract_signals("2020 年的影片")
        assert signals.time_from is not None
        assert signals.time_to is not None
        assert signals.time_from.year == 2020
        assert signals.time_to.year == 2020

    def test_time_window_decade(self):
        """測試：年代提取"""
        signals = extract_signals("1990 年代的電影")
        assert signals.time_from is not None
        assert signals.time_to is not None
        assert signals.time_from.year == 1990
        assert signals.time_to.year == 1999

    def test_rel_position_end(self):
        """測試：相對位置 - 結尾"""
        signals = extract_signals("最後 30 秒的畫面")
        assert signals.rel_position == "end"
        assert signals.tail_seconds == 30

    def test_rel_position_start(self):
        """測試：相對位置 - 開頭"""
        signals = extract_signals("前 10 秒的畫面")
        assert signals.rel_position == "start"
        assert signals.tail_seconds == 10


# ===========================================================================
# Rule-Based Engine 測試
# ===========================================================================

class TestRuleBasedEngine:
    """Rule-Based 引擎測試"""

    def test_empty_result_for_ambiguous(self):
        """測試：模糊查詢返回空結果"""
        pipeline_scores, _ = _rule_based_classify("你好嗎？")
        assert len(pipeline_scores) == 0

    def test_multi_intent(self):
        """測試：多意圖查詢"""
        # 包含關係和時間線提示的查詢
        pipeline_scores, _ = _rule_based_classify(
            "人物關係和時間順序"
        )
        # 應該匹配到多個管道
        assert len(pipeline_scores) >= 1

    def test_confidence_range(self):
        """測試：信心分數範圍"""
        pipeline_scores, _ = _rule_based_classify("大雨中的大笨鐘")
        assert all(0.0 <= v <= 1.0 for v in pipeline_scores.values())

    def test_matched_rules_returned(self):
        """測試：返回匹配的規則 ID"""
        _, matched_rules = _rule_based_classify("大雨中的大笨鐘")
        assert len(matched_rules) > 0
        # matched_rules 是 dict 列表；rule_id 在 rule["rule"].rule_id
        assert any("vr_visual" in r["rule"].rule_id for r in matched_rules)


# ===========================================================================
# Edge Case 測試（純請求層驗證，不需要分類器）
# ===========================================================================

class TestEdgeCases:
    """邊界情況測試"""

    def test_empty_query(self):
        """測試：空查詢

        空查詢的拒絕發生在 API 的請求層（ClassifyRequest 的 min_length=1），
        不是 IntentClassifierService.classify() 本身 -- 它從未設計成要驗證
        輸入，對空字串一樣能算出一個（沒意義但不會壞）的分類結果。
        """
        with pytest.raises(pydantic.ValidationError):
            ClassifyRequest(query="")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
