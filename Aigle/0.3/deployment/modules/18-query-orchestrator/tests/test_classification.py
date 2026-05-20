"""
單元測試 - 根據 Video_Search_Sample_Question_1.0.pdf 的 26 個查詢樣本

測試覆蓋四條查詢路徑：
- VideoRAG (找片段): 測試 1-6, 21-22
- RDBMS (找文件): 測試 7-10
- GraphRAG (問關係): 測試 11-15, 24-25
- TKG (問時序): 測試 16-20, 26
"""
from __future__ import annotations

import pytest

from app.services.signal_extractor import extract_signals, QuerySignals
from app.services.intent_classifier import IntentClassifierService, _rule_based_classify
from app.models.classification import ClassificationResult


# ===========================================================================
# 輔助函式
# ===========================================================================

def _classify(query: str) -> ClassificationResult:
    """輔助函式：分類查詢"""
    classifier = IntentClassifierService()
    return classifier.classify(query)


def _assert_pipeline(query: str, expected_pipeline: str, description: str = ""):
    """輔助函式：檢查分類結果是否包含預期管道"""
    result = _classify(query)
    msg = f"Query: {query} ({description})"
    assert expected_pipeline in result.intent_type, f"{msg} - Expected {expected_pipeline}, got {result.intent_type}"
    return result


# ===========================================================================
# VideoRAG 分類測試（找片段）
# ===========================================================================

class TestVideoRAGClassification:
    """VideoRAG 分類測試 - 找片段"""

    def test_01_visual_scene(self):
        """測試 1: 視覺場景查詢 - 大雨中的大笨鐘"""
        result = _assert_pipeline(
            "大雨中的大笨鐘",
            "VideoRAG",
            "視覺場景",
        )
        assert result.routing_strategy in ("qic", "rule_based")

    def test_02_recurring_event(self):
        """測試 2: 重複事件查詢 - 之前兩次媽祖遶境經過清水的片段"""
        result = _assert_pipeline(
            "之前兩次媽祖遶境經過清水的片段",
            "VideoRAG",
            "重複事件",
        )

    def test_03_artist_music(self):
        """測試 3: 藝術家/音樂查詢 - 五月天最紅的歌曲表演片段"""
        result = _assert_pipeline(
            "五月天最紅的歌曲表演片段",
            "VideoRAG",
            "藝術家/音樂",
        )

    def test_04_fictional_scene(self):
        """測試 4: 虛構場景查詢 - 白雪公主和七個小矮人一起唱歌的場景"""
        result = _assert_pipeline(
            "白雪公主和七個小矮人一起唱歌的場景",
            "VideoRAG",
            "虛構場景",
        )

    def test_05_documentary_location(self):
        """測試 5: 紀錄片/地點查詢 - 南極大陸探險紀錄片中登陸海岸的畫面"""
        result = _assert_pipeline(
            "南極大陸探險紀錄片中登陸海岸的畫面",
            "VideoRAG",
            "紀錄片/地點",
        )

    def test_06_sports_time_filter(self):
        """測試 6: 體育/時間篩選 - NBA 2024 總決賽最後 30 秒的關鍵進球"""
        result = _assert_pipeline(
            "NBA 2024 總決賽最後 30 秒的關鍵進球",
            "VideoRAG",
            "體育/時間篩選",
        )
        # 檢查是否提取了時間訊號
        signals = extract_signals("NBA 2024 總決賽最後 30 秒的關鍵進球")
        assert signals.rel_position == "end"
        assert signals.tail_seconds == 30

    def test_21_fictional_advanced(self):
        """測試 21: 虛構場景查詢（進階）- 一個矮人把魔法戒指丟進火山"""
        result = _assert_pipeline(
            "一個矮人把魔法戒指丟進火山",
            "VideoRAG",
            "虛構場景（進階）",
        )

    def test_22_weather_advanced(self):
        """測試 22: 天氣場景查詢（進階）- 所有在暴風雪中拍攝的戶外直播片段"""
        result = _assert_pipeline(
            "所有在暴風雪中拍攝的戶外直播片段",
            "VideoRAG",
            "天氣場景（進階）",
        )


# ===========================================================================
# RDBMS 分類測試（找文件）
# ===========================================================================

class TestRDBMSClassification:
    """RDBMS 分類測試 - 找文件"""

    def test_07_keyword_search(self):
        """測試 7: 關鍵字搜尋 - 所有包含 GPU 散熱模組拆解教學的產品影片"""
        result = _assert_pipeline(
            "所有包含 GPU 散熱模組拆解教學的產品影片",
            "RDBMS",
            "關鍵字搜尋",
        )
        assert result.routing_strategy in ("qic", "rule_based")

    def test_08_date_filter(self):
        """測試 8: 日期篩選 - 2020 年總統大選開票夜各家電視台的新聞報導"""
        result = _assert_pipeline(
            "2020 年總統大選開票夜各家電視台的新聞報導",
            "RDBMS",
            "日期篩選",
        )
        # 檢查是否提取了時間範圍
        signals = extract_signals("2020 年總統大選開票夜各家電視台的新聞報導")
        assert signals.time_from is not None
        assert signals.time_to is not None
        assert signals.time_from.year == 2020
        assert signals.time_to.year == 2020

    def test_09_document_list(self):
        """測試 9: 文件清單查詢 - 所有提到量子糾纏的學術演講影片清單"""
        result = _assert_pipeline(
            "所有提到量子糾纏的學術演講影片清單",
            "RDBMS",
            "文件清單",
        )

    def test_10_metadata_query(self):
        """測試 10: 元資料查詢 - 產品發表會上 CEO 提到的所有合作夥伴名稱"""
        result = _assert_pipeline(
            "產品發表會上 CEO 提到的所有合作夥伴名稱",
            "RDBMS",
            "元資料查詢",
        )


# ===========================================================================
# GraphRAG 分類測試（問關係）
# ===========================================================================

class TestGraphRAGClassification:
    """GraphRAG 分類測試 - 問關係"""

    def test_11_entity_relation(self):
        """測試 11: 歷史實體關係 - 十九世紀二條城本丸御殿的歷史脈絡與相關影片"""
        result = _assert_pipeline(
            "十九世紀二條城本丸御殿的歷史脈絡與相關影片",
            "GraphRAG",
            "歷史實體關係",
        )

    def test_12_competitor_relation(self):
        """測試 12: 競爭關係 - 受訪者提到的公司與其競爭對手之間的關係"""
        result = _assert_pipeline(
            "受訪者提到的公司與其競爭對手之間的關係",
            "GraphRAG",
            "競爭關係",
        )
        assert result.routing_strategy in ("qic", "rule_based")

    def test_13_co_occurrence(self):
        """測試 13: 共同出現 - 這位導演執導過哪些作品？有哪些共同演員？"""
        result = _assert_pipeline(
            "這位導演執導過哪些作品？有哪些共同演員？",
            "GraphRAG",
            "共同出現",
        )

    def test_14_cross_film(self):
        """測試 14: 跨影片關係 - 這首曲子在哪些影片中被使用，分別是什麼情境？"""
        result = _assert_pipeline(
            "這首曲子在哪些影片中被使用，分別是什麼情境？",
            "GraphRAG",
            "跨影片關係",
        )

    def test_15_topic_cooccurrence(self):
        """測試 15: 主題共現 - 台積電在各新聞報導中被提及時，通常與哪些主題相關？"""
        result = _assert_pipeline(
            "台積電在各新聞報導中被提及時，通常與哪些主題相關？",
            "GraphRAG",
            "主題共現",
        )
        signals = extract_signals("台積電在各新聞報導中被提及時，通常與哪些主題相關？")
        assert signals.hint_graphrag is True

    def test_24_cross_film_advanced(self):
        """測試 24: 跨電影關係查詢（進階）- 鋼鐵人第一次穿上盔甲的所有版本片段"""
        result = _assert_pipeline(
            "鋼鐵人第一次穿上盔甲的所有版本片段",
            "GraphRAG",
            "跨電影關係（進階）",
        )

    def test_25_co_occurrence_advanced(self):
        """測試 25: 共同出現+關係查詢（進階）- 賈伯斯與比爾蓋茲在公開場合同台的所有影像，以及他們之間的商業恩怨脈絡"""
        result = _assert_pipeline(
            "賈伯斯與比爾蓋茲在公開場合同台的所有影像，以及他們之間的商業恩怨脈絡",
            "GraphRAG",
            "共同出現+關係（進階）",
        )


# ===========================================================================
# TKG 分類測試（問時序）
# ===========================================================================

class TestTKGClassification:
    """TKG 分類測試 - 問時序"""

    def test_16_timeline(self):
        """測試 16: 時間軸 - iPhone 從發表到現在各代的演進順序與關鍵功能變化"""
        result = _assert_pipeline(
            "iPhone 從發表到現在各代的演進順序與關鍵功能變化",
            "TKG",
            "時間軸",
        )
        assert result.routing_strategy in ("qic", "rule_based")
        signals = extract_signals("iPhone 從發表到現在各代的演進順序與關鍵功能變化")
        assert signals.hint_tkg is True

    def test_17_temporal_order(self):
        """測試 17: 時間順序 - COVID-19 疫情期間各國封城措施的時間順序"""
        result = _assert_pipeline(
            "COVID-19 疫情期間各國封城措施的時間順序",
            "TKG",
            "時間順序",
        )
        signals = extract_signals("COVID-19 疫情期間各國封城措施的時間順序")
        assert signals.hint_tkg is True

    def test_18_event_chain(self):
        """測試 18: 事件鏈 - 這場訴訟從起訴、審判到判決的完整時間軸"""
        result = _assert_pipeline(
            "這場訴訟從起訴、審判到判決的完整時間軸",
            "TKG",
            "事件鏈",
        )
        signals = extract_signals("這場訴訟從起訴、審判到判決的完整時間軸")
        assert signals.hint_tkg is True

    def test_19_before_after(self):
        """測試 19: 前後關係 - AWS 公司宣布裁員之前發生了哪些重要事件？"""
        result = _assert_pipeline(
            "AWS 公司宣布裁員之前發生了哪些重要事件？",
            "TKG",
            "前後關係",
        )
        signals = extract_signals("AWS 公司宣布裁員之前發生了哪些重要事件？")
        assert signals.hint_tkg is True

    def test_20_during(self):
        """測試 20: 期間查詢 - 火箭升空後到太空艙分離期間的所有影片片段，按時間排序"""
        result = _assert_pipeline(
            "火箭升空後到太空艙分離期間的所有影片片段，按時間排序",
            "TKG",
            "期間查詢",
        )
        signals = extract_signals("火箭升空後到太空艙分離期間的所有影片片段，按時間排序")
        assert signals.hint_tkg is True

    def test_26_capability_evolution_advanced(self):
        """測試 26: 能力躍升查詢（進階）- GPT 系列模型從 GPT-1 到 GPT-4 的發布時間與能力躍升"""
        result = _assert_pipeline(
            "GPT 系列模型從 GPT-1 到 GPT-4 的發布時間與能力躍升",
            "TKG",
            "能力躍升（進階）",
        )
        signals = extract_signals("GPT 系列模型從 GPT-1 到 GPT-4 的發布時間與能力躍升")
        assert signals.hint_tkg is True


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
        pipelines, confidence, patterns, rules = _rule_based_classify("你好嗎？")
        assert len(pipelines) == 0

    def test_multi_intent(self):
        """測試：多意圖查詢"""
        # 包含關係和時間線提示的查詢
        pipelines, confidence, patterns, rules = _rule_based_classify(
            "人物關係和時間順序"
        )
        # 應該匹配到多個管道
        assert len(pipelines) >= 1

    def test_confidence_range(self):
        """測試：信心分數範圍"""
        pipelines, confidence, _, _ = _rule_based_classify("大雨中的大笨鐘")
        assert 0.0 <= confidence <= 1.0

    def test_matched_rules_returned(self):
        """測試：返回匹配的規則 ID"""
        _, _, _, matched_rules = _rule_based_classify("大雨中的大笨鐘")
        assert len(matched_rules) > 0
        # matched_rules 是 dict 列表；rule_id 在 rule["rule"].rule_id
        assert any("vr_visual" in r["rule"].rule_id for r in matched_rules)


# ===========================================================================
# Edge Case 測試
# ===========================================================================

class TestEdgeCases:
    """邊界情況測試"""

    def test_empty_query(self):
        """測試：空查詢"""
        with pytest.raises(Exception):
            _classify("")

    def test_single_char_query(self):
        """測試：單字查詢"""
        result = _classify("影")
        # 單字應該 fallback 到 VideoRAG
        assert "VideoRAG" in result.intent_type

    def test_mixed_language_query(self):
        """測試：混合語言查詢"""
        result = _classify("GPT-1 到 GPT-4 的關係和時間軸")
        # 應該匹配到 TKG（因為有時間線提示）
        assert "TKG" in result.intent_type

    def test_long_query(self):
        """測試：長查詢"""
        long_query = "請幫我找到所有關於人工智能在醫療領域應用的影片，包括診斷、治療、藥物開發、基因體學等各個方面，並且要按照時間順序排列，列出前 10 個最相關的片段。"
        result = _classify(long_query)
        # 應該匹配到 RDBMS（因為有清單、關鍵字提示）或 VideoRAG
        assert len(result.intent_type) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
