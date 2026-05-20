# 特徵工程設計文件

## 1. 現有特徵分析

### 1.1 目前特徵清單（35 個特徵）

| 分類 | 特徵名稱 | 說明 | 類型 | 價值評估 |
|------|---------|------|------|---------|
| **BM25 統計** | `bm25_max` | BM25 最高分 | float | ⭐⭐⭐⭐⭐ |
| | `bm25_mean` | BM25 平均分 | float | ⭐⭐⭐ |
| | `bm25_min` | BM25 最低分 | float | ⭐⭐ |
| | `bm25_std` | BM25 標準差 | float | ⭐⭐ |
| **BM25 排序** | `bm25_top1` | BM25 top-1 分數 | float | ⭐⭐⭐⭐⭐ |
| | `bm25_top2` | BM25 top-2 分數 | float | ⭐⭐⭐⭐ |
| | `bm25_max_diff` | top-1 - top-2 差距 | float | ⭐⭐⭐⭐⭐ |
| | `bm25_top1_idx` | top-1 pipeline 索引 (0-3) | int | ⭐⭐⭐⭐ |
| **FAISS 統計** | `faiss_max` | FAISS 最高分 | float | ⭐⭐⭐⭐⭐ |
| | `faiss_mean` | FAISS 平均分 | float | ⭐⭐⭐ |
| | `faiss_min` | FAISS 最低分 | float | ⭐⭐ |
| | `faiss_std` | FAISS 標準差 | float | ⭐⭐ |
| **FAISS 排序** | `faiss_top1` | FAISS top-1 分數 | float | ⭐⭐⭐⭐⭐ |
| | `faiss_top2` | FAISS top-2 分數 | float | ⭐⭐⭐⭐ |
| | `faiss_max_diff` | top-1 - top-2 差距 | float | ⭐⭐⭐⭐⭐ |
| | `faiss_top1_idx` | top-1 pipeline 索引 (0-3) | int | ⭐⭐⭐⭐ |
| **Rule 統計** | `rule_max` | Rule 最高分 | float | ⭐⭐⭐⭐⭐ |
| | `rule_mean` | Rule 平均分 | float | ⭐⭐⭐ |
| | `rule_min` | Rule 最低分 | float | ⭐⭐ |
| | `rule_std` | Rule 標準差 | float | ⭐⭐ |
| **Rule 排序** | `rule_top1` | Rule top-1 分數 | float | ⭐⭐⭐⭐⭐ |
| | `rule_top2` | Rule top-2 分數 | float | ⭐⭐⭐⭐ |
| | `rule_max_diff` | top-1 - top-2 差距 | float | ⭐⭐⭐⭐⭐ |
| | `rule_top1_idx` | top-1 pipeline 索引 (0-3) | int | ⭐⭐⭐⭐ |
| **跨方法一致性** | `bm25_faiss_gap` | BM25 vs FAISS 差距 | float | ⭐⭐⭐ |
| | `bm25_rule_gap` | BM25 vs Rule 差距 | float | ⭐⭐⭐ |
| | `faiss_rule_gap` | FAISS vs Rule 差距 | float | ⭐⭐⭐ |
| **Query 特徵** | `query_length` | Query 字元數 | int | ⭐⭐ |
| | `num_bm25_pipelines` | BM25 有分數的 pipeline 數 | int | ⭐⭐ |
| | `num_faiss_pipelines` | FAISS 有分數的 pipeline 數 | int | ⭐⭐ |
| | `num_rule_pipelines` | Rule 有分數的 pipeline 數 | int | ⭐⭐ |
| | `num_matched_rules` | 匹配的規則數量 | int | ⭐⭐⭐ |

### 1.2 現有特徵問題分析

#### 問題 1：Classifier 特徵 vs Ranker 特徵不匹配

Classifier 的特徵設計是針對「預測單一 label」，但 Ranker 需要學習「相對排序」。

**Classifier 的特徵問題**：
- `bm25_top1_idx`、`faiss_top1_idx` 等索引特徵在 Ranker 中沒有意義（因為 Ranker 會為每個 pipeline 計算獨立分數）
- `bm25_top1_is_target` 等特徵只能在訓練時使用（需要知道 target），推理時無法計算

**Ranker 需要的特徵**：
- 每個 pipeline 的獨立特徵（讓模型學習哪個 pipeline 更好）
- 跨 pipeline 的相對特徵（讓模型學習差距）
- Query 層級的全域特徵

#### 問題 2：缺少排序相關特徵

現有特徵缺少：
- **排名特徵**：每個 pipeline 在各自方法中的排名
- **分佈特徵**：分數的熵、偏度、峰度
- **一致性特徵**：各方法對 top-k 的同意度

#### 問題 3：缺少語義特徵

- Query 長度太簡單，沒有考慮詞頻、疑問詞等
- 沒有考慮 query 的複雜度（如是否包含時間詞、關係詞等）

---

## 2. Ranker 特徵設計

### 2.1 設計原則

對於 Ranker，每個 query 會被展開為 4 個樣本（每個 pipeline 一個），因此特徵需要分為兩類：

1. **Query-level 特徵**：對所有 4 個樣本相同
2. **Pipeline-level 特徵**：對每個樣本不同

```
Query: "檢索影片中所有提到『生成式 AI』的語義片段。"
┌─────────────────────────────────────────────────┐
│ Query-level 特徵（4 個樣本共用）                  │
│ - query_length, query_word_count, ...           │
├─────────────────────────────────────────────────┤
│ Sample 1 (VideoRAG):                            │
│   Pipeline-level 特徵：                          │
│   - bm25_score_vr, faiss_score_vr, rule_score_vr │
│   - bm25_rank_vr, faiss_rank_vr, rule_rank_vr   │
│   - is_top1_vr, is_target_vr                    │
├─────────────────────────────────────────────────┤
│ Sample 2 (GraphRAG):                            │
│   Pipeline-level 特徵：                          │
│   - bm25_score_gr, faiss_score_gr, rule_score_gr │
│   - bm25_rank_gr, faiss_rank_gr, rule_rank_gr   │
│   - is_top1_gr, is_target_gr                    │
├─────────────────────────────────────────────────┤
│ Sample 3 (TKG): ...                             │
│ Sample 4 (RDBMS): ...                           │
└─────────────────────────────────────────────────┘
```

### 2.2 完整特徵清單

#### A. Query-level 特徵（12 個）

| 特徵名稱 | 類型 | 說明 | 範例值 |
|---------|------|------|--------|
| `query_length` | float | Query 字元數 | 35 |
| `query_word_count` | float | Query 分詞後的詞數 | 8 |
| `query_has_question` | float | 是否包含疑問詞 (0/1) | 1 |
| `query_has_time_expr` | float | 是否包含時間表達 (0/1) | 0 |
| `query_has_relation_word` | float | 是否包含關係詞 (0/1) | 0 |
| `query_has_list_word` | float | 是否包含清單詞 (0/1) | 0 |
| `query_has_visual_word` | float | 是否包含視覺詞 (0/1) | 1 |
| `query_complexity_score` | float | Query 複雜度評分 (0-1) | 0.6 |
| `bm25_max` | float | BM25 最高分 | 12.5 |
| `faiss_max` | float | FAISS 最高分 | 0.85 |
| `rule_max` | float | Rule 最高分 | 1.5 |
| `num_matched_rules` | float | 匹配的規則數量 | 2 |

#### B. Pipeline-level 特徵（每 pipeline 18 個 × 4 pipelines = 72 個）

以 VideoRAG 為例（其他 pipeline 類似）：

| 特徵名稱 | 類型 | 說明 | 範例值 |
|---------|------|------|--------|
| **分數特徵** | | | |
| `bm25_score_vr` | float | BM25 對 VideoRAG 的分數 | 12.5 |
| `faiss_score_vr` | float | FAISS 對 VideoRAG 的分數 | 0.85 |
| `rule_score_vr` | float | Rule 對 VideoRAG 的分數 | 1.5 |
| **排名特徵** | | | |
| `bm25_rank_vr` | float | BM25 中 VideoRAG 的排名 (1-4) | 1 |
| `faiss_rank_vr` | float | FAISS 中 VideoRAG 的排名 (1-4) | 1 |
| `rule_rank_vr` | float | Rule 中 VideoRAG 的排名 (1-4) | 1 |
| **Top-k 特徵** | | | |
| `bm25_top1_vr` | float | BM25 top-1 是否是 VideoRAG (0/1) | 1 |
| `faiss_top1_vr` | float | FAISS top-1 是否是 VideoRAG (0/1) | 1 |
| `rule_top1_vr` | float | Rule top-1 是否是 VideoRAG (0/1) | 1 |
| `bm25_top3_ratio_vr` | float | VideoRAG 在 BM25 top-3 中的佔比 | 1.0 |
| `faiss_top3_ratio_vr` | float | VideoRAG 在 FAISS top-3 中的佔比 | 1.0 |
| `rule_top3_ratio_vr` | float | VideoRAG 在 Rule top-3 中的佔比 | 1.0 |
| **差距特徵** | | | |
| `bm25_gap_vr` | float | BM25 中 top-1 - top-2 差距 | 3.2 |
| `faiss_gap_vr` | float | FAISS 中 top-1 - top-2 差距 | 0.15 |
| `rule_gap_vr` | float | Rule 中 top-1 - top-2 差距 | 0.5 |
| **相對特徵** | | | |
| `bm25_rel_vr` | float | BM25 中 VideoRAG 分數佔比 (self/sum) | 0.45 |
| `faiss_rel_vr` | float | FAISS 中 VideoRAG 分數佔比 | 0.35 |
| `rule_rel_vr` | float | Rule 中 VideoRAG 分數佔比 | 0.40 |

#### C. Cross-pipeline 互動特徵（12 個）

這些特徵在每個 sample 中計算，但會參考其他 pipeline 的分數：

| 特徵名稱 | 類型 | 說明 |
|---------|------|------|
| `agreement_top1` | float | 三個方法 top-1 一致的數量 (0-3) |
| `agreement_top3` | float | 三個方法 top-3 重疊的 pipeline 數量 (0-4) |
| `score_consistency` | float | 三個方法分數標準差的反向指標 |
| `vote_count` | float | 此 pipeline 獲得 top-1 投票的次數 (0-3) |
| `max_min_score_gap` | float | 所有 pipeline 中最高分與最低分的差距 |
| `score_entropy` | float | 所有 pipeline 分數的資訊熵 |
| `is_consensus_winner` | float | 是否為所有方法共識的 top-1 (0/1) |
| `is_disagreement_candidate` | float | 是否在方法間有分歧 (0/1) |
| `rank_agreement` | float | 三個方法排名的標準差反向指標 |
| `top1_confidence` | float | top-1 相對於 top-2 的優勢 |
| `top3_diversity` | float | top-3 中不同 pipeline 的數量 |
| `cross_method_agreement` | float | 跨方法的一致性得分 |

### 2.3 總計

| 類別 | 特徵數量 | 說明 |
|------|---------|------|
| Query-level | 12 | 所有 sample 共用 |
| Pipeline-level | 72 (18 × 4) | 每個 pipeline 獨立 |
| Cross-pipeline | 12 | 參考所有 pipeline |
| **總計** | **96** | |

---

## 3. 特徵計算邏輯詳細設計

### 3.1 Query-level 特徵計算

```python
def extract_query_features(query: str, bm25_scores: dict, faiss_scores: dict, rule_scores: dict) -> dict:
    """提取 Query-level 特徵"""
    
    # 基本特徵
    query_length = len(query)
    tokens = jieba.lcut(query)
    query_word_count = len(tokens)
    
    # 疑問詞檢測
    question_words = re.compile(r'(什麼|怎麼|如何|哪個|哪些|誰|哪段|哪家的)')
    query_has_question = 1.0 if question_words.search(query) else 0.0
    
    # 時間表達檢測
    time_words = re.compile(r'(時間|之前|之後|順序|演進|歷史|timeline|evolution|before|after)')
    query_has_time_expr = 1.0 if time_words.search(query) else 0.0
    
    # 關係詞檢測
    relation_words = re.compile(r'(關係|關聯|合作|競爭|連接|relationship|connection|collaborat|competitor)')
    query_has_relation_word = 1.0 if relation_words.search(query) else 0.0
    
    # 清單詞檢測
    list_words = re.compile(r'(清單|所有|列出|統計|list|count|how many|all|total)')
    query_has_list_word = 1.0 if list_words.search(query) else 0.0
    
    # 視覺詞檢測
    visual_words = re.compile(r'(影片|畫面|影像|片段|視覺|video|clip|footage|visual|scene|frame)')
    query_has_visual_word = 1.0 if visual_words.search(query) else 0.0
    
    # Query 複雜度評分（基於詞數和特殊詞出現）
    query_complexity_score = min(1.0, (query_word_count / 20) * 0.5 + 
                                  (query_has_question + query_has_relation_word + 
                                   query_has_time_expr + query_has_list_word + query_has_visual_word) / 5)
    
    # 各方法最高分
    bm25_max = max(bm25_scores.values()) if bm25_scores else 0.0
    faiss_max = max(faiss_scores.values()) if faiss_scores else 0.0
    rule_max = max(rule_scores.values()) if rule_scores else 0.0
    
    # 匹配的規則數量
    num_matched_rules = len(rule_scores)  # 或從 matched_rules 獲取
    
    return {
        'query_length': query_length,
        'query_word_count': query_word_count,
        'query_has_question': query_has_question,
        'query_has_time_expr': query_has_time_expr,
        'query_has_relation_word': query_has_relation_word,
        'query_has_list_word': query_has_list_word,
        'query_has_visual_word': query_has_visual_word,
        'query_complexity_score': query_complexity_score,
        'bm25_max': bm25_max,
        'faiss_max': faiss_max,
        'rule_max': rule_max,
        'num_matched_rules': num_matched_rules,
    }
```

### 3.2 Pipeline-level 特徵計算

```python
def extract_pipeline_features(pipeline: str, bm25_scores: dict, faiss_scores: dict, rule_scores: dict) -> dict:
    """提取特定 pipeline 的 Pipeline-level 特徵"""
    
    pipelines = ["VideoRAG", "GraphRAG", "TKG", "RDBMS"]
    
    # 分數特徵
    bm25_score = bm25_scores.get(pipeline, 0.0)
    faiss_score = faiss_scores.get(pipeline, 0.0)
    rule_score = rule_scores.get(pipeline, 0.0)
    
    # 計算排名
    bm25_sorted = sorted(bm25_scores.values(), reverse=True)
    faiss_sorted = sorted(faiss_scores.values(), reverse=True)
    rule_sorted = sorted(rule_scores.values(), reverse=True)
    
    bm25_rank = bm25_sorted.index(bm25_score) + 1 if bm25_score > 0 else 5
    faiss_rank = faiss_sorted.index(faiss_score) + 1 if faiss_score > 0 else 5
    rule_rank = rule_sorted.index(rule_score) + 1 if rule_score > 0 else 5
    
    # Top-1 特徵
    bm25_top1_pipeline = max(bm25_scores, key=bm25_scores.get) if bm25_scores else ""
    faiss_top1_pipeline = max(faiss_scores, key=faiss_scores.get) if faiss_scores else ""
    rule_top1_pipeline = max(rule_scores, key=rule_scores.get) if rule_scores else ""
    
    bm25_top1 = 1.0 if bm25_top1_pipeline == pipeline else 0.0
    faiss_top1 = 1.0 if faiss_top1_pipeline == pipeline else 0.0
    rule_top1 = 1.0 if rule_top1_pipeline == pipeline else 0.0
    
    # Top-3 佔比
    bm25_top3 = sorted(bm25_scores.keys(), key=lambda k: bm25_scores[k], reverse=True)[:3]
    faiss_top3 = sorted(faiss_scores.keys(), key=lambda k: faiss_scores[k], reverse=True)[:3]
    rule_top3 = sorted(rule_scores.keys(), key=lambda k: rule_scores[k], reverse=True)[:3]
    
    bm25_top3_ratio = 1.0 if pipeline in bm25_top3 else 0.0
    faiss_top3_ratio = 1.0 if pipeline in faiss_top3 else 0.0
    rule_top3_ratio = 1.0 if pipeline in rule_top3 else 0.0
    
    # 差距特徵（top-1 - top-2）
    bm25_gap = (bm25_sorted[0] - bm25_sorted[1]) if len(bm25_sorted) > 1 else bm25_score
    faiss_gap = (faiss_sorted[0] - faiss_sorted[1]) if len(faiss_sorted) > 1 else faiss_score
    rule_gap = (rule_sorted[0] - rule_sorted[1]) if len(rule_sorted) > 1 else rule_score
    
    # 相對分數（self / sum）
    bm25_total = sum(bm25_scores.values())
    faiss_total = sum(faiss_scores.values())
    rule_total = sum(rule_scores.values())
    
    bm25_rel = bm25_score / bm25_total if bm25_total > 0 else 0.0
    faiss_rel = faiss_score / faiss_total if faiss_total > 0 else 0.0
    rule_rel = rule_score / rule_total if rule_total > 0 else 0.0
    
    return {
        'bm25_score_' + pipeline: bm25_score,
        'faiss_score_' + pipeline: faiss_score,
        'rule_score_' + pipeline: rule_score,
        'bm25_rank_' + pipeline: bm25_rank,
        'faiss_rank_' + pipeline: faiss_rank,
        'rule_rank_' + pipeline: rule_rank,
        'bm25_top1_' + pipeline: bm25_top1,
        'faiss_top1_' + pipeline: faiss_top1,
        'rule_top1_' + pipeline: rule_top1,
        'bm25_top3_ratio_' + pipeline: bm25_top3_ratio,
        'faiss_top3_ratio_' + pipeline: faiss_top3_ratio,
        'rule_top3_ratio_' + pipeline: rule_top3_ratio,
        'bm25_gap_' + pipeline: bm25_gap,
        'faiss_gap_' + pipeline: faiss_gap,
        'rule_gap_' + pipeline: rule_gap,
        'bm25_rel_' + pipeline: bm25_rel,
        'faiss_rel_' + pipeline: faiss_rel,
        'rule_rel_' + pipeline: rule_rel,
    }
```

### 3.3 Cross-pipeline 互動特徵計算

```python
def extract_cross_pipeline_features(bm25_scores: dict, faiss_scores: dict, rule_scores: dict, 
                                     target_pipeline: str) -> dict:
    """提取 Cross-pipeline 互動特徵"""
    
    pipelines = ["VideoRAG", "GraphRAG", "TKG", "RDBMS"]
    
    # 計算各方法的 top-1
    bm25_top1 = max(bm25_scores, key=bm25_scores.get) if bm25_scores else None
    faiss_top1 = max(faiss_scores, key=faiss_scores.get) if faiss_scores else None
    rule_top1 = max(rule_scores, key=rule_scores.get) if rule_scores else None
    
    # Agreement 特徵
    top1_votes = [bm25_top1, faiss_top1, rule_top1]
    agreement_top1 = top1_votes.count(target_pipeline)
    
    # Top-3 重疊
    bm25_top3 = set(sorted(bm25_scores.keys(), key=lambda k: bm25_scores[k], reverse=True)[:3])
    faiss_top3 = set(sorted(faiss_scores.keys(), key=lambda k: faiss_scores[k], reverse=True)[:3])
    rule_top3 = set(sorted(rule_scores.keys(), key=lambda k: rule_scores[k], reverse=True)[:3])
    
    target_in_top3 = target_pipeline in (bm25_top3 & faiss_top3 & rule_top3)
    agreement_top3 = len(bm25_top3 & faiss_top3 & rule_top3)
    
    # 分數一致性（標準差反向）
    all_scores = [bm25_scores.get(target_pipeline, 0), 
                  faiss_scores.get(target_pipeline, 0), 
                  rule_scores.get(target_pipeline, 0)]
    score_std = np.std(all_scores) if len(all_scores) > 1 else 0
    score_consistency = 1.0 / (1.0 + score_std)
    
    # Vote count
    vote_count = sum(1 for v in top1_votes if v == target_pipeline)
    
    # Max-min gap
    all_max_scores = [max(bm25_scores.values()), max(faiss_scores.values()), max(rule_scores.values())]
    max_min_score_gap = max(all_max_scores) - min(all_max_scores)
    
    # Score entropy
    total_score = sum(all_max_scores)
    if total_score > 0:
        probs = [s / total_score for s in all_max_scores]
        score_entropy = -sum(p * np.log2(p + 1e-8) for p in probs)
    else:
        score_entropy = 0
    
    # Consensus winner
    is_consensus_winner = (bm25_top1 == faiss_top1 == rule_top1 == target_pipeline)
    
    # Disagreement candidate
    unique_top1 = set(top1_votes)
    is_disagreement_candidate = len(unique_top1) > 2 and target_pipeline in unique_top1
    
    # Rank agreement
    bm25_rank = sorted(bm25_scores.keys(), key=lambda k: bm25_scores[k], reverse=True).index(target_pipeline) + 1
    faiss_rank = sorted(faiss_scores.keys(), key=lambda k: faiss_scores[k], reverse=True).index(target_pipeline) + 1
    rule_rank = sorted(rule_scores.keys(), key=lambda k: rule_scores[k], reverse=True).index(target_pipeline) + 1
    ranks = [bm25_rank, faiss_rank, rule_rank]
    rank_agreement = 1.0 / (1.0 + np.std(ranks))
    
    # Top-1 confidence
    sorted_bm25 = sorted(bm25_scores.values(), reverse=True)
    sorted_faiss = sorted(faiss_scores.values(), reverse=True)
    sorted_rule = sorted(rule_scores.values(), reverse=True)
    
    bm25_gap = sorted_bm25[0] - sorted_bm25[1] if len(sorted_bm25) > 1 else sorted_bm25[0]
    faiss_gap = sorted_faiss[0] - sorted_faiss[1] if len(sorted_faiss) > 1 else sorted_faiss[0]
    rule_gap = sorted_rule[0] - sorted_rule[1] if len(sorted_rule) > 1 else sorted_rule[0]
    
    top1_confidence = max(bm25_gap, faiss_gap, rule_gap)
    
    # Top-3 diversity
    top3_sets = [bm25_top3, faiss_top3, rule_top3]
    union_top3 = set()
    for s in top3_sets:
        union_top3.update(s)
    top3_diversity = len(union_top3)
    
    # Cross-method agreement
    cross_method_agreement = (agreement_top1 / 3.0) * 0.5 + (agreement_top3 / 4.0) * 0.5
    
    return {
        'agreement_top1': agreement_top1,
        'agreement_top3': agreement_top3,
        'score_consistency': score_consistency,
        'vote_count': vote_count,
        'max_min_score_gap': max_min_score_gap,
        'score_entropy': score_entropy,
        'is_consensus_winner': 1.0 if is_consensus_winner else 0.0,
        'is_disagreement_candidate': 1.0 if is_disagreement_candidate else 0.0,
        'rank_agreement': rank_agreement,
        'top1_confidence': top1_confidence,
        'top3_diversity': top3_diversity,
        'cross_method_agreement': cross_method_agreement,
    }
```

---

## 4. 特徵重要性預測與選擇策略

### 4.1 預測重要特徵（按重要性排序）

| 排名 | 特徵類別 | 預測重要性 | 原因 |
|------|---------|-----------|------|
| 1 | Pipeline-level 分數 | ⭐⭐⭐⭐⭐ | 直接反映各方法對該 pipeline 的置信度 |
| 2 | Cross-pipeline agreement | ⭐⭐⭐⭐⭐ | 高度一致時模型會更自信 |
| 3 | Pipeline-level 排名 | ⭐⭐⭐⭐ | 排名比絕對分數更具可比性 |
| 4 | Query-level 詞彙 | ⭐⭐⭐⭐ | 特定詞彙（如關係詞、時間詞）是強訊號 |
| 5 | Top-1 投票 | ⭐⭐⭐⭐ | 獲得多少方法共識是重要指標 |
| 6 | 差距特徵 | ⭐⭐⭐ | top-1 與 top-2 的差距反映確定性 |
| 7 | 其他 Query 特徵 | ⭐⭐ | 輔助訊號，單獨使用效果有限 |

### 4.2 特徵選擇策略

#### 階段 1：基礎模型（所有特徵）
- 使用全部 ~96 個特徵訓練基礎模型
- 記錄特徵重要性（LightGBM 內建）

#### 階段 2：特徵分析
- 分析訓練後的特徵重要性
- 移除重要性 < 0.001 的特徵
- 檢查 multicollinearity（相關係數 > 0.95 的特徵保留一個）

#### 階段 3：精簡模型
- 使用篩選後的特徵重新訓練
- 比較與基礎模型的效能差異

#### 階段 4：最終模型
- 選擇效能最佳且特徵數最少的版本

---

## 5. 特徵工程實施時程

| 步驟 | 任務 | 產出 | 依賴 |
|------|------|------|------|
| 1 | 實作 `extract_query_features()` | Query-level 特徵函數 | 無 |
| 2 | 實作 `extract_pipeline_features()` | Pipeline-level 特徵函數 | 無 |
| 3 | 實作 `extract_cross_pipeline_features()` | Cross-pipeline 特徵函數 | 無 |
| 4 | 重構 `extract_features()` | 整合所有特徵的函數 | 1, 2, 3 |
| 5 | 重構 `QueryFeatures` dataclass | 新的特徵欄位 | 4 |
| 6 | 更新 `to_array()` 方法 | 特徵陣列轉換 | 5 |
| 7 | 特徵重要性分析 | 特徵重要性報告 | 訓練後 |
| 8 | 特徵選擇與精簡 | 精簡特徵集 | 7 |

---

## 6. 特徵範例輸出

### 6.1 Query-level 範例

對於 query: `"檢索影片中所有提到『生成式 AI』的語義片段。"`

```python
{
    'query_length': 30,
    'query_word_count': 12,
    'query_has_question': 0.0,        # 無疑問詞
    'query_has_time_expr': 0.0,       # 無時間詞
    'query_has_relation_word': 0.0,   # 無關係詞
    'query_has_list_word': 1.0,       # 有「所有」
    'query_has_visual_word': 1.0,     # 有「影片」
    'query_complexity_score': 0.52,
    'bm25_max': 15.2,
    'faiss_max': 0.87,
    'rule_max': 1.5,
    'num_matched_rules': 2,
}
```

### 6.2 Pipeline-level 範例（VideoRAG）

```python
{
    'bm25_score_VideoRAG': 15.2,
    'faiss_score_VideoRAG': 0.87,
    'rule_score_VideoRAG': 1.5,
    'bm25_rank_VideoRAG': 1.0,       # BM25 中排第 1
    'faiss_rank_VideoRAG': 1.0,      # FAISS 中排第 1
    'rule_rank_VideoRAG': 1.0,       # Rule 中排第 1
    'bm25_top1_VideoRAG': 1.0,      # BM25 top-1 是 VideoRAG
    'faiss_top1_VideoRAG': 1.0,     # FAISS top-1 是 VideoRAG
    'rule_top1_VideoRAG': 1.0,      # Rule top-1 是 VideoRAG
    'bm25_top3_ratio_VideoRAG': 1.0,
    'faiss_top3_ratio_VideoRAG': 1.0,
    'rule_top3_ratio_VideoRAG': 1.0,
    'bm25_gap_VideoRAG': 3.5,       # top-1 - top-2
    'faiss_gap_VideoRAG': 0.12,
    'rule_gap_VideoRAG': 0.5,
    'bm25_rel_VideoRAG': 0.48,      # 15.2 / 31.5
    'faiss_rel_VideoRAG': 0.42,
    'rule_rel_VideoRAG': 0.50,
}
```

### 6.3 Cross-pipeline 範例

```python
{
    'agreement_top1': 3.0,          # 三個方法都選 VideoRAG
    'agreement_top3': 4,            # top-3 完全重疊
    'score_consistency': 0.85,      # 分數一致度高
    'vote_count': 3.0,              # 獲得 3 票
    'max_min_score_gap': 5.2,
    'score_entropy': 0.95,
    'is_consensus_winner': 1.0,    # 共識贏家
    'is_disagreement_candidate': 0.0,
    'rank_agreement': 1.0,         # 排名完全一致
    'top1_confidence': 3.5,
    'top3_diversity': 2,           # top-3 中有 2 個不同 pipeline
    'cross_method_agreement': 0.875,
}
```

---

## 7. 訓練數據格式詳細設計

### 7.1 JSON 格式

```json
{
    "query_id": "q001",
    "query": "檢索影片中所有提到『生成式 AI』的語義片段。",
    "labels": [
        {"pipeline": "VideoRAG", "relevance": 3},
        {"pipeline": "GraphRAG", "relevance": 0},
        {"pipeline": "TKG", "relevance": 0},
        {"pipeline": "RDBMS", "relevance": 0}
    ]
}
```

### 7.2 Relevance 標籤指南

| Relevance | 定義 | 範例關鍵字 |
|-----------|------|-----------|
| **3 (Perfect)** | 查詢完全符合此 pipeline 的核心能力 | VideoRAG: "影片", "畫面", "片段"<br>GraphRAG: "關係", "網絡", "合作"<br>TKG: "時間軸", "演進", "之前/之後"<br>RDBMS: "清單", "統計", "所有" |
| **2 (Good)** | 查詢適合此 pipeline，但不是最佳選擇 | VideoRAG: 包含視覺描述但無明確「影片」<br>GraphRAG: 涉及實體但關係不明確<br>TKG: 涉及時間但無明確時間順序<br>RDBMS: 涉及數據但無明確清單需求 |
| **1 (Related)** | 查詢與此 pipeline 有些關聯 | 跨領域查詢，可能適合多個 pipeline |
| **0 (Irrelevant)** | 查詢完全不適合此 pipeline | 明確指向其他 pipeline 的查詢 |

### 7.3 標籤範例

```json
[
    {
        "query_id": "q001",
        "query": "檢索影片中所有提到『生成式 AI』的語義片段。",
        "labels": [
            {"pipeline": "VideoRAG", "relevance": 3},
            {"pipeline": "GraphRAG", "relevance": 0},
            {"pipeline": "TKG", "relevance": 0},
            {"pipeline": "RDBMS", "relevance": 0}
        ]
    },
    {
        "query_id": "q002",
        "query": "分析這兩家公司在股權投資上的多層級交叉關聯。",
        "labels": [
            {"pipeline": "VideoRAG", "relevance": 0},
            {"pipeline": "GraphRAG", "relevance": 3},
            {"pipeline": "TKG", "relevance": 1},
            {"pipeline": "RDBMS", "relevance": 0}
        ]
    },
    {
        "query_id": "q003",
        "query": "追蹤這項技術從研究實驗室到商業生產的關鍵時間節點。",
        "labels": [
            {"pipeline": "VideoRAG", "relevance": 0},
            {"pipeline": "GraphRAG", "relevance": 1},
            {"pipeline": "TKG", "relevance": 3},
            {"pipeline": "RDBMS", "relevance": 0}
        ]
    },
    {
        "query_id": "q004",
        "query": "從訂單表中檢索 2024 年 Q1 消費額前 5% 的客戶清單。",
        "labels": [
            {"pipeline": "VideoRAG", "relevance": 0},
            {"pipeline": "GraphRAG", "relevance": 0},
            {"pipeline": "TKG", "relevance": 0},
            {"pipeline": "RDBMS", "relevance": 3}
        ]
    }
]
```

---

## 8. 評估指標詳細設計

### 8.1 主要評估指標

| 指標 | 說明 | 計算方式 | 目標值 |
|------|------|---------|--------|
| **NDCG@1** | 第一個預測是否正確 | 如果 top-1 relevance > 0 則為 1 | > 0.85 |
| **NDCG@3** | top-3 排序品質 | 考慮相關性權重的排序品質 | > 0.90 |
| **MAP** | 平均精確度 | 所有相關結果的平均精確度 | > 0.80 |
| **MRR** | 第一個相關結果的排名 | 1 / rank_of_first_relevant | > 0.90 |
| **Top-1 Accuracy** | 第一個預測正確的比率 | correct / total | > 0.85 |
| **Top-3 Recall** | 正確結果在 top-3 的比率 | relevant_in_top3 / total_relevant | > 0.95 |

### 8.2 NDCG 計算公式

```python
def compute_ndcg(ranked_results, relevance_scores, k=3):
    """
    計算 NDCG@k
    
    Args:
        ranked_results: 排序後的結果 list
        relevance_scores: 每個結果的相關性分數 (0-3)
        k: top-k
    
    Returns:
        NDCG score (0-1)
    """
    # DCG@k
    dcg = 0.0
    for i in range(min(k, len(ranked_results))):
        rel = relevance_scores[i]
        dcg += (2**rel - 1) / np.log2(i + 2)  # +2 because i is 0-indexed
    
    # 計算理想 DCG（最佳排序）
    sorted_relevance = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(sorted_relevance))):
        rel = sorted_relevance[i]
        idcg += (2**rel - 1) / np.log2(i + 2)
    
    # NDCG
    return dcg / idcg if idcg > 0 else 0.0
```

---

## 9. 總結

### 9.1 特徵工程關鍵改變

1. **從 Classifier 特徵改為 Ranker 特徵**：每個 pipeline 獨立計算特徵
2. **增加排序相關特徵**：排名、top-k ratio、差距等
3. **增加語義特徵**：疑問詞、時間詞、關係詞等
4. **增加一致性特徵**：跨方法 agreement、vote count 等

### 9.2 預期效益

1. **更好的排序品質**：Ranker 直接優化排序指標
2. **更準確的預測**：更多相關特徵提升模型判斷能力
3. **更好的泛化能力**：Query-level 特徵幫助處理 unseen queries

### 9.3 下一步

1. 確認特徵設計是否符合需求
2. 開始實作特徵提取函數
3. 轉換訓練數據格式
4. 訓練 Ranker 模型並評估
