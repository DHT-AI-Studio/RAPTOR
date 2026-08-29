# Feature Engineering Design Document

> Note: this module classifies and ranks queries for a Traditional-Chinese-language product. Example queries, Chinese question-word/time-word/relation-word regex patterns, and Chinese-language JSON label examples below are kept in Chinese on purpose — they are the actual functional content being detected/classified, not documentation prose, and translating them would break what they demonstrate. An English gloss is added next to each.

## 1. Analysis of Existing Features

### 1.1 Current Feature List (35 features)

| Category | Feature Name | Description | Type | Value Assessment |
|------|---------|------|------|---------|
| **BM25 stats** | `bm25_max` | Highest BM25 score | float | ⭐⭐⭐⭐⭐ |
| | `bm25_mean` | Mean BM25 score | float | ⭐⭐⭐ |
| | `bm25_min` | Lowest BM25 score | float | ⭐⭐ |
| | `bm25_std` | BM25 standard deviation | float | ⭐⭐ |
| **BM25 ranking** | `bm25_top1` | BM25 top-1 score | float | ⭐⭐⭐⭐⭐ |
| | `bm25_top2` | BM25 top-2 score | float | ⭐⭐⭐⭐ |
| | `bm25_max_diff` | Gap between top-1 and top-2 | float | ⭐⭐⭐⭐⭐ |
| | `bm25_top1_idx` | Pipeline index of the top-1 (0-3) | int | ⭐⭐⭐⭐ |
| **FAISS stats** | `faiss_max` | Highest FAISS score | float | ⭐⭐⭐⭐⭐ |
| | `faiss_mean` | Mean FAISS score | float | ⭐⭐⭐ |
| | `faiss_min` | Lowest FAISS score | float | ⭐⭐ |
| | `faiss_std` | FAISS standard deviation | float | ⭐⭐ |
| **FAISS ranking** | `faiss_top1` | FAISS top-1 score | float | ⭐⭐⭐⭐⭐ |
| | `faiss_top2` | FAISS top-2 score | float | ⭐⭐⭐⭐ |
| | `faiss_max_diff` | Gap between top-1 and top-2 | float | ⭐⭐⭐⭐⭐ |
| | `faiss_top1_idx` | Pipeline index of the top-1 (0-3) | int | ⭐⭐⭐⭐ |
| **Rule stats** | `rule_max` | Highest rule score | float | ⭐⭐⭐⭐⭐ |
| | `rule_mean` | Mean rule score | float | ⭐⭐⭐ |
| | `rule_min` | Lowest rule score | float | ⭐⭐ |
| | `rule_std` | Rule standard deviation | float | ⭐⭐ |
| **Rule ranking** | `rule_top1` | Rule top-1 score | float | ⭐⭐⭐⭐⭐ |
| | `rule_top2` | Rule top-2 score | float | ⭐⭐⭐⭐ |
| | `rule_max_diff` | Gap between top-1 and top-2 | float | ⭐⭐⭐⭐⭐ |
| | `rule_top1_idx` | Pipeline index of the top-1 (0-3) | int | ⭐⭐⭐⭐ |
| **Cross-method consistency** | `bm25_faiss_gap` | BM25 vs FAISS gap | float | ⭐⭐⭐ |
| | `bm25_rule_gap` | BM25 vs rule gap | float | ⭐⭐⭐ |
| | `faiss_rule_gap` | FAISS vs rule gap | float | ⭐⭐⭐ |
| **Query features** | `query_length` | Query character count | int | ⭐⭐ |
| | `num_bm25_pipelines` | Number of pipelines with a BM25 score | int | ⭐⭐ |
| | `num_faiss_pipelines` | Number of pipelines with a FAISS score | int | ⭐⭐ |
| | `num_rule_pipelines` | Number of pipelines with a rule score | int | ⭐⭐ |
| | `num_matched_rules` | Number of matched rules | int | ⭐⭐⭐ |

### 1.2 Analysis of Problems with Existing Features

#### Problem 1: Classifier features don't match ranker features

The classifier's features are designed for "predicting a single label," but a ranker needs to learn "relative ordering."

**Problems with the classifier's features**:
- Index-style features like `bm25_top1_idx`, `faiss_top1_idx` are meaningless for a ranker (since the ranker computes an independent score for every pipeline)
- Features like `bm25_top1_is_target` can only be used at training time (they need to know the target) — they can't be computed at inference time

**What the ranker needs**:
- Independent features per pipeline (so the model can learn which pipeline is better)
- Relative features across pipelines (so the model can learn the gap)
- Query-level global features

#### Problem 2: Missing ranking-related features

The existing features are missing:
- **Rank features**: each pipeline's rank within each individual method
- **Distribution features**: entropy, skewness, kurtosis of the scores
- **Consistency features**: how much the different methods agree on the top-k

#### Problem 3: Missing semantic features

- Query length is too simplistic — it doesn't account for word frequency, question words, etc.
- Query complexity isn't considered (e.g. whether it contains time expressions, relational words, etc.)

---

## 2. Ranker Feature Design

### 2.1 Design Principles

For the ranker, each query is expanded into 4 samples (one per pipeline), so features need to fall into two categories:

1. **Query-level features**: the same across all 4 samples
2. **Pipeline-level features**: different for each sample

```
Query: "Retrieve every semantic segment in the video that mentions 'generative AI'." (原文: 檢索影片中所有提到『生成式 AI』的語義片段。)
┌─────────────────────────────────────────────────┐
│ Query-level features (shared across all 4 samples) │
│ - query_length, query_word_count, ...           │
├─────────────────────────────────────────────────┤
│ Sample 1 (VideoRAG):                            │
│   Pipeline-level features:                       │
│   - bm25_score_vr, faiss_score_vr, rule_score_vr │
│   - bm25_rank_vr, faiss_rank_vr, rule_rank_vr   │
│   - is_top1_vr, is_target_vr                    │
├─────────────────────────────────────────────────┤
│ Sample 2 (GraphRAG):                            │
│   Pipeline-level features:                       │
│   - bm25_score_gr, faiss_score_gr, rule_score_gr │
│   - bm25_rank_gr, faiss_rank_gr, rule_rank_gr   │
│   - is_top1_gr, is_target_gr                    │
├─────────────────────────────────────────────────┤
│ Sample 3 (TKG): ...                             │
│ Sample 4 (RDBMS): ...                           │
└─────────────────────────────────────────────────┘
```

### 2.2 Complete Feature List

#### A. Query-level features (12)

| Feature Name | Type | Description | Example value |
|---------|------|------|--------|
| `query_length` | float | Query character count | 35 |
| `query_word_count` | float | Word count after tokenizing the query | 8 |
| `query_has_question` | float | Whether it contains a question word (0/1) | 1 |
| `query_has_time_expr` | float | Whether it contains a time expression (0/1) | 0 |
| `query_has_relation_word` | float | Whether it contains a relational word (0/1) | 0 |
| `query_has_list_word` | float | Whether it contains a list word (0/1) | 0 |
| `query_has_visual_word` | float | Whether it contains a visual word (0/1) | 1 |
| `query_complexity_score` | float | Query complexity score (0-1) | 0.6 |
| `bm25_max` | float | Highest BM25 score | 12.5 |
| `faiss_max` | float | Highest FAISS score | 0.85 |
| `rule_max` | float | Highest rule score | 1.5 |
| `num_matched_rules` | float | Number of matched rules | 2 |

#### B. Pipeline-level features (18 per pipeline × 4 pipelines = 72)

Using VideoRAG as the example (the other pipelines are analogous):

| Feature Name | Type | Description | Example value |
|---------|------|------|--------|
| **Score features** | | | |
| `bm25_score_vr` | float | BM25's score for VideoRAG | 12.5 |
| `faiss_score_vr` | float | FAISS's score for VideoRAG | 0.85 |
| `rule_score_vr` | float | Rule's score for VideoRAG | 1.5 |
| **Rank features** | | | |
| `bm25_rank_vr` | float | VideoRAG's rank within BM25 (1-4) | 1 |
| `faiss_rank_vr` | float | VideoRAG's rank within FAISS (1-4) | 1 |
| `rule_rank_vr` | float | VideoRAG's rank within rule (1-4) | 1 |
| **Top-k features** | | | |
| `bm25_top1_vr` | float | Whether BM25's top-1 is VideoRAG (0/1) | 1 |
| `faiss_top1_vr` | float | Whether FAISS's top-1 is VideoRAG (0/1) | 1 |
| `rule_top1_vr` | float | Whether rule's top-1 is VideoRAG (0/1) | 1 |
| `bm25_top3_ratio_vr` | float | Whether VideoRAG is in BM25's top-3 | 1.0 |
| `faiss_top3_ratio_vr` | float | Whether VideoRAG is in FAISS's top-3 | 1.0 |
| `rule_top3_ratio_vr` | float | Whether VideoRAG is in rule's top-3 | 1.0 |
| **Gap features** | | | |
| `bm25_gap_vr` | float | BM25's top-1 - top-2 gap | 3.2 |
| `faiss_gap_vr` | float | FAISS's top-1 - top-2 gap | 0.15 |
| `rule_gap_vr` | float | Rule's top-1 - top-2 gap | 0.5 |
| **Relative features** | | | |
| `bm25_rel_vr` | float | VideoRAG's share of BM25's total score (self/sum) | 0.45 |
| `faiss_rel_vr` | float | VideoRAG's share of FAISS's total score | 0.35 |
| `rule_rel_vr` | float | VideoRAG's share of rule's total score | 0.40 |

#### C. Cross-pipeline interaction features (12)

These features are computed per sample, but reference the scores of the other pipelines:

| Feature Name | Type | Description |
|---------|------|------|
| `agreement_top1` | float | Number of methods (out of 3) whose top-1 agrees (0-3) |
| `agreement_top3` | float | Number of pipelines overlapping across the three methods' top-3 (0-4) |
| `score_consistency` | float | Inverse indicator of the three methods' score standard deviation |
| `vote_count` | float | Number of times this pipeline received a top-1 vote (0-3) |
| `max_min_score_gap` | float | Gap between the highest and lowest score across all pipelines |
| `score_entropy` | float | Information entropy of all pipelines' scores |
| `is_consensus_winner` | float | Whether this is the top-1 all methods agree on (0/1) |
| `is_disagreement_candidate` | float | Whether there's disagreement between methods (0/1) |
| `rank_agreement` | float | Inverse indicator of the three methods' rank standard deviation |
| `top1_confidence` | float | Top-1's advantage over top-2 |
| `top3_diversity` | float | Number of distinct pipelines in the top-3 |
| `cross_method_agreement` | float | Cross-method agreement score |

### 2.3 Total

| Category | Feature Count | Description |
|------|---------|------|
| Query-level | 12 | Shared across all samples |
| Pipeline-level | 72 (18 × 4) | Independent per pipeline |
| Cross-pipeline | 12 | References all pipelines |
| **Total** | **96** | |

---

## 3. Detailed Feature Computation Design

### 3.1 Query-level Feature Computation

```python
def extract_query_features(query: str, bm25_scores: dict, faiss_scores: dict, rule_scores: dict) -> dict:
    """Extract query-level features"""

    # Basic features
    query_length = len(query)
    tokens = jieba.lcut(query)
    query_word_count = len(tokens)

    # Question-word detection (Chinese question words: what/how/which/who/which segment/which vendor's)
    question_words = re.compile(r'(什麼|怎麼|如何|哪個|哪些|誰|哪段|哪家的)')
    query_has_question = 1.0 if question_words.search(query) else 0.0

    # Time-expression detection (Chinese: time/before/after/order/evolution/history + EN equivalents)
    time_words = re.compile(r'(時間|之前|之後|順序|演進|歷史|timeline|evolution|before|after)')
    query_has_time_expr = 1.0 if time_words.search(query) else 0.0

    # Relation-word detection (Chinese: relationship/association/collaboration/competition/connection + EN equivalents)
    relation_words = re.compile(r'(關係|關聯|合作|競爭|連接|relationship|connection|collaborat|competitor)')
    query_has_relation_word = 1.0 if relation_words.search(query) else 0.0

    # List-word detection (Chinese: list/all/enumerate/count + EN equivalents)
    list_words = re.compile(r'(清單|所有|列出|統計|list|count|how many|all|total)')
    query_has_list_word = 1.0 if list_words.search(query) else 0.0

    # Visual-word detection (Chinese: video/scene/footage/segment/visual + EN equivalents)
    visual_words = re.compile(r'(影片|畫面|影像|片段|視覺|video|clip|footage|visual|scene|frame)')
    query_has_visual_word = 1.0 if visual_words.search(query) else 0.0

    # Query complexity score (based on word count and presence of special word types)
    query_complexity_score = min(1.0, (query_word_count / 20) * 0.5 + 
                                  (query_has_question + query_has_relation_word + 
                                   query_has_time_expr + query_has_list_word + query_has_visual_word) / 5)

    # Highest score per method
    bm25_max = max(bm25_scores.values()) if bm25_scores else 0.0
    faiss_max = max(faiss_scores.values()) if faiss_scores else 0.0
    rule_max = max(rule_scores.values()) if rule_scores else 0.0

    # Number of matched rules
    num_matched_rules = len(rule_scores)  # or obtained from matched_rules

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

### 3.2 Pipeline-level Feature Computation

```python
def extract_pipeline_features(pipeline: str, bm25_scores: dict, faiss_scores: dict, rule_scores: dict) -> dict:
    """Extract pipeline-level features for a specific pipeline"""

    pipelines = ["VideoRAG", "GraphRAG", "TKG", "RDBMS"]

    # Score features
    bm25_score = bm25_scores.get(pipeline, 0.0)
    faiss_score = faiss_scores.get(pipeline, 0.0)
    rule_score = rule_scores.get(pipeline, 0.0)

    # Compute rank
    bm25_sorted = sorted(bm25_scores.values(), reverse=True)
    faiss_sorted = sorted(faiss_scores.values(), reverse=True)
    rule_sorted = sorted(rule_scores.values(), reverse=True)

    bm25_rank = bm25_sorted.index(bm25_score) + 1 if bm25_score > 0 else 5
    faiss_rank = faiss_sorted.index(faiss_score) + 1 if faiss_score > 0 else 5
    rule_rank = rule_sorted.index(rule_score) + 1 if rule_score > 0 else 5

    # Top-1 features
    bm25_top1_pipeline = max(bm25_scores, key=bm25_scores.get) if bm25_scores else ""
    faiss_top1_pipeline = max(faiss_scores, key=faiss_scores.get) if faiss_scores else ""
    rule_top1_pipeline = max(rule_scores, key=rule_scores.get) if rule_scores else ""

    bm25_top1 = 1.0 if bm25_top1_pipeline == pipeline else 0.0
    faiss_top1 = 1.0 if faiss_top1_pipeline == pipeline else 0.0
    rule_top1 = 1.0 if rule_top1_pipeline == pipeline else 0.0

    # Top-3 membership
    bm25_top3 = sorted(bm25_scores.keys(), key=lambda k: bm25_scores[k], reverse=True)[:3]
    faiss_top3 = sorted(faiss_scores.keys(), key=lambda k: faiss_scores[k], reverse=True)[:3]
    rule_top3 = sorted(rule_scores.keys(), key=lambda k: rule_scores[k], reverse=True)[:3]

    bm25_top3_ratio = 1.0 if pipeline in bm25_top3 else 0.0
    faiss_top3_ratio = 1.0 if pipeline in faiss_top3 else 0.0
    rule_top3_ratio = 1.0 if pipeline in rule_top3 else 0.0

    # Gap features (top-1 - top-2)
    bm25_gap = (bm25_sorted[0] - bm25_sorted[1]) if len(bm25_sorted) > 1 else bm25_score
    faiss_gap = (faiss_sorted[0] - faiss_sorted[1]) if len(faiss_sorted) > 1 else faiss_score
    rule_gap = (rule_sorted[0] - rule_sorted[1]) if len(rule_sorted) > 1 else rule_score

    # Relative score (self / sum)
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

### 3.3 Cross-pipeline Interaction Feature Computation

```python
def extract_cross_pipeline_features(bm25_scores: dict, faiss_scores: dict, rule_scores: dict, 
                                     target_pipeline: str) -> dict:
    """Extract cross-pipeline interaction features"""

    pipelines = ["VideoRAG", "GraphRAG", "TKG", "RDBMS"]

    # Compute each method's top-1
    bm25_top1 = max(bm25_scores, key=bm25_scores.get) if bm25_scores else None
    faiss_top1 = max(faiss_scores, key=faiss_scores.get) if faiss_scores else None
    rule_top1 = max(rule_scores, key=rule_scores.get) if rule_scores else None

    # Agreement features
    top1_votes = [bm25_top1, faiss_top1, rule_top1]
    agreement_top1 = top1_votes.count(target_pipeline)

    # Top-3 overlap
    bm25_top3 = set(sorted(bm25_scores.keys(), key=lambda k: bm25_scores[k], reverse=True)[:3])
    faiss_top3 = set(sorted(faiss_scores.keys(), key=lambda k: faiss_scores[k], reverse=True)[:3])
    rule_top3 = set(sorted(rule_scores.keys(), key=lambda k: rule_scores[k], reverse=True)[:3])

    target_in_top3 = target_pipeline in (bm25_top3 & faiss_top3 & rule_top3)
    agreement_top3 = len(bm25_top3 & faiss_top3 & rule_top3)

    # Score consistency (inverse of standard deviation)
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

## 4. Feature Importance Prediction and Selection Strategy

### 4.1 Predicted Important Features (ranked by importance)

| Rank | Feature Category | Predicted Importance | Reason |
|------|---------|-----------|------|
| 1 | Pipeline-level scores | ⭐⭐⭐⭐⭐ | Directly reflects each method's confidence in that pipeline |
| 2 | Cross-pipeline agreement | ⭐⭐⭐⭐⭐ | The model gains confidence when there's high agreement |
| 3 | Pipeline-level rank | ⭐⭐⭐⭐ | Rank is more comparable than an absolute score |
| 4 | Query-level lexical features | ⭐⭐⭐⭐ | Specific words (e.g. relational words, time words) are strong signals |
| 5 | Top-1 vote | ⭐⭐⭐⭐ | How many methods agree is an important indicator |
| 6 | Gap features | ⭐⭐⭐ | The gap between top-1 and top-2 reflects certainty |
| 7 | Other query features | ⭐⭐ | Auxiliary signals, limited effect on their own |

### 4.2 Feature Selection Strategy

#### Phase 1: Baseline model (all features)
- Train a baseline model using all ~96 features
- Record feature importance (LightGBM's built-in reporting)

#### Phase 2: Feature analysis
- Analyze the trained model's feature importance
- Remove features with importance < 0.001
- Check for multicollinearity (keep only one of any pair of features with correlation > 0.95)

#### Phase 3: Reduced model
- Retrain using the filtered feature set
- Compare performance against the baseline model

#### Phase 4: Final model
- Select the version with the best performance and fewest features

---

## 5. Feature Engineering Implementation Timeline

| Step | Task | Output | Dependency |
|------|------|------|------|
| 1 | Implement `extract_query_features()` | Query-level feature function | None |
| 2 | Implement `extract_pipeline_features()` | Pipeline-level feature function | None |
| 3 | Implement `extract_cross_pipeline_features()` | Cross-pipeline feature function | None |
| 4 | Refactor `extract_features()` | Function integrating all features | 1, 2, 3 |
| 5 | Refactor the `QueryFeatures` dataclass | New feature fields | 4 |
| 6 | Update the `to_array()` method | Feature array conversion | 5 |
| 7 | Feature importance analysis | Feature importance report | After training |
| 8 | Feature selection and reduction | Reduced feature set | 7 |

---

## 6. Example Feature Output

### 6.1 Query-level Example

For the query: `"Retrieve every semantic segment in the video that mentions 'generative AI'."` (原文: `"檢索影片中所有提到『生成式 AI』的語義片段。"`)

```python
{
    'query_length': 30,
    'query_word_count': 12,
    'query_has_question': 0.0,        # no question word
    'query_has_time_expr': 0.0,       # no time word
    'query_has_relation_word': 0.0,   # no relational word
    'query_has_list_word': 1.0,       # contains "all" (所有)
    'query_has_visual_word': 1.0,     # contains "video" (影片)
    'query_complexity_score': 0.52,
    'bm25_max': 15.2,
    'faiss_max': 0.87,
    'rule_max': 1.5,
    'num_matched_rules': 2,
}
```

### 6.2 Pipeline-level Example (VideoRAG)

```python
{
    'bm25_score_VideoRAG': 15.2,
    'faiss_score_VideoRAG': 0.87,
    'rule_score_VideoRAG': 1.5,
    'bm25_rank_VideoRAG': 1.0,       # ranked #1 in BM25
    'faiss_rank_VideoRAG': 1.0,      # ranked #1 in FAISS
    'rule_rank_VideoRAG': 1.0,       # ranked #1 in rule
    'bm25_top1_VideoRAG': 1.0,      # BM25's top-1 is VideoRAG
    'faiss_top1_VideoRAG': 1.0,     # FAISS's top-1 is VideoRAG
    'rule_top1_VideoRAG': 1.0,      # rule's top-1 is VideoRAG
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

### 6.3 Cross-pipeline Example

```python
{
    'agreement_top1': 3.0,          # all three methods picked VideoRAG
    'agreement_top3': 4,            # top-3 fully overlaps
    'score_consistency': 0.85,      # high score consistency
    'vote_count': 3.0,              # received 3 votes
    'max_min_score_gap': 5.2,
    'score_entropy': 0.95,
    'is_consensus_winner': 1.0,    # the consensus winner
    'is_disagreement_candidate': 0.0,
    'rank_agreement': 1.0,         # ranks fully agree
    'top1_confidence': 3.5,
    'top3_diversity': 2,           # 2 distinct pipelines in the top-3
    'cross_method_agreement': 0.875,
}
```

---

## 7. Detailed Training Data Format Design

### 7.1 JSON Format

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
(EN: `"query"` = "Retrieve every semantic segment in the video that mentions 'generative AI'.")

### 7.2 Relevance Label Guidelines

| Relevance | Definition | Example keywords |
|-----------|------|-----------|
| **3 (Perfect)** | The query fully matches this pipeline's core capability | VideoRAG: "video," "footage," "clip" (影片, 畫面, 片段)<br>GraphRAG: "relationship," "network," "collaboration" (關係, 網絡, 合作)<br>TKG: "timeline," "evolution," "before/after" (時間軸, 演進, 之前/之後)<br>RDBMS: "list," "count," "all" (清單, 統計, 所有) |
| **2 (Good)** | The query suits this pipeline, but it's not the best choice | VideoRAG: contains a visual description but no explicit "video"<br>GraphRAG: involves entities but the relationship isn't explicit<br>TKG: involves time but no explicit chronological order<br>RDBMS: involves data but no explicit list requirement |
| **1 (Related)** | The query has some connection to this pipeline | Cross-domain queries that may suit multiple pipelines |
| **0 (Irrelevant)** | The query doesn't suit this pipeline at all | Queries that clearly target a different pipeline |

### 7.3 Label Examples

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
(EN, in order — q001: "Retrieve every semantic segment in the video that mentions 'generative AI'." (VideoRAG); q002: "Analyze the multi-level cross-holdings between these two companies' equity investments." (GraphRAG); q003: "Trace the key milestones of this technology from research lab to commercial production." (TKG); q004: "Retrieve the list of customers in the top 5% of Q1 2024 spending from the orders table." (RDBMS))

---

## 8. Detailed Evaluation Metrics Design

### 8.1 Primary Evaluation Metrics

| Metric | Description | Calculation | Target |
|------|------|---------|--------|
| **NDCG@1** | Whether the first prediction is correct | 1 if the top-1's relevance > 0 | > 0.85 |
| **NDCG@3** | Quality of the top-3 ranking | Ranking quality weighted by relevance | > 0.90 |
| **MAP** | Mean average precision | Average precision across all relevant results | > 0.80 |
| **MRR** | Rank of the first relevant result | 1 / rank_of_first_relevant | > 0.90 |
| **Top-1 Accuracy** | Rate at which the first prediction is correct | correct / total | > 0.85 |
| **Top-3 Recall** | Rate at which the correct result appears in the top-3 | relevant_in_top3 / total_relevant | > 0.95 |

### 8.2 NDCG Calculation Formula

```python
def compute_ndcg(ranked_results, relevance_scores, k=3):
    """
    Compute NDCG@k

    Args:
        ranked_results: the ranked list of results
        relevance_scores: relevance score for each result (0-3)
        k: top-k

    Returns:
        NDCG score (0-1)
    """
    # DCG@k
    dcg = 0.0
    for i in range(min(k, len(ranked_results))):
        rel = relevance_scores[i]
        dcg += (2**rel - 1) / np.log2(i + 2)  # +2 because i is 0-indexed

    # Compute the ideal DCG (best possible ordering)
    sorted_relevance = sorted(relevance_scores, reverse=True)
    idcg = 0.0
    for i in range(min(k, len(sorted_relevance))):
        rel = sorted_relevance[i]
        idcg += (2**rel - 1) / np.log2(i + 2)

    # NDCG
    return dcg / idcg if idcg > 0 else 0.0
```

---

## 9. Summary

### 9.1 Key Changes in Feature Engineering

1. **Switching from classifier features to ranker features**: each pipeline's features are computed independently
2. **Added ranking-related features**: rank, top-k ratio, gap, etc.
3. **Added semantic features**: question words, time words, relational words, etc.
4. **Added consistency features**: cross-method agreement, vote count, etc.

### 9.2 Expected Benefits

1. **Better ranking quality**: the ranker directly optimizes ranking metrics
2. **More accurate predictions**: more relevant features improve the model's judgment
3. **Better generalization**: query-level features help handle unseen queries

### 9.3 Next Steps

1. Confirm the feature design meets requirements
2. Begin implementing the feature-extraction functions
3. Convert the training data format
4. Train the ranker model and evaluate
