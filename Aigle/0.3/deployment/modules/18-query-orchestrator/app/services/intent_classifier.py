"""
LightGBM-based pipeline classifier.

Tier 0  Pre-QIC guard (regex)               — unambiguous VideoRAG
Tier 1  QIC Hybrid (BM25 + FAISS/bge-m3) — outputs scores per pipeline
Tier 2  Regex Rules                          — outputs scores per pipeline
Tier 3  LightGBM classifier                 — direct pipeline prediction
Tier 4  LLM API                             — re-rank when scores are close
Tier 5  LLM API                             — fallback when no signals from QIC or Rules
Final VideoRAG                              — catch-all fallback

Flow:
     1. Get BM25 scores per pipeline
     2. Get FAISS scores per pipeline
     3. Get Rule scores per pipeline
     4. Use LightGBM to directly classify pipeline (or fixed rules fallback)
     5. Select top-1 pipeline
     6. If top-2 scores are close -> LLM re-rank
     7. If no signals -> LLM decides first, then fallback
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import httpx

from ..models.classification import ClassificationResult, Candidate
from ..classifiers.hybrid_classifier import HybridClassifier
from ..classifiers.lightgbm_fusion import (
    LightGBMRanker,
    extract_features,
    classify_with_lightgbm_ranker,
    classify_with_fixed_rules,
    PIPELINE_CLASSES,
)
from ..core.config import get_settings

_logger = logging.getLogger(__name__)

_VALID_PIPELINES = {"VideoRAG", "GraphRAG", "TKG", "RDBMS"}

# Pre-QIC guard: "哪段影片/新聞拍到/記錄了..." is unambiguously VideoRAG.
_VR_SEGMENT_RE = re.compile(
    r"哪段(影片|新聞|片段|報導)(拍到|記錄了?|捕捉到?|呈現了?|展示了?|拍攝了?)",
    re.IGNORECASE,
)

# Pre-QIC guard: 在/從/根據/檢索 [媒體來源] 中找片段/時段 → 無歧義 VideoRAG
_VR_IN_MEDIA_RE = re.compile(
    # 模式 A：有明確取得動詞（找/提取/定位）
    r"(根據|在|從|檢索)[^。?！]*?(這段|這份|這個)?[^。?！]*?"
    r"(錄音|影音|影片|監控|視訊|錄影|音訊|直播|錄影檔|錄音檔)[^。?！]*?"
    r"(找|檢索|提取|定位|找出)[^。?！]*?"
    r"(片段|時段|位置|區間|語義|段落|畫面|瞬間|對話|音訊|語音|內容)"
    r"|"
    # 模式 B：媒體 + 中討論/中提到 ... 的所有 [媒體內容詞]
    r"(錄音檔?|影音檔?|影片|監控紀錄|視訊|訪談|這段錄音|這段影)[^。?！]*?"
    r"(中討論|中提到|中提及|中出現|中談到)[^。?！]*?"
    r"(音訊|語音|片段|內容|位置|時段|區間)"
    r"|"
    # 模式 C：檢索/搜尋 + 媒體 + 中提到/具體位置（捕捉訪談+談論+位置 及 影片+中提到+非標準詞）
    r"(檢索|搜尋)[^。?！]*?(訪談|影片|錄音|影音)[^。?！]*(中提到|中談到|中討論|具體位置|的位置)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Rule-based classification engine (score-based)
# ---------------------------------------------------------------------------

@dataclass
class Rule:
    rule_id: str = ""
    target_pipeline: str = ""
    patterns: List[re.Pattern] = None
    weight: float = 0.0
    description: str = ""

    def __post_init__(self):
        if self.patterns is None:
            self.patterns = []


def _load_default_rules() -> List[Rule]:
    """
    通用規則設計（基於意圖特徵，中英雙語）

    VideoRAG (4 rules): 視覺/聽覺內容查詢
    RDBMS (2 rules): 文件/影片清單查詢
    GraphRAG (3 rules): 實體關係查詢
    TKG (2 rules): 時間順序查詢
    """
    rules: List[Rule] = []

    # ---- VideoRAG rules ----
    rules.append(Rule(
        rule_id="vr_visual_content",
        target_pipeline="VideoRAG",
        patterns=[
            re.compile(
                r"\b(big ben|eiffel tower|statue of liberty|landmark|scenery|weather|"
                r"environment|scene|landscape|view|sight|visual|footage|broadcast|"
                r"環境|場景|畫面|影像|景色|風景|街景|城市景觀|天氣|暴風雪|大雨|颶風|颱風)\b",
                re.IGNORECASE,
            ),
        ],
        weight=0.75,
        description="視覺場景查詢",
    ))
    rules.append(Rule(
        rule_id="vr_action_person",
        target_pipeline="VideoRAG",
        patterns=[
            re.compile(
                r"\b(what (did|does|do)|who (said|did|performed|acted|played|appeared|sang|danced)|"
                r"action|behavior|goal|score|shoot|kick|throw|gesture|expression|"
                r"畫面|影像|場景|片段|誰說|誰演|動作|行為|進球|得分|射門)\b",
                re.IGNORECASE,
            ),
        ],
        weight=0.75,
        description="動作/人物行為查詢",
    ))
    rules.append(Rule(
        rule_id="vr_scene_moment",
        target_pipeline="VideoRAG",
        patterns=[
            re.compile(
                r"\b(moment|instant|second|when|while|during|the time (when|that)|"
                r"the (exact|specific) (moment|time)|at the (same|exact) (moment|time)|"
                r"那個瞬間|那個時刻|那個時候|...的時候|...時|...期間)\b",
                re.IGNORECASE,
            ),
        ],
        weight=0.80,
        description="特定時刻/場景查詢",
    ))
    rules.append(Rule(
        rule_id="vr_search_in_media",
        target_pipeline="VideoRAG",
        patterns=[
            re.compile(
                r"(根據|在|從|檢索)[^。?！]*?(這段|這份|這個)?[^。?！]*?"
                r"(錄音|影音|影片|監控|視訊|錄影|音訊|直播|錄影檔|錄音檔)[^。?！]*?"
                r"(找|檢索|提取|定位|找出)[^。?！]*?"
                r"(片段|時段|位置|區間|語義|段落|畫面|瞬間|對話|音訊|語音|內容)",
                re.IGNORECASE,
            ),
        ],
        weight=1.10,
        description="在影片/錄音/監控內搜尋特定時刻或內容",
    ))
    rules.append(Rule(
        rule_id="vr_find_clip",
        target_pipeline="VideoRAG",
        patterns=[
            re.compile(
                r"\b(find (the|that|which)? (clip|segment|footage|video|scene|shot|moment))|"
                r"(locate (the|that)? (video|clip|segment|footage))|"
                r"(show (me )?(the|that)? (video|clip|footage|scene|shot))|"
                r"找(到)?(.*?)(的)?(畫面|片段|影片|影像|鏡頭|瞬間)|"
                r"(.*?)(的)?(那個)?(瞬間|時刻|時候)(的)?(影片|畫面|片段)?",
                re.IGNORECASE,
            ),
        ],
        weight=0.85,
        description="找片段/畫面查詢",
    ))

    # ---- RDBMS rules ----
    rules.append(Rule(
        rule_id="rdbms_db_query",
        target_pipeline="RDBMS",
        patterns=[
            re.compile(
                # 明確 SQL/DB 操作（有 table/系統 上下文）
                r"(查詢|檢索|從)[^。?！]*?(資料庫|數據庫|系統|表|名冊|庫)[^。?！]*?(清單|名單|記錄|明細|清冊|編號|資料)|"
                # 列出/找出 + 業務實體
                r"(統計|計算|列出|找出|提取|篩選)[^。?！]*?(訂單|客戶|會員|用戶|帳號|員工|庫存|帳款|薪資|採購|合約|保險|票據|商品|產品)|"
                # 業務實體 + 清單/數量
                r"(訂單|客戶|會員|用戶|帳號|帳戶)[^。?！]*?(清單|名單|記錄|明細|統計|數量|篩選|到期)|"
                r"(員工|人員|職員)[^。?！]*?(名冊|清單|年資|薪資|考勤|績效)|"
                r"(庫存|倉儲|進貨|出貨)[^。?！]*?(數量|清單|記錄|批次|明細)|"
                # 統計/平均 + 分組維度（不需要特定業務詞）
                r"統計[^。?！]*(各|每個|每家|各部門|各分店|各區域|各類)[^。?！]*(平均|總計|成長|佔比|數量|金額)|"
                # 登入/帳號記錄（登入本身是結構化 log，不含黑名單以免打到圖譜查詢）
                r"(登入|登錄|異地登入)[^。?！]*(記錄|帳號|清單)|"
                # English 業務查詢
                r"\b(retrieve|query|select|count|calculate|list|find)\b[^.!?]*"
                r"\b(records?|databases?|tables?|inventor(?:y|ies)|orders?|customers?|employees?|transactions?|invoices?|contracts?|products?|sales?|revenues?|accounts?|users?)\b",
                re.IGNORECASE,
            ),
        ],
        weight=1.00,
        description="資料庫/業務記錄結構化查詢（訂單/客戶/員工/庫存）",
    ))
    rules.append(Rule(
        rule_id="rdbms_file_document",
        target_pipeline="RDBMS",
        patterns=[
            re.compile(
                r"\b(PDF|Word|Excel|Powerpoint|PowerPoint|csv|xlsx|docx|pptx)\b|"
                r"(檔案|文件|文檔|表單|合約|申請單|憑證|發票|報表|清冊)"
                r"[^。?！]*?(清單|名單|列表|目錄|索引|記錄|編號)|"
                r"(統計|計算|佔比|百分比|總數|數量|份數|筆數)"
                r"[^。?！]*?(文件|檔案|記錄|報告|表單|合約)|"
                r"(上傳|修改|存取|下載|刪除)[^。?！]*?(檔案|文件|記錄)|"
                r"(元數據|metadata|標籤|標記|版本號|大小|格式)[^。?！]*?(檔案|文件)|"
                r"\b(how many|count|total|filter by|sort by|list all|find all)\b"
                r"[^.!?]*\b(document|file|report|contract|form|invoice|record)\b",
                re.IGNORECASE,
            ),
        ],
        weight=0.90,
        description="文件/檔案系統 metadata 查詢（PDF/合約/報表）",
    ))

    # ---- GraphRAG rules ----
    rules.append(Rule(
        rule_id="graphrag_graph_topology",
        target_pipeline="GraphRAG",
        patterns=[
            re.compile(
                # 圖譜/知識圖 + 圖操作詞（domain-agnostic，捕捉圖結構操作意圖）
                r"圖譜[^。?！]*?(路徑|節點|交集|遍歷|關聯|相關|核心|影響範圍)|"
                r"(知識圖|社群網路|關係網路|實體網路)[^。?！]*?(節點|核心|路徑|關聯)|"
                # 圖論術語本身即強信號
                r"(核心節點|關鍵節點|中心節點|樞紐節點)|"
                r"\b(graph traversal|network traversal|node centrality|path analysis|shortest path)\b|"
                # 交集/重疊 + 圖論物件
                r"(交集|重疊路徑)[^。?！]*(實體|關係|節點)|"
                # 圖譜 + 英文圖操作
                r"\b(graph|knowledge graph)[^.!?]*(path|node|edge|traversal|centrality)\b",
                re.IGNORECASE,
            ),
        ],
        weight=1.10,
        description="圖結構操作查詢（圖譜路徑/節點/交集，不限 domain）",
    ))
    rules.append(Rule(
        rule_id="graphrag_relationship",
        target_pipeline="GraphRAG",
        patterns=[
            re.compile(
                r"\b(relation|related|connected|connection|between|associated|linked|"
                r"relationship|competitor|rival|partner|collaborator|co-occur|together|"
                r"network|ties|affili|coalit|co-star|same scene|together|"
                r"關聯|相關|連結|競爭|夥伴|合作|同台|共現|共同|網絡|派系|盟友|"
                r"合演|合作對象|商業恩怨|恩怨脈絡|合作關係|關係)\b|"
                r"(與.*的關係|.*和.*的|誰與.*|.*之間的)\b",
                re.IGNORECASE,
            ),
        ],
        weight=0.85,
        description="實體關係查詢",
    ))
    rules.append(Rule(
        rule_id="graphrag_collaboration",
        target_pipeline="GraphRAG",
        patterns=[
            re.compile(
                r"\b(collaborat|partner|co-(star|found|host|organize)|joint|together ("
                r"organized|hosted|produced|released)|working (with|alongside)|"
                r"business (relationship|connection|tie|往来)|"
                r"supply chain|long-term (supplier|partner)|"
                r"合作|往來|聯繫|聯結|共事|供貨|資金)\b|"
                r"(.*?)(和|與|跟)(.*?)(的)?(關係|合作|往來|聯繫)\b",
                re.IGNORECASE,
            ),
        ],
        weight=0.80,
        description="合作/往來查詢",
    ))
    rules.append(Rule(
        rule_id="graphrag_conflict",
        target_pipeline="GraphRAG",
        patterns=[
            re.compile(
                r"\b(competitor|rival|rivalry|conflict|dispute|feud|opposition|"
                r"clash|standoff|tension|controversy|land (dispute)|"
                r"commercial feud|business rivalry|legal (battle|fight|dispute)|"
                r"競爭|對手|爭議|恩怨|對立|衝突|糾紛|對陣)\b",
                re.IGNORECASE,
            ),
        ],
        weight=0.80,
        description="競爭/爭議查詢",
    ))
    rules.append(Rule(
        rule_id="graphrag_citation",
        target_pipeline="GraphRAG",
        patterns=[
            re.compile(
                r"\b(cite|reference|cited|referenced|quote|quoted|mutual citation|"
                r"cross-reference|bibliography|academic (network|connection)|scholar|forum|conference|"
                r"引用|被引用|互相引用|引用關係|學術網路|論文引用|學者|論壇|會議)\b",
                re.IGNORECASE,
            ),
        ],
        weight=0.85,
        description="引用/學術關係查詢",
    ))
    rules.append(Rule(
        rule_id="graphrag_hierarchy",
        target_pipeline="GraphRAG",
        patterns=[
            re.compile(
                r"\b(subsidiary|affiliate|parent company|holding company|corporate structure|"
                r"ownership|stakeholder|shareholder|board member|executive|"
                r"子公司|母公司|控股公司|企業結構|股權|持股|股東|董事|高層)\b|"
                r"(旗下|隸屬|所屬|集團|旗下所有|企業集團)\b",
                re.IGNORECASE,
            ),
        ],
        weight=0.85,
        description="企業階層/所有權查詢",
    ))
    rules.append(Rule(
        rule_id="graphrag_career",
        target_pipeline="GraphRAG",
        patterns=[
            re.compile(
                r"\b(career|work history|employment|worked at|joined|left|transition|"
                r"move to|switch to|former (employee|executive|colleague)|tech company|senior executive|"
                r"職業|工作經歷|就職|離職|轉職|跳槽|前.*員工|前同事|科技公司|高層)\b",
                re.IGNORECASE,
            ),
        ],
        weight=0.85,
        description="職業/工作轉換查詢",
    ))

    # ---- TKG rules ----
    rules.append(Rule(
        rule_id="tkg_temporal",
        target_pipeline="TKG",
        patterns=[
            re.compile(
                r"\b(before|after|during|when|while|then|next|prior|earlier|later|"
                r"timeline|sequence|order of events|evolution|progression|chronolog|"
                r"先後|時間軸|順序|演進|演變|期間|到現在|以來|歷程|時間順序)\b|"
                r"(從.*到.*的|在.*之前|在.*之後|當.*時|隨著|逐漸|發展過程)\b",
                re.IGNORECASE,
            ),
        ],
        weight=0.85,
        description="時間順序/演進查詢",
    ))
    rules.append(Rule(
        rule_id="tkg_lifecycle_sequence",
        target_pipeline="TKG",
        patterns=[
            re.compile(
                # 從X到Y的時序/歷程/生命週期
                r"從[^。?！]*?到[^。?！]*?(時序|歷程|生命週期|演進|發展史|大事紀)|"
                # 完整生命週期 / 賽事時序
                r"(完整|全部)[^。?！]*?(生命週期|歷程|發展史|時序)|"
                # 追蹤 + 時間軸概念
                r"追蹤[^。?！]*?(時序|歷程|演變|大事紀|生命週期|歷任)|"
                # 歷任 / 成員加入退出史
                r"(歷任|歷代)[^。?！]*(負責人|主管|執行長|成員|領導)|"
                r"(加入|退出)[^。?！]*?(史|順序|時序|歷程)|"
                # 時態圖譜/時態關係圖 → TKG
                r"時態(圖譜|關係圖|知識圖|網路)|"
                # 在X期間所有發生過的事件/記錄
                r"(在|於)[^。?！]*?(期間|運行期間)[^。?！]*?(所有|發生過的|歷次)[^。?！]*(事件|記錄|操作|維護)",
                re.IGNORECASE,
            ),
        ],
        weight=1.05,
        description="時序/生命週期/歷任清單/時態圖譜查詢",
    ))
    rules.append(Rule(
        rule_id="tkg_before_after",
        target_pipeline="TKG",
        patterns=[
            re.compile(
                r"\b(what (happened|occurred|took place) (before|prior to|leading up to|"
                r"beforehand))|"
                r"(what (came|happened) (after|following|subsequently))|"
                r"(what|which) (event|incident|announcement) (came|happened) (first|before|prior)|"
                r"(.*?)(之前|以前|先前)(.*?)(的)?(事件|事情|發生)|"
                r"(.*?)(之後|以後|隨後)(.*?)(的)?(事件|事情|發生)|"
                r"(.*?)(隨著|逐漸|發展)(過程|歷程|演變)",
                re.IGNORECASE,
            ),
        ],
        weight=0.82,
        description="之前/之後時間查詢",
    ))
    return rules


def _rule_based_classify(query: str) -> Tuple[Dict[str, float], List[dict]]:
    """
    Rule-based classifier that returns scores per pipeline.

    Returns:
         (pipeline_scores, matched_rules)
        pipeline_scores: Dict[pipeline_name, score] e.g. {"VideoRAG": 0.75, "GraphRAG": 0.6}
        matched_rules: List of matched rule details
    """
    rules = _load_default_rules()
    pipeline_scores: Dict[str, float] = {}
    matched_rules: List[dict] = []

    for rule in rules:
        if not rule or not rule.patterns:
            continue
        score = 0.0
        matched_patterns: List[str] = []
        for pattern in rule.patterns:
            if pattern.search(query):
                score += rule.weight
                matched_patterns.append(pattern.pattern)
        if score > 0:
            matched_rules.append({
                "rule": rule,
                "score": score,
                "matched_patterns": matched_patterns,
            })
            # Accumulate scores per pipeline
            pipeline_scores[rule.target_pipeline] = pipeline_scores.get(rule.target_pipeline, 0.0) + score

    # Normalize scores to [0, 1]
    if pipeline_scores:
        max_score = max(pipeline_scores.values())
        if max_score > 0:
            pipeline_scores = {k: min(v / max_score, 1.0) for k, v in pipeline_scores.items()}

    return pipeline_scores, matched_rules


# ---------------------------------------------------------------------------
# Tier 3/4: LLM re-ranker and fallback decision
# ---------------------------------------------------------------------------

_LLM_RERANK_PROMPT = """\
You are a query routing expert. Given a user query and pipeline scores from three classifiers,
determine the final ranking of pipelines.

## Available Pipelines
VideoRAG - Search for visual/auditory content: scenes, clips, footage, who did what, who said what, specific moments, actions, behaviors
GraphRAG - Query relationships BETWEEN entities: collaborations, competitions, connections, networks, citations, hierarchies, career moves
TKG       - Query temporal sequences: timelines, evolution, before/after, chronological order, progression, history
RDBMS      - Query lists/filters: find all, list, catalog, filter by metadata (date, channel, reporter), statistics

## Key Distinctions
- VideoRAG vs GraphRAG: VideoRAG = "what happened in the video" (visual content). GraphRAG = "how are entities related" (relationships, networks).
- GraphRAG vs RDBMS: GraphRAG = "what is the relationship between X and Y" or "network of connections". RDBMS = "list all videos about X".
- TKG vs VideoRAG: TKG = "what happened before/after X" or "timeline of events". VideoRAG = "find the clip of X happening".
- If query mentions "relationship", "connection", "network", "citation", "hierarchy", "career", "worked at" -> GraphRAG
- If query mentions "timeline", "evolution", "before/after", "sequence", "progression" -> TKG
- If query mentions "list", "all", "find all", "catalog", "statistics" -> RDBMS
- Otherwise, default to VideoRAG for visual content queries

## Input
Query: {query}

Pipeline scores from BM25:
{bm25_scores}

Pipeline scores from FAISS:
{faiss_scores}

Pipeline scores from Rule-based:
{rule_scores}

## Task
1. Analyze the query intent based on keywords and context
2. Consider all three score sources
3. If scores are zero or low, use the key distinctions above
4. Output final pipeline ranking (comma-separated, highest first)

## Output Format
Analysis: [brief 1-2 sentence analysis]
Pipelines: [pipeline1,pipeline2,...]

Example:
Query: 找著名古堡在節日夜晚被燈光點亮的影片
BM25 scores: VideoRAG=0.85, RDBMS=0.3
FAISS scores: VideoRAG=0.8, GraphRAG=0.4
Rule scores: VideoRAG=0.9, GraphRAG=0.4
Analysis: Query asks for visual footage of a castle with lights at night. Strong VideoRAG signal.
Pipelines: VideoRAG,GraphRAG
"""


def _call_llm_rerank(query: str, bm25_scores: Dict[str, float], faiss_scores: Dict[str, float], rule_scores: Dict[str, float]) -> List[str]:
    """Use LLM to re-rank pipelines when scores are close or when no signals exist."""
    bm25_str = "\n".join(f"    {k}={v:.3f}" for k, v in sorted(bm25_scores.items(), key=lambda x: -x[1]))
    faiss_str = "\n".join(f"    {k}={v:.3f}" for k, v in sorted(faiss_scores.items(), key=lambda x: -x[1]))
    rule_str = "\n".join(f"    {k}={v:.3f}" for k, v in sorted(rule_scores.items(), key=lambda x: -x[1]))

    payload = {
        "task": "text-generation-ollama",
        "engine": "ollama",
        "model_name": get_settings().INFERENCE_MODEL,
        "data": {"inputs": _LLM_RERANK_PROMPT.format(
            query=query,
            bm25_scores=bm25_str,
            faiss_scores=faiss_str,
            rule_scores=rule_str,
        )},
        "options": {"temperature": 0, "max_length": 256, "think": False},
    }
    try:
        resp = httpx.post(
            f"{get_settings().INFERENCE_URL}/inference/infer",
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=get_settings().INFERENCE_TIMEOUT,
        )
        resp.raise_for_status()
        raw = resp.json().get("result", {}).get("generated_text", "").strip()
        _logger.debug("LLM re-rank response: %r", raw)
    except Exception as e:
        _logger.warning("LLM re-rank unavailable (%s), using fused scores", e)
        return []

    # Extract pipelines from response
    for line in raw.split("\n"):
        if line.strip().startswith("Pipelines:"):
            tokens = re.split(r"[,\s]+", line.split(":", 1)[1].strip())
            return [t.strip() for t in tokens if t.strip() in _VALID_PIPELINES]

    return []


# ---------------------------------------------------------------------------
# Score fusion helper (fallback when LightGBM is not available)
# ---------------------------------------------------------------------------

def _fuse_with_fixed_weights(
    bm25_scores: Dict[str, float],
    faiss_scores: Dict[str, float],
    rule_scores: Dict[str, float],
    w_bm25: float = 0.4,
    w_faiss: float = 0.3,
    w_rule: float = 0.3,
) -> Dict[str, float]:
    """
    Fuse BM25, FAISS, and Rule-based scores with fixed weights.

    Args:
        bm25_scores: Scores from BM25 classifier
        faiss_scores: Scores from FAISS classifier
        rule_scores: Scores from Rule-based
        w_bm25: Weight for BM25 scores
        w_faiss: Weight for FAISS scores
        w_rule: Weight for Rule-based scores

    Returns:
        Fused scores per pipeline
    """
    fused: Dict[str, float] = {}
    all_pipelines = set(bm25_scores.keys()) | set(faiss_scores.keys()) | set(rule_scores.keys())

    for pipeline in all_pipelines:
        bm25_val = bm25_scores.get(pipeline, 0.0)
        faiss_val = faiss_scores.get(pipeline, 0.0)
        rule_val = rule_scores.get(pipeline, 0.0)
        fused[pipeline] = w_bm25 * bm25_val + w_faiss * faiss_val + w_rule * rule_val

    return fused


def _select_top_k(
    fused_scores: Dict[str, float],
    k: int = 3,
    threshold: float = 0.15,
) -> tuple:
    """
    Select top pipeline(s), returning whether LLM re-rank is needed.

    In normal cases, return only the single highest-scoring pipeline.
    LLM re-rank is triggered only when:
        - Case 2: Top score is too low (< 0.3) → return 1 pipeline + LLM
        - Case 3: Top-2 scores are too close (gap < threshold) → return 2 pipelines + LLM

    Args:
        fused_scores: Fused scores per pipeline
        k: Maximum number of top pipelines to select (only used in edge cases)
        threshold: Score gap threshold for triggering LLM re-rank

    Returns:
         (selected_pipelines, need_llm_rerank)
    """
    sorted_pipelines = sorted(fused_scores.items(), key=lambda x: -x[1])

    if not sorted_pipelines:
        return [], False

    top_pipeline, top_score = sorted_pipelines[0]

    # Case 2: If top score is very low, need LLM help to decide
    if top_score < 0.3:
        return [top_pipeline], True

    # Case 3: Check if top-2 scores are too close
    if len(sorted_pipelines) >= 2:
        second_pipeline, second_score = sorted_pipelines[1]
        gap = top_score - second_score

        if gap < threshold:
            # Scores are close, need LLM re-rank to disambiguate
            return [top_pipeline, second_pipeline], True

    # Case 4 (normal): Return only the single highest-scoring pipeline
    return [top_pipeline], False


# ---------------------------------------------------------------------------
# Main classifier service
# ---------------------------------------------------------------------------

class IntentClassifierService:
    """
    LightGBM-based pipeline classifier with fallback to fixed rules.

    Flow:
         1. Get BM25 scores per pipeline
         2. Get FAISS scores per pipeline
         3. Get Rule scores per pipeline
         4. Use LightGBM to directly classify pipeline (or fixed rules fallback)
         5. Select top-1 pipeline
         6. If top-2 scores are close -> LLM re-rank
         7. If no signals from QIC or Rules -> LLM decides first, then fallback
    """

    def __init__(self) -> None:
        _logger.info("Initializing LightGBM-based Pipeline Classifier...")
        self._hybrid = HybridClassifier()
        self._rules = _load_default_rules()
        self._lightgbm_model = None
        s = get_settings()
        self._use_lightgbm = s.USE_LIGHTGBM_FUSION

        if self._use_lightgbm:
            try:
                model_path = s.LIGHTGBM_MODEL_PATH
                self._lightgbm_model = LightGBMRanker(model_path=model_path)
                _logger.info(f"LightGBM ranker model loaded from {model_path}")
            except Exception as e:
                _logger.warning(
                    f"Failed to load LightGBM model at {model_path}: {e}. "
                    "Falling back to fixed rules."
                )
                self._use_lightgbm = False
        else:
            _logger.info("Using fixed rules (LightGBM disabled)")

        _logger.info(
            "Intent classifier ready -- BM25 + FAISS + Rule scores -> LightGBM Ranker -> top-1 (LLM re-rank when gap < threshold)"
        )

    def classify(self, query: str, k: int = 3) -> ClassificationResult:
        # ------------------------------------------------------------------
        # Tier 0: Pre-QIC guard for unambiguous VideoRAG patterns
        # ------------------------------------------------------------------
        if _VR_SEGMENT_RE.search(query) or _VR_IN_MEDIA_RE.search(query):
            _logger.debug("Pre-QIC guard matched: VideoRAG segment/in-media query")
            return _build_result(["VideoRAG"], 0.9, "pre_qic_guard", [], k)

        # ------------------------------------------------------------------
        # Tier 1: Get BM25 and FAISS scores (QIC components)
        # ------------------------------------------------------------------
        bm25_scores: Dict[str, float] = {}
        faiss_scores: Dict[str, float] = {}
        faiss_intents: List[Candidate] = []

         # 新增：完整分數（用於分佈特徵）
        bm25_all_scores: Dict[str, List[float]] = {}
        faiss_all_scores: Dict[str, List[float]] = {}

        try:
             # Get raw BM25 scores (Max-Pooling for LightGBM input)
            bm25_scores = self._hybrid.bm25_classifier._max_pooling(
                self._hybrid.bm25_classifier._compute_scores(query)
             ) or {}

             # 新增：獲取所有 BM25 分數（按 pipeline 分組）
            try:
                bm25_all_scores = self._hybrid.bm25_classifier._group_scores_by_pipeline(
                    self._hybrid.bm25_classifier._compute_scores(query)
                 )
            except Exception as e:
                 _logger.warning("Failed to get all BM25 scores: %s", e)

            # Get raw FAISS scores
            faiss_result = self._hybrid.faiss_classifier.classify(query, k=k)
            faiss_scores = {
                c.target_pipeline: c.confidence
                for c in faiss_result.intents if c.target_pipeline
                }
            faiss_intents = faiss_result.intents

            # 新增：獲取所有 FAISS 分數（按 pipeline 分組）
            try:
                faiss_all_scores = self._hybrid.faiss_classifier.get_all_faiss_scores(query)
            except Exception as e:
                  _logger.warning("Failed to get all FAISS scores: %s", e)

            _logger.debug(
                  "BM25 scores: %s",
                  dict(sorted(bm25_scores.items(), key=lambda x: -x[1])),
            )
            _logger.debug(
                  "FAISS scores: %s",
                  dict(sorted(faiss_scores.items(), key=lambda x: -x[1])),
            )
        except Exception as exc:
               _logger.warning("QIC components failed (%s); using Rule-based only", exc)

        # ------------------------------------------------------------------
        # Tier 2: Rule-based (get scores per pipeline)
        # ------------------------------------------------------------------
        rule_scores, matched_rules = _rule_based_classify(query)
        if rule_scores:
            _logger.debug(
                "Rule-based scores: %s",
                dict(sorted(rule_scores.items(), key=lambda x: -x[1])),
            )

        # ------------------------------------------------------------------
        # Tier 3: Pipeline classification (LightGBM or fixed rules)
        # ------------------------------------------------------------------
        if not bm25_scores and not faiss_scores and not rule_scores:
            # No signals from any source -> try LLM first (if enabled), then fallback to VideoRAG
            _logger.warning("No signals from any source; falling back to VideoRAG")
            if get_settings().USE_LLM_RERANK:
                llm_pipelines = _call_llm_rerank(query, {}, {}, {})
                if llm_pipelines:
                    _logger.info("LLM decided: %s", llm_pipelines)
                    return _build_result(llm_pipelines, 0.5, "llm_fallback", [], k)
            return _build_result(["VideoRAG"], 0.0, "fallback", [], k)

        # Use LightGBM Ranker or fixed rules
        ranked_pipelines: List[Tuple[str, float]] = []

        if self._use_lightgbm and self._lightgbm_model:
              # 傳遞完整分數給 LightGBM（用於分佈特徵計算）
            ranked_pipelines = classify_with_lightgbm_ranker(
                bm25_scores=bm25_scores,
                faiss_scores=faiss_scores,
                rule_scores=rule_scores,
                ranker=self._lightgbm_model,
                query=query,
                matched_rules=matched_rules,
                bm25_all_scores=bm25_all_scores,
                faiss_all_scores=faiss_all_scores,
              )
            predicted_pipeline = ranked_pipelines[0][0]
            confidence = float(ranked_pipelines[0][1])
            fused_scores = {p: float(s) for p, s in ranked_pipelines}
            _logger.debug(
                "LightGBM ranker: pipeline=%s, confidence=%.3f for query: %s",
                predicted_pipeline, confidence, query[:50],
            )
        else:
            predicted_pipeline, confidence = classify_with_fixed_rules(
                bm25_scores, faiss_scores, rule_scores
            )
            fused_scores = {predicted_pipeline: confidence}
            _logger.debug(
                "Fixed rules classified: pipeline=%s, confidence=%.3f",
                predicted_pipeline, confidence,
            )

        # ------------------------------------------------------------------
        # Tier 4: LLM re-rank
        # Ranker mode: trigger when top-1/top-2 score gap < threshold
        # Fixed-rules mode: trigger when cross-source agreement is low
        # ------------------------------------------------------------------
        need_llm = False

        if self._use_lightgbm and len(ranked_pipelines) >= 2:
            gap = ranked_pipelines[0][1] - ranked_pipelines[1][1]
            if gap < get_settings().LIGHTGBM_RERANK_THRESHOLD:
                need_llm = True
                _logger.debug(
                    "Ranker top-1/top-2 gap=%.3f < threshold=%.3f; triggering LLM re-rank",
                    gap, get_settings().LIGHTGBM_RERANK_THRESHOLD,
                )
        elif not self._use_lightgbm:
            bm25_top1 = max(bm25_scores, key=bm25_scores.get) if bm25_scores else None
            faiss_top1 = max(faiss_scores, key=faiss_scores.get) if faiss_scores else None
            rule_top1 = max(rule_scores, key=rule_scores.get) if rule_scores else None
            agreement_count = sum([
                bm25_top1 == predicted_pipeline if bm25_top1 else False,
                faiss_top1 == predicted_pipeline if faiss_top1 else False,
                rule_top1 == predicted_pipeline if rule_top1 else False,
            ])
            total_sources = sum([bool(bm25_scores), bool(faiss_scores), bool(rule_scores)])
            if total_sources > 0 and agreement_count < total_sources / 2:
                need_llm = True
                _logger.debug(
                    "Fixed rules disagree (%d/%d agree); triggering LLM re-rank",
                    agreement_count, total_sources,
                )

        if need_llm and get_settings().USE_LLM_RERANK:
            llm_pipelines = _call_llm_rerank(query, bm25_scores, faiss_scores, rule_scores)
            if llm_pipelines:
                predicted_pipeline = llm_pipelines[0]
                fused_scores = {predicted_pipeline: 0.7}
            else:
                _logger.debug("LLM re-rank failed; keeping ranker prediction")
        elif need_llm:
            _logger.debug("LLM re-rank skipped (USE_LLM_RERANK=False); keeping ranker prediction")
            need_llm = False

        # ------------------------------------------------------------------
        # Build result
        # ------------------------------------------------------------------
        selected_pipelines = [predicted_pipeline]

        _logger.info(
            "Classified: pipeline=%s, confidence=%.3f, strategy=%s",
            predicted_pipeline,
            confidence,
            "llm_rerank" if need_llm else ("ranker" if self._use_lightgbm else "fixed_rules"),
        )

        intents = [
            Candidate(
                intent_id=predicted_pipeline,
                confidence=confidence,
                target_pipeline=predicted_pipeline,
                rank=1,
            )
        ]

        routing_strategy = "llm_rerank" if need_llm else ("ranker" if self._use_lightgbm else "fixed_rules")
        return ClassificationResult(
            confidence=confidence,
            intent_type=selected_pipelines,
            intents=intents,
            routing_strategy=routing_strategy,
            matched_rules=[r["rule"].rule_id for r in matched_rules[:3]] if matched_rules else [],
        )


def _build_result(
    pipelines: List[str],
    confidence: float,
    routing_strategy: str,
    matched_rules: List[dict],
    k: int = 3,
) -> ClassificationResult:
    unique_pipelines = list(dict.fromkeys(pipelines))
    intents = [
        Candidate(
            intent_id=p,
            confidence=confidence,
            target_pipeline=p,
            rank=i + 1,
        )
        for i, p in enumerate(unique_pipelines[:k])
    ]
    return ClassificationResult(
        confidence=confidence,
        intent_type=unique_pipelines if unique_pipelines else ["VideoRAG"],
        intents=intents,
        routing_strategy=routing_strategy,
        matched_rules=[r["rule"].rule_id for r in matched_rules[:3]] if matched_rules else [],
    )
