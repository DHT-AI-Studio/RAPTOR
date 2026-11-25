"""
Signal extraction from natural-language queries.
Adapted from mmrag_prototype/app/query/router.py — pure regex, no ML.

Extracts orthogonal parameters that complement QIC intent classification:
  sort_by           - "relevance" | "time_asc" | "time_desc"
  top_k_override    - explicit count hint ("前兩次" → 2)
  time_from/to      - year / decade / century phrases, with after / before / since / until
                      (以後 / 之前 / 以來 / 為止 after the period, TMP-2)
  keywords          - Latin-script terms a TKG search should find in the evidence
  reference         - the event an answer is relative to ("after Google acquired Android"), with
                      reference_relations -- the Allen relations a fact must have to it (P16)
  rel_position      - "end" | "start" (for "last N seconds" queries)
  tail_seconds      - number of seconds for rel_position
  group_by_doc      - "all videos" type queries

Mode hint flags (from Video_Search fast_classify_modes):
  hint_tkg          - temporal cues detected beyond explicit dates
  hint_graphrag     - relational / network cues detected
  hint_rdbms        - metadata / catalog cues detected

Generalized rules based on query intent patterns (not specific keywords):
   - VideoRAG: Visual/auditory content queries (what was seen/heard/done)
   - RDBMS:    Document/video list queries (find, list, filter by metadata)
   - GraphRAG: Entity relationship queries (relations, collaborations, competitions)
   - TKG:      Temporal sequence queries (timeline, evolution, before/after)
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta

# ---------------------------------------------------------------------------
# Signal patterns
# ---------------------------------------------------------------------------

_COUNT_WORDS = {
    "一": 1, "兩": 2, "二": 2, "三": 3, "四": 4, "五": 5,
    "六": 6, "七": 7, "八": 8, "九": 9, "十": 10,
    "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10
}

_COUNT_HINT = re.compile(
    r"(前|最早的?|最近的?|最後的?|最新的?|first|last|earliest|latest|top)\s*"
    r"(\d+|[一二兩三四五六七八九十]|one|two|three|four|five)?"
    r"\s*(次|個|段|部|集|times|videos|clips|moments)?",
    re.IGNORECASE,
)

_REL_POSITION = re.compile(
    r"(最後|最末|末段|後|last|final)\s*(\d+)?\s*(秒|sec|seconds)|"
    r"(最前|開頭|開始|前|first|opening)\s*(\d+)?\s*(秒|sec|seconds)",
    re.IGNORECASE,
)

_CENTURY = re.compile(
    r"(\d{1,2})\s*(世紀|st century|nd century|rd century|th century)",
    re.IGNORECASE,
)
_DECADE = re.compile(r"(\d{4})\s*年代|(\d{3})0s\b")
# Not \b: in 「2011年」 the year is followed by a word character, and \b would miss it (TMP-2)
_YEAR = re.compile(r"(?<![0-9A-Za-z])(?:19|20)\d{2}(?![0-9A-Za-z])")

_GROUP_BY_DOC = re.compile(
    r"\b(all (videos|films|movies|documents|docs|clips|sources)|list of |"
    r"which (videos|films))\b|"
    r"(所有[^。?!]*?(影片|片段|版本|報導|作品|清單|名單|節目)|"
    r"哪些影片|哪些作品|哪些電影|哪些片段|哪些版本|影片清單)",
    re.IGNORECASE,
)

# ---------------------------------------------------------------------------
# Generalized Mode hint patterns (intent-based, not keyword-specific)
# ---------------------------------------------------------------------------

# TKG: Temporal Knowledge Graph hints
# Focus on temporal reasoning patterns, not specific events
_TEMPORAL_CUES = re.compile(
    r"\b(before|after|during|when|while|then|next|prior|earlier|later|"
    r"timeline|sequence|order of events|evolution|progression|chronolog)\b|"
    r"(先後|時間軸|順序|演進|演變|期間|到現在|以來|歷程|時間順序|之前|之後|"
    r"從.*到.*的|在.*之前|在.*之後|當.*時|隨著|逐漸|發展過程)",
    re.IGNORECASE,
)

# GraphRAG: Relational hints
# Focus on relationship query patterns, not specific entities
_RELATIONAL_CUES = re.compile(
    r"\b(relation|related|connected|connection|between|associated|linked|"
    r"relationship|competitor|rival|partner|collaborator|co-occur|together|"
    r"network|ties|affili|coalit|co-star|same scene|together)\b|"
    r"(關聯|相關|連結|競爭|競爭對手|夥伴|合作|同台|共現|共同|網絡|派系|盟友|"
    r"合演|合作對象|商業恩怨|恩怨脈絡|合作關係|關係|"
    r"與.*的關係|.*和.*的|誰與.*|.*之間的)",
    re.IGNORECASE,
)

# RDBMS: Metadata/catalog hints
# Focus on list/filter query patterns
_METADATA_CUES = re.compile(
    r"\b(list (all|every)|find all|how many|count|total|uploaded|published|"
    r"by (channel|reporter|anchor|author|correspondent)|archive|catalog|"
    r"all (videos|reports|clips|broadcasts|segments)|filter by|sort by date)\b|"
    r"(所有.*的|列出|幾則|幾部|幾個|多少|存檔|頻道|電視台|各家|記者|主播|特派|"
    r"查詢.*清單|搜尋.*找到|篩選|依.*查詢|查詢.*資料|包含.*的)",
    re.IGNORECASE,
)

# VideoRAG: Visual/auditory content hints
# Focus on content retrieval patterns, not specific scenes
_VIDEO_CONTENT = re.compile(
    r"\b(what (did|does|do)|who (said|did|performed|acted|played|appeared)|"
    r"action|behavior|goal|score|shoot|kick|throw|"
    r"visual|scene|footage|clip|segment|broadcast)\b|"
    r"(畫面|影像|場景|片段|誰|做了什麼|誰說|誰演|動作|行為|進球|得分|射門|"
    r"什麼|怎麼|如何|哪個|哪些)",
    re.IGNORECASE,
)

_VIDEO_SCENE = re.compile(
    r"\b(big ben|eiffel tower|statue of liberty|landmark|scenery|weather)\b|"
    r"(環境|場景|畫面|影像|景色|風景|街景|城市景觀|天氣|暴風雪|大雨|颶風|颱風)",
    re.IGNORECASE,
)

_VIDEO_FICTIONAL = re.compile(
    r"\b(elf|dwarf|magic|ring|volcano|snow white|fairy|fictional|animated|fantasy)\b|"
    r"(童話|動畫|虛構|奇幻|魔法|矮人|小矮人|火山|戒指|白雪公主)",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class QuerySignals:
    """查詢訊號"""
    # 排序方式
    sort_by: str = "relevance"              # "relevance" | "time_asc" | "time_desc"

    # 數量提示
    top_k_override: int | None = None

    # 時間範圍
    time_from: datetime | None = None
    time_to: datetime | None = None

    # TKG 證據關鍵字（TMP-4）
    keywords: list[str] = field(default_factory=list)

    # TKG 的 Allen 比對方式（P07）：見 tkg_match()
    match: str | None = None

    # 參照事件（P16）：見 extract_reference()
    reference: str | None = None
    reference_relations: list[str] | None = None

    # 相對位置
    rel_position: str | None = None      # "end" | "start"
    tail_seconds: int | None = None

    # 文件分組
    group_by_doc: bool = False

    # 管道提示旗標
    hint_tkg: bool = False            # 時間線提示
    hint_graphrag: bool = False       # 關係提示
    hint_rdbms: bool = False          # 元資料提示
    hint_videorag: bool = False       # 視覺/聽覺內容提示


# ---------------------------------------------------------------------------
# Extractors
# ---------------------------------------------------------------------------

def _parse_count_hint(q: str) -> tuple[int | None, str]:
    m = _COUNT_HINT.search(q)
    if not m:
        return None, "relevance"

    direction = (m.group(1) or "").lower()
    num_raw = m.group(2)

    n: int | None = None
    if num_raw:
        if num_raw.isdigit():
            n = int(num_raw)
        else:
            n = _COUNT_WORDS.get(num_raw.lower())

    if any(w in direction for w in ("前", "最早", "first", "earliest")):
        return n, "time_asc"
    if any(w in direction for w in ("最後", "最近", "最新", "last", "latest")):
        return n, "time_desc"
    return n, "relevance"


def _parse_rel_position(q: str) -> tuple[str | None, int | None]:
    m = _REL_POSITION.search(q)
    if not m:
        return None, None
    if m.group(1):
        sec = int(m.group(2)) if m.group(2) else None
        return "end", sec
    if m.group(4):
        sec = int(m.group(5)) if m.group(5) else None
        return "start", sec
    return None, None


# TMP-4 / temporal plan P02: an operator right before a single period turns it into a
# one-sided window -- parsed before the period is reduced to its calendar days.
# "after 2011" starts 2012-01-01; "before 2013" ends 2012-12-31.
_OPERATOR = re.compile(r"\b(after|before|since|until|till)\s+(?:the\s+)?(?:year\s+)?$", re.IGNORECASE)
# TMP-2: in Chinese the operator follows the period -- 「2011年以後」 is "after 2011", 「2013年之前」 "before 2013",
# 「2010年以來」 / 「2010年起」 "since 2010", 「到2012年為止」 "until 2012". An interim parser: TMP-7 moves this into
# Module 25's grammar.
_ZH_OPERATOR = re.compile(r"\s*(?:年代|年)?\s*(以後|之後|之前|以前|以來|起|開始|為止)")
_ZH_OPERATORS = {"以後": "after", "之後": "after", "之前": "before", "以前": "before",
                 "以來": "since", "起": "since", "開始": "since", "為止": "until"}


def _apply_operator(q: str, pos: int, first: datetime, last: datetime,
                    end: int | None = None) -> tuple[datetime | None, datetime | None]:
    """*pos* / *end*: where the period starts and ends in *q*."""
    if m := _OPERATOR.search(q[:pos]):
        op = m.group(1).lower()
    elif end is not None and (m := _ZH_OPERATOR.match(q, end)):
        op = _ZH_OPERATORS[m.group(1)]
    else:
        return first, last
    if op == "after":
        return last + timedelta(days=1), None
    if op == "before":
        return None, first - timedelta(days=1)
    if op == "since":
        return first, None
    return None, last                                   # until / till


def _parse_time_window(q: str) -> tuple[datetime | None, datetime | None]:
    m = _CENTURY.search(q)
    if m:
        c = int(m.group(1))
        return _apply_operator(q, m.start(), datetime(year=(c - 1) * 100 + 1, month=1, day=1),
                               datetime(year=c * 100, month=12, day=31), m.end())
    m = _DECADE.search(q)
    if m:
        y = int(m.group(1) or (m.group(2) + "0"))
        return _apply_operator(q, m.start(), datetime(year=y, month=1, day=1),
                               datetime(year=y + 9, month=12, day=31), m.end())
    years = list(_YEAR.finditer(q))
    if len(years) >= 2:
        # in any order ("in 2016 and 2010"): the earliest to the latest, never an inverted window
        ys = [int(y.group()) for y in years]
        return datetime(min(ys), 1, 1), datetime(max(ys), 12, 31)
    if len(years) == 1:
        y = int(years[0].group())
        return _apply_operator(q, years[0].start(), datetime(y, 1, 1), datetime(y, 12, 31), years[0].end())
    return None, None


# TMP-4: the words a TKG search should find in a fact's evidence. Latin-script terms
# only -- names like "LG", "Hugo Barra", "Android" -- minus question words and the
# temporal phrase. Chinese text is not segmented here: a wrongly cut term would have to
# appear in the evidence verbatim and silently empty the result, so a Chinese question
# falls back to temporal-only retrieval instead.
_TERM = re.compile(r"[A-Za-z][A-Za-z0-9&'.+-]*[A-Za-z0-9+]|[A-Za-z]")
_STOPWORDS = frozenset({
    "a", "about", "after", "all", "an", "and", "any", "are", "as", "at", "be", "been", "before", "between",
    "by", "can", "could", "did", "do", "does", "done", "during", "event", "events", "for", "from", "give",
    "had", "has", "have", "how", "i", "in", "is", "it", "its", "list", "me", "news", "of", "on", "or",
    "please", "s", "show", "since", "tell", "that", "the", "their", "there", "these", "thing", "things",
    "this", "those", "till", "to", "until", "was", "were", "what", "whats", "when", "where", "which", "who",
    "whom", "why", "will", "with", "would", "year", "years", "decade", "decades", "century", "centuries",
    "happen", "happened", "happening", "happens", "occur", "occurred", "going", "go", "went",
    "st", "nd", "rd", "th",
})


def extract_keywords(q: str) -> list[str]:
    text = _CENTURY.sub(" ", q)
    text = _DECADE.sub(" ", text)
    text = _YEAR.sub(" ", text)
    seen: dict[str, str] = {}
    for term in _TERM.findall(text):
        key = term.casefold().strip(".'")
        if key and key not in _STOPWORDS and key not in seen:
            seen[key] = term.strip(".'")
    return list(seen.values())


# P16 (temporal plan §5.11): a relation word followed by an EVENT, not a date, names a reference event.
# "after 2011" stays a window (_OPERATOR); "after Google acquired Android" asks for facts after that event.
# The phrase runs to the end of the clause. Relations, strictest reading (§5.9): "after" is the Allen
# `after` -- an event meeting the reference's end is not after it; "immediately after" is `met-by`.
_DURING = ["starts", "during", "finishes", "equals"]
_REFERENCE_EN = re.compile(
    r"\b(?P<op>immediately after|right after|immediately before|right before|after|before|during|while)\s+"
    r"(?P<ref>[^?.!,;]+)", re.IGNORECASE)
_REFERENCE_ZH = re.compile(r"(?:在|當)?(?P<ref>[^，。？！,.?!；;]+?)(?P<op>之後|以後|之前|以前|期間)")
_EN_RELATIONS = {"after": ["after"], "before": ["before"], "during": _DURING, "while": _DURING,
                 "immediately after": ["met-by"], "right after": ["met-by"],
                 "immediately before": ["meets"], "right before": ["meets"]}
_ZH_RELATIONS = {"之後": ["after"], "以後": ["after"], "之前": ["before"], "以前": ["before"], "期間": _DURING}
_DATE_ONLY = re.compile(r"^\s*(?:the\s+)?(?:year\s+)?(?:early\s+|mid\s+|late\s+)?"
                        r"(?:\d{1,2}\s*(?:st|nd|rd|th)\s+century|(?:19|20)\d{2}s?|\d{3}0s)\b", re.IGNORECASE)
_ZH_DATE_ONLY = re.compile(r"^\s*(?:\d{4}\s*年代?|\d{1,2}\s*世紀)\s*$")


# A pronoun after the relation word ("after that", "在那之後") points back into the question; it names no
# event that could be looked up, so it is not a reference.
_ANAPHORA_EN = frozenset({"that", "this", "it", "then", "these", "those", "which", "there", "them", "him", "her",
                          "so", "now"})
_ANAPHORA_ZH = frozenset({"那", "這", "此", "其", "那時", "這時", "那個", "這個", "那件事", "這件事"})


def _names_an_event(ref: str) -> bool:
    words = re.findall(r"[A-Za-z]+", ref.casefold())
    if words and all(w in _ANAPHORA_EN for w in words) and not re.search(r"[\u4e00-\u9fff]", ref):
        return False
    if ref in _ANAPHORA_ZH:
        return False
    return bool(extract_keywords(ref)) or bool(re.search(r"[\u4e00-\u9fff]{2,}", ref))


def extract_reference(q: str) -> tuple[str | None, list[str] | None, tuple[int, int] | None]:
    """(reference text, Allen relations, span of the whole phrase in q) -- or Nones when the question
    names no reference event. A date after the relation word is a window, not a reference; a pronoun
    ("after that") names nothing, so the next relation phrase is tried."""
    for m in _REFERENCE_EN.finditer(q):
        ref = m.group("ref").strip()
        if ref and not _DATE_ONLY.match(ref) and _names_an_event(ref):
            return ref, list(_EN_RELATIONS[m.group("op").lower()]), m.span()
    for m in _REFERENCE_ZH.finditer(q):
        ref = m.group("ref").strip()
        if ref and not _ZH_DATE_ONLY.match(ref) and _names_an_event(ref):
            return ref, list(_ZH_RELATIONS[m.group("op")]), m.span()
    return None, None, None


def tkg_match(time_from: datetime | None, time_to: datetime | None) -> str | None:
    """P07: how a TKG search should relate facts to the window. A period named in full
    ("in 2010", "2010-2016") asks what was going on then, so a fact that only shares time
    with it counts -- a 2008-2013 role answers "what was X doing in 2010" (`overlaps`).
    One open side ("after 2011", "before 2013") asks for what lies beyond the bound: the fact
    inside the window (`inside`, TMP-2) -- with [2012-01-01, unbounded) that is the strict
    `{after}` of 2011. No window: nothing to match."""
    if time_from is not None and time_to is not None:
        return "overlaps"
    return "inside" if time_from is not None or time_to is not None else None


def _detect_tkg_hints(query: str) -> bool:
    """檢測 TKG (Temporal Knowledge Graph) 提示

    通用檢測邏輯：
      - 時間軸查詢: 詢問事件的時間順序、發展過程
      - 前後關係: 詢問某事件發生前/後的狀況
      - 事件鏈: 詢問多個事件的完整過程
      - 期間查詢: 詢問某段期間內發生的事
      - 演變追蹤: 詢問某事物的演進過程
    """
    # 檢查時間線提示（包含「從...到...」結構）
    return bool(_TEMPORAL_CUES.search(query))


def _detect_graphrag_hints(query: str) -> bool:
    """檢測 GraphRAG 提示

    通用檢測邏輯：
      - 實體關係: 詢問兩個或多個實體之間的關係
      - 共同出現: 詢問實體共同出現的場景
      - 跨作品關係: 詢問跨電影/作品的角色關係
      - 主題共現: 詢問主題之間的關聯
      - 競爭關係: 詢問競爭對手之間的關係
    """
    # 檢查關係提示（包含「關係」結構）
    return bool(_RELATIONAL_CUES.search(query))


def _detect_rdbms_hints(query: str) -> bool:
    """檢測 RDBMS 提示

    通用檢測邏輯：
      - 關鍵字搜尋: 查詢包含特定內容的文件
      - 日期篩選: 依時間範圍篩選
      - 文件清單: 要求列出所有符合條件的文件
      - 元資料查詢: 查詢特定元資料（頻道、記者等）
    """
    # 檢查元資料提示（包含清單/篩選結構）
    return bool(_METADATA_CUES.search(query))


def _detect_videorag_hints(query: str) -> bool:
    """檢測 VideoRAG 提示

    通用檢測邏輯：
      - 視覺場景: 查詢特定視覺畫面或場景
      - 動作/行為: 查詢影片中發生的動作
      - 人物行為: 查詢誰做了什麼、誰說了什麼
      - 重複事件: 查詢歷屆/每年發生的事件
      - 天氣/環境: 查詢特定環境下的影片
      - 虛構場景: 查詢影視作品中的虛構場景
    """
    # 檢查視覺/聽覺內容提示
    if _VIDEO_CONTENT.search(query):
        return True
    # 檢查視覺場景提示
    if _VIDEO_SCENE.search(query):
        return True
    # 檢查虛構場景提示
    return bool(_VIDEO_FICTIONAL.search(query))


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def extract_signals(query: str) -> QuerySignals:
    """
    從查詢中提取訊號

    使用通用意圖特徵檢測四條查詢路徑：
      - VideoRAG: 視覺/聽覺內容查詢（看到什麼、聽到什麼、誰做了什麼）
      - RDBMS:    文件/影片清單查詢（所有、列出、關鍵字篩選）
      - GraphRAG: 實體關係查詢（關係、合作、競爭、共同出現）
      - TKG:      時間順序查詢（時間軸、先後、演進、從...到...）

    Args:
        query: 用戶查詢文本

    Returns:
        QuerySignals: 提取的訊號物件
    """
    top_k, sort_by = _parse_count_hint(query)
    rel_pos, tail_sec = _parse_rel_position(query)
    t_from, t_to = _parse_time_window(query)
    reference, reference_relations, ref_span = extract_reference(query)
    # the reference phrase names the anchor event, not what the answer's facts must mention
    rest = query if ref_span is None else query[:ref_span[0]] + " " + query[ref_span[1]:]

    # 檢測各管道提示（通用意圖特徵）
    hint_tkg = _detect_tkg_hints(query) or reference is not None
    hint_graphrag = _detect_graphrag_hints(query)
    hint_rdbms = _detect_rdbms_hints(query)
    hint_videorag = _detect_videorag_hints(query)

    return QuerySignals(
        sort_by=sort_by,
        top_k_override=top_k,
        time_from=t_from,
        time_to=t_to,
        keywords=extract_keywords(rest),
        match=tkg_match(t_from, t_to),
        reference=reference,
        reference_relations=reference_relations,
        rel_position=rel_pos,
        tail_seconds=tail_sec,
        group_by_doc=bool(_GROUP_BY_DOC.search(query)),
        hint_tkg=hint_tkg,
        hint_graphrag=hint_graphrag,
        hint_rdbms=hint_rdbms,
        hint_videorag=hint_videorag,
    )
