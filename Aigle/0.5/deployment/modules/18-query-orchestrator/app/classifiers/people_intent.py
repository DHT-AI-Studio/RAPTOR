"""
People intent (MTR-25): is the question about WHO appears in the user's videos, and what comes with it.

People is a label on top of the pipeline IntentClassifierService picks, not a fifth pipeline: a "who"
can come with a "what" ("Peter discussing the budget") or a "when" ("Peter's first appearance in
2023"). The four-pipeline LightGBM ranker is left as it is.

Signals, the same three as the main classifier:
- BM25 over the people intents (people_intent.yaml) and the main taxonomy's intents, max-pooled per side
- FAISS: how much closer the question is to the nearest people example than to the nearest of the main
  taxonomy's 2400 examples (the main FAISS index itself, so only the people examples are embedded)
- rules: a name-like subject being searched for (a capitalised name, a title or a kinship word, "X出現的"),
  or words only a people question uses (who appeared, 入鏡, 哪些人, ...)

A question is a people question when a rule matches, or when BM25 and FAISS both lean to the people
side. A false positive costs a PeopleAgent call that finds no one -- Module 21 then answers as usual.

The intent returned:
- "people"           -- the person alone: PeopleAgent
- "people+semantic"  -- a person and a topic ("discussing X"): PeopleAgent + vector search, RRF-fused
- "people+temporal"  -- a person and a time or an event: PeopleAgent + TKG
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

import faiss
import jieba
import numpy as np
from rank_bm25 import BM25Okapi

from ..data.intent_taxonomy import load_intent_taxonomy
from ..models.intent import IntentTaxonomy

logger = logging.getLogger(__name__)

PEOPLE = "People"
PEOPLE_TAXONOMY_PATH = str(Path(__file__).resolve().parent.parent / "data" / "people_intent.yaml")
_TASK_DESC = "Determine the user's intent by understanding the full semantic meaning of the query."  # FAISSClassifier's
_GAP = 0.0              # FAISS: nearest people example closer than the nearest other one by more than this,
_STRONG_GAP = 0.03      # ... (with BM25 agreeing) / or by more than this alone

_CJK = r"[一-鿿]"
_SURNAMES = ("陳林黃張李王吳劉蔡楊許鄭謝郭洪曾邱廖賴周徐蘇葉莊呂江何蕭羅高潘簡朱鍾游彭詹胡施沈余盧梁趙顏柯翁魏"
             "孫戴范方宋鄧杜傅侯曹薛丁卓阮馬董溫唐藍石蔣古紀姚連馮歐程湯田康姜白汪鄒尤巫鐘黎涂龔嚴韓袁金童陸夏柳"
             "邵錢伍倪于譚駱熊任甘秦顧毛章史萬俞雷饒凌崔尹孔辛武陶段龍韋葛池孟殷麥賀賈莫")
_KIN = (r"(my|our) (son|daughter|kids?|children|mom|mum|mother|dad|father|parents|grandma|grandmother|grandpa|"
        r"grandfather|wife|husband|brother|sister|uncle|aunt|baby|family)")
_RULES = re.compile(
    r"\b(who (appears|appeared|shows? up|showed up|is in|was in|are in|were in)|"
    r"which (people|persons?) (appear|appeared|show up|showed up|are in|were in)|"
    r"appear(s|ed|ing)? in (my|the|this|these|those) (videos?|clips?|footage|recordings?)|"
    r"appearances? (of|in)|people (seen|shown) in|(this|that) person'?s? face|" + _KIN + r")\b|"
    # a name-like subject before 出現 -- not 出現『an object』 or 『... 出現』
    + _CJK + r"{1,4}有?出現(?!過?[『」』「])(的|過的|是|在|於|過)|"
    r"(入鏡|露臉|現身|拍到了?誰|有誰|哪些人|誰出現|人臉|有沒有拍到|有沒有" + _CJK + r"{2,3}$)|"
    # a kinship word as the one searched for (外公 inside 境外公司, 家人 inside 畫家人脈 are not)
    r"(^|我的?|我們的?|有|拍到|找出?|看)(爸爸|媽媽|阿嬤|阿公|爺爺|奶奶|外婆|外公|兒子|女兒|老公|老婆|哥哥|姊姊|"
    r"姐姐|弟弟|妹妹|小孩|孩子|家人)|"
    # a surname with a title (陳醫師, 王老師) -- not 出老師 in 找出老師示範
    r"[" + _SURNAMES + r"](老師|醫師|醫生|經理|董事長|總經理|教授|先生|小姐|女士)",
    re.IGNORECASE,
)
# a capitalised name next to a search for it -- case matters here ("with" is too common to count)
_NAME = r"[A-Z][a-z]+(?: [A-Z][a-z]+)?"
_RULES_NAME = re.compile(
    rf"\b(?i:find|show(?: me)?|where(?: is|'s| does)|clips? of|videos? of|footage of|featuring|starring)"
    rf"(?: all| every| the)?(?: (?:clips?|videos?|footage|scenes?) (?:of|with))? ({_NAME})\b|"
    rf"^(?:{_NAME})(?:'s)? (?:discuss|talk|speak|spoke|explain|present|first|appear|interview)|"
    rf"\b(?i:did|does|do|is|was|has) {_NAME} (?i:(?:first |last )?(?:appear|show up|come up|show))"
)
# a time asked about -- the main classifier's TKG guess is not used: it calls "discussing climate change" TKG
_TEMPORAL = re.compile(
    r"\b(when|first|last|earliest|latest|oldest|newest|before|after|since|until|timeline|in (time|chronological) "
    r"order|chronolog\w*|over time|appearances? (over|through))\b|"
    r"(何時|什麼時候|最早|最晚|最近|第一次|最後一次|之前|之後|以前|以後|以來|時間順序|依時間|按時間|時間軸)",
    re.IGNORECASE,
)
_TOPIC = re.compile(
    r"\b(discuss(es|ed|ing)?|talk(s|ed|ing)? about|speak(s|ing)? about|spoke about|"
    r"mention(s|ed|ing)?|say(s|ing)? about|said about|explain(s|ed|ing)?|present(s|ed|ing)?|"
    r"on the (topic|subject) of|regarding|about)\b|"
    r"(談論|討論|談到|提到|提及|說到|講到|聊到|講解|介紹|說明|關於)",
    re.IGNORECASE,
)


def _tokenize(text: str) -> list[str]:          # BM25Classifier's
    return [t.strip() for t in jieba.lcut((text or "").lower()) if t.strip()]


class PeopleIntentClassifier:
    def __init__(self, model, main_faiss_index, main_taxonomy: IntentTaxonomy | None = None,
                 people_taxonomy: IntentTaxonomy | None = None):
        """model: the main classifier's SentenceTransformer; main_faiss_index: its index, whose
        vectors (the main taxonomy's examples) are the "not people" side here."""
        self.model = model
        people = people_taxonomy or load_intent_taxonomy(PEOPLE_TAXONOMY_PATH)
        main = main_taxonomy or load_intent_taxonomy()

        docs = [" ".join([i.description] + (i.keywords or [])) for i in people.intents + main.intents]
        self._bm25 = BM25Okapi([_tokenize(d) for d in docs])
        self._bm25_people = np.array([True] * len(people.intents) + [False] * len(main.intents))

        examples = [e for i in people.intents for e in i.examples]
        vectors = model.encode(examples, convert_to_numpy=True, normalize_embeddings=True).astype("float32")
        self._people_index = faiss.IndexFlatIP(vectors.shape[1])
        self._people_index.add(vectors)
        self._other_index = main_faiss_index
        logger.info("People intent ready: %d people examples against %d others", len(examples),
                    main_faiss_index.ntotal)

    def scores(self, query: str) -> dict:
        bm25 = self._bm25.get_scores(_tokenize(query))
        vec = self.model.encode([f"Instruct: {_TASK_DESC}\nQuery: {query}"], convert_to_numpy=True,
                                normalize_embeddings=True).astype("float32")
        people, _ = self._people_index.search(vec, 1)
        other, _ = self._other_index.search(vec, 1)
        return {
            "bm25_people": float(bm25[self._bm25_people].max(initial=0.0)),
            "bm25_other": float(bm25[~self._bm25_people].max(initial=0.0)),
            "faiss_gap": float(people[0][0] - other[0][0]),
            "rule": bool(_RULES.search(query) or _RULES_NAME.search(query)),
        }

    def is_people(self, query: str) -> bool:
        s = self.scores(query)
        bm25_leans = s["bm25_people"] > s["bm25_other"]
        return s["rule"] or s["faiss_gap"] > _STRONG_GAP or (s["faiss_gap"] > _GAP and bm25_leans)

    def intent(self, query: str, *, temporal: bool) -> str | None:
        """None, or "people" / "people+semantic" / "people+temporal". `temporal`: the caller found a date
        range or a reference event in the question; a time word ("first", "最早") counts too."""
        if not self.is_people(query):
            return None
        if temporal or _TEMPORAL.search(query):
            return "people+temporal"
        if _TOPIC.search(query):
            return "people+semantic"
        return "people"
