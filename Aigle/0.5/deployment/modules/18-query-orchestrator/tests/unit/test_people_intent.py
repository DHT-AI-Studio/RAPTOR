"""MTR-25 people intent (app/classifiers/people_intent.py) -- the rules and the compound intent, with a
stand-in embedding model; the real classifier on real questions is tests/integration/test_people_intent.py."""
from __future__ import annotations

import faiss
import numpy as np
import pytest
from app.classifiers import people_intent as pi
from app.models.intent import Intent, IntentTaxonomy


class Model:
    """Every text to the same vector: FAISS never tips the decision, so only the rules and BM25 do."""

    def encode(self, texts, **kw):
        return np.ones((len(texts), 4), dtype="float32") / 2


@pytest.fixture(scope="module")
def clf():
    other = faiss.IndexFlatIP(4)
    other.add(np.ones((3, 4), dtype="float32") / 2)
    people = IntentTaxonomy([Intent(id="pp", description="find a person in my videos", target_pipeline="People",
                                    keywords=["person"], examples=["find Peter in the video"])])
    main = IntentTaxonomy([Intent(id="vr", description="find a scene of a storm", target_pipeline="VideoRAG",
                                  keywords=["storm", "scene"], examples=["find the storm"])])
    return pi.PeopleIntentClassifier(Model(), other, main_taxonomy=main, people_taxonomy=people)


@pytest.mark.parametrize("q", [
    "Find Peter Parker in the video", "Show me clips of Mary Jane", "Who appeared in the wedding video?",
    "Peter Parker discussing climate change", "show me all videos of my son",
    "找出王小明出現的片段", "王老師談論微積分的影片", "影片裡有哪些人", "有沒有拍到媽媽", "阿明有入鏡嗎",
])
def test_people_questions(clf, q):
    assert clf.is_people(q)


@pytest.mark.parametrize("q", [
    "find the storm scene",
    "定位影片中所有出現過『紅色警示燈』閃爍的時段。",          # 出現 an object, not a person
    "在極地紀錄片中檢索『極光出現』且伴隨『環境音』的視覺內容。",
    "找出該政治人物家屬持股的所有境外公司環狀路徑。",          # 外公 inside 境外公司
    "在視訊教學中找出老師示範『複數平面運算』的白板書寫過程。",  # 出老師, not a surname + title
    "List all patients with Dr. Smith tomorrow.",
])
def test_not_people_questions(clf, q):
    assert not clf.is_people(q)


@pytest.mark.parametrize("q, temporal, expected", [
    ("Find Peter Parker in the video", False, "people"),
    ("Peter Parker discussing climate change", False, "people+semantic"),
    ("王老師談論微積分的影片", False, "people+semantic"),
    ("When did Peter Parker first appear?", False, "people+temporal"),
    ("小傑第一次出現是在哪支影片", False, "people+temporal"),
    ("找出王小明出現的片段", True, "people+temporal"),        # a date range the signals found
    ("find the storm scene", True, None),
])
def test_compound_intent(clf, q, temporal, expected):
    assert clf.intent(q, temporal=temporal) == expected
