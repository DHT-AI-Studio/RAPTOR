"""MTR-25 people intent through the real classifier (bge-m3, the main FAISS index) and POST /classify.

- the Jira examples: "find [person] in the video" -> people, "[person] discussing [topic]" -> people+semantic
- held-out people questions (none of them in people_intent.yaml): at least 85% found
- the 800 training_data.json questions (none about a person): under 1% called people
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from app.routers.query import router
from app.services.intent_classifier import IntentClassifierService
from fastapi import FastAPI
from fastapi.testclient import TestClient

HELD_OUT = [
    "Find Bruce Wayne in my videos", "Every clip where Lisa appears", "Who showed up at the wedding video?",
    "Show me scenes featuring Uncle Bob", "Alice talking about the merger", "Kevin discussing football",
    "When did Bob first appear in the recordings?", "Peter's appearances in 2021", "Show me videos of my daughter",
    "Grace explaining the plan", "Find Oliver in last summer's clips", "who is in the meeting recording",
    "找出大雄出現的影片", "小明談論考試的片段", "哪些影片有拍到爸爸", "婚禮影片裡有哪些人", "林醫師 2023 年出現的片段",
    "影片中有沒有王大明", "找出所有有小華的畫面", "陳老師講解物理的影片", "找出黃經理出現的影片", "有哪些影片拍到小美",
    "阿明在影片中說了什麼", "王醫師討論手術的片段", "影片裡的人是誰",
]


@pytest.fixture(scope="module")
def service():
    return IntentClassifierService()


@pytest.fixture(scope="module")
def client(service):
    app = FastAPI()
    app.include_router(router)
    app.state.intent_classifier = service
    return TestClient(app)


@pytest.mark.parametrize("q, expected", [
    ("Find Peter Parker in the video", "people"),
    ("找出王小明出現的片段", "people"),
    ("Peter Parker discussing climate change", "people+semantic"),
    ("王老師談論微積分的影片", "people+semantic"),
    ("When did Peter Parker first appear in 2023?", "people+temporal"),
    ("王小明 2022 年出現的影片", "people+temporal"),
    ("Find the clip where the anchor cries on air", None),
    ("What happened after Google acquired Android?", None),
    ("列出所有合約檔案", None),
])
def test_classify_returns_the_people_intent(client, q, expected):
    r = client.post("/classify", json={"query": q})
    assert r.status_code == 200 and r.json()["people_intent"] == expected


def test_held_out_people_questions_are_found(service):
    missed = [q for q in HELD_OUT if not service.people_intent(q, temporal=False)]
    assert len(missed) <= len(HELD_OUT) * 0.15, missed


def test_the_existing_questions_are_left_alone(service):
    data = json.loads((Path(__file__).parents[2] / "app" / "data" / "training_data.json").read_text(encoding="utf-8"))
    flagged = [r["query"] for r in data if service.people_intent(r["query"], temporal=False)]
    assert len(flagged) < len(data) * 0.01, flagged
