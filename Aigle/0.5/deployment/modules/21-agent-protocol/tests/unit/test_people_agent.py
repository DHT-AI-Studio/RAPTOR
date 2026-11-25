"""
MTR-25: PeopleAgent (agents/people_search_agent.py) -- the people a question names, by the names and
aliases the user gave them, and their segments from Module 13's /search/people, as the end user.

Run from the 21-agent-protocol/ directory:
    python -m pytest tests/unit/test_people_agent.py -v
"""
import sys
from pathlib import Path
from typing import ClassVar

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT / "app"))
sys.path.insert(0, str(ROOT / "agents"))

import people_search_agent as agent
from models import PeopleSearchRequest

PEOPLE = [
    {"identity_id": "p1", "name": "Peter Parker", "aliases": {"zh-TW": "彼得"}},
    {"identity_id": "p2", "name": "Peter", "aliases": {}},
    {"identity_id": "p3", "name": "Al", "aliases": {}},
    {"identity_id": "p4", "name": "王小明", "aliases": {}},
]


def segment(name, version, trace, first, last, confidence):
    return {"identity_id": name, "identity_name": name, "version_id": version, "trace_id": trace,
            "first_sec": first, "last_sec": last, "confidence": confidence, "filename": f"{version}.mp4",
            "asset_path": f"u/videos/{version}", "keyframe": {"image_path": f"u/videos/{version}/face_{trace}.jpg"}}


SEGMENTS = {
    "Peter Parker": [segment("Peter Parker", "v1", "t1", 0.0, 3.5, 0.7), segment("Peter Parker", "v2", "t3", 10.0, 12.0, 0.95)],
    "Peter": [segment("Peter", "v3", "t9", 1.0, 2.0, 0.9), segment("Peter Parker", "v1", "t1", 0.0, 3.5, 0.7)],
    "王小明": [segment("王小明", "v4", "t4", 5.0, 9.0, 0.8)],
}


class Resp:
    def __init__(self, status, body):
        self.status_code, self._body, self.text = status, body, str(body)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)

    def json(self):
        return self._body


class Client:
    """Module 13: /identity/identities (two pages) and /search/people."""
    calls: ClassVar[list] = []
    people_status = 200

    def __init__(self, *a, **kw):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def get(self, url, headers=None, params=None):
        Client.calls.append(("GET", url, headers["Authorization"], params))
        page = PEOPLE[params["offset"]:params["offset"] + 2]
        return Resp(200, {"total": len(PEOPLE), "items": page})

    def post(self, url, headers=None, json=None):
        Client.calls.append(("POST", url, headers["Authorization"], json))
        if Client.people_status != 200:
            return Resp(Client.people_status, {"detail": {"error": "PERMISSION_DENIED"}})
        return Resp(200, {"results": SEGMENTS.get(json["query"], [])})


def run(monkeypatch, question, auth="Bearer jwt-1", top_k=10, status=200):
    Client.calls, Client.people_status = [], status
    monkeypatch.setattr(agent, "_PAGE", 2)
    monkeypatch.setattr(agent.httpx, "Client", Client)
    monkeypatch.setenv("API_GATEWAY_URL", "http://gw/api/0.5")
    return agent._do_people_search(PeopleSearchRequest(query=question, top_k=top_k, branch_id="u1", auth=auth))


def test_names_are_matched_as_words_for_latin_and_anywhere_for_cjk():
    assert agent._named("find Peter Parker in the video", ["Peter Parker"])
    assert not agent._named("find Alice in the video", ["Al"])                 # a whole word only
    assert agent._named("找出王小明出現的片段", ["王小明"])
    assert agent._named("彼得談論氣候", ["Peter Parker", "彼得"])               # an alias counts


def test_the_people_a_question_names_are_searched_as_the_end_user(monkeypatch):
    res = run(monkeypatch, "find Peter Parker in the video")
    gets = [c for c in Client.calls if c[0] == "GET"]
    assert [c[3]["offset"] for c in gets] == [0, 2]                            # every page of the user's people
    assert all(c[1] == "http://gw/api/0.5/identity/identities" and c[2] == "Bearer jwt-1" for c in gets)
    posts = [c for c in Client.calls if c[0] == "POST"]
    # "Peter Parker" and "Peter" are both named; the longer name first
    assert [(c[1], c[3]["query"]) for c in posts] == [("http://gw/api/0.5/search/people", "Peter Parker"),
                                                      ("http://gw/api/0.5/search/people", "Peter")]
    assert res.source == "people"
    # Peter Parker's segments (surest first), then Peter's; t1 only once
    assert [h.metadata["trace_id"] for h in res.hits] == ["t3", "t1", "t9"]
    h = res.hits[0]
    assert (h.version_id, h.metadata["trace_id"], h.start_time, h.end_time, h.metadata["identity_name"], h.score) == (
        "v2", "t3", 10.0, 12.0, "Peter Parker", 0.95)
    assert (h.metadata["first_sec"], h.metadata["last_sec"]) == (10.0, 12.0)
    assert h.id == "people:v2:t3" and h.asset_path == "u/videos/v2"
    assert h.metadata["keyframe_path"] == "u/videos/v2/face_t3.jpg"
    assert "Peter Parker appears in v2.mp4 from 10.0s to 12.0s" in h.content


def test_cjk_names_and_top_k(monkeypatch):
    assert [h.metadata["identity_name"] for h in run(monkeypatch, "找出王小明出現的片段").hits] == ["王小明"]
    assert len(run(monkeypatch, "find Peter Parker in the video", top_k=1).hits) == 1


def test_a_raw_token_gets_its_bearer_prefix(monkeypatch):
    run(monkeypatch, "find Peter Parker", auth="jwt-raw")
    assert {c[2] for c in Client.calls} == {"Bearer jwt-raw"}


def test_no_one_named_no_token_or_no_consent_finds_no_one(monkeypatch):
    assert run(monkeypatch, "find the explosion in the video").hits == []
    assert not [c for c in Client.calls if c[0] == "POST"]                    # nobody to search for
    assert run(monkeypatch, "find Peter Parker", auth=None).hits == [] and Client.calls == []
    assert run(monkeypatch, "find Peter Parker", status=403).hits == []       # MTR-24 consent not granted
