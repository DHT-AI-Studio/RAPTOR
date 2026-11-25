"""VideoSearchService's behaviour when module 25 fails, with PersonalDBService faked.

Regression for: module 25 down -> every retriever raised, the service swallowed
all four and returned an empty result, so SearchByPerson answered OK with zero
hits ("nobody matched") instead of reporting the outage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))

from app.services.video_search_service import VideoSearchService

USER = {"user_id": "u1", "branch_id": "u1"}


def _down():
    return HTTPException(status_code=503, detail="Personal DB service unreachable")


class _FakePersonalDB:
    """Each retriever either returns a canned payload or raises."""

    def __init__(self, bm25=None, vector=None, graphrag=None, tkg=None):
        self._r = {"bm25": bm25, "vector": vector, "graphrag": graphrag, "tkg": tkg}

    async def ensure_database(self, user_id):
        return None

    def _answer(self, name):
        value = self._r[name]
        if isinstance(value, Exception):
            raise value
        return value if value is not None else {"results": []}

    async def bm25_search(self, user_id, body):
        return self._answer("bm25")

    async def vector_search(self, user_id, body):
        return self._answer("vector")

    async def graphrag_search(self, user_id, body):
        return self._answer("graphrag")

    async def temporal_search(self, user_id, body):
        return self._answer("tkg")

    async def rerank(self, user_id, body):
        return {"results": []}


class _FakeResolver:
    async def resolve_version_urls(self, version_map, user):
        return {}


def _service(db):
    return VideoSearchService(personal_db=db, url_resolver=_FakeResolver())


@pytest.mark.asyncio
async def test_all_retrievers_failing_raises_instead_of_returning_empty():
    svc = _service(_FakePersonalDB(bm25=_down(), vector=_down(), graphrag=_down(), tkg=_down()))
    with pytest.raises(HTTPException) as exc_info:
        await svc.search_by_person("alice", USER)
    assert exc_info.value.status_code == 503


@pytest.mark.asyncio
@pytest.mark.parametrize("failing", ["bm25", "vector", "graphrag", "tkg"])
async def test_one_retriever_failing_still_returns_a_result(failing):
    svc = _service(_FakePersonalDB(**{failing: _down()}))
    result = await svc.search_by_person("alice", USER)
    assert result.videos == []  # the others legitimately found nothing -- and that is not an error


@pytest.mark.asyncio
async def test_three_of_four_failing_is_still_tolerated():
    svc = _service(_FakePersonalDB(bm25=_down(), vector=_down(), graphrag=_down()))
    result = await svc.search_by_person("alice", USER)
    assert result.videos == []


@pytest.mark.asyncio
async def test_all_retrievers_succeeding_with_no_hits_is_a_normal_empty_result():
    svc = _service(_FakePersonalDB())
    result = await svc.search_by_person("nobody", USER)
    assert result.videos == []
