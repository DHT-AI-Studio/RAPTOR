"""PA-10 — search: index 5 docs, run hybrid / vector / BM25; result count <= top_k
and scores are descending. The query embedder is mocked (deterministic fake
vectors) so the local BGE-M3 model never has to load."""
from __future__ import annotations

import pytest

from app.models.index import ChunkIndexRequest, SourceSummaryIndexRequest
from app.models.search import SearchRequest
from app.services import indexer, searcher
from tests.conftest import fake_vector

pytestmark = pytest.mark.asyncio


def _chunk(cid: str, text: str) -> ChunkIndexRequest:
    return ChunkIndexRequest(chunk_id=cid, type="documents", embedding_type="text",
                             embedding=fake_vector(text), text=text)


async def _seed_five(client, branch) -> None:
    for i in range(5):
        await indexer.index_chunk(client, branch, _chunk(f"c{i}", f"Samsung report chunk {i}"))


def _descending(scores: list[float]) -> bool:
    return all(scores[i] >= scores[i + 1] for i in range(len(scores) - 1))


async def test_vector_search(client, make_db, mock_embed):
    branch = await make_db("ittest_vec")
    await _seed_five(client, branch)
    resp = await searcher.vector_search(client, branch, SearchRequest(query="Samsung", top_k=3))
    assert 0 < len(resp.results) <= 3
    assert _descending([r.score for r in resp.results])


async def test_bm25_search(client, make_db, mock_embed):
    branch = await make_db("ittest_bm25")
    await _seed_five(client, branch)
    resp = await searcher.bm25_search(client, branch, SearchRequest(query="Samsung", top_k=3))
    assert 0 < len(resp.results) <= 3
    assert _descending([r.score for r in resp.results])


async def test_hybrid_search(client, make_db, mock_embed):
    branch = await make_db("ittest_hybrid")
    await _seed_five(client, branch)
    resp = await searcher.hybrid_search(client, branch, SearchRequest(query="Samsung", top_k=3))
    assert 0 < len(resp.results) <= 3
    assert _descending([r.score for r in resp.results])
    assert "embed_sec" in resp.timing and "fusion_sec" in resp.timing and "rerank_sec" in resp.timing


async def test_filter_by_type(client, make_db, mock_embed):
    branch = await make_db("ittest_filter")
    await _seed_five(client, branch)                                   # all documents
    resp = await searcher.vector_search(
        client, branch, SearchRequest(query="Samsung", top_k=5, type="videos"))
    assert resp.results == []                                          # no video chunks


async def test_filter_by_type_list_form_vector(client, make_db, mock_embed):
    """type as a list (IN clause, not the bare-string = clause above) with zero
    matches -- vectorNeighbors() historically ignored an empty filter RID
    subquery and returned unfiltered neighbors instead of none; regression
    guard for that (see _rid_filter_matches() in searcher.py)."""
    branch = await make_db("ittest_filter_list")
    await _seed_five(client, branch)                                   # all documents
    resp = await searcher.vector_search(
        client, branch, SearchRequest(query="Samsung", top_k=5, type=["videos", "audios"]))
    assert resp.results == []


async def test_filter_by_type_zero_match_hybrid(client, make_db, mock_embed):
    """Same empty-filter regression guard as above, but through hybrid_search()'s
    fuse() (vectorNeighbors + BM25 combined) rather than vector_search() alone --
    a filter that empties the vectorNeighbors half must not leak its unfiltered
    neighbors into the RRF-fused result just because the BM25 half correctly
    contributed nothing."""
    branch = await make_db("ittest_filter_hybrid")
    await _seed_five(client, branch)                                   # all documents
    resp = await searcher.hybrid_search(
        client, branch, SearchRequest(query="Samsung", top_k=5, type=["videos"]))
    assert resp.results == []


# ---------------------------------------------------------------------------
# Every SearchRequest filter field, both a real-match and a legitimate
# zero-match case, run through both vector_search() and hybrid_search() --
# both use vectorNeighbors() and were vulnerable to the same empty-filter
# leak regardless of *which* field emptied the candidate set, not just
# `type` (the field the bug was originally found through).
# ---------------------------------------------------------------------------

async def _seed_mixed(client, branch) -> None:
    """One document chunk, one video chunk, one audio chunk (each with its own
    speaker/source/version_id), one archived document chunk, and one
    whole-document Source summary -- covers every filter field, and every
    pairwise/triple combination below, with both a real match and a
    legitimate non-match."""
    await indexer.index_chunk(client, branch, ChunkIndexRequest(
        chunk_id="doc1", type="documents", embedding_type="text",
        embedding=fake_vector("doc1"), text="quarterly revenue report",
        source="pdf", version_id="v-doc-1", status="active", filename="report.pdf",
    ))
    await indexer.index_chunk(client, branch, ChunkIndexRequest(
        chunk_id="vid1", type="videos", embedding_type="text",
        embedding=fake_vector("vid1"), text="interview clip transcript",
        source="mp4", version_id="v-vid-1", status="active", filename="clip.mp4",
        speaker="SPEAKER_00",
    ))
    await indexer.index_chunk(client, branch, ChunkIndexRequest(
        chunk_id="doc2", type="documents", embedding_type="text",
        embedding=fake_vector("doc2"), text="archived old memo",
        source="pdf", version_id="v-doc-2", status="archived", filename="old.pdf",
    ))
    await indexer.index_chunk(client, branch, ChunkIndexRequest(
        chunk_id="aud1", type="audios", embedding_type="text",
        embedding=fake_vector("aud1"), text="voicemail transcript",
        source="mp3", version_id="v-aud-1", status="active", filename="voice.mp3",
        speaker="SPEAKER_01",
    ))
    await indexer.index_source_summary(client, branch, SourceSummaryIndexRequest(
        version_id="v-doc-1", summary="whole document summary text",
        embedding=fake_vector("summary-doc1"), media_type="documents",
        filename="report.pdf", status="active",
    ))


# (filter kwargs, expect at least one result). Speaker/source only exist on
# Chunk rows (not Source's whole-asset summary), so their zero-match cases
# pin embedding_type="text" to keep the Source branch (which has nothing to
# filter speaker/source *by* and would otherwise still contribute the
# unrelated doc1 summary) out of the comparison.
_FILTER_CASES = [
    ("type_documents", {"type": ["documents"]}, True),
    ("type_videos", {"type": ["videos"]}, True),
    ("type_images_zero", {"type": ["images"]}, False),
    ("filename_match", {"filename": ["report.pdf"]}, True),
    ("filename_zero", {"filename": ["nonexistent.pdf"]}, False),
    ("speaker_match", {"speaker": ["SPEAKER_00"], "embedding_type": "text"}, True),
    ("speaker_zero", {"speaker": ["SPEAKER_99"], "embedding_type": "text"}, False),
    ("source_match", {"source": "mp4", "embedding_type": "text"}, True),
    ("source_zero", {"source": "wav", "embedding_type": "text"}, False),
    ("version_id_match", {"version_id": "v-doc-1"}, True),
    ("version_id_zero", {"version_id": "00000000-0000-0000-0000-000000000000"}, False),
    ("embedding_type_text", {"embedding_type": "text"}, True),
    ("embedding_type_summary", {"embedding_type": "summary"}, True),
    ("status_active", {"status": "active"}, True),
    ("status_archived", {"status": "archived"}, True),
]


@pytest.mark.parametrize("name,filters,expect_nonempty", _FILTER_CASES)
async def test_vector_search_every_filter(client, make_db, mock_embed, name, filters, expect_nonempty):
    branch = await make_db(f"ittest_vf_{name}")
    await _seed_mixed(client, branch)
    resp = await searcher.vector_search(
        client, branch, SearchRequest(query="report", top_k=10, **filters))
    if expect_nonempty:
        assert resp.results, f"expected a match for {filters}, got none"
    else:
        assert resp.results == [], f"expected zero matches for {filters}, got {resp.results}"


@pytest.mark.parametrize("name,filters,expect_nonempty", _FILTER_CASES)
async def test_hybrid_search_every_filter(client, make_db, mock_embed, name, filters, expect_nonempty):
    branch = await make_db(f"ittest_hf_{name}")
    await _seed_mixed(client, branch)
    resp = await searcher.hybrid_search(
        client, branch, SearchRequest(query="report", top_k=10, **filters))
    if expect_nonempty:
        assert resp.results, f"expected a match for {filters}, got none"
    else:
        assert resp.results == [], f"expected zero matches for {filters}, got {resp.results}"


# ---------------------------------------------------------------------------
# Filter *combinations* -- 2-4 fields ANDed together. The single-field cases
# above only prove each field narrows correctly on its own; they can't catch
# a bug that only shows up once multiple WHERE clauses combine (e.g. every
# individual field matches something, but the AND of all of them legitimately
# matches nothing). _rid_filter_matches() checks the exact same combined
# WHERE clause the main query uses, so this should generalize -- these cases
# exist to prove that empirically rather than take it on faith.
# ---------------------------------------------------------------------------
_COMBO_CASES = [
    # Two matching fields narrowing to a real, non-empty result.
    ("doc_active", {"type": ["documents"], "status": "active"}, True),
    ("doc_archived", {"type": ["documents"], "status": "archived"}, True),
    ("doc_report_filename", {"type": ["documents"], "filename": ["report.pdf"]}, True),
    ("vid_speaker00", {"type": ["videos"], "speaker": ["SPEAKER_00"]}, True),
    ("aud_speaker01", {"type": ["audios"], "speaker": ["SPEAKER_01"]}, True),
    ("doc_pdf_active", {"type": ["documents"], "source": "pdf", "status": "active"}, True),
    ("doc_pdf_archived", {"type": ["documents"], "source": "pdf", "status": "archived"}, True),
    ("doc_or_vid_active", {"type": ["documents", "videos"], "status": "active"}, True),
    ("v_doc1_text", {"version_id": "v-doc-1", "embedding_type": "text"}, True),
    ("v_doc1_summary", {"version_id": "v-doc-1", "embedding_type": "summary"}, True),
    ("pdf_old_filename", {"source": "pdf", "filename": ["old.pdf"], "status": "archived"}, True),
    ("doc1_full_combo", {"type": ["documents"], "source": "pdf", "filename": ["report.pdf"],
                          "status": "active"}, True),

    # Each field alone would match something, but the AND of them is a real,
    # legitimate empty set -- the exact shape of bug this regression guards.
    ("vid_archived_zero", {"type": ["videos"], "status": "archived"}, False),
    ("doc_clip_filename_zero", {"type": ["documents"], "filename": ["clip.mp4"]}, False),
    ("vid_speaker01_zero", {"type": ["videos"], "speaker": ["SPEAKER_01"]}, False),
    ("aud_speaker00_zero", {"type": ["audios"], "speaker": ["SPEAKER_00"]}, False),
    ("doc_speaker00_zero", {"type": ["documents"], "speaker": ["SPEAKER_00"],
                             "embedding_type": "text"}, False),
    ("doc_mp4_source_zero", {"type": ["documents"], "source": "mp4", "embedding_type": "text"}, False),
    ("vidaud_archived_zero", {"type": ["videos", "audios"], "status": "archived"}, False),
    ("v_vid1_summary_zero", {"version_id": "v-vid-1", "embedding_type": "summary"}, False),
    ("report_speaker00_zero", {"filename": ["report.pdf"], "speaker": ["SPEAKER_00"],
                                "embedding_type": "text"}, False),
    ("mp3_report_filename_zero", {"source": "mp3", "filename": ["report.pdf"],
                                   "embedding_type": "text"}, False),
    ("vid_pdf_source_zero", {"type": ["videos"], "source": "pdf", "embedding_type": "text"}, False),
]


@pytest.mark.parametrize("name,filters,expect_nonempty", _COMBO_CASES)
async def test_vector_search_filter_combinations(client, make_db, mock_embed, name, filters, expect_nonempty):
    branch = await make_db(f"ittest_vc_{name}")
    await _seed_mixed(client, branch)
    resp = await searcher.vector_search(
        client, branch, SearchRequest(query="report", top_k=10, **filters))
    if expect_nonempty:
        assert resp.results, f"expected a match for {filters}, got none"
    else:
        assert resp.results == [], f"expected zero matches for {filters}, got {resp.results}"


@pytest.mark.parametrize("name,filters,expect_nonempty", _COMBO_CASES)
async def test_hybrid_search_filter_combinations(client, make_db, mock_embed, name, filters, expect_nonempty):
    branch = await make_db(f"ittest_hc_{name}")
    await _seed_mixed(client, branch)
    resp = await searcher.hybrid_search(
        client, branch, SearchRequest(query="report", top_k=10, **filters))
    if expect_nonempty:
        assert resp.results, f"expected a match for {filters}, got none"
    else:
        assert resp.results == [], f"expected zero matches for {filters}, got {resp.results}"
