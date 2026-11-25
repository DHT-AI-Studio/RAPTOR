"""PA-10 — graph: index 3 entities with relations, traverse, verify paths returned."""
from __future__ import annotations

import pytest
from app.models.graph_index import EntityIndexRequest, RelationshipIndexRequest
from app.models.graph_search import GraphSearchRequest
from app.services import graph_indexer, searcher

pytestmark = pytest.mark.asyncio


async def test_graph_traversal_returns_entities_edges_paths(client, make_db):
    branch = await make_db("ittest_graph")

    # Alpha -knows-> Beta -knows-> Gamma  (chain of 3 entities). Not "A"/"B"/"C": the
    # seed is resolved through the Entity(name) full-text index, which drops English
    # stop words -- "a" is one, so an entity named "A" can never be found as a seed.
    for eid, name in [("a", "Alpha"), ("b", "Beta"), ("c", "Gamma")]:
        await graph_indexer.index_entity(client, branch, EntityIndexRequest(
            entity_id=eid, name=name, type="ORG"))
    await graph_indexer.index_relationship(client, branch, RelationshipIndexRequest(
        from_entity_id="a", to_entity_id="b", relation="knows"))
    await graph_indexer.index_relationship(client, branch, RelationshipIndexRequest(
        from_entity_id="b", to_entity_id="c", relation="knows"))

    resp = await searcher.graph_search(client, branch, GraphSearchRequest(entity_name="Alpha", max_depth=3))

    names = {e["name"] for e in resp.entities}
    assert {"Alpha", "Beta", "Gamma"} <= names            # whole chain reachable from Alpha
    assert len(resp.edges) >= 2                            # both RELATION edges surfaced
    assert any(path and path[-1] == "Gamma" for path in resp.paths)   # a path Alpha -> ... -> Gamma exists
