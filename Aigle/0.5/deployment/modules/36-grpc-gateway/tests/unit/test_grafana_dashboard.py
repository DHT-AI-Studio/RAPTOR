"""grafana/raptor-grpc-gateway.json must stay importable and keep pointing at metrics that exist."""
from __future__ import annotations

import json
import re
from pathlib import Path

from interceptors import metrics

_DASHBOARD = Path(__file__).parent.parent.parent / "grafana" / "raptor-grpc-gateway.json"


def _exported_metric_names() -> set[str]:
    names = set()
    for counter in (metrics.REQUEST_COUNTER, metrics.STREAM_BYTES_COUNTER, metrics.STREAM_CANCELLATIONS_COUNTER):
        names.add(counter._name + "_total")
    names.add(metrics.ACTIVE_STREAMS_GAUGE._name)
    for suffix in ("_bucket", "_sum", "_count"):
        names.add(metrics.LATENCY_HISTOGRAM._name + suffix)
    return names


def _dashboard() -> dict:
    return json.loads(_DASHBOARD.read_text())


def test_every_metric_the_dashboard_queries_is_exported_by_the_gateway():
    exported = _exported_metric_names()
    named = set()
    for panel in _dashboard()["panels"]:
        for target in panel["targets"]:
            named |= set(re.findall(r"\bgrpc_server_[a-z_]+\b", target["expr"]))
    assert named, "no gateway metric found in any query"
    assert named <= exported, f"queries use metrics the gateway does not export: {sorted(named - exported)}"


def test_the_dashboard_covers_what_vie01_165_asks_for():
    titles = " | ".join(p["title"].lower() for p in _dashboard()["panels"])
    for needed in ("active streams", "throughput", "latency", "cancellation", "resource_exhausted"):
        assert needed in titles, needed


def test_panels_have_unique_ids_stay_on_the_24_column_grid_and_do_not_overlap():
    seen_ids: set[int] = set()
    cells: set[tuple[int, int]] = set()
    for panel in _dashboard()["panels"]:
        assert panel["id"] not in seen_ids
        seen_ids.add(panel["id"])
        g = panel["gridPos"]
        assert g["x"] + g["w"] <= 24
        for x in range(g["x"], g["x"] + g["w"]):
            for y in range(g["y"], g["y"] + g["h"]):
                assert (x, y) not in cells, f"{panel['title']} overlaps another panel"
                cells.add((x, y))


def test_every_query_uses_the_method_variable_that_the_dashboard_defines():
    dashboard = _dashboard()
    assert [v["name"] for v in dashboard["templating"]["list"]] == ["method"]
    for panel in dashboard["panels"]:
        for target in panel["targets"]:
            assert "$method" in target["expr"], f"{panel['title']}: query ignores the method filter"
