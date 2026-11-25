"""TMP-7: Module 18 reads the query's time condition through Module 25's `parse` once it is proven available
-- PD_TEMPORAL_V2 on and a parse route -- and a failed parse makes the query UNRESOLVED, never answered
without its time condition and never read by the interim regex instead (temporal plan §5.2, Appendix D F2)."""
from types import SimpleNamespace

import httpx
import pytest
from app.core.config import Settings, get_settings
from app.routers.query import ClassifyRequest, classify_endpoint
from app.services import temporal_parse
from app.services.temporal_parse import ParseFailure, TemporalParseClient, parse_client

_REQUEST = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(intent_classifier=SimpleNamespace(
    people_intent=lambda q, temporal: None,
    classify=lambda q, k=3: SimpleNamespace(intent_type=["TKG"], confidence=1.0, routing_strategy="rule_based",
                                            matched_rules=[])))))


def _classify(query):
    return classify_endpoint(ClassifyRequest(query=query), _REQUEST)


def _using_parse(monkeypatch, reply):
    """`parse` proven available; each call answers with *reply* (a dict, or an exception to raise)."""
    calls = []

    def parse(text):
        calls.append(text)
        if isinstance(reply, Exception):
            raise reply
        return reply

    monkeypatch.setattr(parse_client, "available", lambda: True)
    monkeypatch.setattr(parse_client, "parse", parse)
    return calls


def test_the_window_comes_from_module_25_once_parse_is_available(monkeypatch):
    calls = _using_parse(monkeypatch, {"status": "RESOLVED", "match": "inside", "keywords": ["Google"],
                                       "window": {"start": "2012-01-01T00:00:00.000+08:00", "end": None},
                                       "reference": None, "reference_relations": None})
    plan = _classify("What did Google acquire after 2011?")
    assert calls == ["What did Google acquire after 2011?"]
    assert (plan.temporal_source, plan.temporal_status, plan.temporal_constraint_applied) == ("parse", "RESOLVED", True)
    assert (plan.time_from, plan.time_to, plan.match, plan.keywords) == (
        "2012-01-01T00:00:00.000+08:00", None, "inside", ["Google"])


def test_a_reference_event_from_parse_makes_it_a_tkg_question(monkeypatch):
    _using_parse(monkeypatch, {"status": "RESOLVED", "window": None, "match": None, "keywords": [],
                               "reference": "Google acquired Android", "reference_relations": ["after"]})
    plan = _classify("What happened after Google acquired Android?")
    assert (plan.reference, plan.reference_relations, plan.modes) == ("Google acquired Android", ["after"], ["TKG"])


@pytest.mark.parametrize("failure", [ParseFailure("timeout after 1000 ms"), ParseFailure("Module 25 returned 503")])
def test_a_failed_parse_is_unresolved_and_never_the_regex_window(monkeypatch, failure):
    _using_parse(monkeypatch, failure)
    plan = _classify("what happened after 2011")          # the regex would read [2012-01-01, unbounded)
    assert (plan.temporal_status, plan.temporal_reason, plan.temporal_constraint_applied, plan.answer) == (
        "UNRESOLVED", "parse_failure", False, None)
    assert (plan.time_from, plan.time_to, plan.match, plan.reference, plan.keywords) == (None, None, None, None, [])


def test_a_date_module_25_cannot_read_is_unresolved_with_its_reason(monkeypatch):
    _using_parse(monkeypatch, {"status": "UNRESOLVED", "reason": "invalid_calendar", "window": None})
    plan = _classify("What happened after 2011-02-30?")
    assert (plan.temporal_status, plan.temporal_reason, plan.time_from) == ("UNRESOLVED", "invalid_calendar", None)


def test_until_parse_is_available_the_interim_regex_reads_the_window():
    plan = _classify("What happened after 2011?")
    assert (plan.temporal_source, plan.temporal_status, plan.match) == ("regex", "RESOLVED", "inside")
    assert plan.time_from.startswith("2012-01-01T00:00:00.000")
    assert _classify("What is Android?").temporal_status == "NO_TEMPORAL"


# ── the client ──────────────────────────────────────────────────────────────

@pytest.fixture
def client(monkeypatch):
    c = TemporalParseClient()
    monkeypatch.setattr(get_settings(), "personal_db_url", "http://m25")
    monkeypatch.setattr(get_settings(), "parse_capability_ttl", 60)
    return c


@pytest.mark.parametrize("caps, expected", [
    ({"temporal_v2": True, "operations": {"parse": "/personal/temporal/parse"}}, True),
    ({"temporal_v2": False, "operations": {"parse": "/personal/temporal/parse"}}, False),   # PD_TEMPORAL_V2 off
    ({"temporal_v2": True, "operations": {"parse": None}}, False),                          # no parse route
    ({"operations": {}}, False),                                                             # an older Module 25
])
def test_parse_is_used_only_with_v2_on_and_a_parse_route(monkeypatch, client, caps, expected):
    monkeypatch.setattr(temporal_parse.httpx, "get", lambda url, timeout: httpx.Response(
        200, json=caps, request=httpx.Request("GET", url)))
    assert client.available() is expected


def test_an_unreachable_module_25_keeps_the_regex_and_the_check_is_cached(monkeypatch, client):
    calls = []

    def get(url, timeout):
        calls.append(url)
        raise httpx.ConnectError("refused")

    monkeypatch.setattr(temporal_parse.httpx, "get", get)
    assert client.available() is False and client.available() is False
    assert calls == ["http://m25/personal/temporal/capabilities"]          # once per parse_capability_ttl


@pytest.mark.parametrize("response, error", [
    (httpx.TimeoutException("slow"), "timeout"),
    (httpx.Response(503, json={"detail": "busy"}), "503"),
    (httpx.Response(200, json={"window": None}), "status"),              # a reply without a status
])
def test_parse_failures(monkeypatch, client, response, error):
    def post(url, json, timeout):
        assert timeout == get_settings().parse_timeout_ms / 1000 and json["timezone"] and json["anchor"]
        if isinstance(response, Exception):
            raise response
        response.request = httpx.Request("POST", url)
        return response

    monkeypatch.setattr(temporal_parse.httpx, "post", post)
    with pytest.raises(ParseFailure, match=error):
        client.parse("what happened after 2011")


def test_parse_timeout_ms_is_read_from_the_environment(monkeypatch):
    monkeypatch.setenv("PARSE_TIMEOUT_MS", "250")
    assert Settings().parse_timeout_ms == 250
    monkeypatch.delenv("PARSE_TIMEOUT_MS")
    monkeypatch.setenv("QO_PARSE_TIMEOUT_MS", "400")
    assert Settings().parse_timeout_ms == 400
