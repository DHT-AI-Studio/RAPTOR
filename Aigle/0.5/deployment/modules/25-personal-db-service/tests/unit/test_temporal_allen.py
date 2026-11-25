"""Independent predicate oracle and exhaustive finite-domain regression checks."""
from itertools import product
from time import perf_counter_ns

import pytest
from app.temporal.allen import (
    INVERSE,
    RELATIONS,
    certainty,
    composition_table,
    possible_relations,
    relate,
    window_relations,
)


def oracle(a, b):
    s, e = a
    u, v = b
    predicates = {
        "before": e < u, "after": v < s,
        "equals": s == u and e == v,
        "starts": s == u and e < v, "started-by": s == u and e > v,
        "finishes": s > u and e == v, "finished-by": s < u and e == v,
        "during": u < s <= e < v, "contains": s < u <= v < e,
        "meets": s < e == u < v, "met-by": u < v == s < e,
        "overlaps": s < u < e < v, "overlapped-by": u < s < v < e,
    }
    matches = [name for name, valid in predicates.items() if valid]
    assert len(matches) == 1
    return matches[0]


def intervals(n):
    return [(s, e) for s in range(n) for e in range(s, n)]


def domains(n):
    for starts, ends in product(intervals(n), repeat=2):
        box = (*starts, *ends)
        concrete = [(s, e) for s in range(starts[0], starts[1] + 1)
                    for e in range(ends[0], ends[1] + 1) if s <= e]
        yield box, concrete
    # A point with uncertain time has correlated endpoints s=e, not two independent ranges.
    for lo, hi in intervals(n):
        yield (lo, hi, lo, hi, "point"), [(t, t) for t in range(lo, hi + 1)]


def test_classifier_all_441_pairs():
    pairs = list(product(intervals(6), repeat=2))
    assert len(pairs) == 441
    for a, b in pairs:
        assert relate(*a, *b) == oracle(a, b)
        assert relate(*b, *a) == INVERSE[relate(*a, *b)]


def test_composition_all_169_entries_and_points():
    expected = {(a, b): set() for a, b in product(RELATIONS, repeat=2)}
    for a, b, c in product(intervals(6), repeat=3):
        expected[oracle(a, b), oracle(b, c)].add(oracle(a, c))
    assert len(composition_table()) == 169
    assert dict(composition_table()) == expected
    a, b, c = (0, 1), (1, 1), (1, 2)
    assert relate(*a, *b) == "finished-by"
    assert relate(*b, *c) == "starts"
    assert relate(*a, *c) == "meets"
    assert "meets" in composition_table()["finished-by", "starts"]


def test_window_all_19208_feasible_cases_and_infeasible_boxes():
    feasible = 0
    for box, concrete in domains(7):
        for window in intervals(7):
            expected = {oracle(a, window) for a in concrete}
            assert window_relations(box, *window) == expected, (box, window)
            feasible += bool(concrete)
    assert feasible == 19208


def test_general_solver_all_9025_feasible_pairs_and_infeasible_boxes():
    boxes = [(box, concrete) for box, concrete in domains(4) if concrete]
    assert len(boxes) ** 2 == 9025
    for (a, av), (b, bv) in product(boxes, repeat=2):
        expected = {oracle(x, y) for x, y in product(av, bv)}
        assert possible_relations(a, b) == expected, (a, b)
    assert not possible_relations((5, 4, 0, 1), (0, 2, 0, 3))
    assert not possible_relations((5, 6, 0, 1), (0, 2, 0, 3))
    with pytest.raises(ValueError, match="infeasible"):
        certainty(set())


def test_unbounded_and_unresolved_domains():
    assert possible_relations((None, None, None, None), (None, None, None, None)) == set(RELATIONS)
    assert window_relations((None, None, None, None), 0, 1) == set(RELATIONS) - {"overlaps", "overlapped-by", "during"}
    fact = {"start": {"state": "unresolved", "lo_ms": None, "hi_ms": None},
            "end": {"state": "unresolved", "lo_ms": None, "hi_ms": None}}
    assert not possible_relations(fact, (0, 1, 0, 1))


def test_ongoing_confirmation_constrains_both_solver_paths():
    from app.temporal.allen import EndpointRange, IntervalDomain
    fact = {"start": {"state": "resolved", "lo_ms": 0, "hi_ms": 2},
            "end": {"state": "ongoing", "lo_ms": None, "hi_ms": None},
            "extent": "ongoing", "confirmed_active_at": 10}
    domain = IntervalDomain(EndpointRange(0, 2), EndpointRange(None, None), "ongoing", 10)
    assert window_relations(fact, 4, 6) == {"contains"}
    assert window_relations(domain, 4, 6) == {"contains"}
    assert possible_relations(fact, (4, 4, 6, 6)) == {"contains"}
    assert possible_relations((4, 4, 6, 6), domain) == {"during"}
    # Confirmation proves activity only at the source instant, not indefinitely.
    assert "before" in window_relations(fact, 12, 15)
    invalid = {**fact, "confirmed_active_at": True}
    with pytest.raises(TypeError, match="confirmation"):
        window_relations(invalid, 4, 6)


def test_uncached_p95_latency():
    fast, general = [], []
    # Translate domains each iteration so none of these calls hits the cache.
    for offset in range(1000, 3000):
        box = (offset, offset + 3, offset + 1, offset + 5)
        other = (offset + 1, offset + 4, offset + 2, offset + 6)
        start = perf_counter_ns()
        window_relations(box, offset + 2, offset + 4)
        fast.append(perf_counter_ns() - start)
        start = perf_counter_ns()
        possible_relations(box, other)
        general.append(perf_counter_ns() - start)
    fast_p95 = sorted(fast)[int(len(fast) * .95)] / 1000
    general_p95 = sorted(general)[int(len(general) * .95)] / 1000
    print(f"uncached p95: window={fast_p95:.1f}us general={general_p95:.1f}us")
    assert fast_p95 < 100
    assert general_p95 < 5000
