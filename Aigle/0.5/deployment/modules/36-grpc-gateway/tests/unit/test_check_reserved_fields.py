"""Tests for scripts/check_reserved_fields.py: the CI contract check that a
proto field removed since a baseline version must have its number (and
name) reserved, so nothing can ever claim it again."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "scripts"))
from check_reserved_fields import check

_BASELINE = """
syntax = "proto3";
package test.v1;
message Job {
  string id = 1;
  string status = 2;
  string detail = 3;
}
"""

_REMOVED_UNRESERVED = """
syntax = "proto3";
package test.v1;
message Job {
  string id = 1;
  string status = 2;
}
"""

_REMOVED_RESERVED = """
syntax = "proto3";
package test.v1;
message Job {
  reserved 3;
  reserved "detail";
  string id = 1;
  string status = 2;
}
"""

_FIELD_STILL_PRESENT = """
syntax = "proto3";
package test.v1;
message Job {
  string id = 1;
  string status = 2;
  string detail = 3;
}
"""

_NEW_MESSAGE_ADDED = """
syntax = "proto3";
package test.v1;
message Job {
  string id = 1;
  string status = 2;
  string detail = 3;
}
message NewThing {
  string name = 1;
}
"""


def _write_proto(tmp_path: Path, name: str, content: str) -> Path:
    d = tmp_path / name / "raptor" / "v1"
    d.mkdir(parents=True)
    p = d / "raptor.proto"
    p.write_text(content)
    return p


def test_flags_removed_field_number_that_is_not_reserved(tmp_path):
    baseline = _write_proto(tmp_path, "baseline", _BASELINE)
    current = _write_proto(tmp_path, "current", _REMOVED_UNRESERVED)

    violations = check(baseline, current)

    assert len(violations) == 2  # one for the number, one for the name
    assert any("field 3" in v and "not in a `reserved` range" in v for v in violations)
    assert any("field 3" in v and "not reserved" in v for v in violations)


def test_allows_removed_field_that_is_properly_reserved(tmp_path):
    baseline = _write_proto(tmp_path, "baseline", _BASELINE)
    current = _write_proto(tmp_path, "current", _REMOVED_RESERVED)

    assert check(baseline, current) == []


def test_allows_unchanged_field(tmp_path):
    baseline = _write_proto(tmp_path, "baseline", _BASELINE)
    current = _write_proto(tmp_path, "current", _FIELD_STILL_PRESENT)

    assert check(baseline, current) == []


def test_allows_adding_a_new_message(tmp_path):
    baseline = _write_proto(tmp_path, "baseline", _BASELINE)
    current = _write_proto(tmp_path, "current", _NEW_MESSAGE_ADDED)

    assert check(baseline, current) == []


_RESERVED_BASELINE = """
syntax = "proto3";
package test.v1;
message Job {
  reserved 3;
  reserved "detail";
  string id = 1;
  string status = 2;
}
"""

_RESERVATION_DROPPED_AND_NUMBER_REUSED = """
syntax = "proto3";
package test.v1;
message Job {
  string id = 1;
  string status = 2;
  int32 retries = 3;
}
"""

_RESERVATION_DROPPED_AND_NAME_REUSED = """
syntax = "proto3";
package test.v1;
message Job {
  reserved 3;
  string id = 1;
  string status = 2;
  string detail = 4;
}
"""

_RESERVATION_KEPT_AND_FIELD_ADDED = """
syntax = "proto3";
package test.v1;
message Job {
  reserved 3;
  reserved "detail";
  string id = 1;
  string status = 2;
  int32 retries = 4;
}
"""


def test_flags_reusing_a_reserved_number(tmp_path):
    baseline = _write_proto(tmp_path, "baseline", _RESERVED_BASELINE)
    current = _write_proto(tmp_path, "current", _RESERVATION_DROPPED_AND_NUMBER_REUSED)

    violations = check(baseline, current)

    assert any("reservation of number 3 was removed" in v and "'retries' now uses it" in v for v in violations)
    assert any("reservation of name 'detail' was removed" in v for v in violations)


def test_flags_reusing_a_reserved_name(tmp_path):
    baseline = _write_proto(tmp_path, "baseline", _RESERVED_BASELINE)
    current = _write_proto(tmp_path, "current", _RESERVATION_DROPPED_AND_NAME_REUSED)

    violations = check(baseline, current)

    assert any("reservation of name 'detail' was removed and a field now uses it" in v for v in violations)
    assert not any("number 3" in v for v in violations)  # that reservation was kept


def test_allows_new_fields_next_to_kept_reservations(tmp_path):
    baseline = _write_proto(tmp_path, "baseline", _RESERVED_BASELINE)
    current = _write_proto(tmp_path, "current", _RESERVATION_KEPT_AND_FIELD_ADDED)

    assert check(baseline, current) == []


def test_the_real_contract_passes_against_itself():
    """The checked-in raptor.proto against itself: nothing removed, no reservation dropped."""
    proto = Path(__file__).parent.parent.parent / "proto" / "raptor" / "v1" / "raptor.proto"
    assert check(proto, proto) == []
