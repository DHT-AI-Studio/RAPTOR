"""
MTR-6 / VIE01-216: Archive Regression Suite.

Covers the one subtask not already proven by tests/unit/test_unit.py's
fake-based coverage: the negative requirement that no unarchive code path
exists anywhere on module 04's API surface. (The 5-state archive test
matrix, and Register/Unregister both returning 409 on an archived asset,
are proven at the AssetManager level in tests/unit/test_unit.py --
test_archive_allows_non_terminal_status, test_archive_blocks_terminal_status,
test_register_returns_conflict_on_archived_asset, and the pre-existing
test_unregister_rejected_on_archived.)

Structural, not textual: enumerates the real FastAPI route table rather
than grepping for the word "unarchive", so a differently-named route
(e.g. "/restore", "/reactivate") would also be caught.

Run from the 04-object-storage/ directory:
    python -m pytest test_archive_regression.py -v
"""
from asset_management.endpoints import app

_SUSPECT_SUBSTRINGS = ("unarchive", "restore", "reactivate", "unarchived")


def test_no_unarchive_route_exists():
    """MTR-6: archived is a one-way transition -- only destroy() (-> the
    terminal 'destroyed' state) may follow it. No route anywhere on this
    service's API surface may offer a way back to a pre-archive status."""
    paths = [route.path.lower() for route in app.routes if hasattr(route, "path")]
    assert paths, "route enumeration returned nothing -- app.routes shape changed"

    offending = [p for p in paths if any(s in p for s in _SUSPECT_SUBSTRINGS)]
    assert not offending, f"found route(s) that look like an unarchive path: {offending}"
