"""GD-4 / VIE01-249: Module 04's count-only file summary for the DSAR
(GET /internal/gdpr/user/{user_id}/summary). Mocked Postgres -- no live
infra needed."""
import json

import pytest
from asset_management.client import AssetManager
from asset_management.database import Database, summarize_file_rows


def _row(primary, assoc=None, as_json_string=False):
    assoc = assoc if assoc is not None else []
    return {"primary_filename": primary, "associated_filenames": json.dumps(assoc) if as_json_string else assoc}


def test_counts_primary_and_associated_files():
    rows = [
        _row("report.PDF", [["notes.txt", "v1"], ["slides.pptx", "v1"]]),
        _row("clip.mp4"),
        _row("report.pdf"),  # a second version of the same asset still counts
    ]
    assert summarize_file_rows(rows) == {"count": 5, "file_types": ["mp4", "pdf", "pptx", "txt"]}


def test_associated_files_stored_as_json_string_are_parsed():
    rows = [_row("a.wav", [["b.srt", "v2"]], as_json_string=True)]
    assert summarize_file_rows(rows) == {"count": 2, "file_types": ["srt", "wav"]}


def test_skips_associated_entries_metadata_parsing_also_drops():
    """Same filter as Database._row_to_metadata: malformed pairs and pairs
    with no version_id are not real files."""
    rows = [_row("a.txt", [["b.txt", None], ["c.txt"], "d.txt", [None, "v1"]])]
    assert summarize_file_rows(rows) == {"count": 1, "file_types": ["txt"]}


def test_extensionless_and_dotfiles_add_no_type():
    rows = [_row("README"), _row(".hidden"), _row("archive.tar.gz")]
    assert summarize_file_rows(rows) == {"count": 3, "file_types": ["gz"]}


def test_no_files():
    assert summarize_file_rows([]) == {"count": 0, "file_types": []}


class _FakeConn:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def fetch(self, sql, *args):
        self.calls.append((sql, args))
        return self.rows


class _FakePool:
    def __init__(self, conn):
        self.conn = conn

    def acquire(self):
        conn = self.conn

        class _Ctx:
            async def __aenter__(self):
                return conn

            async def __aexit__(self, *exc):
                return False

        return _Ctx()


@pytest.mark.asyncio
async def test_db_summary_scopes_by_branch_and_selects_only_filename_columns():
    conn = _FakeConn([_row("a.pdf")])
    db = Database()
    db.pool = _FakePool(conn)

    assert await db.get_gdpr_file_summary("user-1") == {"count": 1, "file_types": ["pdf"]}
    sql, args = conn.calls[0]
    # Same scope as gdpr_erase_user (branch only, every status) ...
    assert "WHERE branch = $1" in sql and "status" not in sql
    assert args == ("user-1",)
    # ... but never full rows.
    assert "SELECT primary_filename, associated_filenames FROM commit_history" in " ".join(sql.split())


@pytest.mark.asyncio
async def test_manager_delegates_to_db_without_touching_object_store():
    class _Db:
        async def get_gdpr_file_summary(self, branch):
            return {"count": 2, "file_types": ["mp4"], "branch": branch}

    class _ExplodingObjectStore:
        def __getattr__(self, name):
            raise AssertionError(f"summary must not touch the object store ({name})")

    manager = AssetManager.__new__(AssetManager)
    manager.db = _Db()
    manager.object_store = _ExplodingObjectStore()
    assert await manager.gdpr_summarize_user("user-1") == {"count": 2, "file_types": ["mp4"], "branch": "user-1"}
