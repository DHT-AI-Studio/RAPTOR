import os

# asset_management.config.Settings() is constructed eagerly at import time
# and requires these even though nothing in tests/unit/ actually uses live
# infra -- set harmless placeholders before any test module imports the
# package, so `pytest tests/unit/` runs standalone in CI with no .env.
for key, default in {
    "S3_ENDPOINT": "http://unused", "S3_PUBLIC_URL": "http://unused",
    "AWS_ACCESS_KEY": "unused", "AWS_SECRET_KEY": "unused", "S3_BUCKET": "unused",
    "POSTGRES_HOST": "unused", "POSTGRES_PORT": "5432", "POSTGRES_USER": "unused",
    "POSTGRES_PASSWORD": "unused", "POSTGRES_DB": "unused",
    "LAKEFS_ENDPOINT": "http://unused", "LAKEFS_ACCESS_KEY": "unused", "LAKEFS_SECRET_KEY": "unused",
}.items():
    os.environ.setdefault(key, default)

from datetime import datetime
from zoneinfo import ZoneInfo

from asset_management.models import AssetMetadata

# ── Shared across test_asset_manager_core.py / test_lifecycle.py /
# test_associated_files.py. Hand-written fakes for Database/SearchSync --
# matches the "capturing fake" style already used in 13-api-services' tests,
# not unittest.mock. Each fake only implements what the method under test
# actually calls. Fakes local to a single concern (upload_files(),
# destroy_v2(), auto_archive/auto_destroy(), update_expiration(),
# _upload_assoc_batch()) stay defined in their own file instead.


def _base_metadata_kwargs(**overrides):
    kwargs = {
        "asset_path": "document/txt/foo",
        "version_id": "v1",
        "primary_filename": "foo.txt",
        "associated_filenames": [],
        "upload_date": datetime.now(tz=ZoneInfo("Asia/Taipei")),
        "user": "tester",
        "branch": "test_branch",
        "status": "uploaded",
        "checksum": "deadbeef",
    }
    kwargs.update(overrides)
    return kwargs


class _FakeDB:
    """Tracks one asset's status across calls, so AssetManager's own
    read-modify-read flow (e.g. archive()'s completion poll) behaves like
    a real DB would, without a real Postgres connection."""

    def __init__(self, initial_status: str):
        self.metadata = AssetMetadata(**_base_metadata_kwargs(status=initial_status))
        self.log_calls = []
        self.saved = []

    async def get_asset_by_path_and_version(self, asset_path, version_id, branch):
        return self.metadata

    async def get_latest_asset(self, asset_path, branch):
        return self.metadata

    async def update_status(self, asset_path, version_id, status, branch):
        self.metadata.status = status

    async def save_metadata(self, metadata):
        self.saved.append(metadata)

    async def log_access(self, *args, **kwargs):
        self.log_calls.append((args, kwargs))


class _FakeSearchSync:
    def __init__(self, indexed: bool = True):
        self.personal_db_calls = []
        self._indexed = indexed

    async def archive_metadata(self, *args, **kwargs):
        pass

    async def _personal_db_set_status(self, asset_path, version_id, status, branch_id=None):
        self.personal_db_calls.append((asset_path, version_id, status, branch_id))

    async def check_indexed(self, asset_path, version_id, branch, status=None):
        return self._indexed
