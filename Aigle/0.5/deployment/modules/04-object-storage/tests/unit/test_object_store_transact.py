"""ObjectStore._sync_transact_write()/batch_upload() -- covers the
ephemeral-branch transaction mechanics directly, since these are LakeFS SDK
internals that _FakeObjectStore (used everywhere else in tests/unit/)
deliberately doesn't exercise.

Run from the 04-object-storage/ directory:
    python -m pytest tests/unit/test_object_store_transact.py -v
"""
import pytest
from asset_management.object_store import ObjectStore
from fastapi import HTTPException
from lakefs.exceptions import ConflictException


class _FakeStat:
    def __init__(self, checksum):
        self.checksum = checksum


class _FakeObject:
    def __init__(self, branch, path):
        self.branch = branch
        self.path = path

    def upload(self, data, content_type, pre_sign=False):
        self.branch.staged[self.path] = data

    def stat(self):
        return _FakeStat(checksum=f"checksum:{self.path}")


class _FakeListedObject:
    def __init__(self, path):
        self.path = path


class _FakeBranch:
    """One fake per ephemeral branch created during a transaction."""

    def __init__(self, repo, name):
        self.repo = repo
        self.name = name
        self.staged = {}
        self.deleted_paths = []
        self.committed = False
        self.deleted = False

    def create(self, source=None, hidden=True, **kwargs):
        # Accepts both call shapes seen in object_store.py: positional
        # (branch_id, hidden=True) from the transact-equivalent path, and
        # source_reference=/exist_ok= keywords from create_branch().
        self.repo.created_branches.append(self.name)
        return self

    def object(self, path):
        return _FakeObject(self, path)

    def objects(self, prefix=""):
        return [_FakeListedObject(p) for p in self.repo.existing_paths if p.startswith(prefix)]

    def delete_objects(self, paths):
        self.deleted_paths.extend(paths)

    def commit(self, message):
        self.committed = True
        self.commit_message = message
        # Queue of per-attempt overrides consumed in order, for simulating
        # a group that's byte-identical to what's already on the branch
        # (a real dedup no-op) even though it has staged content -- the
        # default (no override queued) only raises when truly nothing
        # was staged or deleted.
        outcome = self.repo.commit_outcomes.pop(0) if self.repo.commit_outcomes else None
        if outcome == "no_change" or (outcome is None and not self.staged and not self.deleted_paths):
            raise Exception("commit: no changes")

    def merge_into(self, target, message):
        self.repo.merge_calls.append((self.name, target))
        outcome = self.repo.merge_outcomes.pop(0) if self.repo.merge_outcomes else "ok"
        if outcome == "conflict":
            raise ConflictException(0, "conflict")
        return f"merge-commit-{self.name}"

    def delete(self):
        self.deleted = True
        self.repo.deleted_branches.append(self.name)


class _FakeRepo:
    def __init__(self):
        self.created_branches = []
        self.deleted_branches = []
        self.merge_calls = []
        # Queue of "ok"/"conflict" outcomes consumed in order across
        # successive merge_into() calls (one per retry attempt).
        self.merge_outcomes = []
        # Queue of "no_change"/None outcomes consumed in order across
        # successive commit() calls (one per group/attempt).
        self.commit_outcomes = []
        # Paths visible to .branch(...).objects(prefix=...) listings, for
        # delete_associated_files()'s "list what's there first" step.
        self.existing_paths = []
        self._branches = {}

    def branch(self, name):
        b = _FakeBranch(self, name)
        self._branches[name] = b
        return b


@pytest.fixture
def store(monkeypatch):
    import asset_management.object_store as object_store_module
    s = ObjectStore()
    fake_repo = _FakeRepo()
    monkeypatch.setattr(object_store_module.lakefs, "repository", lambda **kw: fake_repo)
    return s, fake_repo


def test_batch_upload_is_one_commit_and_one_merge(store):
    s, repo = store
    merge_id, checksums = s._sync_transact_write(
        "user-branch", "Batch upload 3 files",
        uploads=[(b"a", "k1", "text/plain"), (b"b", "k2", "text/plain"), (b"c", "k3", "text/plain")],
    )
    assert merge_id == f"merge-commit-{repo.created_branches[0]}"
    assert checksums == {"k1": "checksum:k1", "k2": "checksum:k2", "k3": "checksum:k3"}
    assert len(repo.merge_calls) == 1  # one merge for the whole batch, not per-file
    assert repo.deleted_branches == [repo.created_branches[0]]  # ephemeral branch cleaned up


def test_conflict_on_merge_retries_and_succeeds(store):
    s, repo = store
    repo.merge_outcomes = ["conflict", "ok"]
    merge_id, _ = s._sync_transact_write("user-branch", "msg", uploads=[(b"a", "k1", "text/plain")])
    assert merge_id == f"merge-commit-{repo.created_branches[1]}"
    assert len(repo.created_branches) == 2  # fresh ephemeral branch on retry
    assert repo.deleted_branches == repo.created_branches  # both cleaned up, success and failed attempt alike


def test_conflict_exhausts_retries_and_raises(store):
    s, repo = store
    repo.merge_outcomes = ["conflict"] * 10  # more than MAX_TX_RETRIES
    with pytest.raises(ConflictException):
        s._sync_transact_write("user-branch", "msg", uploads=[(b"a", "k1", "text/plain")])
    from asset_management.config import settings
    assert len(repo.created_branches) == settings.lakefs_tx_max_retries
    assert repo.deleted_branches == repo.created_branches  # every attempt's branch cleaned up


@pytest.mark.asyncio
async def test_batch_upload_public_api_returns_per_file_version_id(store):
    s, repo = store
    result = await s.batch_upload(
        [{"key": "k1", "content": b"a", "content_type": "text/plain"},
         {"key": "k2", "content": b"b", "content_type": "text/plain"}],
        branch="user-branch",
    )
    # branch="user-branch" (non-default) also triggers create_branch()'s own
    # "ensure it exists" call, which shows up in created_branches too -- the
    # ephemeral tx branch is the one prefixed "tx-".
    tx_branches = [b for b in repo.created_branches if b.startswith("tx-")]
    assert len(tx_branches) == 1
    expected_version_id = f"merge-commit-{tx_branches[0]}"
    assert {(f["key"], f["version_id"]) for f in result["files"]} == {("k1", expected_version_id), ("k2", expected_version_id)}


@pytest.mark.asyncio
async def test_batch_upload_empty_raises_400():
    s = ObjectStore()
    with pytest.raises(Exception) as exc_info:
        await s.batch_upload([], branch="user-branch")
    assert exc_info.value.status_code == 400


@pytest.mark.asyncio
async def test_batch_upload_no_caller_facing_file_count_limit(store):
    """Exceeding lakefs_batch_group_max_files splits into more than one
    LakeFS transaction internally -- it never rejects the caller."""
    s, repo = store
    from asset_management.config import settings
    n = settings.lakefs_batch_group_max_files + 1
    objects = [{"key": f"k{i}", "content": b"x", "content_type": "text/plain"} for i in range(n)]
    result = await s.batch_upload(objects, branch="user-branch")
    assert len(result["files"]) == n
    assert len(repo.merge_calls) == 2  # split into exactly 2 groups for n = limit + 1


@pytest.mark.asyncio
async def test_batch_upload_no_caller_facing_byte_limit(store):
    """Exceeding lakefs_batch_group_max_bytes splits into more than one
    LakeFS transaction internally -- it never rejects the caller."""
    s, repo = store
    from asset_management.config import settings
    big = b"x" * (settings.lakefs_batch_group_max_bytes // 2 + 1)
    objects = [
        {"key": "k1", "content": big, "content_type": "application/octet-stream"},
        {"key": "k2", "content": big, "content_type": "application/octet-stream"},
    ]
    result = await s.batch_upload(objects, branch="user-branch")
    assert len(result["files"]) == 2
    assert len(repo.merge_calls) == 2  # each object alone is over half the threshold -> its own group


@pytest.mark.asyncio
async def test_batch_upload_a_single_file_is_never_split_across_groups(store):
    """A single object always lands whole in one group, even if it alone
    exceeds the byte threshold -- the threshold only decides when to *stop
    accumulating more objects*, never where to cut one object's bytes."""
    s, repo = store
    from asset_management.config import settings
    huge = b"x" * (settings.lakefs_batch_group_max_bytes * 3)
    result = await s.batch_upload(
        [{"key": "k1", "content": huge, "content_type": "application/octet-stream"}],
        branch="user-branch",
    )
    assert len(repo.merge_calls) == 1
    assert result["files"][0]["checksum"] == "checksum:k1"


@pytest.mark.asyncio
async def test_batch_upload_one_noop_group_among_others_is_skipped_not_fatal(store):
    """A multi-group batch where one group's content happens to be
    byte-identical to what's already on the branch (a real LakeFS dedup
    no-op, not a failure) must not abort the other groups -- its keys are
    simply absent from the result, same as the single-group case."""
    s, repo = store
    from asset_management.config import settings
    n = settings.lakefs_batch_group_max_files + 1  # forces exactly 2 groups
    repo.commit_outcomes = [None, "no_change"]  # group 1 commits, group 2 is a dedup no-op
    objects = [{"key": f"k{i}", "content": b"x", "content_type": "text/plain"} for i in range(n)]
    result = await s.batch_upload(objects, branch="user-branch")
    returned_keys = {f["key"] for f in result["files"]}
    assert returned_keys == {f"k{i}" for i in range(settings.lakefs_batch_group_max_files)}
    assert len(repo.merge_calls) == 1  # group 2's commit raised before it ever reached merge_into


@pytest.mark.asyncio
async def test_upload_file_success(store):
    s, repo = store
    result = await s.upload_file(b"hello", "some/key.txt", "text/plain", branch="user-branch")
    assert result == {"key": "some/key.txt", "version_id": f"merge-commit-{repo.created_branches[-1]}", "checksum": "checksum:some/key.txt"}
    tx_branches = [b for b in repo.created_branches if b.startswith("tx-")]
    assert len(tx_branches) == 1
    assert repo.deleted_branches == tx_branches  # only the ephemeral tx branch is cleaned up, not "user-branch" itself


@pytest.mark.asyncio
async def test_upload_file_no_changes_raises_409(store):
    """Byte-identical re-upload of an existing key surfaces as a 409
    LakeFS/DB-inconsistency signal, not a generic 500 -- distinct from
    batch_upload()'s per-group silent skip, since a single-file caller has
    no other files in the same call to fall back on."""
    s, repo = store
    repo.commit_outcomes = ["no_change"]
    with pytest.raises(HTTPException) as exc_info:
        await s.upload_file(b"hello", "some/key.txt", "text/plain", branch="user-branch")
    assert exc_info.value.status_code == 409


@pytest.mark.asyncio
async def test_delete_file_routes_through_transact_write(store):
    s, repo = store
    await s.delete_file("some/key.txt", branch="user-branch")
    tx_branches = [b for b in repo.created_branches if b.startswith("tx-")]
    assert len(tx_branches) == 1
    assert repo._branches[tx_branches[0]].deleted_paths == ["some/key.txt"]
    assert len(repo.merge_calls) == 1
    assert repo.deleted_branches == tx_branches  # only the ephemeral tx branch is cleaned up, not "user-branch" itself


@pytest.mark.asyncio
async def test_delete_file_swallows_not_found_and_no_changes(store):
    """delete_file() treats "already gone" the same as "nothing to
    delete" -- both are a silent no-op, not an error surfaced to the
    caller (MTR-1's original behavior, unaffected by routing through
    _sync_transact_write now)."""
    s, repo = store
    repo.commit_outcomes = ["no_change"]
    await s.delete_file("already/gone.txt", branch="user-branch")  # must not raise


@pytest.mark.asyncio
async def test_delete_associated_files_deletes_only_non_primary_paths(store):
    s, repo = store
    repo.existing_paths = ["document/txt/foo/primary.txt", "document/txt/foo/extra1.txt", "document/txt/foo/extra2.txt"]
    await s.delete_associated_files("document/txt/foo", "primary.txt", branch="user-branch")
    tx_branches = [b for b in repo.created_branches if b.startswith("tx-")]
    assert len(tx_branches) == 1
    deleted = set(repo._branches[tx_branches[0]].deleted_paths)
    assert deleted == {"document/txt/foo/extra1.txt", "document/txt/foo/extra2.txt"}


@pytest.mark.asyncio
async def test_delete_associated_files_none_found_is_a_noop(store):
    """No associated files besides the primary -- must return early
    without ever starting a transaction (nothing to delete)."""
    s, repo = store
    repo.existing_paths = ["document/txt/foo/primary.txt"]
    await s.delete_associated_files("document/txt/foo", "primary.txt", branch="user-branch")
    tx_branches = [b for b in repo.created_branches if b.startswith("tx-")]
    assert tx_branches == []


# ── _bulk_delete_from_s3_bucket() -- no caller-facing count limit, chunks
# at S3's own 1000-key-per-request hard limit internally. ──────────────────

class _FakeS3Client:
    """Doubles as its own async context manager -- matches aioboto3's own
    shape, where session.client(...).__aenter__() returns an object that
    itself has __aexit__ (object_store.py awaits __aenter__() once up
    front and calls __aexit__() on its result, not on the wrapper)."""

    def __init__(self, errors_for=()):
        self.delete_calls = []
        self._errors_for = set(errors_for)

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        pass

    async def delete_objects(self, Bucket, Delete):
        keys = [o["Key"] for o in Delete["Objects"]]
        self.delete_calls.append(keys)
        errors = [{"Key": k, "Message": "boom"} for k in keys if k in self._errors_for]
        return {"Deleted": [{"Key": k} for k in keys if k not in self._errors_for], "Errors": errors}


class _FakeSession:
    def __init__(self, client):
        self._client = client

    def client(self, service_name, **kwargs):
        return self._client


@pytest.mark.asyncio
async def test_bulk_delete_from_s3_bucket_chunks_at_1000():
    s = ObjectStore()
    fake_client = _FakeS3Client()
    s.session = _FakeSession(fake_client)
    keys = [f"k{i}" for i in range(2500)]
    result = await s._bulk_delete_from_s3_bucket(keys)
    assert result == {"deleted": 2500, "errors": 0}
    # Chunks are dispatched concurrently now, so call order isn't guaranteed -- only the sizes are.
    assert sorted(len(c) for c in fake_client.delete_calls) == [500, 1000, 1000]


@pytest.mark.asyncio
async def test_bulk_delete_from_s3_bucket_empty_input_makes_no_calls():
    s = ObjectStore()
    fake_client = _FakeS3Client()
    s.session = _FakeSession(fake_client)
    result = await s._bulk_delete_from_s3_bucket([])
    assert result == {"deleted": 0, "errors": 0}
    assert fake_client.delete_calls == []


@pytest.mark.asyncio
async def test_bulk_delete_from_s3_bucket_counts_per_key_errors():
    s = ObjectStore()
    fake_client = _FakeS3Client(errors_for={"bad1", "bad2"})
    s.session = _FakeSession(fake_client)
    result = await s._bulk_delete_from_s3_bucket(["ok1", "bad1", "ok2", "bad2"])
    assert result == {"deleted": 2, "errors": 2}


class _SlowThenFastS3Client(_FakeS3Client):
    """chunk 0 is artificially slow; if chunks ran sequentially, chunk 1 could
    never finish before chunk 0 does."""

    def __init__(self):
        super().__init__()
        self.order: list[int] = []

    async def delete_objects(self, Bucket, Delete):
        keys = [o["Key"] for o in Delete["Objects"]]
        if keys[0] == "slow-0":
            import asyncio as _asyncio
            await _asyncio.sleep(0.2)
        self.order.append(keys[0])
        return await super().delete_objects(Bucket=Bucket, Delete=Delete)


@pytest.mark.asyncio
async def test_chunks_are_dispatched_concurrently_not_one_at_a_time():
    s = ObjectStore()
    fake_client = _SlowThenFastS3Client()
    s.session = _FakeSession(fake_client)
    # 1001 keys -> two chunks (1000 + 1); the slow one is the first chunk's first key.
    keys = ["slow-0"] + [f"fast-{i}" for i in range(1000)]
    await s._bulk_delete_from_s3_bucket(keys)
    assert fake_client.order[0] != "slow-0"      # the fast chunk finished first


@pytest.mark.asyncio
async def test_bulk_delete_concurrency_is_bounded_by_the_semaphore():
    in_flight = 0
    max_in_flight = 0

    class _TrackingS3Client(_FakeS3Client):
        async def delete_objects(self, Bucket, Delete):
            nonlocal in_flight, max_in_flight
            in_flight += 1
            max_in_flight = max(max_in_flight, in_flight)
            import asyncio as _asyncio
            await _asyncio.sleep(0.05)
            in_flight -= 1
            return await super().delete_objects(Bucket=Bucket, Delete=Delete)

    s = ObjectStore()
    fake_client = _TrackingS3Client()
    s.session = _FakeSession(fake_client)
    keys = [f"k{i}" for i in range(20_000)]        # 20 chunks of 1000
    await s._bulk_delete_from_s3_bucket(keys, concurrency=3)
    assert max_in_flight <= 3
