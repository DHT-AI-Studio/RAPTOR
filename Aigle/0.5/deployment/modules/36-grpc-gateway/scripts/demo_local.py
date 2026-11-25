#!/usr/bin/env python3
"""End-to-end local demo of every real RPC on the grpc-gateway — runs
entirely against fakes (no module 04/06/25, Kafka, or Redis required), the
same fake-double pattern the pytest suite uses. Prints narrated output for
each step so it can be run live.

Usage:
    pip install -r requirements.txt
    bash scripts/gen_proto.sh   # once, if generated/ is empty
    python scripts/demo_local.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

import grpc
from grpc_health.v1 import health_pb2, health_pb2_grpc

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import dev_paths

dev_paths.setup()

from app.services.video_search_service import SegmentHit, VideoHit, VideoSearchResult
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server

AUTH = (("authorization", "Bearer demo-token"),)
DEMO_VIDEO_BYTES = b"DEMO-VIDEO-BYTES-" * 4096  # ~68 KB fake payload


def _banner(title: str) -> None:
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ── Fakes -- same doubles tests/test_*.py use, wired together for one run ──

class FakeVideoSearchService:
    async def search_by_person(self, person_name, user, top_k=10, candidate_multiplier=5, score_threshold=0.52):
        return VideoSearchResult(
            query=person_name,
            videos=[
                VideoHit(
                    video_id="v-demo-1", filename="team_meeting.mp4", score=0.91,
                    asset_url="https://storage.example/team_meeting.mp4", upload_time="2026-08-01T10:00:00Z",
                    segments=[
                        SegmentHit(start_time=12.5, end_time=18.0, score=0.91,
                                   text=f"{person_name} speaking about the roadmap", sources=["bm25", "vector"]),
                        SegmentHit(start_time=145.0, end_time=150.0, score=0.77,
                                   text=f"{person_name} answers a question", sources=["graphrag"]),
                    ],
                )
            ],
        )


class FakeStorageService:
    """Backs DownloadAsset (get_presign_url), UploadAsset (upload_file), and
    SubmitJob's asset lookup (download_asset)."""

    def __init__(self):
        self.uploads = []

    async def get_presign_url(self, asset_path, version_id, user):
        return f"https://storage.example/{asset_path}/{version_id}"

    async def download_asset(self, asset_path, version_id, user, return_file_content=False):
        return {"metadata": {
            "asset_path": asset_path, "version_id": version_id,
            "primary_filename": "team_meeting.mp4", "status": "success",
        }}

    async def upload_file(self, primary_file, archive_date_or_ttl, destroy_date_or_ttl, user):
        size = os.fstat(primary_file.file.fileno()).st_size
        self.uploads.append((primary_file.filename, size))
        return {
            "asset_path": "assets/demo-upload.bin", "version_id": f"v{len(self.uploads)}",
            "primary_filename": primary_file.filename, "status": "success",
        }


class _FakeStreamResp:
    def __init__(self, body: bytes, offset: int):
        self._body = body
        self.status_code = 206 if offset else 200
        self.headers = {
            "content-type": "video/mp4",
            "content-range": f"bytes {offset}-{len(DEMO_VIDEO_BYTES) - 1}/{len(DEMO_VIDEO_BYTES)}",
            "etag": "demo-etag",
        }

    def raise_for_status(self):
        pass

    async def aiter_bytes(self, chunk_size):
        for i in range(0, len(self._body), chunk_size):
            yield self._body[i:i + chunk_size]


class _FakeStreamCtx:
    def __init__(self, body: bytes, offset: int):
        self._resp = _FakeStreamResp(body, offset)

    async def __aenter__(self):
        return self._resp

    async def __aexit__(self, *exc):
        return False


class FakeHttpClient:
    """Backs DownloadAsset's ranged GET against the (fake) presigned URL --
    serves DEMO_VIDEO_BYTES, honouring the Range header for offset-resume."""

    def stream(self, method, url, headers=None):
        offset = 0
        range_header = (headers or {}).get("Range", "")
        if range_header.startswith("bytes="):
            offset = int(range_header[len("bytes="):].split("-")[0] or 0)
        return _FakeStreamCtx(DEMO_VIDEO_BYTES[offset:], offset)


class FakeRedis:
    def __init__(self):
        self._kv = {}
        self._lists = {}

    async def get(self, key):
        return self._kv.get(key)

    async def setex(self, key, ttl, value):
        self._kv[key] = value

    async def incr(self, key):
        self._kv[key] = int(self._kv.get(key, 0)) + 1
        return self._kv[key]

    async def decr(self, key):
        self._kv[key] = int(self._kv.get(key, 0)) - 1
        return self._kv[key]

    async def expire(self, key, ttl):
        pass

    async def rpush(self, key, value):
        self._lists.setdefault(key, []).append(value)

    async def lrange(self, key, start, end):
        items = self._lists.get(key, [])
        return items[start:] if end == -1 else items[start:end + 1]


class FakeKafkaProducer:
    def __init__(self):
        self.sent = []

    async def send_and_wait(self, topic, message):
        self.sent.append((topic, message))


async def main() -> None:
    fake_storage = FakeStorageService()
    fake_redis = FakeRedis()
    fake_kafka = FakeKafkaProducer()

    server = build_server(
        video_search_service=FakeVideoSearchService(),
        storage_service=fake_storage,
        http_client=FakeHttpClient(),
        redis_client=fake_redis,
        kafka_producer=fake_kafka,
    )
    port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{port}")
    stub = raptor_pb2_grpc.RaptorServiceStub(channel)
    health_stub = health_pb2_grpc.HealthStub(channel)

    try:
        _banner("1. Health check")
        r = await health_stub.Check(health_pb2.HealthCheckRequest(service=""))
        print("status:", health_pb2.HealthCheckResponse.ServingStatus.Name(r.status))

        _banner("2. Ping -- full interceptor chain (auth required)")
        resp = await stub.Ping(raptor_pb2.PingRequest(message="hello"), metadata=AUTH)
        print(f"authenticated -> message={resp.message!r} request_id={resp.request_id}")
        try:
            await stub.Ping(raptor_pb2.PingRequest(message="hello"))
        except grpc.aio.AioRpcError as e:
            print(f"no credentials    -> {e.code()}: {e.details()}")

        _banner("3. SearchByPerson")
        reply = await stub.SearchByPerson(raptor_pb2.SearchByPersonRequest(person_name="Alice", top_k=5), metadata=AUTH)
        print(f"total={reply.total}")
        for hit in reply.hits:
            print(f"  clip_id={hit.clip_id} score={hit.score:.2f} text={hit.text!r}")

        _banner("4. StreamSearchByPerson -- incremental delivery")
        async for hit in stub.StreamSearchByPerson(raptor_pb2.SearchByPersonRequest(person_name="Alice"), metadata=AUTH):
            print(f"  [stream] clip_id={hit.clip_id} score={hit.score:.2f}")

        _banner("5. DownloadAsset -- offset resume, constant-memory streaming")
        total = 0
        async for msg in stub.DownloadAsset(
            raptor_pb2.DownloadAssetRequest(asset_path="assets/team_meeting.mp4", version_id="v1", offset=1000),
            metadata=AUTH,
        ):
            if msg.HasField("header"):
                h = msg.header
                print(f"  header: content_type={h.content_type} content_length={h.content_length} seekable={h.seekable}")
            else:
                total += len(msg.chunk.data)
        print(f"  downloaded {total} bytes starting at offset 1000 (full file is {len(DEMO_VIDEO_BYTES)} bytes)")

        _banner("6. UploadAsset -- idempotency")

        def upload_gen(key: str):
            async def _g():
                yield raptor_pb2.UploadAssetRequest(metadata=raptor_pb2.UploadAssetMetadata(
                    filename="notes.txt", content_type="text/plain", idempotency_key=key))
                yield raptor_pb2.UploadAssetRequest(chunk_data=b"hello from the demo upload\n")
            return _g()

        resp1 = await stub.UploadAsset(upload_gen("demo-key-1"), metadata=AUTH)
        print(f"  first upload  -> asset_id={resp1.asset_id} bytes_received={resp1.bytes_received}")
        try:
            await stub.UploadAsset(upload_gen("demo-key-1"), metadata=AUTH)
        except grpc.aio.AioRpcError as e:
            trailing = e.trailing_metadata()
            print(f"  duplicate     -> {e.code()}, reused asset-id={trailing['asset-id']} (not re-written)")

        _banner("7. SubmitJob -> GetJob -> WatchJob")
        job = await stub.SubmitJob(
            raptor_pb2.SubmitJobRequest(
                job_type="fast",
                input_asset_ids=[raptor_pb2.AssetRef(asset_path="assets/team_meeting.mp4", version_id="v1")],
                request_id="demo-submit-1",
            ),
            metadata=AUTH,
        )
        print(f"  SubmitJob -> job_id={job.job_id} media_type={job.media_type} state={raptor_pb2.JobState.Name(job.state)}")

        got = await stub.GetJob(raptor_pb2.GetJobRequest(job_id=job.job_id), metadata=AUTH)
        print(f"  GetJob    -> state={raptor_pb2.JobState.Name(got.state)}")

        print("  WatchJob  -> streaming state changes (a background task stands in for")
        print("               the real Kafka-consumer worker, which isn't running here):")

        async def simulate_worker_progress():
            # Sleeps longer than WatchJob's poll interval (default 1s, see
            # GRPC_GATEWAY_WATCH_JOB_POLL_INTERVAL_S) so each transition is
            # observed on its own poll tick instead of being skipped over.
            await asyncio.sleep(1.5)
            await fake_redis.setex(
                f"{job.media_type}_orchestrator:{job.job_id}", 86400, json.dumps({"step": "transcribing"})
            )
            await asyncio.sleep(1.5)
            await fake_redis.setex(
                f"{job.media_type}_orchestrator:{job.job_id}", 86400, json.dumps({"step": "complete"})
            )

        asyncio.create_task(simulate_worker_progress())
        async for event in stub.WatchJob(raptor_pb2.WatchJobRequest(job_id=job.job_id), metadata=AUTH):
            print(f"    [event seq={event.sequence}] state={raptor_pb2.JobState.Name(event.state)}")

        _banner("Done")
        print("Every RPC above ran against the real interceptor-wired grpc.aio server --")
        print("only the downstream I/O (SeaweedFS/LakeFS, Redis, Kafka, module 25) is faked,")
        print("the same doubles tests/test_*.py use. Swap the Fake* classes above for the")
        print("real app.services.* classes (see server.py's serve()) to run against live infra.")

    finally:
        await channel.close()
        await server.stop(grace=None)


if __name__ == "__main__":
    asyncio.run(main())
