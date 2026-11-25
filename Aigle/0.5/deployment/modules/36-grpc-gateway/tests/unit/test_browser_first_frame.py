"""VIE01-163: "first chunk playable by an HTML5 <video> element before FFmpeg finishes the full
transcode ... first frame renders within 3 s of RPC start", checked with a real browser.

The other StreamVideoSegment tests prove the bytes are structurally valid (ffprobe); this one proves
a browser actually shows a frame. A real gateway streams a real FFmpeg transcode; a minimal HTTP
bridge (what Content AI's media endpoint does) re-exposes it as `video/mp4`; headless Chrome loads a
page whose <video> points at the bridge and reports when its first frame is decoded and drawn.

Skipped where there is no ffmpeg or no Chrome/Chromium (CI's unit-test runner has neither).
"""
from __future__ import annotations

import asyncio
import functools
import http.server
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import grpc
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent / "generated"))
from conftest import authz_http_client, identity_http_client, make_token
from raptor.v1 import raptor_pb2, raptor_pb2_grpc
from server import build_server
from test_media_stream_service import _asset_ref, _FakeRedis, _FakeStorageService

_BROWSERS = ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser")
_BUDGET_S = 3.0
_PER_CHUNK_DELAY_S = 0.3

_PAGE = b"""<!doctype html><meta charset=utf-8><body><script>
const post = (o) => fetch('/result', {method: 'POST', body: JSON.stringify(o)});
const v = document.createElement('video');
v.muted = true; v.autoplay = true; v.playsInline = true;
const t0 = performance.now();
v.addEventListener('error', () => post({ok: false, error: 'media error ' + (v.error && v.error.code)}));
let decodedAt = null;
function lit() {
  const c = document.createElement('canvas'); c.width = 32; c.height = 24;
  const g = c.getContext('2d'); g.drawImage(v, 0, 0, 32, 24);
  const px = g.getImageData(0, 0, 32, 24).data;
  let n = 0; for (let i = 0; i < px.length; i += 4) if (px[i] + px[i+1] + px[i+2] > 30) n++;
  return n;
}
function check() {
  // The first frame is "shown" once drawing the <video> yields real pixels, not before.
  const n = lit();
  if (n > 0) post({ok: true, ms: performance.now() - t0, decodedMs: decodedAt, w: v.videoWidth, h: v.videoHeight, litPixels: n});
  else requestAnimationFrame(check);
}
v.addEventListener('loadeddata', () => { decodedAt = performance.now() - t0; check(); });
document.body.appendChild(v);
v.src = '/video';
v.play().catch(() => {});
setTimeout(() => post({ok: false, error: 'no frame after 20 s'}), 20000);
</script>"""


@pytest.fixture(scope="module")
def long_video(tmp_path_factory):
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg binary not available on this host")
    tmp_dir = tmp_path_factory.mktemp("first_frame")
    subprocess.run(
        ["ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=30:size=640x360:rate=25",
         "-f", "lavfi", "-i", "sine=frequency=440:duration=30", "-shortest",
         # faststart: the test server below has no Range support, and FFmpeg cannot read a file whose
         # moov atom is at the end without seeking (it then outputs only the 1.3 KB header).
         "-pix_fmt", "yuv420p", "-c:v", "libx264", "-g", "25", "-c:a", "aac", "-movflags", "+faststart",
         str(tmp_dir / "long.mp4")],
        check=True, capture_output=True,
    )
    handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=str(tmp_dir))
    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{httpd.server_address[1]}/long.mp4"
    finally:
        httpd.shutdown()


def _browser() -> str:
    for name in _BROWSERS:
        path = shutil.which(name)
        if path:
            return path
    pytest.skip("no Chrome/Chromium on this host")


async def _bridge(stub, auth, state: dict):
    """Tiny HTTP server: /page, /video (chunked video/mp4 fed from the gRPC stream), /result."""

    async def handle(reader, writer):
        try:
            head = await reader.readuntil(b"\r\n\r\n")
            line = head.split(b"\r\n", 1)[0].decode()
            method, path, _ = line.split(" ", 2)
            length = 0
            for h in head.split(b"\r\n")[1:]:
                if h.lower().startswith(b"content-length:"):
                    length = int(h.split(b":", 1)[1])
            body = await reader.readexactly(length) if length else b""

            if path == "/page":
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\nContent-Length: %d\r\n\r\n" % len(_PAGE) + _PAGE)
            elif path == "/video":
                state["rpc_start"] = time.monotonic()
                writer.write(b"HTTP/1.1 200 OK\r\nContent-Type: video/mp4\r\nTransfer-Encoding: chunked\r\n\r\n")
                call = stub.StreamVideoSegment(
                    raptor_pb2.StreamVideoSegmentRequest(
                        asset=_asset_ref(), format=raptor_pb2.VIDEO_STREAM_FORMAT_MP4_H264,
                        preferred_chunk_size=64 * 1024,
                    ),
                    metadata=auth,
                )
                try:
                    async for chunk in call:
                        state.setdefault("first_chunk", time.monotonic())
                        writer.write(b"%x\r\n" % len(chunk.data) + chunk.data + b"\r\n")
                        await writer.drain()
                        # A slow consumer: the gateway's bounded queue then holds FFmpeg back, so the
                        # transcode is still running while the browser already has its first frame.
                        await asyncio.sleep(_PER_CHUNK_DELAY_S)
                    state["stream_end"] = time.monotonic()
                    writer.write(b"0\r\n\r\n")
                finally:
                    call.cancel()
            elif path == "/result" and method == "POST":
                state["result"] = json.loads(body)
                state["result_at"] = time.monotonic()
                state["done"].set()
                writer.write(b"HTTP/1.1 204 No Content\r\n\r\n")
            else:
                writer.write(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\n\r\n")
            await writer.drain()
        except (ConnectionError, asyncio.IncompleteReadError):
            pass  # the browser closed the connection (normal once it has what it needs)
        finally:
            writer.close()

    return await asyncio.start_server(handle, "127.0.0.1", 0)


@pytest.mark.asyncio
async def test_a_real_browser_shows_the_first_frame_of_the_stream_within_3_seconds(long_video):
    browser = _browser()
    server = build_server(
        identity_http_client=identity_http_client(), authz_http_client=authz_http_client(),
        storage_service=_FakeStorageService(long_video), redis_client=_FakeRedis(),
    )
    grpc_port = server.add_insecure_port("localhost:0")
    await server.start()
    channel = grpc.aio.insecure_channel(f"localhost:{grpc_port}")
    state: dict = {"done": asyncio.Event()}
    bridge = await _bridge(raptor_pb2_grpc.RaptorServiceStub(channel), (("authorization", f"Bearer {make_token()}"),), state)
    port = bridge.sockets[0].getsockname()[1]
    profile = tempfile.mkdtemp(prefix="first-frame-chrome-")
    proc = await asyncio.create_subprocess_exec(
        browser, "--headless=new", "--no-sandbox", "--disable-gpu", "--mute-audio", "--no-first-run",
        "--autoplay-policy=no-user-gesture-required", f"--user-data-dir={profile}", f"http://127.0.0.1:{port}/page",
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
    )
    try:
        await asyncio.wait_for(state["done"].wait(), timeout=40)
    finally:
        proc.kill()
        await proc.wait()
        bridge.close()
        await channel.close()
        await server.stop(grace=None)
        shutil.rmtree(profile, ignore_errors=True)

    result = state["result"]
    assert result["ok"], f"the browser did not show a frame: {result}"
    assert result["w"] == 640 and result["h"] == 360
    assert result["litPixels"] > 100, "the first frame was blank"
    elapsed = state["result_at"] - state["rpc_start"]
    print(f"first frame {elapsed:.2f}s after the RPC started; first gRPC chunk "
          f"{state['first_chunk'] - state['rpc_start']:.2f}s; browser: decoded at {result['decodedMs']:.0f} ms, "
          f"drawn at {result['ms']:.0f} ms")
    assert elapsed < _BUDGET_S, f"first frame took {elapsed:.2f}s (budget {_BUDGET_S}s)"
    # "before FFmpeg finishes the full transcode": with the consumer held to ~0.3 s per 64 KiB chunk the
    # stream takes seconds, and the first frame must not have waited for it.
    # No "stream_end" at all means the stream was still running when the browser had its frame.
    assert "stream_end" not in state or state["stream_end"] - state["result_at"] > 1.0, (
        "the first frame only appeared when the stream was about to end"
    )
