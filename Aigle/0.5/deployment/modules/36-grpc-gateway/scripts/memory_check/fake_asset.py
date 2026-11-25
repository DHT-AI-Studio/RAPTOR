#!/usr/bin/env python3
"""A stand-in for module 04 (asset management) for memory measurements.

Serves one generated blob of --size-mib MiB at the three endpoints the gateway uses for
DownloadAsset and UploadAsset, and reads and discards uploads. Nothing is stored, so a
multi-gigabyte transfer costs no disk and touches no shared service.

    python fake_asset.py --port 19000 --size-mib 1200
"""
from __future__ import annotations

import argparse
import os
import re

import uvicorn
from starlette.applications import Starlette
from starlette.responses import JSONResponse, StreamingResponse
from starlette.routing import Route

_BLOCK = os.urandom(1 << 20)


def build_app(blob_size: int, port: int) -> Starlette:
    async def filedownload(request):
        return JSONResponse({"metadata": {"primary_filename": "big.bin", "asset_path": request.path_params["p"]}})

    async def presign(request):
        return JSONResponse({"url": f"http://127.0.0.1:{port}/blob"})

    async def blob(request):
        start, end = 0, blob_size - 1
        ranged = re.match(r"bytes=(\d*)-(\d*)", request.headers.get("range", ""))
        if ranged:
            if ranged.group(1):
                start = int(ranged.group(1))
            if ranged.group(2):
                end = min(int(ranged.group(2)), blob_size - 1)
        length = end - start + 1

        async def body():
            left = length
            while left > 0:
                take = min(left, len(_BLOCK))
                yield _BLOCK[:take]
                left -= take

        headers = {"Accept-Ranges": "bytes", "Content-Length": str(length), "Content-Type": "application/octet-stream"}
        if ranged:
            headers["Content-Range"] = f"bytes {start}-{end}/{blob_size}"
        return StreamingResponse(body(), status_code=206 if ranged else 200, headers=headers)

    async def fileupload(request):
        received = 0
        async for chunk in request.stream():
            received += len(chunk)
        return JSONResponse({
            "asset_path": "video/mp4/memtest", "version_id": "v-memtest",
            "primary_filename": "big.bin", "status": "uploaded", "received": received,
        })

    return Starlette(routes=[
        Route("/filedownload/{p:path}/{v}", filedownload),
        Route("/presign-url/{k:path}/{v}", presign),
        Route("/blob", blob),
        Route("/fileupload", fileupload, methods=["POST"]),
    ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=19000)
    parser.add_argument("--size-mib", type=int, default=1200)
    args = parser.parse_args()
    uvicorn.run(build_app(args.size_mib << 20, args.port), host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
