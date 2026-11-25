"""A Module 15 stand-in for Module 34 load and session tests (RTA-X4 / RTA-X8).

Answers POST /api/v1/chat the way Module 15 does with "stream": true -- SSE ``start``, one
``delta`` per sentence, ``end`` -- after a think time of --delay-s (+ up to --jitter-s), so Module
34's own capacity (VAD, STT, TTS, output) can be measured without loading the LLM, the guardrail
or Module 15 itself.  Not for production use.

  python -m benchmarks.fake_chat --port 8021 --delay-s 1.5 --jitter-s 1.0
  (Module 34: RT_CHAT_SERVICE_URL=http://<this host>:8021/api/v1)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import random
import uuid

from fastapi import (  # module level: FastAPI resolves the annotations here
    FastAPI,
    Header,
    Request,
)
from fastapi.responses import JSONResponse, StreamingResponse

ANSWER = "好的，我幫您查一下。您的訂單已經出貨，預計明天下午送達。還有其他需要幫忙的嗎？"


def create_app(answer: str, delay_s: float, jitter_s: float) -> FastAPI:
    from app.llm_bridge import SentenceSplitter

    app = FastAPI(title="fake Module 15")
    stats = {"requests": 0, "active": 0, "max_active": 0}

    def sentences() -> list[str]:
        sp = SentenceSplitter()
        return sp.feed(answer) + sp.flush()

    @app.get("/stats")
    async def get_stats() -> dict:
        return stats

    @app.post("/api/v1/chat")
    async def chat(request: Request, x_request_id: str | None = Header(None, alias="X-Request-ID")):
        body = await request.json()
        rid = x_request_id or uuid.uuid4().hex
        stats["requests"] += 1
        think = delay_s + random.random() * jitter_s
        if not body.get("stream"):
            await asyncio.sleep(think)
            return JSONResponse({"response": answer, "request_id": rid, "session_id": body.get("session_id")})

        async def events():
            stats["active"] += 1
            stats["max_active"] = max(stats["max_active"], stats["active"])
            try:
                yield f"event: start\ndata: {json.dumps({'request_id': rid, 'session_id': body.get('session_id')})}\n\n"
                await asyncio.sleep(think)
                for s in sentences():
                    yield f"event: delta\ndata: {json.dumps({'text': s}, ensure_ascii=False)}\n\n"
                yield f"event: end\ndata: {json.dumps({'request_id': rid, 'response': answer}, ensure_ascii=False)}\n\n"
            finally:
                stats["active"] -= 1

        return StreamingResponse(events(), media_type="text/event-stream", headers={"X-Request-ID": rid})

    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8021)
    parser.add_argument("--answer", default=ANSWER)
    parser.add_argument("--delay-s", type=float, default=1.5, help="think time before the first sentence")
    parser.add_argument("--jitter-s", type=float, default=1.0, help="random extra think time, 0..this")
    args = parser.parse_args()
    import uvicorn

    uvicorn.run(create_app(args.answer, args.delay_s, args.jitter_s), host=args.host, port=args.port,
                log_level="warning")


if __name__ == "__main__":
    main()
