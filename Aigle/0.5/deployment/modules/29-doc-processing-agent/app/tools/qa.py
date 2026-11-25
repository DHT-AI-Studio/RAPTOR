"""DA-09a: qa task tool."""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseLLMTool


class QATool(BaseLLMTool):
    task_name = "qa"

    async def run(self, full_text: str, question: str | None = None, **_: Any) -> dict[str, Any]:
        if not question:
            return {"error": "qa task requires options.question"}

        prompt = (
            "你是文件問答專家。請根據以下文件內容，回答問題。\n"
            "只根據文件內容作答，若文件中找不到答案請明確說明。\n"
            "輸出格式：\n"
            '{"answer": "...", "confidence": <0.0-1.0>, "source_hint": "文件中的相關段落摘要"}\n\n'
            f"問題：{question}\n\n"
            f"文件內容（前 5000 字）：\n{full_text[:5000]}"
        )
        raw = await self._infer(prompt)
        import json
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else {"answer": raw, "raw": True}
        except ValueError:
            return {"answer": raw, "raw": True}
