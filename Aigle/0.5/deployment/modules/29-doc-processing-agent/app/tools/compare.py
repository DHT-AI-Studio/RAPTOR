"""DA-09a: compare task tool."""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseLLMTool


class CompareTool(BaseLLMTool):
    task_name = "compare"

    async def run(
        self,
        full_text: str,
        compare_text: str | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        if not compare_text:
            return {"error": "compare task requires a second document via options.compare_text"}

        prompt = (
            "你是文件比較專家。請比較以下兩份文件的異同。\n"
            "只輸出一個 JSON 物件，格式：\n"
            '{"similarities": ["..."], "differences": ["..."], "summary": "..."}\n\n'
            f"【文件 A】（前 2500 字）：\n{full_text[:2500]}\n\n"
            f"【文件 B】（前 2500 字）：\n{compare_text[:2500]}"
        )
        raw = await self._infer(prompt)
        import json
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else {"result": raw, "raw": True}
        except ValueError:
            return {"result": raw, "raw": True}
