"""DA-09a: summarize task tool."""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseLLMTool


class SummarizeTool(BaseLLMTool):
    task_name = "summarize"

    async def run(self, full_text: str, max_length: int = 500, **_: Any) -> dict[str, Any]:
        prompt = (
            f"請用繁體中文為以下文件撰寫摘要，字數約 {max_length} 字以內。\n"
            "摘要應涵蓋：主要主題、關鍵結論、重要數據。\n"
            "直接輸出摘要文字，不需要標題或前言。\n\n"
            f"文件內容：\n{full_text[:6000]}"
        )
        summary = await self._infer(prompt)
        return {"summary": summary, "char_count": len(summary)}
