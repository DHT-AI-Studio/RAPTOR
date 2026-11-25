"""DA-09a: classify task tool."""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseLLMTool


class ClassifyTool(BaseLLMTool):
    task_name = "classify"

    async def run(self, full_text: str, **_: Any) -> dict[str, Any]:
        prompt = (
            "你是文件分類專家。請根據以下文件內容，判斷文件類型。\n"
            "只輸出一個 JSON 物件，格式：\n"
            '{"category": "<類型>", "subcategory": "<子類型>", "confidence": <0.0-1.0>, "reason": "<簡短說明>"}\n'
            "常見類型：財務報告、合約、技術文件、學術論文、新聞稿、電子郵件、其他\n\n"
            f"文件內容（前 3000 字）：\n{full_text[:3000]}"
        )
        raw = await self._infer(prompt)
        import json
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else {"category": raw, "raw": True}
        except ValueError:
            return {"category": raw, "raw": True}
