"""DA-09a: extract task tool."""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseLLMTool


class ExtractTool(BaseLLMTool):
    task_name = "extract"

    async def run(self, full_text: str, fields: list[str] | None = None, **_: Any) -> dict[str, Any]:
        fields_hint = (
            f"需要提取的欄位：{', '.join(fields)}\n" if fields
            else "請提取所有重要資訊（日期、金額、人名、地點、關鍵數據等）。\n"
        )
        prompt = (
            "你是資訊提取專家。請從以下文件中提取結構化資訊。\n"
            f"{fields_hint}"
            "只輸出一個 JSON 物件，鍵為欄位名，值為提取到的內容（未找到填 null）。\n\n"
            f"文件內容（前 4000 字）：\n{full_text[:4000]}"
        )
        raw = await self._infer(prompt)
        import json
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            return {"extracted": json.loads(m.group(0))} if m else {"extracted": raw, "raw": True}
        except ValueError:
            return {"extracted": raw, "raw": True}
