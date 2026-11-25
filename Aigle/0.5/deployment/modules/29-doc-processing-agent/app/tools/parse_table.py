"""DA-09a: parse_table task tool."""
from __future__ import annotations

from typing import Any

from app.tools.base import BaseLLMTool


class ParseTableTool(BaseLLMTool):
    task_name = "parse_table"

    async def run(self, full_text: str, chunks: list[dict] | None = None, **_: Any) -> dict[str, Any]:
        # Prefer chunks that already contain table markup from Module 12
        table_chunks = []
        if chunks:
            table_chunks = [
                c.get("payload", {}).get("text", "")
                for c in chunks
                if "<TABLE>" in c.get("payload", {}).get("text", "")
            ]

        source_text = "\n\n".join(table_chunks) if table_chunks else full_text[:5000]

        prompt = (
            "你是表格解析專家。請從以下文件內容中提取所有表格，並轉換為結構化 JSON。\n"
            "輸出格式：\n"
            '{"tables": [{"title": "...", "headers": ["col1","col2",...], '
            '"rows": [["val1","val2",...], ...]}]}\n'
            "若無表格，回傳 {\"tables\": []}。\n\n"
            f"文件內容：\n{source_text}"
        )
        raw = await self._infer(prompt)
        import json
        import re
        m = re.search(r"\{.*\}", raw, re.DOTALL)
        try:
            return json.loads(m.group(0)) if m else {"tables": [], "raw": raw}
        except ValueError:
            return {"tables": [], "raw": raw}
