# services/document_graph_service/graph_builder.py

import json
import logging
import aiohttp
from typing import List, Dict, Any
from dotenv import load_dotenv
import os

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
load_dotenv(os.path.join(parent_dir, ".env"))

INFERENCE_URL = os.getenv("INFERENCE_URL", "http://raptor-ai-lifecycle-api:8010/inference/infer")
INFERENCE_MODEL = os.getenv("INFERENCE_MODEL", "qwen3.5:9b")
TEMPORAL_MODEL_URL = os.getenv("TEMPORAL_MODEL_URL", "http://raptor-temporal-model-service:8841")

logger = logging.getLogger(__name__)

ENTITY_EXTRACTION_PROMPT = """Extract entities and relationships from the following text.
Return ONLY valid JSON with this exact structure:
{{
  "entities": [
    {{"name": "Entity Name", "type": "PERSON|ORG|LOCATION|EVENT|CONCEPT|OTHER"}}
  ],
  "relations": [
    {{
      "subject": "Entity A",
      "relation": "RELATION_TYPE",
      "object": "Entity B",
      "time_start": "YYYY-MM-DD or null",
      "time_end": "YYYY-MM-DD or null"
    }}
  ]
}}

Text:
{text}"""


class GraphBuilder:
    async def extract_entities_and_relations(
        self, chunks: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Call LLM to extract entities and relations from document chunks."""
        combined_text = "\n\n".join(
            c.get("text", "") for c in chunks[:20]  # limit to first 20 chunks
        )
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=combined_text[:6000])

        async with aiohttp.ClientSession() as session:
            payload = {
                "task": "text-generation",
                "engine": "ollama",
                "model_name": INFERENCE_MODEL,
                "data": {"inputs": prompt},
                "options": {"max_tokens": 2000, "temperature": 0.1},
            }
            async with session.post(INFERENCE_URL, json=payload, timeout=aiohttp.ClientTimeout(total=120)) as resp:
                if resp.status != 200:
                    raise RuntimeError(f"Inference API error: {resp.status}")
                result = await resp.json()

        raw_text = result.get("result", {}).get("response", "")
        return self._parse_json_response(raw_text)

    def _parse_json_response(self, raw: str) -> Dict[str, Any]:
        """Extract JSON block from LLM response."""
        try:
            start = raw.find("{")
            end = raw.rfind("}") + 1
            if start >= 0 and end > start:
                return json.loads(raw[start:end])
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse entity extraction JSON: {e}")
        return {"entities": [], "relations": []}

    async def post_temporal_facts(
        self, relations: List[Dict[str, Any]], source_document_id: str
    ) -> None:
        """Push temporally-anchored relations to the temporal model service."""
        temporal = [
            r for r in relations
            if r.get("time_start") or r.get("time_end")
        ]
        if not temporal:
            return

        facts = [
            {
                "entity": r["subject"],
                "relation": r.get("relation", "RELATED_TO"),
                "value": r["object"],
                "time_start": r.get("time_start"),
                "time_end": r.get("time_end"),
                "source_document_id": source_document_id,
            }
            for r in temporal
        ]

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{TEMPORAL_MODEL_URL}/temporal/facts",
                    json={"facts": facts},
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    if resp.status not in (200, 201):
                        logger.warning(f"Temporal model service returned {resp.status}")
            except Exception as e:
                logger.warning(f"Could not reach temporal model service: {e}")
