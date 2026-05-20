# app/services/agent_card_service.py
"""Builds the Agent Card JSON from environment variables."""

from __future__ import annotations

from typing import Any, Dict

from app.core.config import get_settings


class AgentCardService:
    def __init__(self):
        self.settings = get_settings()
    
    def build(self) -> Dict[str, Any]:
        return {
            "id":          self.settings.agent_id,
            "name":        self.settings.agent_name,
            "version":     "0.3.0",
            "description": "Multimodal RAG + Knowledge Graph platform gateway",
            "endpoint":    self.settings.agent_endpoint,
            "capabilities": [
                "document_rag",
                "video_rag",
                "graph_rag",
                "tkg_query",
                "tts",
            ],
            "protocols":       ["a2a/1.0", "jsonrpc/2.0"],
            "authentication":  {"type": "bearer"},
        }
