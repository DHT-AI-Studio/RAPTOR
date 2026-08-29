# agent_protocol/jsonrpc_handler.py

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# JSON-RPC 2.0 error codes
PARSE_ERROR      = -32700
INVALID_REQUEST  = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS   = -32602
INTERNAL_ERROR   = -32603


def _ok(id_: Any, result: Any) -> Dict:
    return {"jsonrpc": "2.0", "id": id_, "result": result}


def _err(id_: Any, code: int, message: str, data: Any = None) -> Dict:
    error: Dict = {"code": code, "message": message}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": id_, "error": error}


class JSONRPCHandler:
    """Dispatch JSON-RPC 2.0 method calls for the A2A protocol."""

    def __init__(self, agent_card_service, discovery_service):
        self.agent_card = agent_card_service
        self.discovery  = discovery_service

    async def handle(self, body: Dict) -> Dict:
        rpc_id = body.get("id")
        method = body.get("method", "")
        params = body.get("params", {})

        if body.get("jsonrpc") != "2.0" or not method:
            return _err(rpc_id, INVALID_REQUEST, "Invalid JSON-RPC 2.0 request")

        try:
            if method == "agent.discover":
                result = self.agent_card.build()
                return _ok(rpc_id, result)

            elif method == "agent.delegate":
                task        = params.get("task", "")
                payload     = params.get("payload", {})
                target_id   = params.get("target_agent_id")
                result = await self.discovery.delegate(task, payload, target_id)
                return _ok(rpc_id, result)

            elif method == "agent.query":
                query  = params.get("query", "")
                # Session identity for memory archiving — explicit
                # params.session_id wins, else fall back to the JSON-RPC id.
                if not params.get("session_id") and rpc_id is not None:
                    params["session_id"] = str(rpc_id)
                result = await self.discovery.query(query, params)
                return _ok(rpc_id, result)

            else:
                return _err(rpc_id, METHOD_NOT_FOUND, f"Method not found: {method}")

        except Exception as exc:
            logger.exception(f"JSON-RPC handler error for method={method}")
            return _err(rpc_id, INTERNAL_ERROR, str(exc))
