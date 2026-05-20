"""
app/a2a_helpers.py
Utility functions for packing Pydantic models into A2A text messages
and unpacking them on the other side.
Ported from 22-agent-protocol/answer/a2a_helpers.py.
"""
from __future__ import annotations
import uuid
from typing import Type, TypeVar
from pydantic import BaseModel

from a2a.types import SendMessageRequest, MessageSendParams, Task, Message
from a2a.utils import get_message_text, new_agent_text_message
from a2a.client.helpers import create_text_message_object

T = TypeVar("T", bound=BaseModel)


def pack_request(model: BaseModel) -> str:
    return model.model_dump_json()


def unpack_request(text: str, model_cls: Type[T]) -> T:
    return model_cls.model_validate_json(text)


def pack_result(model: BaseModel) -> str:
    return model.model_dump_json()


def unpack_result(text: str, model_cls: Type[T]) -> T:
    return model_cls.model_validate_json(text)


def make_send_request(payload: BaseModel) -> SendMessageRequest:
    return SendMessageRequest(
        id=str(uuid.uuid4()),
        params=MessageSendParams(
            message=create_text_message_object(content=pack_request(payload))
        ),
    )


def extract_text_from_response(response) -> str:
    result = response.root.result
    if isinstance(result, Message):
        return get_message_text(result)
    if isinstance(result, Task):
        history = result.history or []
        for msg in reversed(history):
            if msg.role.value == "agent":
                return get_message_text(msg)
    return ""
