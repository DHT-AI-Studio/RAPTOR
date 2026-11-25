"""The gateway's Kafka producer must be stopped when it fails to start.

A producer whose start() failed (broker unreachable) was just dropped. It still owned a client,
and aiokafka logged `Unclosed AIOKafkaProducer` at exit -- noise on every start without Kafka,
which is a supported configuration (UploadAsset skips its event, SubmitJob answers UNAVAILABLE).
"""
from __future__ import annotations

from types import SimpleNamespace
from typing import ClassVar

import pytest
import server


class _Producer:
    instances: ClassVar[list[_Producer]] = []

    def __init__(self, *, start_error=None, stop_error=None, **kwargs):
        self.kwargs = kwargs
        self.start_error = start_error
        self.stop_error = stop_error
        self.started = False
        self.stopped = 0
        type(self).instances.append(self)

    async def start(self):
        if self.start_error:
            raise self.start_error
        self.started = True

    async def stop(self):
        self.stopped += 1
        if self.stop_error:
            raise self.stop_error


def _install(monkeypatch, **producer_kwargs):
    _Producer.instances = []
    monkeypatch.setattr(server, "AIOKafkaProducer", lambda **kw: _Producer(**producer_kwargs, **kw))


def _settings(servers):
    return SimpleNamespace(kafka_bootstrap_servers=servers)


@pytest.mark.asyncio
async def test_no_bootstrap_servers_means_no_producer(monkeypatch):
    _install(monkeypatch)
    assert await server._start_kafka_producer(_settings("")) is None
    assert _Producer.instances == []


@pytest.mark.asyncio
async def test_a_started_producer_is_returned_running(monkeypatch):
    _install(monkeypatch)
    producer = await server._start_kafka_producer(_settings("kafka:9092"))
    assert producer is _Producer.instances[0] and producer.started and producer.stopped == 0
    assert producer.kwargs["bootstrap_servers"] == "kafka:9092"


@pytest.mark.asyncio
@pytest.mark.parametrize("error", [OSError("broker unreachable"), TimeoutError("start timed out")])
async def test_a_producer_that_failed_to_start_is_stopped_and_dropped(monkeypatch, error):
    _install(monkeypatch, start_error=error)
    assert await server._start_kafka_producer(_settings("kafka:9092")) is None
    (producer,) = _Producer.instances
    assert producer.stopped == 1, "the failed producer was dropped without being stopped (Unclosed AIOKafkaProducer)"


@pytest.mark.asyncio
async def test_stopping_a_failed_producer_cannot_prevent_startup(monkeypatch):
    _install(monkeypatch, start_error=OSError("down"), stop_error=RuntimeError("stop failed too"))
    assert await server._start_kafka_producer(_settings("kafka:9092")) is None
