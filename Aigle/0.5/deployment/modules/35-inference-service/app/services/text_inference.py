"""INF-5..8: text tasks on top of Module 07 ``POST /inference/infer``.

Shared path for every task: resolve the model (INF-3), build system/user messages,
call 07 with the registry-selected version, map errors, and return
``{task, model, model_version, output, usage, latency_ms}``.  Tasks only add their
own prompt and post-processing.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any

from app.config import Settings
from app.errors import ServiceError
from app.services import prompts
from app.services.guardrail import Guardrail
from app.services.model_defaults import ModelResolver, model_not_found
from app.services.model_registry import ModelEntry
from app.services.module07_client import Module07Client

logger = logging.getLogger(__name__)

TEXT_TASK = "text"
# Sampling used by the prompt-driven tasks; `complete` forwards the caller's own values.
TASK_TEMPERATURE = 0.2
_CHUNK_CONCURRENCY = 2

_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL)
_BULLET = re.compile(r"^\s*(?:[-*•+]|\d+[.)、])\s+(.*\S)\s*$")
_SENTENCE_SPLIT = re.compile(r"(?<=[。！？])|(?<=[.!?])(?=\s)")
_CJK = "぀-ヿ㐀-鿿豈-﫿가-힯"
_WORD_UNIT = re.compile(rf"[{_CJK}]|[^\s{_CJK}]+")
_DETECTED = re.compile(
    rf"^\s*`?{prompts.DETECTED_LANGUAGE_MARKER}\s*[:：]\s*([A-Za-z]{{2,3}}(?:[-_][A-Za-z0-9]{{2,8}})*)`?\s*$",
    re.MULTILINE,
)
HEADLINE_MAX_WORDS = 20


# ---------------------------------------------------------------- post-processing helpers

def clean_output(text: str) -> str:
    return _THINK_BLOCK.sub("", text or "").strip()


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s and s.strip()]


def paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text.strip()) if p.strip()]


def count_words(text: str) -> int:
    """Headline length unit: one per CJK character, one per whitespace-separated word
    elsewhere; tokens without letters/digits (punctuation) are not counted."""
    return sum(1 for m in _WORD_UNIT.finditer(text) if re.search(r"\w", m.group()))


def truncate_words(text: str, limit: int) -> str:
    count = 0
    for m in _WORD_UNIT.finditer(text):
        if re.search(r"\w", m.group()):
            count += 1
            if count == limit:
                return text[: m.end()].rstrip(",，、;；:：-–—")
    return text


def format_bullets(text: str, max_items: int | None) -> str:
    lines = [ln for ln in text.splitlines() if ln.strip()]
    items = [m.group(1) for ln in lines if (m := _BULLET.match(ln))]
    if not items:  # model answered in prose: one bullet per sentence
        items = split_sentences(" ".join(ln.strip() for ln in lines))
    if max_items:
        items = items[:max_items]
    return "\n".join(f"- {item}" for item in items)


def format_paragraphs(text: str, max_sentences: int | None) -> str:
    paras = [" ".join(ln.strip() for ln in p.splitlines()) for p in paragraphs(text)][:3]
    if max_sentences:
        kept: list[str] = []
        budget = max_sentences
        for p in paras:
            if budget <= 0:
                break
            sentences = split_sentences(p)[:budget]
            budget -= len(sentences)
            kept.append(_join_sentences(sentences))
        paras = kept
    return "\n\n".join(paras)


def _join_sentences(sentences: list[str]) -> str:
    out = ""
    for s in sentences:
        if out and not re.search(rf"[{_CJK}。！？]$", out):
            out += " "
        out += s
    return out


def format_headline(text: str) -> str:
    first = next((ln for ln in text.splitlines() if ln.strip()), "")
    first = re.sub(r"^\s*(?:#+|[-*•+]|\d+[.)])\s*", "", first).strip().strip("\"'“”「」*").strip()
    sentences = split_sentences(first)
    headline = sentences[0] if sentences else first
    if count_words(headline) > HEADLINE_MAX_WORDS:
        cut = truncate_words(headline, HEADLINE_MAX_WORDS)
        # Prefer ending at the last clause boundary inside the limit over cutting mid-phrase,
        # as long as that keeps most of the headline.
        clauses = [m.start() for m in re.finditer(r"[,，、;；:：—–]", cut)]
        if clauses and count_words(cut[: clauses[-1]]) >= HEADLINE_MAX_WORDS * 0.6:
            cut = cut[: clauses[-1]]
        headline = cut.rstrip()
    return headline


def parse_detected_language(text: str) -> tuple[str | None, str]:
    """Split the ``DETECTED_LANGUAGE: xx`` header line off a translation."""
    match = _DETECTED.search(text)
    if not match or text[: match.start()].strip():
        return None, text.strip()
    code = match.group(1).replace("_", "-")
    return code, text[match.end():].strip()


def structure_report(source: str, output: str) -> dict[str, Any]:
    def headings(t: str) -> int:
        return sum(1 for ln in t.splitlines() if re.match(r"^\s*#{1,6}\s", ln))

    src_p, out_p = len(paragraphs(source)), len(paragraphs(output))
    src_h, out_h = headings(source), headings(output)
    return {
        "preserved": src_p == out_p and src_h == out_h,
        "input_paragraphs": src_p,
        "output_paragraphs": out_p,
        "input_headings": src_h,
        "output_headings": out_h,
    }


def split_chunks(text: str, max_chars: int) -> list[str]:
    """Split on paragraph, then line, then sentence boundaries; hard-cut as a last resort."""
    if len(text) <= max_chars:
        return [text]

    def pieces(block: str) -> list[str]:
        if len(block) <= max_chars:
            return [block]
        for sep_split in (lambda b: re.split(r"\n\s*\n", b), lambda b: b.split("\n"), split_sentences):
            parts = [p for p in sep_split(block) if p.strip()]
            if len(parts) > 1:
                return [piece for part in parts for piece in pieces(part)]
        return [block[i:i + max_chars] for i in range(0, len(block), max_chars)]

    chunks: list[str] = []
    current = ""
    for piece in pieces(text):
        if current and len(current) + len(piece) + 2 > max_chars:
            chunks.append(current)
            current = piece
        else:
            current = f"{current}\n\n{piece}" if current else piece
    if current:
        chunks.append(current)
    return chunks


def sum_usage(usages: list[dict[str, int] | None]) -> dict[str, int] | None:
    if not usages or any(u is None for u in usages):
        return None
    keys = ("prompt_tokens", "completion_tokens", "total_tokens")
    return {k: sum(int(u.get(k) or 0) for u in usages) for k in keys}


# ---------------------------------------------------------------- service

class TextInference:
    def __init__(
        self, client: Module07Client, resolver: ModelResolver, settings: Settings,
        guardrail: Guardrail | None = None,
    ) -> None:
        self._client = client
        self._resolver = resolver
        self._settings = settings
        self._guardrail = guardrail
        self._info_cache: dict[tuple[str, str | None, str | None], tuple[float, dict[str, Any]]] = {}

    async def _model_info(self, entry: ModelEntry) -> dict[str, Any]:
        key = (entry.name, entry.version, entry.engine)
        cached = self._info_cache.get(key)
        if cached and time.monotonic() - cached[0] < self._settings.model_cache_ttl_s:
            return cached[1]
        info = await self._call(entry, self._client.model_info(entry.name, entry.version, entry.engine))
        self._info_cache[key] = (time.monotonic(), info)
        return info

    async def _call(self, entry: ModelEntry, coro):
        try:
            return await coro
        except ServiceError as exc:
            if exc.error == "model_not_found":
                # Registry is stale (model or selected version removed): refresh and report.
                self._resolver.registry.invalidate()
                self._info_cache.clear()
                raise await model_not_found(self._resolver.registry, TEXT_TASK, entry.name, exc.message) from exc
            raise

    async def _generate(
        self,
        entry: ModelEntry,
        system_prompt: str | None,
        user_text: str,
        options: dict[str, Any],
    ) -> tuple[str, dict[str, int] | None]:
        if self._guardrail:
            user_text = await self._screen_input(system_prompt, user_text)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        body = await self._call(
            entry, self._client.infer(entry.name, entry.version, entry.engine, messages, options)
        )
        result = body.get("result") or {}
        text = result.get("response") if isinstance(result, dict) else result
        output = clean_output(text if isinstance(text, str) else "")
        if self._guardrail:
            output = await self._guardrail.check_output(output)
        return output, body.get("usage")

    async def _screen_input(self, system_prompt: str | None, user_text: str) -> str:
        """Check the system prompt and user text together, as one input; return
        the user text to generate from. The system prompt goes to Module 07 as
        its own message, so a redaction is applied to the user text only; one
        that reached into the system prompt can't be split back apart and is
        rejected instead."""
        prefix = f"{system_prompt}\n" if system_prompt and user_text else ""
        checked_text = prefix + user_text if user_text else (system_prompt or "")
        checked = await self._guardrail.check_input(checked_text)
        if checked == checked_text or not user_text:
            return user_text
        if not checked.startswith(prefix):
            raise ServiceError(400, "guardrail_rejection",
                               reason="redaction reached the system prompt", direction="input")
        return checked[len(prefix):]

    async def _task_options(self, entry: ModelEntry) -> dict[str, Any]:
        """Prompt-driven tasks use the model's full output budget when 07 knows it, so long
        texts are not cut at a runtime default (HF: 512 tokens)."""
        options: dict[str, Any] = {"temperature": TASK_TEMPERATURE}
        limit = ((await self._model_info(entry)).get("limits") or {}).get("max_output_tokens")
        if limit:
            options["max_tokens"] = int(limit)
        return options

    @staticmethod
    def _response(task: str, entry: ModelEntry, output: str, usage, started: float, **extra) -> dict[str, Any]:
        return {
            "task": task,
            "model": entry.name,
            "model_version": entry.version,
            "output": output,
            "usage": usage,
            "latency_ms": int((time.perf_counter() - started) * 1000),
            **extra,
        }

    # ----- INF-5 -----
    async def translate(self, model: str | None, text: str, options) -> dict[str, Any]:
        started = time.perf_counter()
        entry = await self._resolver.resolve(TEXT_TASK, model)
        system = prompts.translate_prompt(
            self._settings.translate_system_prompt,
            options.target_language, options.source_language, options.formality,
        )
        output, usage = await self._generate(entry, system, text, await self._task_options(entry))
        extra: dict[str, Any] = {"target_language": options.target_language}
        if options.source_language:
            extra["source_language"] = options.source_language
        else:
            detected, output = parse_detected_language(output)
            extra["detected_language"] = detected
        return self._response("translate", entry, output, usage, started, **extra)

    # ----- INF-6 -----
    async def summarize(self, model: str | None, text: str, options) -> dict[str, Any]:
        started = time.perf_counter()
        entry = await self._resolver.resolve(TEXT_TASK, model)
        chunks = split_chunks(text, self._settings.summarize_chunk_chars)
        if len(chunks) > self._settings.summarize_max_chunks:
            raise ServiceError(
                413, "input_too_long",
                f"input splits into {len(chunks)} chunks of {self._settings.summarize_chunk_chars} chars; "
                f"the limit is {self._settings.summarize_max_chunks}",
                max_chars=self._settings.summarize_chunk_chars * self._settings.summarize_max_chunks,
            )
        task_options = await self._task_options(entry)
        usages: list[dict[str, int] | None] = []
        source = text
        if len(chunks) > 1:
            lang = (
                f"Write it in {prompts.language_label(options.language)}."
                if options.language else "Write it in the same language as the text."
            )
            chunk_prompt = prompts.CHUNK_SUMMARY_PROMPT.replace("{language_instruction}", lang)
            gate = asyncio.Semaphore(_CHUNK_CONCURRENCY)

            async def one(chunk: str):
                async with gate:
                    return await self._generate(entry, chunk_prompt, chunk, task_options)

            partials = await asyncio.gather(*(one(c) for c in chunks))
            usages.extend(u for _, u in partials)
            source = "\n\n".join(p for p, _ in partials)

        system = prompts.summarize_prompt(
            self._settings.summarize_system_prompt, options.style, options.max_sentences, options.language
        )
        raw, usage = await self._generate(entry, system, source, task_options)
        usages.append(usage)
        if options.style == "bullet":
            output = format_bullets(raw, options.max_sentences)
        elif options.style == "paragraph":
            output = format_paragraphs(raw, options.max_sentences)
        else:
            output = format_headline(raw)
        return self._response(
            "summarize", entry, output, sum_usage(usages), started, style=options.style, chunks=len(chunks)
        )

    # ----- INF-7 -----
    async def complete(self, model: str | None, text: str, options) -> dict[str, Any]:
        started = time.perf_counter()
        entry = await self._resolver.resolve(TEXT_TASK, model)
        info = await self._model_info(entry)
        limit = (info.get("limits") or {}).get("max_output_tokens")
        if limit is None:
            raise ServiceError(
                503, "model_limits_not_configured",
                f"max_output_tokens is not configured for model '{entry.name}' in Module 07; "
                "set it via POST /models/set_model_limits before using /inference/text/complete",
                model=entry.name, model_version=entry.version,
            )
        if options.max_tokens > int(limit):
            raise ServiceError(
                422, "max_tokens_exceeded", model=entry.name, model_limit=int(limit),
                requested=options.max_tokens,
            )
        # Forwarded exactly as received: same field names, same values.
        forward: dict[str, Any] = {"max_tokens": options.max_tokens, "temperature": options.temperature}
        if options.stop_sequences is not None:
            forward["stop_sequences"] = options.stop_sequences
        output, usage = await self._generate(entry, options.system_prompt, text, forward)
        return self._response("complete", entry, output, usage, started)

    # ----- INF-8 -----
    async def transform(self, model: str | None, text: str, options) -> dict[str, Any]:
        started = time.perf_counter()
        entry = await self._resolver.resolve(TEXT_TASK, model)
        system = prompts.transform_prompt(
            self._settings.transform_system_prompt, options.instruction, options.preserve_structure
        )
        output, usage = await self._generate(entry, system, text, await self._task_options(entry))
        extra: dict[str, Any] = {}
        if options.preserve_structure:
            extra["structure"] = structure_report(text, output)
        return self._response("transform", entry, output, usage, started, **extra)

    # ----- INF-18 -----
    async def stream_complete(
        self,
        model: str | None,
        text: str,
        options,  # CompleteOptions
    ):
        """Async generator: yields SSE-formatted bytes for POST /inference/text/complete/stream.

        Lifecycle:
          1. Resolve model + validate limits  (before any bytes are sent)
          2. Guardrail input check            (before any bytes are sent)
          3. Open streaming connection to 07  (before any bytes are sent)
          4. Yield token events               (one per JSONL line from 07)
          5. Yield final event + [DONE]       (on done=true line)
          6. On mid-stream upstream error → yield error event then return

        Output guardrail is intentionally omitted: the full output is never
        assembled in memory, which is the entire point of streaming.
        """
        import json as _json

        # ── Step 1: model resolution + limit validation (mirrors complete()) ──
        entry = await self._resolver.resolve(TEXT_TASK, model)
        info = await self._model_info(entry)
        limit = (info.get("limits") or {}).get("max_output_tokens")
        if limit is None:
            raise ServiceError(
                503, "model_limits_not_configured",
                f"max_output_tokens is not configured for model '{entry.name}' in Module 07; "
                "set it via POST /models/set_model_limits before using /inference/text/complete/stream",
                model=entry.name, model_version=entry.version,
            )
        if options.max_tokens > int(limit):
            raise ServiceError(
                422, "max_tokens_exceeded",
                model=entry.name, model_limit=int(limit), requested=options.max_tokens,
            )

        # ── Step 2: guardrail input check ──
        if self._guardrail:
            combined = "\n".join(filter(None, [options.system_prompt, text]))
            await self._guardrail.check_input(combined)

        # ── Step 3: build messages + options, open streaming connection ──
        messages: list[dict[str, str]] = []
        if options.system_prompt:
            messages.append({"role": "system", "content": options.system_prompt})
        messages.append({"role": "user", "content": text})

        forward: dict[str, Any] = {
            "max_tokens": options.max_tokens,
            "temperature": options.temperature,
        }
        if options.stop_sequences is not None:
            forward["stop_sequences"] = options.stop_sequences

        # _call() wraps model_not_found registry refresh logic
        response = await self._call(
            entry,
            self._client.infer_stream(entry.name, entry.version, entry.engine, messages, forward),
        )

        # ── Steps 4-6: consume JSONL stream, yield SSE events ──
        async def _generate():
            try:
                async for raw_line in response.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue

                    payload_str = line[len("data:"):].strip()

                    if payload_str == "[DONE]":
                        yield b"data: [DONE]\n\n"
                        return

                    try:
                        body = _json.loads(payload_str)
                    except _json.JSONDecodeError:
                        continue

                    # OpenAI SSE chunk: choices[0].delta.content
                    choices = body.get("choices") or []
                    if not choices:
                        continue

                    choice = choices[0]
                    delta_obj = choice.get("delta") or {}
                    delta = delta_obj.get("content") or ""
                    finish_reason = choice.get("finish_reason")  # "stop" | "length" | None

                    if finish_reason is not None:
                        # Emit finish event; [DONE] arrives in the next iteration
                        finish = "length" if finish_reason == "length" else "stop"
                        yield (
                            f'data: {_json.dumps({"token": "", "finish_reason": finish})}\n\n'
                        ).encode()
                        continue

                    if delta:
                        yield (
                            f'data: {_json.dumps({"token": delta, "finish_reason": None})}\n\n'
                        ).encode()

                # Stream ended without [DONE]
                yield b'data: {"error": "upstream_error", "message": "stream ended without [DONE] sentinel"}\n\n'

            except Exception as exc:
                logger.error("Unexpected error during SSE stream: %s", exc, exc_info=True)
                try:
                    import json as _j
                    yield (
                        f'data: {_j.dumps({"error": "upstream_error", "message": str(exc)})}\n\n'
                    ).encode()
                except Exception as e:  # noqa: BLE001
                    logger.debug("Stream cleanup error (ignored): %s", e)
            finally:
                await response.aclose()


        return _generate()
