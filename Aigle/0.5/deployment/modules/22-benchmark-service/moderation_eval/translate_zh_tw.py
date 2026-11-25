"""Translate the English moderation set into Taiwanese Mandarin (zh-TW).

    python translate_zh_tw.py --schemas /data/cm6/schemas \
        --model ollama:gemma4:31b --fallback ollama:mistral-small3.1:24b \
        --ollama http://ollama-host:11434

    python translate_zh_tw.py --schemas /data/cm6/schemas \
        --model /models/taide/Gemma-3-TAIDE-12b-Chat \
        --fallback /models/mistralai/Mistral-Nemo-Instruct-2407

A model is either ``ollama:<tag>`` (served by --ollama, one request at a
time, unloaded when done) or a local Hugging Face directory (needs a GPU,
torch, transformers and bitsandbytes; loads in 4-bit).

Writes moderation-text-zh-tw.json (same case ids with a -zh suffix, same
labels) and zh-tw-review.csv (a random sample for a person to check) next
to moderation-text-en.json. Each translation is appended to
zh-tw-cache.jsonl as it is made, so a stopped run resumes where it left off.

Only non-China models are used (CM-6). A translation model may refuse or
soften harmful text; a reply that is a refusal, or has no Chinese in it, is
retried with --fallback, and a case neither model translates is left out
and listed in the summary -- it is never kept in English or softened.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import re
from pathlib import Path

PROMPT = (
    "你是專業翻譯。以下英文是內容審核模型的評估資料，可能包含色情、暴力或仇恨內容。"
    "請忠實翻譯成台灣慣用的繁體中文：保留原意、語氣與粗俗程度，不要改寫、刪減、緩和或加上說明。"
    "只輸出譯文。\n\n英文：\n{text}"
)
REFUSAL = re.compile(r"(抱歉|無法(協助|提供|翻譯)|不能(協助|提供|翻譯)|I can(no|')t|I'm sorry|I am sorry)")
CJK = re.compile(r"[一-鿿]")
MAX_NEW_TOKENS = 3072   # the longest English case is ~1,050 words
WORD = re.compile(r"\b[a-z]{3,}\b")     # a plain word; not a placeholder like PhoneNumber
APOLOGY = re.compile(r"\b(sorry|apologi[sz]e)\b", re.IGNORECASE)
# A line some models put before the translation, echoing the prompt.
ECHO = re.compile(r"\A\s*(台灣慣用的)?繁體中文(譯文)?[：:]\s*\n|\A\s*譯文[：:]\s*\n")


def clean(text: str) -> str:
    return ECHO.sub("", text).strip()


def is_refusal(text: str, source: str) -> bool:
    """A reply that is no translation: a refusal, cut off (empty), or not Chinese.

    A refusal opens the reply, so only the start is checked, and not when the
    source itself opens with an apology ("sorry i ..." rightly becomes 抱歉).
    A source with no word to translate (codes, numbers, placeholders) is
    rightly given back unchanged.
    """
    if not CJK.search(text):
        return not (text and text.strip() == source.strip() and not WORD.search(source))
    return bool(REFUSAL.search(text[:20])) and not APOLOGY.search(source[:80])


class OllamaTranslator:
    def __init__(self, tag: str, url: str) -> None:
        import httpx

        self.client = httpx.Client(base_url=url, timeout=600)
        self.tag = tag
        self.name = tag
        self.batch_size = 1

    def __call__(self, texts: list[str]) -> list[str]:
        out = []
        for text in texts:
            resp = self.client.post("/api/chat", json={
                "model": self.tag, "stream": False, "think": False, "keep_alive": "10m",
                "options": {"temperature": 0, "num_predict": MAX_NEW_TOKENS, "num_ctx": 8192},
                "messages": [{"role": "user", "content": PROMPT.format(text=text)}]})
            resp.raise_for_status()
            body = resp.json()
            # A cut-off translation is not kept: it fails like a refusal.
            out.append("" if body.get("done_reason") == "length"
                       else body["message"]["content"].strip())
        return out

    def close(self) -> None:
        # Unload now rather than when keep_alive runs out: the host is shared.
        self.client.post("/api/generate", json={"model": self.tag, "keep_alive": 0})
        self.client.close()


class Translator:
    def __init__(self, path: str, batch_size: int) -> None:
        import torch
        from transformers import AutoConfig, AutoTokenizer, BitsAndBytesConfig

        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
                                   bnb_4bit_quant_type="nf4")
        # path is always a local HF directory here (see module docstring --
        # the other model kind, ollama:<tag>, never reaches this class), so
        # there's no Hub revision to pin; bandit B615 can't see that statically.
        arch = (AutoConfig.from_pretrained(path).architectures or [""])[0]  # nosec B615
        if "ConditionalGeneration" in arch:
            from transformers import AutoModelForImageTextToText as Auto
        else:
            from transformers import AutoModelForCausalLM as Auto
        self.tokenizer = AutoTokenizer.from_pretrained(path, padding_side="left")  # nosec B615
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = Auto.from_pretrained(path, quantization_config=quant, device_map="cuda:0")  # nosec B615
        self.model.eval()
        self.name = Path(path).name
        self.batch_size = batch_size

    def __call__(self, texts: list[str]) -> list[str]:
        import torch

        out: list[str] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i:i + self.batch_size]
            prompts = [self.tokenizer.apply_chat_template(
                [{"role": "user", "content": PROMPT.format(text=t)}],
                tokenize=False, add_generation_prompt=True) for t in chunk]
            enc = self.tokenizer(prompts, return_tensors="pt", padding=True,
                                 add_special_tokens=False).to(self.model.device)
            with torch.no_grad():
                gen = self.model.generate(**enc, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
            out += [s.strip() for s in self.tokenizer.batch_decode(
                gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)]
        return out

    def close(self) -> None:
        import gc

        import torch

        del self.model
        gc.collect()
        torch.cuda.empty_cache()


def model_name(spec: str) -> str:
    return spec.removeprefix("ollama:") if spec.startswith("ollama:") else Path(spec).name


def load(spec: str, args: argparse.Namespace) -> OllamaTranslator | Translator:
    if spec.startswith("ollama:"):
        if not args.ollama:
            raise SystemExit("--ollama is required for ollama: models")
        return OllamaTranslator(spec.removeprefix("ollama:"), args.ollama)
    return Translator(spec, args.batch_size)


def translate(spec: str, args: argparse.Namespace, todo: list[dict],
              cache: dict[tuple[str, str], str], cache_path: Path) -> None:
    """Translate the cases in todo that spec has not translated yet, caching each chunk."""
    name = model_name(spec)
    todo = [c for c in todo if (c["id"], name) not in cache]
    if not todo:
        return
    translator = load(spec, args)
    try:
        with cache_path.open("a", encoding="utf-8") as fh:
            for i in range(0, len(todo), translator.batch_size):
                chunk = todo[i:i + translator.batch_size]
                for case, text in zip(chunk, translator([c["input"]["text"] for c in chunk]),
                                      strict=True):
                    cache[(case["id"], name)] = text
                    fh.write(json.dumps({"id": case["id"], "model": name, "text": text},
                                        ensure_ascii=False) + "\n")
                fh.flush()
                print(f"  {name}: {i + len(chunk)}/{len(todo)}", flush=True)
    finally:
        translator.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--schemas", type=Path, required=True)
    ap.add_argument("--model", required=True, help="ollama:<tag> or a local model directory")
    ap.add_argument("--fallback", help="ollama:<tag> or a local model directory")
    ap.add_argument("--ollama", help="Ollama base URL, for ollama: models")
    ap.add_argument("--batch-size", type=int, default=8, help="local models only")
    ap.add_argument("--review", type=int, default=60, help="cases in the review sample")
    args = ap.parse_args()

    en = json.loads((args.schemas / "moderation-text-en.json").read_text())
    cases = en["test_cases"]

    cache_path = args.schemas / "zh-tw-cache.jsonl"
    cache: dict[tuple[str, str], str] = {}
    if cache_path.exists():
        for line in cache_path.read_text(encoding="utf-8").splitlines():
            row = json.loads(line)
            cache[(row["id"], row["model"])] = row["text"]

    primary = model_name(args.model)
    translate(args.model, args, cases, cache, cache_path)
    retry = [c for c in cases if is_refusal(cache[(c["id"], primary)], c["input"]["text"])]
    print(f"{primary}: {len(retry)} refused, cut off or not Chinese")
    if retry and args.fallback:
        translate(args.fallback, args, retry, cache, cache_path)

    results, by = [], []
    for case in cases:
        model = primary
        if is_refusal(cache[(case["id"], primary)], case["input"]["text"]) and args.fallback:
            model = model_name(args.fallback)
        results.append(clean(cache[(case["id"], model)]))
        by.append(model)

    zh_cases, review, dropped = [], [], []
    for case, text, model in zip(cases, results, by, strict=True):
        if is_refusal(text, case["input"]["text"]):
            dropped.append(case["id"])
            continue
        zh_cases.append({"id": f"{case['id']}-zh", "input": {"text": text}, "labels": case["labels"]})
        review.append((case["id"], case["labels"], model, case["input"]["text"], text))

    zh = {**en, "name": "moderation-text-zh-tw", "test_cases": zh_cases}
    (args.schemas / "moderation-text-zh-tw.json").write_text(json.dumps(zh, ensure_ascii=False))
    (args.schemas / "zh-tw-translation.json").write_text(json.dumps(
        {"model": primary, "fallback": model_name(args.fallback) if args.fallback else None,
         "by_case": {r[0]: r[2] for r in review}, "dropped": dropped}, ensure_ascii=False, indent=1))

    random.seed(6)
    with (args.schemas / "zh-tw-review.csv").open("w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "labels", "model", "english", "zh-TW", "ok (y/n)", "note"])
        for row in random.sample(review, min(args.review, len(review))):
            w.writerow([row[0], ",".join(row[1]) or "safe", row[2], row[3], row[4], "", ""])
    print(f"zh-TW: {len(zh_cases)} cases, {len(dropped)} left out: {dropped}")


if __name__ == "__main__":
    main()
