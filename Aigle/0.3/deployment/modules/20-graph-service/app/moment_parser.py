# kafka/services/video_graph_service/moment_parser.py
#
# Parse Fung's combined contextual / OCR / ASR / LVLM text field.
#
# Old format (Multimodal Contextual Process only):
#   "ocr:{frame1:{...}, frame2:{...}} / asr:{...} / lvlm:{...}"
#
# New format (post Text Contextual RAG):
#   "contextual:{這段影片展示了...} / ocr:{...} / asr:{...} / lvlm:{...}"
#
# Both formats are accepted. Missing sections return empty string.

from typing import Tuple


def parse_moment_text(text: str) -> Tuple[str, str, str, str]:
    """
    Parse the combined text field. Returns
    (contextual_text, ocr_text, asr_text, lvlm_description).

    Backward-compatible: callers expecting the 3-tuple should switch to
    4-tuple unpacking. If the input has no `contextual:` prefix, the
    contextual_text element is an empty string and the other three
    fields behave identically to the previous parser.
    """
    contextual_text   = ""
    ocr_text          = ""
    asr_text          = ""
    lvlm_description  = ""

    try:
        # contextual: ... / ocr:    (explicit contextual section)
        if text.startswith("contextual:{"):
            after_ctx = text[len("contextual:{"):]
            for delim in ("} / ocr:", "} / asr:", "} / lvlm:"):
                if delim in after_ctx:
                    contextual_text = after_ctx.split(delim, 1)[0].strip()
                    text = delim.lstrip("} ") + after_ctx.split(delim, 1)[1]
                    break
            else:
                contextual_text = after_ctx.rstrip("}").strip()
                text = ""

        else:
            # Generic free-text prefix (e.g. "[上下文補充] ...", "[context] ...", or any prefix)
            # Everything before the first structured marker is the contextual text.
            for marker in ("ocr:{", "asr:{", "lvlm:{"):
                idx = text.find(marker)
                if idx > 0:
                    contextual_text = text[:idx].strip()
                    text = text[idx:]
                    break

        # asr:{ ... } / lvlm:
        if "/ asr:{" in text or text.startswith("asr:{"):
            anchor = "/ asr:{" if "/ asr:{" in text else "asr:{"
            after_asr = text.split(anchor, 1)[1]
            if "} / lvlm:{" in after_asr:
                asr_text = after_asr.split("} / lvlm:{", 1)[0].strip()
            else:
                asr_text = after_asr.rstrip("}").strip()

        # lvlm:{ ... }
        if "/ lvlm:{" in text or text.startswith("lvlm:{"):
            anchor = "/ lvlm:{" if "/ lvlm:{" in text else "lvlm:{"
            lvlm_part = text.split(anchor, 1)[1]
            lvlm_description = lvlm_part.rstrip("}").strip()

        # ocr:{ ... } / asr:    (only when ocr is the first section)
        if text.startswith("ocr:{"):
            ocr_raw = text.split("/ asr:", 1)[0]
            ocr_text = ocr_raw[len("ocr:{"):].rstrip("} ").strip()

    except Exception:
        pass

    return contextual_text, ocr_text, asr_text, lvlm_description


def extract_search_text(text: str) -> str:
    """
    Combined search text for hybrid entity-name keyword matching.
    Includes contextual_text (if present) on top of asr + lvlm so that
    LLM-synthesized entity mentions also contribute to recall.
    """
    contextual, _ocr, asr, lvlm = parse_moment_text(text)
    return " ".join(part for part in (contextual, asr, lvlm) if part).strip()


def extract_moment_fields(payload: dict) -> dict:
    """
    Read raw text from a Qdrant payload that may have either:
      - 'text'        — Multimodal Contextual Process output (raw)
      - 'contextual'  — Text Contextual RAG output (lvlm part is contextually enriched)
      - both          — text=raw, contextual=enriched (preserve both)

    Returns dict with keys: asr_text, lvlm_description, contextual_text,
    raw_text (for hybrid keyword matching).

    Mapping rules:
      - asr_text         := asr from text  OR asr from contextual (raw preferred,
                            both should be identical anyway since speech doesn't
                            change between stages)
      - lvlm_description := lvlm from text only (raw visual description)
      - contextual_text  := lvlm from contextual only (LLM-synthesized narrative)
      - raw_text         := text or contextual (for hybrid entity-name matching)
    """
    text_raw       = payload.get("text", "") or ""
    contextual_raw = payload.get("contextual", "") or ""

    ctx_t, asr_t, lvlm_t = "", "", ""
    if text_raw:
        ctx_t, _ocr, asr_t, lvlm_t = parse_moment_text(text_raw)

    ctx_c, asr_c, lvlm_c = "", "", ""
    if contextual_raw:
        ctx_c, _ocr, asr_c, lvlm_c = parse_moment_text(contextual_raw)

    ocr_t = ""
    if text_raw:
        _, ocr_t, _, _ = parse_moment_text(text_raw)

    return {
        "asr_text":         asr_t or asr_c,
        "lvlm_description": lvlm_t,
        "contextual_text":  ctx_t or ctx_c or lvlm_c,
        "ocr_text":         ocr_t,
        "raw_text":         text_raw or contextual_raw,
    }
