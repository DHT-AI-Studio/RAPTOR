"""P21: a .txt upload -- and the inline text the gateway's temporal/ingest sends down the same path -- is chunked
without losing what Module 25 reads sentences and headings by: the sentence punctuation and the blank lines."""
import importlib.util
import os
import re

# Loaded from its file, not via sys.path: the analysis service has its own config.py (see test_source_dates.py).
_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                     "worker", "services", "document_analysis_service", "processors", "txt_processor.py")
_spec = importlib.util.spec_from_file_location("m12_txt_processor", _PATH)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def _processor(max_tokens=400):
    """TxtProcessor without loading bge-m3: a token is a word or a CJK character."""
    p = _mod.TxtProcessor.__new__(_mod.TxtProcessor)
    p.max_chunk_tokens = max_tokens
    p.count_tokens = lambda text: len(re.findall(r"[A-Za-z0-9]+|[^\sA-Za-z0-9]", text))
    return p


EN = ("Android Timeline\n\n"
      "Android Inc. was founded in Palo Alto in October 2003 by Andy Rubin and others.\n\n"
      "Google acquired Android Inc. in July 2005.")
ZH = "科技公司大事記\n\n蘋果公司於2007年1月發表iPhone。\n\n三星電子於2010年6月發表Galaxy S。台積電於2020年5月宣布在亞利桑那設廠！"


def test_a_heading_stays_on_its_own_line():
    assert _processor().smart_chunk_text(EN) == [EN]


def test_chinese_sentence_punctuation_is_kept():
    [chunk] = _processor().smart_chunk_text(ZH)
    assert chunk == ZH
    assert "。" in chunk and "！" in chunk


def test_a_line_break_inside_a_paragraph_is_kept():
    text = "The first line of a wrapped paragraph\ncontinues here. A second sentence follows."
    assert _processor().smart_chunk_text(text) == [text]


def test_chunks_still_respect_the_token_limit_and_split_between_paragraphs():
    p = _processor(max_tokens=20)
    chunks = p.smart_chunk_text(EN)
    assert chunks == [EN.rsplit("\n\n", 1)[0], "Google acquired Android Inc. in July 2005."]
    assert all(p.count_tokens(c) <= 20 for c in chunks)


def test_an_abbreviation_is_not_the_end_of_a_sentence():
    p = _processor(max_tokens=20)
    assert all(not c.endswith("Inc.") for c in p.smart_chunk_text(EN))


def test_a_sentence_longer_than_the_limit_is_still_split():
    p = _processor(max_tokens=5)
    chunks = p.smart_chunk_text("one two three four five six seven eight nine ten.")
    assert len(chunks) > 1 and all(p.count_tokens(c) <= 5 for c in chunks)
