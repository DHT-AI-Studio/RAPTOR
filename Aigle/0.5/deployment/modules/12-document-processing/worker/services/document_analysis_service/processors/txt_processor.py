import logging
import re
import uuid
from datetime import datetime
from pathlib import Path

import opencc
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

class TxtProcessor:
    def __init__(self, max_chunk_tokens: int = 512):
        self.max_chunk_tokens = max_chunk_tokens
        # revision pinned to BAAI/bge-m3's main HEAD as of 2026-10-15 (bandit B615)
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3", revision="5617a9f61b028005a4858fdac845db406aefb181")
        self.cc = opencc.OpenCC('s2twp')
    
    def count_tokens(self, text: str) -> int:
        """計算文本的 token 數量"""
        if not text:
            return 0
        
        # 使用 tokenizer 計算 token 數量
        tokens = self.tokenizer.encode(
            text, 
            truncation=True, 
            max_length=512,
            add_special_tokens=False
        )
        return len(tokens)
    
    # 句子的邊界：中文標點之後、或換行（英文句點不切：「Android Inc.」不是句尾）。
    # 標點本身留在句子裡（P21：Module 25 靠它們斷句）
    _SENTENCE_BOUNDARY = re.compile(r'(?<=[。！？；])[ \t]*|[ \t]*\r?\n[ \t]*')

    def split_by_punctuation(self, text: str) -> list[tuple[str, str]]:
        """根據標點符號分割文本：[(分隔, 句子)]。分隔是句子前面原本的空白：段落之間 "\\n\\n"、換行 "\\n"、
        中文標點之後 ""（後面原本有空白就是 " "）；接回去時照原樣，標題才會留在自己那一行"""
        # 先按段落分割（雙換行符）
        paragraphs = re.split(r'\n\s*\n', text)

        sentences = []
        for paragraph in paragraphs:
            if not paragraph.strip():
                continue

            # 在段落內按句子分割，記下每句前面的分隔
            separator = "\n\n"
            pos = 0
            for m in self._SENTENCE_BOUNDARY.finditer(paragraph):
                sentence = paragraph[pos:m.start()].strip()
                if sentence:
                    sentences.append((separator, sentence))
                    separator = "\n" if "\n" in m.group(0) else (" " if m.group(0) else "")
                pos = m.end()
            sentence = paragraph[pos:].strip()
            if sentence:
                sentences.append((separator, sentence))

        return sentences

    def smart_chunk_text(self, text: str) -> list[str]:
        """智能分塊文本，基於 token 數量和標點符號"""
        if not text.strip():
            return []

        # 先按標點符號分割成句子
        sentences = self.split_by_punctuation(text)

        chunks = []
        current_chunk = ""
        current_token_count = 0

        for separator, sentence in sentences:
            sentence_tokens = self.count_tokens(sentence)

            # 如果單個句子就超過最大 token 數，需要進一步分割
            if sentence_tokens > self.max_chunk_tokens:
                # 保存當前 chunk（如果有內容）
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""
                    current_token_count = 0

                # 按字符長度強制分割長句子
                chunk_parts = self.force_split_long_sentence(sentence)
                chunks.extend(chunk_parts)
                continue

            # 檢查加入這個句子是否會超過限制
            separator_tokens = self.count_tokens(separator) if current_chunk else 0
            new_token_count = current_token_count + separator_tokens + sentence_tokens

            if new_token_count > self.max_chunk_tokens and current_chunk:
                # 超過限制，保存當前 chunk 並開始新的
                chunks.append(current_chunk)
                current_chunk = sentence
                current_token_count = sentence_tokens
            else:
                # 可以加入當前 chunk，照原本的分隔接上
                current_chunk = current_chunk + separator + sentence if current_chunk else sentence
                current_token_count = new_token_count

        # 處理最後一個 chunk
        if current_chunk:
            chunks.append(current_chunk)

        return chunks
    
    def force_split_long_sentence(self, sentence: str) -> list[str]:
        """強制分割超長句子"""
        chunks = []
        words = sentence.split()
        
        current_chunk = []
        current_token_count = 0
        
        for word in words:
            word_tokens = self.count_tokens(word)
            separator_tokens = self.count_tokens(" ") if current_chunk else 0
            new_token_count = current_token_count + separator_tokens + word_tokens
            
            if new_token_count > self.max_chunk_tokens and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_token_count = word_tokens
            else:
                current_chunk.append(word)
                current_token_count = new_token_count
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        
        return chunks
    
    def process_document(self, file_path: Path) -> dict[str, list[dict]] | dict[str, str]:
        """
        處理 txt 文件
        
        Returns:
            成功: {"chunk": [...]}
            失敗: {"error": "錯誤訊息"}
        """
        try:
            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5', 'cp1252']
            content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if content is None:
                logger.error("無法讀取 TXT 文件")
                return {"error": "無法讀取文件"}
            
            # 檢查文件是否為空
            if not content.strip():
                logger.warning(f"TXT 文件為空: {file_path.name}")
                return {"error": "TXT 文件為空"}
            
            content = self.cc.convert(content)

            # 分塊處理
            text_chunks = self.smart_chunk_text(content)
            
            # 檢查是否成功分割出 chunks
            if not text_chunks:
                logger.warning(f"無法從 TXT 文件中提取任何內容: {file_path.name}")
                return {"error": "無法從 TXT 文件中提取任何內容"}
            
            # 生成最終的 chunk 結構
            chunks = []
            document_id = file_path.stem
            filename = file_path.name
            upload_time = datetime.now().isoformat()
            
            for chunk_index, chunk_text in enumerate(text_chunks):
                chunk = {
                    "id": str(uuid.uuid4()),
                    "payload": {
                        "document_id": f"{document_id}_chunk_{chunk_index}",
                        "type": "documents",
                        "text": chunk_text,
                        "filename": filename,
                        "source": "txt",
                        "chunk_index": chunk_index,
                        "page_numbers": [],
                        "element_types": ["text"],
                        "char_count": len(chunk_text),
                        "upload_time": upload_time,
                        "embedding_type": "text"
                    }
                }
                chunks.append(chunk)

            return {"chunk": chunks, "full_text": content}

        except Exception as e:  # noqa: BLE001 -- spans file IO, opencc conversion and tokenizer-based chunking; already logged + returns structured error
            logger.error(f"處理 TXT 時發生錯誤: {e!s}")
            return {"error": str(e)}