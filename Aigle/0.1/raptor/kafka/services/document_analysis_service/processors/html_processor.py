import json
import uuid
from pathlib import Path
from typing import List, Dict
from datetime import datetime
import re

from transformers import AutoTokenizer
import logging
logger = logging.getLogger(__name__)

class HTMLProcessor:
    def __init__(self, max_chunk_tokens: int = 400):
        self.max_chunk_tokens = max_chunk_tokens
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    
    def count_tokens(self, text: str) -> int:
        """計算文本的 token 數量"""
        if not text:
            return 0
        
        try:
            max_chars = 1500
            if len(text) > max_chars:
                text = text[:max_chars]
            
            tokens = self.tokenizer.encode(
                text, 
                truncation=True, 
                max_length=512,
                add_special_tokens=False
            )
            return len(tokens)
        except Exception:
            return len(text) // 4
    
    def split_html_text(self, html_content: str) -> List[str]:
        """分割 HTML 原始碼，保留所有標籤和內容"""
        if not html_content:
            return []
        
        html_content = re.sub(r'\n\s*\n', '\n', html_content)
        lines = [line.strip() for line in html_content.split('\n') if line.strip()]
        html_content = '\n'.join(lines)
        
        chunks = []
        
        # 嘗試按 HTML 標籤自然分割
        # 1. 首先嘗試按完整的 HTML 元素分割（開始標籤到結束標籤）
        tag_pattern = r'(<[^>]+>)'
        parts = re.split(tag_pattern, html_content)
        
        # 重新組合，保持標籤和內容的配對
        reconstructed_parts = []
        current_part = ""
        
        for part in parts:
            if not part.strip():
                continue
                
            current_part += part
            
            # 如果遇到結束標籤或自閉合標籤，或者內容過長，就分割
            if (re.match(r'<\/[^>]+>$', part.strip()) or  # 結束標籤
                re.match(r'<[^>]+\/>$', part.strip()) or   # 自閉合標籤
                self.count_tokens(current_part) > self.max_chunk_tokens * 0.8):
                
                if current_part.strip():
                    reconstructed_parts.append(current_part.strip())
                current_part = ""
        
        # 處理剩餘部分
        if current_part.strip():
            reconstructed_parts.append(current_part.strip())
        
        # 如果分割效果不好，回退到簡單的行分割
        if len(reconstructed_parts) <= 1:
            reconstructed_parts = html_content.split('\n')
            reconstructed_parts = [part.strip() for part in reconstructed_parts if part.strip()]
        
        # 將部分組合成適當大小的塊
        current_chunk = []
        current_tokens = 0
        
        for part in reconstructed_parts:
            if not part:
                continue
                
            part_tokens = self.count_tokens(part)
            
            # 如果單個部分就超過限制
            if part_tokens > self.max_chunk_tokens:
                # 先保存當前塊
                if current_chunk:
                    chunks.append('\n'.join(current_chunk))
                    current_chunk = []
                    current_tokens = 0
                
                # 如果部分太長，按字符強制分割
                if part_tokens > self.max_chunk_tokens * 1.5:
                    max_chars = self.max_chunk_tokens * 3
                    for i in range(0, len(part), max_chars):
                        chunk_part = part[i:i + max_chars]
                        if chunk_part.strip():
                            chunks.append(chunk_part.strip())
                else:
                    chunks.append(part)
            
            elif current_tokens + part_tokens > self.max_chunk_tokens and current_chunk:
                # 完成當前塊
                chunks.append('\n'.join(current_chunk))
                current_chunk = [part]
                current_tokens = part_tokens
            else:
                current_chunk.append(part)
                current_tokens += part_tokens
        
        # 處理最後一個塊
        if current_chunk:
            chunks.append('\n'.join(current_chunk))
        
        return [chunk for chunk in chunks if chunk.strip()]
    
    def process_document(self, file_path: Path) -> Dict[str, List[Dict]]:
        """處理 HTML 文檔，保留完整的 HTML 原始碼"""
        try:

            encodings = ['utf-8', 'utf-8-sig', 'gbk', 'big5', 'cp1252']
            html_content = None
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding) as f:
                        html_content = f.read()
                    break
                except UnicodeDecodeError:
                    continue
            
            if html_content is None:
                logger.error("無法讀取 HTML 文件")
                return {"error": "無法讀取文件"}
            
            # 檢查文件是否為空
            if not html_content.strip():
                logger.warning(f"HTML 文件為空: {file_path.name}")
                return {"error": "HTML 文件為空"}
            
            # 直接分割 HTML 原始碼
            text_chunks = self.split_html_text(html_content)
            
            # 檢查是否成功分割出 chunks
            if not text_chunks:
                logger.warning(f"無法從 HTML 文件中提取任何內容: {file_path.name}")
                return {"error": "無法從 HTML 文件中提取任何內容"}
            
            # 生成最終的 chunk 結構
            chunks = []
            document_id = file_path.stem
            filename = file_path.name
            source = filename.split('.')[-1].lower()
            upload_time = datetime.now().isoformat()
            
            for chunk_index, chunk_text in enumerate(text_chunks):
                chunk = {
                    "id": str(uuid.uuid4()),
                    "payload": {
                        "document_id": f"{document_id}_chunk_{chunk_index}",
                        "type": "documents",
                        "text": chunk_text,
                        "filename": filename,
                        "source": source,
                        "chunk_index": chunk_index,
                        "page_numbers": [],
                        "element_types": ["text"],
                        "char_count": len(chunk_text),
                        "upload_time": upload_time,
                        "embedding_type": "text"
                    }
                }
                chunks.append(chunk)
            
            return {"chunk": chunks}
        
        except Exception as e:
            logger.error(f"處理 HTML 時發生錯誤: {str(e)}")
            return {"error": str(e)}