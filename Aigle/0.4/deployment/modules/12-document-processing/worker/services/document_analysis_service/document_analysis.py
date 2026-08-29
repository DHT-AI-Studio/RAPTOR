# services/document_analysis_service/document_analysis.py

import os
import re
import json
import logging
import concurrent.futures
import requests
from pathlib import Path
from typing import Dict, Any, List
from datetime import datetime

from processors import (
    OfficeDocumentProcessor, 
    VLMAnnotator,
    CSVXLSXProcessor,
    HTMLProcessor,
    TxtProcessor,
    PDFOCRProcessor
)

from config import (
    # VLM_API_URL,
    MAX_CHUNK_TOKENS,
    SUPPORTED_FILE_TYPES,
    ANALYSIS_RESULTS_DIR,
    CONTEXTUAL_DOC_CHARS,
    CONTEXTUAL_BATCH_SIZE,
    ASYNC_PROCESSING_CONFIG,
)

logger = logging.getLogger(__name__)

class DocumentAnalysisClient:
    def __init__(
        self,
        inference_url: str = "http://raptor-ai-lifecycle-api:8010",
        inference_model: str = "qwen3.5:9b",
        inference_think: bool = False,
    ):
        self.inference_url = inference_url
        self.inference_model = inference_model
        self.inference_think = inference_think
        logger.info("Initializing shared VLM annotator...")
        self.vlm_annotator = VLMAnnotator()
        logger.info("VLM annotator initialized successfully")
        self.office_processor = OfficeDocumentProcessor(
            vlm_annotator=self.vlm_annotator,  
            max_chunk_tokens=MAX_CHUNK_TOKENS
        )
        
        self.csv_xlsx_processor = CSVXLSXProcessor()
        
        self.html_processor = HTMLProcessor(max_chunk_tokens=MAX_CHUNK_TOKENS)
        self.txt_processor = TxtProcessor(max_chunk_tokens=MAX_CHUNK_TOKENS)
        
        self.pdf_ocr_processor = PDFOCRProcessor(
            vlm_annotator=self.vlm_annotator,
            max_chunk_tokens=MAX_CHUNK_TOKENS
        )
        
        logger.info("Document analysis client initialized")
    
    def get_file_type(self, filename: str) -> str:
        """根據檔案名稱判斷檔案類型"""
        file_extension = filename.lower().split('.')[-1]
        
        for file_type, extensions in SUPPORTED_FILE_TYPES.items():
            if file_extension in extensions:
                return file_type
        
        raise ValueError(f"Unsupported file type: {file_extension}")
    
    def get_processor(self, file_type: str, processing_mode: str = "default"):
        """根據檔案類型和處理模式獲取對應的處理器"""
        if file_type == "pdf":
            # PDF 有兩種處理模式
            if processing_mode == "ocr":
                return self.pdf_ocr_processor
            else:
                return self.office_processor  # 預設使用 office processor
        elif file_type == "office":
            return self.office_processor
        elif file_type == "spreadsheet":
            return self.csv_xlsx_processor
        elif file_type == "web":
            return self.html_processor
        elif file_type == "text":
            return self.txt_processor
        else:
            raise ValueError(f"No processor available for file type: {file_type}")
    def analyze_document_sync(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        同步版本的文件分析方法（用於線程池執行）
        
        Args:
            request_payload: 包含分析請求的 payload
            
        Returns:
            分析結果
        """
        try:
            # 提取請求參數
            request_id = request_payload["request_id"]
            parameters = request_payload.get("parameters", {})
            
            temp_file_path = parameters.get("temp_file_path")
            
            if not temp_file_path:
                return {
                    "request_id": request_id,
                    "action": "document_analysis",
                    "status": "error",
                    "error": {
                        "code": "MISSING_FILE_PATH",
                        "message": "Missing temp_file_path in parameters",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            file_path = Path(temp_file_path)
            
            if not file_path.exists():
                return {
                    "request_id": request_id,
                    "action": "document_analysis",
                    "status": "error",
                    "error": {
                        "code": "FILE_NOT_FOUND",
                        "message": f"Temporary file not found: {temp_file_path}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # 獲取檔案資訊
            filename = parameters.get("primary_filename", file_path.name)
            processing_mode = "default"
            metadata = request_payload.get("metadata", {})
            if metadata:
                original_metadata = metadata.get("original_metadata", {})
                if isinstance(original_metadata, dict):
                    nested_original_metadata = original_metadata.get("original_metadata", {})
                    if isinstance(nested_original_metadata, dict):
                        processing_mode = nested_original_metadata.get("processing_mode", "default")
            asset_path = parameters.get("asset_path")
            version_id = parameters.get("version_id")
            status = parameters.get("status")
            branch_id = parameters.get("branch_id")
            logger.info(f"[SYNC] Processing document: {filename} (mode: {processing_mode}) from: {temp_file_path}")
            logger.debug(f"[SYNC] Asset path: {asset_path}, Version ID: {version_id}, Status: {status}")
            
            # 判斷檔案類型
            file_type = self.get_file_type(filename)
            
            # 獲取對應的處理器
            processor = self.get_processor(file_type, processing_mode)

 
            
            # 處理文件
            start_time = datetime.now()
            analysis_result = processor.process_document(file_path)
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 檢查處理結果
            if "error" in analysis_result:
                return {
                    "request_id": request_id,
                    "action": "document_analysis",
                    "status": "error",
                    "error": {
                        "code": "PROCESSING_FAILED",
                        "message": f"Document processing failed: {analysis_result['error']}",
                        "timestamp": datetime.now().isoformat()
                    }
                }
            
            # 提取 chunks
            chunks = analysis_result.get("chunk", [])
            full_text = analysis_result.get("full_text", "")

            if len(chunks) > 1:
                self._apply_contextual_chunking(chunks, full_text, filename)

            if not chunks:
                logger.warning(f"No chunks generated for document: {filename}")
                chunks = []

            for chunk in chunks:
                if "payload" in chunk:
                    # 更新 filename
                    chunk["payload"]["filename"] = filename
                    
                    # 更新 document_id（移除副檔名再用 primary_filename）
                    base_name = Path(filename).stem
                    chunk_index = chunk["payload"].get("chunk_index", 0)
                    chunk["payload"]["document_id"] = f"{base_name}_chunk_{chunk_index}"
                    if asset_path is not None:
                        chunk["payload"]["asset_path"] = asset_path
                    if version_id is not None:
                        chunk["payload"]["version_id"] = version_id
                    if status is not None:
                        chunk["payload"]["status"] = status
                    if branch_id is not None:
                        chunk["payload"]["branch_id"] = branch_id
            
            # 生成分析結果檔案路徑
            analysis_result_path = os.path.join(
                ANALYSIS_RESULTS_DIR,
                f"analysis_result_{request_id}.json"
            )
            with open(analysis_result_path, 'w', encoding='utf-8') as f:
                json.dump(chunks, f, ensure_ascii=False, indent=2)
            
            logger.info(f"[SYNC] Analysis result saved to: {analysis_result_path}")

            # 計算檔案大小
            file_size_bytes = os.path.getsize(analysis_result_path)
            
            # 返回成功結果
            return {
                "request_id": request_id,
                "action": "document_analysis",
                "status": "success",
                "parameters": {
                    "analysis_result_path": analysis_result_path,
                    "filename": filename,
                    "file_type": file_type,
                    "processing_mode": processing_mode,
                    "total_chunks": len(chunks),
                    "file_size_bytes": file_size_bytes,
                    "processing_time_seconds": processing_time,
                    "asset_path": asset_path, 
                    "version_id": version_id
                },
                "metadata": {
                    "processed_at": datetime.now().isoformat(),
                    "processing_method": "async_thread_pool"
                }
            }
            
        except Exception as e:
            logger.error(f"[SYNC] Error analyzing document: {e}")
            
            # 清理可能產生的檔案
            try:
                request_id = request_payload.get('request_id', 'unknown')
                filename = request_payload.get('parameters', {}).get('primary_filename', 'unknown')
                analysis_result_path = os.path.join(
                    ANALYSIS_RESULTS_DIR,
                    f"analysis_result_{request_id}.json"
                )
                if os.path.exists(analysis_result_path):
                    os.remove(analysis_result_path)
                    logger.info(f"Cleaned up failed analysis result file: {analysis_result_path}")
            except Exception as cleanup_error:
                logger.warning(f"Failed to cleanup analysis result file: {cleanup_error}")
            
            return {
                "request_id": request_payload.get("request_id"),
                "action": "document_analysis",
                "status": "error",
                "error": {
                    "code": "ANALYSIS_ERROR",
                    "message": str(e),
                    "timestamp": datetime.now().isoformat()
                },
                "metadata": {
                    "error_type": type(e).__name__,
                    "processing_method": "async_thread_pool"
                }
            }  
    async def analyze_document(self, request_payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        異步版本的文件分析方法（向後兼容）
        現在內部使用同步版本
        """
        import asyncio
        import concurrent.futures
        
        loop = asyncio.get_event_loop()
        
        # 使用線程池執行同步版本
        with concurrent.futures.ThreadPoolExecutor(
            max_workers=ASYNC_PROCESSING_CONFIG["max_workers"]
        ) as executor:
            try:
                # 設置超時
                result = await asyncio.wait_for(
                    loop.run_in_executor(executor, self.analyze_document_sync, request_payload),
                    timeout=ASYNC_PROCESSING_CONFIG["task_timeout"]
                )
                return result
            except asyncio.TimeoutError:
                logger.error(f"Document analysis timeout for request: {request_payload.get('request_id')}")
                return {
                    "request_id": request_payload.get("request_id"),
                    "action": "document_analysis",
                    "status": "error",
                    "error": {
                        "code": "PROCESSING_TIMEOUT",
                        "message": f"Processing timeout after {ASYNC_PROCESSING_CONFIG['task_timeout']} seconds",
                        "timestamp": datetime.now().isoformat()
                    }
                }
    def _generate_batch_context(
        self,
        doc_context: str,
        batch_texts: List[str],
        batch_indices: List[int],
        filename: str,
    ) -> List[str]:
        """Call LLM once for a batch of chunks; return context strings aligned with batch_indices."""
        chunks_xml = "\n".join(
            f'<chunk index="{idx}">{text}</chunk>'
            for idx, text in zip(batch_indices, batch_texts)
        )
        prompt = (
            f"<document>\n{doc_context}\n</document>\n\n"
            f"以下是文件「{filename}」中數個段落：\n"
            f"{chunks_xml}\n\n"
            f"請根據上方的文件內容，為每個 <chunk> 用1到2句繁體中文說明該段落在文件中的位置與脈絡"
            f"（屬於哪個章節、前後討論了什麼），不要摘要段落內容。\n\n"
            f"請嚴格只輸出 JSON 陣列，不要有任何其他文字：\n"
            f'[{{"index": <int>, "context": "<str>"}}, ...]'
        )
        try:
            resp = requests.post(
                f"{self.inference_url}/inference/infer",
                json={
                    "task": "text-generation-ollama",
                    "engine": "ollama",
                    "model_name": self.inference_model,
                    "data": {"inputs": prompt},
                    "options": {"max_length": 600, "temperature": 0.1, "max_input_length": 32000, "num_ctx": 16384, "think": self.inference_think},
                },
                headers={"Content-Type": "application/json"},
                timeout=ASYNC_PROCESSING_CONFIG["contextual_timeout"],
            )
            resp.raise_for_status()
            raw = resp.json().get("result", {}).get("response", "").strip()
            return self._parse_batch_response(raw, batch_indices)
        except Exception as e:
            logger.warning(f"Batch context generation failed (indices {batch_indices[0]}–{batch_indices[-1]}): {e}")
            return [""] * len(batch_indices)

    def _parse_batch_response(self, raw: str, batch_indices: List[int]) -> List[str]:
        """Parse JSON array from LLM batch response; falls back to empty strings on failure."""
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                index_map = {item["index"]: item.get("context", "") for item in parsed}
                return [index_map.get(idx, "") for idx in batch_indices]
        except (json.JSONDecodeError, KeyError, TypeError):
            pass

        match = re.search(r"\[.*?\]", raw, re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group())
                index_map = {item["index"]: item.get("context", "") for item in parsed}
                return [index_map.get(idx, "") for idx in batch_indices]
            except Exception:
                pass

        logger.warning(f"JSON parse failed for batch indices {batch_indices}, using empty prefixes")
        return [""] * len(batch_indices)

    def _apply_contextual_chunking(self, chunks: List[Dict], full_text: str = "", filename: str = "") -> None:
        """
        Anthropic-style contextual retrieval — in-place, batch mode.

        Groups chunks into batches of CONTEXTUAL_BATCH_SIZE and sends each batch
        as a single LLM call requesting a JSON array of context sentences.
        Batches are processed concurrently via a thread pool.
        Replaces payload["text"] with:

            [FILENAME] ... [/FILENAME]
            [CONTEXT] ... [/CONTEXT]
            [CHUNK] ... [/CHUNK]
        """
        if not filename:
            filename = chunks[0]["payload"].get("filename", "Document") if chunks else "Document"

        full_doc = full_text or "\n\n".join(c["payload"].get("text", "") for c in chunks)
        use_full_doc = len(full_doc) <= CONTEXTUAL_DOC_CHARS

        raw_texts = [chunk["payload"].get("text", "") for chunk in chunks]
        total_chunks = len(chunks)

        def get_doc_context(indices: List[int]) -> str:
            if use_full_doc:
                return full_doc
            # Expand outward from the batch boundaries until CONTEXTUAL_DOC_CHARS
            parts = [raw_texts[i] for i in indices]
            total = sum(len(raw_texts[i]) for i in indices)
            left, right = indices[0] - 1, indices[-1] + 1
            while left >= 0 or right < total_chunks:
                if left >= 0:
                    candidate = raw_texts[left]
                    if total + len(candidate) > CONTEXTUAL_DOC_CHARS:
                        break
                    parts.insert(0, candidate)
                    total += len(candidate)
                    left -= 1
                if right < total_chunks:
                    candidate = raw_texts[right]
                    if total + len(candidate) > CONTEXTUAL_DOC_CHARS:
                        break
                    parts.append(candidate)
                    total += len(candidate)
                    right += 1
            return "\n\n".join(parts)

        batches = [
            list(range(i, min(i + CONTEXTUAL_BATCH_SIZE, total_chunks)))
            for i in range(0, total_chunks, CONTEXTUAL_BATCH_SIZE)
        ]
        total_batches = len(batches)
        completed_batches = 0

        def _process_batch(batch_indices: List[int]):
            nonlocal completed_batches
            doc_ctx = get_doc_context(batch_indices)
            texts = [raw_texts[i] for i in batch_indices]
            contexts = self._generate_batch_context(doc_ctx, texts, batch_indices, filename)
            completed_batches += 1
            logger.info(
                f"Contextual prefix: batch {completed_batches}/{total_batches} done "
                f"(chunks {batch_indices[0]+1}–{batch_indices[-1]+1}/{total_chunks})"
            )
            return batch_indices, contexts

        logger.info(
            f"Generating contextual prefixes for {total_chunks} chunks "
            f"in {total_batches} batches of {CONTEXTUAL_BATCH_SIZE} "
            f"(max_workers={ASYNC_PROCESSING_CONFIG['max_workers']})"
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=ASYNC_PROCESSING_CONFIG["max_workers"]) as pool:
            batch_results = list(pool.map(_process_batch, batches))
        logger.info(f"Contextual prefixes complete: all {total_batches} batches done")

        context_sentences = [""] * total_chunks
        for batch_indices, contexts in batch_results:
            for idx, ctx in zip(batch_indices, contexts):
                context_sentences[idx] = ctx

        for i, (chunk, raw_text, context_sentence) in enumerate(zip(chunks, raw_texts, context_sentences)):
            if context_sentence:
                chunk["payload"]["text"] = (
                    f"[FILENAME] {filename} [/FILENAME]\n"
                    f"[CONTEXT] {context_sentence} [/CONTEXT]\n"
                    f"[CHUNK] {raw_text} [/CHUNK]"
                )
            else:
                chunk["payload"]["text"] = f"[FILENAME] {filename} [/FILENAME]\n[CHUNK] {raw_text} [/CHUNK]"
            logger.info(f"Contextualised chunk {i + 1}/{total_chunks}: {filename}")

    def _get_page_range(self, chunks: List[Dict]) -> Dict[str, Any]:
        """獲取頁面範圍資訊"""
        all_pages = []
        for chunk in chunks:
            if "payload" in chunk:
                page_numbers = chunk["payload"].get("page_numbers", [])
                all_pages.extend(page_numbers)
        
        if all_pages:
            return {
                "min_page": min(all_pages),
                "max_page": max(all_pages),
                "total_pages": len(set(all_pages))
            }
        else:
            return {
                "min_page": None,
                "max_page": None,
                "total_pages": 0
            }
    
    def _get_unique_element_types(self, chunks: List[Dict]) -> List[str]:
        """獲取所有唯一的元素類型"""
        all_element_types = set()
        for chunk in chunks:
            if "payload" in chunk:
                element_types = chunk["payload"].get("element_types", [])
                all_element_types.update(element_types)
        
        return sorted(list(all_element_types))
