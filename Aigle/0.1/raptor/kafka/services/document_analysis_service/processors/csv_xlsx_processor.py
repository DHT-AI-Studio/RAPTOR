import pandas as pd
import uuid
import json
import logging
from datetime import datetime
from transformers import AutoTokenizer
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
import csv

logger = logging.getLogger(__name__)

class CSVXLSXProcessor:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.max_tokens = 512
    
    def count_tokens(self, text: str) -> int:
        """計算文本的token數量"""
        return len(self.tokenizer.encode(text))
    
    def process_document(self, file_path: Path, upload_time: Optional[str] = None) -> Union[Dict[str, List[Dict]], Dict[str, str]]:
        """
        處理 CSV/XLSX 文件，返回標準格式的 chunks
        
        Returns:
            成功: {"chunk": [...]}
            失敗: {"error": "錯誤訊息"}
        """
        try:
            file_path = Path(file_path)
            
            if upload_time is None:
                upload_time = datetime.now().isoformat()
            
            if file_path.suffix.lower() == '.csv':
                result = self._process_csv(file_path, upload_time)
            elif file_path.suffix.lower() in ['.xlsx', '.xls']:
                result = self._process_xlsx(file_path, upload_time)
            else:
                return {"error": f"不支援的文件格式: {file_path.suffix}"}
            
            return result
            
        except Exception as e:
            logger.error(f"處理文件時發生錯誤: {str(e)}")
            return {"error": str(e)}
    
    def _validate_csv_format(self, file_path: Path, max_check_rows: int = 1000) -> tuple[bool, str]:
        """檢查 CSV 格式"""
        try:
            encodings = ['utf-8', 'gbk', 'big5', 'cp1252']
            
            for encoding in encodings:
                try:
                    with open(file_path, 'r', encoding=encoding, newline='') as f:
                        csv_reader = csv.reader(f, quotechar='"', delimiter=',')
                        
                        try:
                            header_row = next(csv_reader)
                            header_cols = len(header_row)
                        except StopIteration:
                            return False, "檔案為空"
                        
                        # 檢查數據行，但有上限
                        for line_num, row in enumerate(csv_reader, start=2):
                            if line_num > max_check_rows:
                                break  # 超過限制就停止檢查
                                
                            if len(row) != header_cols:
                                return False, f"第{line_num}行有{len(row)}個欄位，但標題有{header_cols}個欄位，非正確的csv格式無法解析。"
                    
                    return True, "格式正常"
                    
                except UnicodeDecodeError:
                    continue
            
            return False, "無法用任何編碼讀取檔案"
            
        except Exception as e:
            return False, f"檢查失敗: {str(e)}"
    
    def _process_csv(self, file_path: Path, upload_time: str) -> Union[Dict[str, List[Dict]], Dict[str, str]]:
        """
        處理 CSV 文件
        
        Returns:
            成功: {"chunk": [...]}
            失敗: {"error": "錯誤訊息"}
        """
        # 格式檢查
        is_valid, error_msg = self._validate_csv_format(file_path)
        
        if not is_valid:
            logger.error(f"CSV格式驗證失敗: {error_msg}")
            return {"error": f"CSV格式錯誤: {error_msg}"}
        
        try:
            encodings = ['utf-8', 'gbk', 'big5', 'cp1252']
            df = None
            
            for encoding in encodings:
                try:
                    # 使用標準 csv 模組先解析
                    rows = []
                    with open(file_path, 'r', encoding=encoding, newline='') as f:
                        csv_reader = csv.reader(f, quotechar='"', delimiter=',')
                        rows = list(csv_reader)
                    
                    # 轉為 DataFrame
                    if rows:
                        headers = rows[0]
                        data = rows[1:]
                        df = pd.DataFrame(data, columns=headers)
                    else:
                        df = pd.DataFrame()
                    
                    break
                except UnicodeDecodeError:
                    continue
            
            # 如果所有編碼都失敗，使用 pandas 的容錯模式
            if df is None:
                df = pd.read_csv(
                    file_path, 
                    encoding='utf-8', 
                    errors='ignore',
                    on_bad_lines='skip',
                    engine='python',
                    quoting=1
                )
            
            # 檢查是否為空
            if df.empty:
                return {"error": "CSV檔案為空或無法讀取任何數據"}
            
            chunks = self.chunk_table(df, sheet_index=None, sheet_name=None)
            
            formatted_chunks = self._format_chunks(
                chunks, 
                filename=file_path.name,
                source="csv",
                upload_time=upload_time
            )
            
            return {"chunk": formatted_chunks}
            
        except Exception as e:
            logger.error(f"處理 CSV 時發生錯誤: {str(e)}")
            return {"error": f"CSV處理錯誤: {str(e)}"}
    
    def _process_xlsx(self, file_path: Path, upload_time: str) -> Union[Dict[str, List[Dict]], Dict[str, str]]:
        """
        處理 XLSX 文件
        
        Returns:
            成功: {"chunk": [...]}
            失敗: {"error": "錯誤訊息"}
        """
        all_chunks = []
        failed_sheets = []
        
        try:
            excel_file = pd.ExcelFile(file_path)
            
            if not excel_file.sheet_names:
                return {"error": "XLSX檔案中沒有任何工作表"}
            
            for sheet_idx, sheet_name in enumerate(excel_file.sheet_names):
                try:
                    df = pd.read_excel(file_path, sheet_name=sheet_name)
                    
                    # 檢查是否為空
                    if df.empty:
                        logger.warning(f"Sheet '{sheet_name}' 為空，跳過處理")
                        continue
                    
                    chunks = self.chunk_table(df, sheet_index=sheet_idx, sheet_name=sheet_name)
                    
                    formatted_chunks = self._format_chunks(
                        chunks,
                        filename=file_path.name,
                        source="xlsx",
                        upload_time=upload_time,
                        sheet_offset=len(all_chunks)
                    )
                    all_chunks.extend(formatted_chunks)
                    
                except Exception as e:
                    # 記錄失敗的 sheet，但繼續處理其他 sheet
                    error_msg = f"Sheet '{sheet_name}' 處理失敗: {str(e)}"
                    logger.warning(error_msg)
                    failed_sheets.append(error_msg)
                    continue
            
            # 如果所有 sheet 都失敗了
            if not all_chunks and failed_sheets:
                return {"error": f"所有工作表處理失敗: {'; '.join(failed_sheets)}"}
            
            # 如果有部分成功，記錄警告但返回成功的 chunks
            if failed_sheets:
                logger.warning(f"部分工作表處理失敗: {'; '.join(failed_sheets)}")
            
            return {"chunk": all_chunks}
        
        except Exception as e:
            logger.error(f"處理 XLSX 時發生錯誤: {str(e)}")
            return {"error": f"XLSX處理錯誤: {str(e)}"}
    
    def chunk_table(self, df: pd.DataFrame, sheet_index: Optional[int], sheet_name: Optional[str]) -> List[Dict[str, Any]]:
        """表格chunking主邏輯"""
        if df.empty:
            return []
        
        df = df.fillna('-')
        headers = df.columns.tolist()
        
        if self._can_fit_entire_table(df, sheet_name):
            return [self._create_single_chunk(df, headers, sheet_name, sheet_index, 0, len(df)-1)]
        
        if self._headers_exceed_limit(headers, sheet_name):
            return self._chunk_by_columns(df, headers, sheet_name, sheet_index)
        
        return self._chunk_by_rows(df, headers, sheet_name, sheet_index)
    
    def _chunk_by_columns(self, df: pd.DataFrame, headers: List[str], sheet_name: Optional[str], sheet_index: Optional[int]) -> List[Dict[str, Any]]:
        """按欄位拆分，每個chunk都保留第一欄"""
        chunks = []
        first_col_name = headers[0]
        
        current_cols = [0]
        current_col_names = [first_col_name]
        
        for i in range(1, len(headers)):
            test_cols = current_cols + [i]
            test_col_names = current_col_names + [headers[i]]
            test_df = df.iloc[:, test_cols]
            
            if self._estimate_chunk_tokens(test_df, test_col_names, sheet_name) <= self.max_tokens:
                current_cols.append(i)
                current_col_names.append(headers[i])
            else:
                if len(current_cols) > 1:
                    chunk_df = df.iloc[:, current_cols]
                    chunks.append(self._create_single_chunk(
                        chunk_df, current_col_names, sheet_name, sheet_index, 
                        0, len(df)-1, col_range=(current_cols[0], current_cols[-1])
                    ))
                
                current_cols = [0, i]
                current_col_names = [first_col_name, headers[i]]
        
        if len(current_cols) > 1:
            chunk_df = df.iloc[:, current_cols]
            chunks.append(self._create_single_chunk(
                chunk_df, current_col_names, sheet_name, sheet_index,
                0, len(df)-1, col_range=(current_cols[0], current_cols[-1])
            ))
        
        return chunks
    
    def _chunk_by_rows(self, df: pd.DataFrame, headers: List[str], sheet_name: Optional[str], sheet_index: Optional[int]) -> List[Dict[str, Any]]:
        """按行拆分，每個chunk都保留完整的欄位名稱"""
        chunks = []
        current_start = 0
        
        for i in range(1, len(df) + 1):
            test_df = df.iloc[current_start:i]
            
            if self._estimate_chunk_tokens(test_df, headers, sheet_name) > self.max_tokens:
                if i - 1 > current_start:
                    chunk_df = df.iloc[current_start:i-1]
                    chunks.append(self._create_single_chunk(
                        chunk_df, headers, sheet_name, sheet_index,
                        current_start, i-2
                    ))
                    current_start = i - 1
                else:
                    chunk_df = df.iloc[current_start:i]
                    chunks.append(self._create_single_chunk(
                        chunk_df, headers, sheet_name, sheet_index,
                        current_start, i-1
                    ))
                    current_start = i
        
        if current_start < len(df):
            chunk_df = df.iloc[current_start:]
            chunks.append(self._create_single_chunk(
                chunk_df, headers, sheet_name, sheet_index,
                current_start, len(df)-1
            ))
        
        return chunks
    
    def _create_single_chunk(self, df: pd.DataFrame, headers: List[str], sheet_name: Optional[str], 
                           sheet_index: Optional[int], start_row: int, end_row: int, 
                           col_range: Optional[tuple] = None) -> Dict[str, Any]:
        """創建單個chunk的數據結構"""
        markdown_text = self._create_markdown_table(df, headers, sheet_name, start_row, end_row, col_range)
        page_numbers = [sheet_index] if sheet_index is not None else []
        
        return {
            "text": markdown_text,
            "page_numbers": page_numbers,
            "start_row": start_row,
            "end_row": end_row,
            "col_range": col_range
        }
    
    def _create_markdown_table(self, df: pd.DataFrame, headers: List[str], sheet_name: Optional[str], 
                             start_row: int, end_row: int, col_range: Optional[tuple] = None) -> str:
        """生成markdown表格文本"""
        markdown_parts = []
        
        if sheet_name:
            title = f"## {sheet_name}"
            
            if start_row > 0 or end_row < len(df) - 1:
                actual_start = start_row + 1
                actual_end = end_row + 1
                title += f" (第{actual_start}-{actual_end}行)"
            
            if col_range and col_range[0] != 0 or (col_range and col_range[1] != len(headers) - 1):
                col_start = col_range[0] + 1
                col_end = col_range[1] + 1
                title += f" (欄位{col_start}-{col_end})"
            
            markdown_parts.append(title)
            markdown_parts.append("")
        
        header_row = "| " + " | ".join(str(h) for h in headers) + " |"
        separator_row = "|" + "|".join(["-" * max(len(str(h)), 3) for h in headers]) + "|"
        
        markdown_parts.append(header_row)
        markdown_parts.append(separator_row)
        
        for _, row in df.iterrows():
            row_values = []
            for val in row.values:
                if pd.isna(val) or val == '' or val is None:
                    row_values.append('-')
                else:
                    row_values.append(str(val))
            
            data_row = "| " + " | ".join(row_values) + " |"
            markdown_parts.append(data_row)
        
        return "\n".join(markdown_parts)
    
    def _estimate_chunk_tokens(self, df: pd.DataFrame, headers: List[str], sheet_name: Optional[str]) -> int:
        """估算chunk的token數量"""
        markdown_text = self._create_markdown_table(df, headers, sheet_name, 0, len(df)-1)
        return self.count_tokens(markdown_text)
    
    def _can_fit_entire_table(self, df: pd.DataFrame, sheet_name: Optional[str]) -> bool:
        """檢查整個表格是否能fit在一個chunk中"""
        headers = df.columns.tolist()
        return self._estimate_chunk_tokens(df, headers, sheet_name) <= self.max_tokens
    
    def _headers_exceed_limit(self, headers: List[str], sheet_name: Optional[str]) -> bool:
        """檢查僅headers是否就超過限制"""
        empty_df = pd.DataFrame(columns=headers)
        test_tokens = self._estimate_chunk_tokens(empty_df, headers, sheet_name)
        return test_tokens > self.max_tokens * 0.6
    
    def _format_chunks(self, chunks: List[Dict[str, Any]], filename: str, source: str, 
                      upload_time: str, sheet_offset: int = 0) -> List[Dict[str, Any]]:
        """將chunk數據格式化為標準輸出格式"""
        formatted_chunks = []
        
        for i, chunk in enumerate(chunks):
            chunk_index = sheet_offset + i
            text = chunk["text"]
            
            formatted_chunk = {
                "id": str(uuid.uuid4()),
                "payload": {
                    "document_id": f"{Path(filename).stem}_chunk_{chunk_index}",
                    "type": "documents",
                    "text": text,
                    "filename": filename,
                    "source": source,
                    "chunk_index": chunk_index,
                    "page_numbers": chunk["page_numbers"],
                    "element_types": ["table"],
                    "char_count": len(text),
                    "upload_time": upload_time,
                    "embedding_type": "text"
                }
            }
            formatted_chunks.append(formatted_chunk)
        
        return formatted_chunks

