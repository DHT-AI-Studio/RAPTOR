# services/document_analysis_service/document_summary.py

import json
import requests
from pathlib import Path
from typing import List, Dict, Any
import re
import opencc

class DocumentSummarizer:
    def __init__(self, ollama_url: str = "http://192.168.157.165:8009", model_name: str = "qwenforsummary"):
        self.ollama_url = ollama_url
        self.model_name = model_name
        self.max_model_tokens = 6000
        self.chunk_group_size = 5
        self.max_summary_tokens = 800
        self.safety_buffer = 0.8
        self.converter = opencc.OpenCC('s2t')
    
    def count_tokens_estimate(self, text: str) -> int:
        """估算文本的token數量"""
        chinese_chars = len(re.findall(r'[\u4e00-\u9fff]', text))
        english_words = len(re.findall(r'[a-zA-Z]+', text))
        other_chars = len(text) - chinese_chars - len(re.findall(r'[a-zA-Z\s]', text))
        
        estimated_tokens = int(chinese_chars * 1.5 + english_words * 1.3 + other_chars * 1.2)
        return estimated_tokens
    
    # def call_ollama(self, prompt: str, system_prompt: str = "") -> str:
    #     """調用Ollama API"""
    #     url = f"{self.ollama_url}/api/generate"
        
    #     payload = {
    #         "model": self.model_name,
    #         "prompt": prompt,
    #         "system": system_prompt,
    #         "stream": False,
    #         "options": {
    #             "temperature": 0.3,
    #             "top_p": 0.9,
    #             "max_tokens": self.max_summary_tokens
    #         }
    #     }
        
    #     # 顯示發送給模型的內容
    #     print("\n" + "="*80)
    #     print("發送給模型的 PROMPT:")
    #     print("="*80)
    #     if system_prompt:
    #         print("SYSTEM PROMPT:")
    #         print("-" * 40)
    #         print(system_prompt)
    #         print("-" * 40)
    #     print("USER PROMPT:")
    #     print("-" * 40)
    #     print(prompt)
    #     print("="*80)
    #     print("等待模型回應...")
        
    #     try:
    #         response = requests.post(url, json=payload, timeout=300)
    #         response.raise_for_status()
    #         result = response.json()
    #         model_response = result.get('response', '').strip()
            
    #         # 顯示模型回應
    #         print("\n模型回應:")
    #         print("-" * 40)
    #         print(model_response)
    #         print("-" * 40)
            
    #         return model_response
    #     except Exception as e:
    #         error_msg = f"摘要生成失敗: {str(e)}"
    #         print(f"\n錯誤: {error_msg}")
    #         return error_msg

    def call_ollama(self, prompt: str, system_prompt: str = "") -> str:
        """調用Ollama API"""
        url = f"{self.ollama_url}/inference/infer"
        
        # 合併 system_prompt 和 prompt
        combined_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
        
        payload = {
            "task": "text-generation-ollama",
            "engine": "ollama",
            "model_name": self.model_name,
            "data": {
                "inputs": combined_prompt
            },
            "options": {
                "max_length": self.max_summary_tokens,
                "temperature": 0.3
            }
        }
        
        # 顯示發送給模型的內容
        print("\n" + "="*80)
        print("發送給模型的 PROMPT:")
        print("="*80)
        print(combined_prompt)
        print("="*80)
        print("等待模型回應...")
        
        try:
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }
            response = requests.post(url, json=payload, headers=headers, timeout=300)
            response.raise_for_status()
            result = response.json()
            
            # 從回應中提取 generated_text
            model_response = result.get('result', {}).get('generated_text', '').strip()
            
            if not model_response:
                print(f"警告: 未找到 generated_text，完整回應: {result}")
                return "模型未返回有效回應"
            
            # 顯示模型回應
            print("\n模型回應:")
            print("-" * 40)
            print(model_response)
            print("-" * 40)
            
            return model_response
        except Exception as e:
            error_msg = f"摘要生成失敗: {str(e)}"
            print(f"\n錯誤: {error_msg}")
            return error_msg

    def create_first_level_prompt(self, chunks: List[Dict], group_index: int) -> str:
        """創建第一層摘要的prompt"""
        system_prompt = """你是一個專業的文檔摘要助手。請仔細閱讀以下內容並生成簡潔而全面的摘要。

要求：
1. 保留所有關鍵信息和重要概念
2. 保持邏輯結構和前後文關係
3. 保留重要的數據、日期、人名、地名等具體信息
4. 摘要長度控制在原文的30-40%
5. 使用繁體中文回答
重要：請直接根據內容提供摘要，不要加入額外的說明文字，只需要摘要本身的內容。"""
        
        combined_text = ""
        for i, chunk in enumerate(chunks):
            chunk_text = chunk['payload']['text']
            combined_text += f"\n=== 段落 {i+1} ===\n{chunk_text}\n"
        
        prompt = f"""請為以下第{group_index+1}組內容生成摘要：

{combined_text}

請生成一個簡潔而完整的摘要，保留所有重要信息："""
        
        return prompt, system_prompt
    
    def create_higher_level_prompt(self, summaries: List[str], level: int, group_index: int) -> str:
        """創建高層摘要的prompt"""
        system_prompt = f"""你是一個專業的文檔摘要助手。請整合以下第{level}層的部分摘要，生成更高層次的整合摘要。

要求：
1. 整合所有摘要中的關鍵信息
2. 消除重複內容，但保留重要信息
3. 保持整體邏輯連貫性和結構
4. 確保摘要涵蓋所有重要主題
5. 使用繁體中文回答"""

        combined_summaries = ""
        for i, summary in enumerate(summaries):
            combined_summaries += f"\n=== 摘要 {i+1} ===\n{summary}\n"
        
        prompt = f"""請整合以下第{level}層的摘要內容：

{combined_summaries}

請生成一個整合性的摘要，涵蓋所有重要信息："""
        
        return prompt, system_prompt
    
    def create_final_prompt(self, summaries: List[str], filename: str) -> str:
        """創建最終摘要的prompt"""
#         system_prompt = """你是一個專業的文檔摘要助手。請基於提供的所有摘要內容，生成一個完整、專業的最終文檔摘要。

# 要求：
# 1. 生成完整的文檔概述
# 2. 包含文檔的主要主題和關鍵結論
# 3. 保持專業性和可讀性
# 4. 結構清晰，邏輯連貫
# 5. 使用繁體中文回答"""
#         system_prompt = """你是一個專業的文檔摘要助手。請基於提供的所有摘要內容，生成一個極簡的文檔概述。

# 要求：
# 1. 用5-8句話概括整個文檔的核心內容
# 2. 語言簡潔明瞭，避免冗長描述
# 3. 直接說明文檔的主要目的或內容
# 4. 使用繁體中文回答"""
        system_prompt = """你是一個專業的文檔摘要助手。請分析文檔的整體目的和核心內容，生成概述。

分析要點：
1. 識別文檔的主要目的和功能
2. 抓取最重要的核心訊息
3. 區分主要內容與次要細節
4. 關注文檔想要傳達的關鍵資訊

輸出要求：
- 用5-8句話概括文檔的核心內容
- 重點說明文檔的主要目的或價值
- 語言簡潔明瞭，避免冗長描述
- 使用繁體中文回答"""

        combined_summaries = ""
        for i, summary in enumerate(summaries):
            combined_summaries += f"\n=== 部分摘要 {i+1} ===\n{summary}\n"
        
        # prompt = f"""請為文檔「{filename}」生成最終摘要。

# 基於以下所有部分摘要：
# {combined_summaries}

# 請生成一個完整的文檔摘要，包含：
# 1. 文檔主要內容概述
# 2. 關鍵主題和重點
# 重要：請只根據內容提供摘要，不要加入額外的說明文字，不要加入任何前綴或後綴說明，只需要摘要本身的內容。"""
#         prompt = f"""請為文檔「{filename}」生成極簡摘要。

# 基於以下所有部分摘要：
# {combined_summaries}

# 請用5-8句話簡潔說明文檔的核心內容。

# 重要：請只用短短幾句話概略說明，不要詳細描述，不要加入額外的說明文字。"""
        prompt = f"""請為文檔「{filename}」生成簡單摘要。

基於以下摘要內容：
{combined_summaries}

請重點說明這份文檔的核心目的和最重要的關鍵資訊。"""
        
        return prompt, system_prompt
    
    def group_chunks(self, chunks: List[Dict], group_size: int) -> List[List[Dict]]:
        """將chunks分組"""
        groups = []
        for i in range(0, len(chunks), group_size):
            group = chunks[i:i + group_size]
            groups.append(group)
        return groups
    
    def calculate_summary_layers(self, total_chunks: int) -> int:
        """計算需要多少層摘要"""
        if total_chunks <= 0:
            return 0
        
        layers = 1
        current_items = (total_chunks + self.chunk_group_size - 1) // self.chunk_group_size
        
        while current_items > 1:
            current_items = (current_items + 3) // 4
            layers += 1
            
            estimated_tokens = current_items * self.max_summary_tokens
            if estimated_tokens <= self.max_model_tokens * self.safety_buffer:
                break
        
        return layers
    
    def summarize_from_json(self, json_file_path: str) -> str:
        """從JSON文件生成摘要"""
        try:
            print(f"正在讀取分析文件: {json_file_path}")
            # 讀取JSON文件
            with open(json_file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if not isinstance(data, list):
                return f"FORMAT ERROR: expect list, get {type(data)}"
                
            
            chunks = data
            
            filename = "unknown"
            if chunks and isinstance(chunks[0], dict):
                payload = chunks[0].get('payload', {})
                filename = payload.get('filename', 'unknown')
            
            print(f"讀取到 {len(chunks)} 個chunks，文件名: {filename}")
            
            if not chunks:
                return "文檔為空，無法生成摘要"
            
            # 驗證chunks格式
            valid_chunks = []
            for i, chunk in enumerate(chunks):
                if isinstance(chunk, dict) and 'payload' in chunk:
                    payload = chunk['payload']
                    if isinstance(payload, dict) and 'text' in payload:
                        valid_chunks.append(chunk)
                    else:
                        print(f"警告: 第 {i+1} 個chunk的payload格式不正確")
                else:
                    print(f"警告: 第 {i+1} 個chunk格式不正確")
            
            if not valid_chunks:
                return "沒有找到有效的文字內容塊"
            
            chunks = valid_chunks
            total_chunks = len(chunks)
            layers_needed = self.calculate_summary_layers(total_chunks)
            
            print(f"\n開始處理文檔: {filename}")
            print(f"有效chunks: {total_chunks} 個，需要 {layers_needed} 層摘要")
            
            current_items = chunks
            current_layer = 1
            
            # 執行多層摘要
            while current_layer <= layers_needed:
                print(f"\n正在處理第 {current_layer} 層摘要...")
                
                if current_layer == 1:
                    # 第一層：處理原始chunks
                    groups = self.group_chunks(current_items, self.chunk_group_size)
                    summaries = []
                    
                    for i, group in enumerate(groups):
                        print(f"\n處理第 {i+1}/{len(groups)} 組 chunks...")
                        prompt, system_prompt = self.create_first_level_prompt(group, i)
                        summary = self.call_ollama(prompt, system_prompt)
                        summaries.append(summary)
                    
                    current_items = summaries
                    
                else:
                    # 後續層：處理摘要
                    groups = self.group_chunks(current_items, 4)
                    new_summaries = []
                    
                    for i, group in enumerate(groups):
                        print(f"\n整合第 {i+1}/{len(groups)} 組摘要...")
                        prompt, system_prompt = self.create_higher_level_prompt(group, current_layer-1, i)
                        summary = self.call_ollama(prompt, system_prompt)
                        new_summaries.append(summary)
                    
                    current_items = new_summaries
                
                current_layer += 1
                
                # 檢查是否可以進行最終摘要
                total_tokens = sum(self.count_tokens_estimate(item) for item in current_items)
                if total_tokens <= self.max_model_tokens * self.safety_buffer:
                    break
            
            # 生成最終摘要
            print(f"\n正在生成最終摘要...")
            if len(current_items) == 1:
                final_summary = current_items[0]
            else:
                prompt, system_prompt = self.create_final_prompt(current_items, filename)
                final_summary = self.call_ollama(prompt, system_prompt)
            
            
            final_summary = self.converter.convert(final_summary)
            print(f"\n摘要生成完成！")
            return final_summary
            
        except json.JSONDecodeError as e:
            error_msg = f"JSON解析錯誤: {str(e)}"
            print(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"處理文件時發生錯誤: {str(e)}"
            print(error_msg)
            return error_msg

