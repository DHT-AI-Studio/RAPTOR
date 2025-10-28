# services/audio_summary_service/audio_summary.py

import json
import requests
from typing import List, Dict, Any
from transformers import AutoTokenizer
import opencc

class AudioSummary:
    def __init__(self, model_name="qwenforsummary", max_tokens_per_batch=1600, max_summary_length=200, ollama_url="http://192.168.157.165:8009"):
        self.model_name = model_name
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_summary_length = max_summary_length
        self.ollama_url = ollama_url
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.cc = opencc.OpenCC('s2t')
    
    def load_audio_data(self, file_path: str) -> List[Dict]:
        """載入音訊分析結果檔案"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def extract_audio_info(self, data: List[Dict]) -> tuple:
        """提取音訊資訊和標籤"""
        audio_info = ""
        all_labels = set()
        
        for item in data:
            payload = item.get("payload", {})
            start_time = payload.get("start_time", "0")
            speaker = payload.get("speaker", "Unknown")
            text = payload.get("text", "")
            labels = payload.get("audio_labels", [])
            
            audio_info += f"- {start_time}s [{speaker}]: {text}\n"
            all_labels.update(labels)
        
        return audio_info, list(all_labels)
    
    def calculate_tokens(self, text: str) -> int:
        """計算文本的token數量"""
        return len(self.tokenizer.encode(text))
    
    def split_into_batches(self, audio_info: str) -> List[str]:
        """將音訊資訊分批處理，確保每批不超過token限制"""
        lines = audio_info.strip().split('\n')
        batches = []
        current_batch = ""
        
        for line in lines:
            test_batch = current_batch + line + '\n'
            if self.calculate_tokens(test_batch) > self.max_tokens_per_batch:
                if current_batch:  # 如果當前批次不為空，保存它
                    batches.append(current_batch.strip())
                    current_batch = line + '\n'
                else:  # 如果單行就超過限制，強制添加
                    batches.append(line)
                    current_batch = ""
            else:
                current_batch = test_batch
        
        if current_batch.strip():
            batches.append(current_batch.strip())
        
        return batches
    
    def generate_summary(self, content: str, is_final=False) -> str:
        """使用本地Ollama生成摘要"""
        if is_final:
            prompt = f"""請將以下內容整合成一個簡潔的摘要，限制在{self.max_summary_length}字以內，概述音訊的主要內容：

{content}

摘要要求：
- 字數限制：{self.max_summary_length}字以內
- 概述音訊的主要內容和重點
- 保持客觀和準確
- 請用繁體中文回答"""
        else:
            prompt = f"""請對以下音訊內容進行詳細摘要，保留重要細節和時間點：

{content}

摘要要求：
- 保留重要的時間點和說話者資訊
- 詳細描述內容要點
- 不要添加額外說明，只根據內容進行摘要
- 請用繁體中文回答
- 不要提及你自己的名字或身份"""
        payload = {
            "task": "text-generation-ollama",
            "engine": "ollama",
            "model_name": self.model_name,
            "data": {
                "inputs": prompt
            },
            "options": {
                "max_length": self.max_summary_length if is_final else 200,
                "temperature": 0.3
            }
        }
        
        try:
            headers = {
                'accept': 'application/json',
                'Content-Type': 'application/json'
            }
            
            response = requests.post(
                f"{self.ollama_url}/inference/infer",
                json=payload,
                headers=headers,
                timeout=300  # 5分鐘超時
            )
            
            if response.status_code == 200:
                result = response.json()
                # 從新的回應格式中提取 generated_text
                summary = result.get('result', {}).get('generated_text', '').strip()
                
                if not summary:
                    print(f"警告: 未找到 generated_text")
                    print(f"完整回應: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    return "模型未返回有效回應"
                
                # 轉換為繁體中文
                summary = self.cc.convert(summary)
                return summary
            else:
                return f"摘要生成失敗: HTTP {response.status_code}"
                
        except requests.exceptions.Timeout:
            return "摘要生成失敗: 請求超時(300秒)"
        except requests.exceptions.RequestException as e:
            return f"摘要生成失敗: 連接錯誤 - {str(e)}"
        except Exception as e:
            return f"摘要生成失敗: {str(e)}"
        
        # try:
        #     response = requests.post(
        #         f"{self.ollama_url}/api/generate",
        #         json={
        #             "model": self.model_name,
        #             "prompt": prompt,
        #             "stream": False
        #         },
        #         timeout=300  # 5分鐘超時
        #     )
            
        #     if response.status_code == 200:
        #         result = response.json()
        #         summary = result.get('response', '').strip()
        #         summary = self.cc.convert(summary)
        #         return summary
        #     else:
        #         return f"摘要生成失敗: HTTP {response.status_code}"
                
        # except requests.exceptions.RequestException as e:
        #     return f"摘要生成失敗: 連接錯誤 - {str(e)}"
        # except Exception as e:
        #     return f"摘要生成失敗: {str(e)}"
    
    def process_summary(self, file_path: str) -> str:
        """主要處理函數"""
        # 載入數據
        try:
            data = self.load_audio_data(file_path)
        except FileNotFoundError:
            return f"錯誤: 找不到檔案 {file_path}"
        except json.JSONDecodeError:
            return f"錯誤: 檔案 {file_path} 不是有效的JSON格式"
        
        if not data:
            return "無音訊數據可供摘要"
        
        # 提取音訊資訊和標籤
        audio_info, labels = self.extract_audio_info(data)
        
        # 添加標籤資訊
        labels_info = f"音訊類別標籤: {', '.join(labels)}\n\n" if labels else ""
        full_content = labels_info + audio_info
        
        # 計算總token數
        total_tokens = self.calculate_tokens(full_content)
        print(f"總token數: {total_tokens}")
        
        if total_tokens <= self.max_tokens_per_batch:
            # 單次摘要
            print("執行單次摘要...")
            return self.generate_summary(full_content, is_final=True)
        else:
            # 多層摘要
            print(f"執行多層摘要，總token數: {total_tokens}")
            batches = self.split_into_batches(audio_info)
            batch_summaries = []
            
            # 第一層：對每個批次進行詳細摘要
            print(f"第一層摘要：處理 {len(batches)} 個批次")
            for i, batch in enumerate(batches):
                batch_content = labels_info + batch if i == 0 else batch
                summary = self.generate_summary(batch_content, is_final=False)
                batch_summaries.append(summary)
                print(f"完成第 {i+1}/{len(batches)} 批次摘要")
            
            # 第二層：整合所有批次摘要
            combined_summaries = labels_info + "\n".join(batch_summaries)
            
            # 檢查是否需要更多層次的摘要
            layer = 2
            while self.calculate_tokens(combined_summaries) > self.max_tokens_per_batch:
                print(f"第{layer}層摘要：token數仍超過限制，繼續分層處理")
                # 將批次摘要再次分批
                summary_batches = self.split_into_batches("\n".join(batch_summaries))
                new_summaries = []
                
                for i, batch in enumerate(summary_batches):
                    summary = self.generate_summary(batch, is_final=False)
                    new_summaries.append(summary)
                    print(f"完成第{layer}層第 {i+1}/{len(summary_batches)} 批次摘要")
                
                batch_summaries = new_summaries
                combined_summaries = labels_info + "\n".join(batch_summaries)
                layer += 1
            
            # 最終摘要
            print("生成最終摘要...")
            final_summary = self.generate_summary(combined_summaries, is_final=True)
            return final_summary
