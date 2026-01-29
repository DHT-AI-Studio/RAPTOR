# services/video_summary_service/video_summary.py

import json
import requests
from typing import List, Dict, Any, Optional
from transformers import AutoTokenizer
import opencc
import logging

logger = logging.getLogger(__name__)

class VideoSummary:
    def __init__(self, model_name="qwenforsummary", max_tokens_per_batch=1600, max_summary_length=300, ollama_url="http://192.168.157.165:8009"):
        self.model_name = model_name
        self.max_tokens_per_batch = max_tokens_per_batch
        self.max_summary_length = max_summary_length
        self.ollama_url = ollama_url
        self.tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-m3")
        self.cc = opencc.OpenCC('s2t')
    
    def load_json_data(self, file_path: str) -> Any:
        """載入JSON檔案"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading JSON file {file_path}: {e}")
            return None
    
    def extract_video_analysis_info(self, video_data: Dict) -> tuple:
        """提取視頻分析資訊"""
        event_summary = video_data.get("event_summary", "")
        frames = video_data.get("frames", [])
        
        scene_info = ""
        for frame in frames:
            timestamp = frame.get('timestamp', 'Unknown')
            ocr_text = frame.get('text', 'No text')
            if ocr_text and ocr_text.strip():
                scene_info += f"- 時間點 {timestamp}s: {ocr_text}\n"
        
        return event_summary, scene_info
    
    def extract_audio_analysis_info(self, audio_data: List[Dict]) -> str:
        """提取音頻分析資訊"""
        audio_info = ""
        
        for segment in audio_data:
            start_time = segment.get('start_time', 0)
            text = segment.get('text', '')
            speaker = segment.get('speaker', 'Unknown')
            if text and text.strip():
                audio_info += f"- {start_time:.1f}s [{speaker}]: {text}\n"
        
        return audio_info if audio_info else "無音頻轉錄資訊。"
    
    def calculate_tokens(self, text: str) -> int:
        """計算文本的token數量"""
        try:
            return len(self.tokenizer.encode(text))
        except Exception as e:
            logger.warning(f"Token calculation error: {e}, using character count estimation")
            return len(text) // 2  # 估算
    
    def create_summary_prompt(self, event_summary: str, scene_info: str, audio_info: str) -> str:
        """創建摘要生成的prompt"""
        prompt = f'''
<background>
你是一個視訊媒體分析專家，負責整合多元資訊並且生成中文總結(Summary)。
</background>

<Goal>
請根據相關資訊的畫面文字資訊(包含時間點及OCR文字)及音頻轉錄資訊以及視覺描述總結這段影片的主要內容，
</Goal>

<Rules>
- 分析視覺描述與畫面文字資訊及音頻轉錄資訊的關聯性
- 生成詳細的中文總結(Summary)，請盡可能的保留細節。
- 以文字敘述的方式呈現，不需要條列。
- 字數限制：{self.max_summary_length}字以內
- 請用繁體中文回答
- 不要提及你自己的名字或身份
</Rules>

<Information>
##以下是相關資訊

畫面文字資訊：
{scene_info if scene_info else '無畫面文字資訊。'}

音頻轉錄資訊：
{audio_info}

視覺描述：
{event_summary if event_summary else 'No visual description available'}
</Infomation>
'''
        return prompt
    def generate_summary(self, prompt: str) -> str:
        """使用本地Ollama生成摘要"""

        
        # 構建新的 API payload
        payload = {
            "task": "text-generation-ollama",
            "engine": "ollama",
            "model_name": self.model_name,
            "data": {
                "inputs": prompt
            },
            "options": {
                "max_length": self.max_summary_length,
                "temperature": 0.3
            }
        }
        
        # 顯示發送給模型的內容
        logger.info("\n" + "="*80)
        logger.info("發送給模型的 PROMPT:")
        logger.info("="*80)
        logger.info(prompt[:500] + "..." if len(prompt) > 500 else prompt)
        logger.info("="*80)
        logger.info("等待模型回應...")
        
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
                    logger.warning(f"未找到 generated_text")
                    logger.warning(f"完整回應: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    return "模型未返回有效回應"
                
                # 顯示模型回應
                logger.info("\n模型回應:")
                logger.info("-" * 40)
                logger.info(summary)
                logger.info("-" * 40)
                
                # 轉換為繁體中文
                summary = self.cc.convert(summary)
                return summary
            else:
                error_msg = f"摘要生成失敗: HTTP {response.status_code}"
                logger.error(error_msg)
                return error_msg
                
        except requests.exceptions.Timeout:
            error_msg = "摘要生成失敗: 請求超時(300秒)"
            logger.error(error_msg)
            return error_msg
        except requests.exceptions.RequestException as e:
            error_msg = f"摘要生成失敗: 連接錯誤 - {str(e)}"
            logger.error(error_msg)
            return error_msg
        except Exception as e:
            error_msg = f"摘要生成失敗: {str(e)}"
            logger.error(error_msg)
            import traceback
            traceback.print_exc()
            return error_msg
    
    # def generate_summary(self, prompt: str) -> str:
    #     """使用本地Ollama生成摘要"""
    #     try:
    #         response = requests.post(
    #             f"{self.ollama_url}/api/generate",
    #             json={
    #                 "model": self.model_name,
    #                 "prompt": prompt,
    #                 "stream": False
    #             },
    #             timeout=300  # 5分鐘超時
    #         )
            
    #         if response.status_code == 200:
    #             result = response.json()
    #             summary = result.get('response', '').strip()
    #             summary = self.cc.convert(summary)
    #             return summary
    #         else:
    #             return f"摘要生成失敗: HTTP {response.status_code}"
                
    #     except requests.exceptions.RequestException as e:
    #         return f"摘要生成失敗: 連接錯誤 - {str(e)}"
    #     except Exception as e:
    #         return f"摘要生成失敗: {str(e)}"
    
    def process_summary(self, video_analysis_path: str, audio_analysis_path: str) -> str:
        """主要處理函數"""
        # 載入視頻分析數據
        video_data = self.load_json_data(video_analysis_path)
        if video_data is None:
            return f"錯誤: 無法載入視頻分析檔案 {video_analysis_path}"
        
        # 載入音頻分析數據
        audio_data = self.load_json_data(audio_analysis_path)
        if audio_data is None:
            return f"錯誤: 無法載入音頻分析檔案 {audio_analysis_path}"
        
        # 提取視頻分析資訊
        event_summary, scene_info = self.extract_video_analysis_info(video_data)
        
        # 提取音頻分析資訊
        audio_info = self.extract_audio_analysis_info(audio_data)
        
        # 創建摘要prompt
        prompt = self.create_summary_prompt(event_summary, scene_info, audio_info)
        
        # 計算token數
        total_tokens = self.calculate_tokens(prompt)
        logger.info(f"總token數: {total_tokens}")
        
        if total_tokens > self.max_tokens_per_batch:
            logger.warning(f"Token數({total_tokens})超過限制({self.max_tokens_per_batch})，將進行內容壓縮")
            # 如果超過限制，可以考慮壓縮內容或分批處理
            # 這裡簡化處理，截取部分內容
            if len(scene_info) > 500:
                scene_info = scene_info[:500] + "..."
            if len(audio_info) > 800:
                audio_info = audio_info[:800] + "..."
            prompt = self.create_summary_prompt(event_summary, scene_info, audio_info)
        
        # 生成摘要
        logger.info("開始生成視頻摘要...")
        summary = self.generate_summary(prompt)
        
        return summary
