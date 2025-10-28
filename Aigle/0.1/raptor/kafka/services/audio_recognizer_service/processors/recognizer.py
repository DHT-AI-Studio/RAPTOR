import whisperx
import torch
import gc
from dotenv import load_dotenv
import os
load_dotenv('.env') 
from langsmith import traceable

import ffmpeg
import logging
from typing import Dict, Any, Tuple

logger = logging.getLogger(__name__)

class SpeechRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"
        
        # 定義模型相關屬性
        self.model_name = "large-v3"
        self.batch_size = 16
        
        logger.info(f"Loading WhisperX model '{self.model_name}' on {self.device}")
        
        try:
            self.model = whisperx.load_model(self.model_name, self.device, compute_type=self.compute_type)
            logger.info("WhisperX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            raise

    @traceable(run_type="llm", name="SpeechRecognizer", project_name=os.getenv("LANGSMITH_PROJECT", "audioprocess"))
    def transcribe(self, audio_path: str) -> Tuple[Dict[str, Any], Any]:
        """ Transcribe audio to text using WhisperX model.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            tuple: (transcription_result, audio_data)
        """
        try:
            logger.info(f"Loading audio file: {audio_path}")
            audio = whisperx.load_audio(audio_path)
            
            logger.info("Starting transcription...")
            result = self.model.transcribe(audio, batch_size=self.batch_size)
            
            logger.info(f"Transcription completed. Language detected: {result.get('language', 'unknown')}")
            return result, audio
            
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

    def extract_audio_if_video(self, file_path: str) -> str:
        """
        如果是視頻檔案則提取音頻，否則返回原檔案路徑
        
        Args:
            file_path (str): 輸入檔案路徑
            
        Returns:
            str: 音頻檔案路徑
        """
        base_name = os.path.splitext(file_path)[0]
        audio_path = f"{base_name}.wav"
        
        try:
            probe = ffmpeg.probe(file_path)
            has_video = any(stream['codec_type'] == 'video' for stream in probe['streams'])
            
            if has_video:
                logger.info(f"Extracting audio from video: {file_path}")
                ffmpeg.input(file_path).output(
                    audio_path, 
                    ac=1,  # 單聲道
                    ar=16000  # 16kHz 採樣率
                ).run(overwrite_output=True, quiet=True)
                return audio_path
            else:
                return file_path
                
        except Exception as e:
            logger.error(f"ffmpeg probe failed for {file_path}: {e}")
            raise RuntimeError(f"ffmpeg probe failed: {e}")
    
    def process_audio_file(self, file_path: str, request_id: str) -> Dict[str, Any]:
        """
        處理音頻檔案並返回結果
        
        Args:
            file_path (str): 輸入檔案路徑（可能是音頻或視頻）
            request_id (str): 請求 ID
            
        Returns:
            dict: 處理結果
        """
        extracted_audio_path = None
        
        try:
            # 檢查是否為視頻檔案，如果是則提取音頻
            audio_path = self.extract_audio_if_video(file_path)
            
            # 記錄是否創建了新的音頻檔案
            if audio_path != file_path:
                extracted_audio_path = audio_path
                logger.info(f"Audio extracted to: {audio_path}")
            
            # 執行轉錄
            result, audio_data = self.transcribe(audio_path)
            
            return result
        except Exception as e:
            logger.error(f"Failed to process audio file {file_path}: {e}")
            # 返回錯誤結果
            return {
                "error": str(e)
            }
            
        finally:
            # 清理提取的音頻檔案（如果有的話）
            if extracted_audio_path and os.path.exists(extracted_audio_path):
                try:
                    os.remove(extracted_audio_path)
                    logger.info(f"Cleaned up extracted audio file: {extracted_audio_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup extracted audio file: {cleanup_error}")
    
    def cleanup(self):
        """清理資源"""
        try:
            if hasattr(self, 'model'):
                del self.model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SpeechRecognizer resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
