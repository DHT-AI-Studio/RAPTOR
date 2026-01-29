# services/audio_diarization_service/processors/diarization.py

import whisperx
import torch
import gc
import numpy as np
import pandas as pd
import ffmpeg
import logging
import os
from typing import Dict, Any, List, Optional, Union
from dotenv import load_dotenv

load_dotenv('.env') 
from langsmith import traceable

logger = logging.getLogger(__name__)

class SpeakerDiarization:
    def __init__(self, device=None, hf_token=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        hf_token = os.getenv('HUGGINGFACE_HUB_TOKEN')
        logger.info(f"Loading WhisperX Diarization Pipeline on {self.device}")
        # self.pipeline = whisperx.diarize.DiarizationPipeline(device=self.device)
        self.pipeline = whisperx.diarize.DiarizationPipeline(
            device=self.device,
            use_auth_token=hf_token)
        logger.info("WhisperX Diarization Pipeline loaded successfully")
    
    @traceable(run_type="tool", name="diarize", project_name=os.getenv("LANGSMITH_PROJECT", "audioprocess"))
    def identify(self, audio):
        """ Identify speakers in the audio.
        Args:
            audio (np.ndarray): Audio data to process.
        Returns:
            pd.DataFrame: DataFrame containing speaker segments with start and end times.
        """
        logger.info("Starting speaker diarization...")
        # 執行說話人分離，返回時間區段
        result = self.pipeline(audio)
        logger.info(f"Diarization completed. Result type: {type(result)}")
        if hasattr(result, '__len__'):
            logger.info(f"Result length: {len(result)}")
        return result

    def extract_audio_if_video(self, file_path: str) -> str:
        """
        如果是視頻檔案則提取音頻，否則返回原檔案路徑
        """
        base_name = os.path.splitext(file_path)[0]
        audio_path = f"{base_name}_diarization.wav"
        
        try:
            probe = ffmpeg.probe(file_path)
            has_video = any(stream['codec_type'] == 'video' for stream in probe['streams'])
            
            if has_video:
                logger.info(f"Extracting audio from video for diarization: {file_path}")
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

    def get_audio_duration(self, audio_path: str) -> float:
        """
        獲取音頻檔案的時長
        """
        try:
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            raise RuntimeError(f"Cannot determine audio duration: {e}")

    def convert_diarization_result_to_serializable(self, diarize_result) -> Dict[str, Any]:
        """
        將 diarization 結果轉換為可序列化的格式
        """
        try:
            if isinstance(diarize_result, pd.DataFrame):
                logger.info("Converting DataFrame to serializable format")
                
                # 轉換為字典列表格式
                segments = []
                speakers = set()
                
                for idx, row in diarize_result.iterrows():
                    segment = {
                        "start": float(row.get('start', 0)),
                        "end": float(row.get('end', 0)),
                        "speaker": str(row.get('speaker', 'UNKNOWN'))
                    }
                    segments.append(segment)
                    speakers.add(segment['speaker'])
                
                return {
                    "segments": segments,
                    "total_segments": len(segments),
                    "unique_speakers": len(speakers),
                    "speakers": list(speakers)
                }
                
            elif hasattr(diarize_result, 'to_dict'):
                logger.info("Converting object with to_dict method")
                return diarize_result.to_dict()
                
            elif isinstance(diarize_result, dict):
                logger.info("Result is already a dictionary")
                return diarize_result
                
            elif isinstance(diarize_result, list):
                logger.info("Result is a list")
                speakers = set()
                for segment in diarize_result:
                    if isinstance(segment, dict) and 'speaker' in segment:
                        speakers.add(segment['speaker'])
                
                return {
                    "segments": diarize_result,
                    "total_segments": len(diarize_result),
                    "unique_speakers": len(speakers),
                    "speakers": list(speakers)
                }
            
            else:
                logger.warning(f"Unknown diarization result type: {type(diarize_result)}")
                return {
                    "segments": [],
                    "total_segments": 0,
                    "unique_speakers": 0,
                    "speakers": [],
                    "raw_result": str(diarize_result)
                }
                
        except Exception as e:
            logger.error(f"Failed to convert diarization result: {e}")
            return {
                "segments": [],
                "total_segments": 0,
                "unique_speakers": 0,
                "speakers": [],
                "conversion_error": str(e)
            }

    def process_audio_file(
        self, 
        file_path: str, 
        request_id: str, 
        **kwargs
    ) -> Dict[str, Any]:
        """
        處理音頻檔案並返回說話人分離結果
        """
        extracted_audio_path = None
        
        try:
            # 檢查是否為視頻檔案，如果是則提取音頻
            audio_path = self.extract_audio_if_video(file_path)
            if audio_path != file_path:
                extracted_audio_path = audio_path
            
            # 獲取音頻時長
            audio_duration = self.get_audio_duration(audio_path)
            
            # 載入音頻
            logger.info(f"Loading audio from: {audio_path}")
            audio = whisperx.load_audio(audio_path)
            
            # 執行說話人分離
            diarize_result = self.identify(audio)
            
            # 轉換結果為可序列化格式
            processed_result = self.convert_diarization_result_to_serializable(diarize_result)
            
            # 清理記憶體
            del audio
            gc.collect()
            
            # 清理提取的音頻檔案
            if extracted_audio_path and os.path.exists(extracted_audio_path):
                try:
                    os.remove(extracted_audio_path)
                    logger.info(f"Cleaned up extracted audio file: {extracted_audio_path}")
                except Exception as cleanup_error:
                    logger.warning(f"Failed to cleanup extracted audio file: {cleanup_error}")
            
            # 返回完整結果
            return {
                "request_id": request_id,
                "status": "success",
                "diarization_result": processed_result["segments"],
                "audio_duration": audio_duration,
                "total_segments": processed_result["total_segments"],
                "unique_speakers": processed_result["unique_speakers"],
                "speakers": processed_result["speakers"],
                "processing_info": {
                    "device": self.device,
                    "original_file": file_path,
                    "processed_file": audio_path
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to process audio file {file_path}: {e}")
            
            # 清理提取的音頻檔案
            if extracted_audio_path and os.path.exists(extracted_audio_path):
                try:
                    os.remove(extracted_audio_path)
                except:
                    pass
            
            return {
                "request_id": request_id,
                "status": "error",
                "error": str(e)
            }
