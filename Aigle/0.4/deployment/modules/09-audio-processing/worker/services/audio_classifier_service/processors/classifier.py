# services/audio_classifier_service/processors/classifier.py

import torch
import threading
import librosa
import numpy as np
from panns_inference import AudioTagging, SoundEventDetection, labels
import ffmpeg
import logging
import os
from typing import Dict, Any, List, Tuple
from dotenv import load_dotenv

load_dotenv('.env')
from langsmith import traceable
from config import ASYNC_PROCESSING_CONFIG

logger = logging.getLogger(__name__)

class AudioClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self._inference_sem = threading.Semaphore(ASYNC_PROCESSING_CONFIG["max_inference_concurrency"])

        logger.info(f"Loading PANNs models on {self.device}")

        try:
            self.audio_tagging_model = AudioTagging(checkpoint_path=None, device=self.device)
            self.sound_event_detection_model = SoundEventDetection(checkpoint_path=None, device=self.device)
            self.labels = labels
            logger.info("PANNs models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PANNs models: {e}")
            raise

    @traceable(run_type="tool", name="classify", project_name=os.getenv("LANGSMITH_PROJECT", "audioprocess"))
    def classify(self, audio_path: str, top_k: int = 5, chunk_sec: float = 10.0) -> List[Dict[str, Any]]:
        """
        對音訊檔案進行分類，並返回可能性最高的幾個標籤。
        以 chunk_sec 為單位逐段載入推理後平均，避免大檔案撐爆 VRAM。
        """
        SR = 32000
        try:
            logger.info(f"Classifying audio (chunk={chunk_sec}s): {audio_path}")
            duration = self.get_audio_duration(audio_path)
            num_chunks = max(1, int(np.ceil(duration / chunk_sec)))

            all_clipwise = []
            for i in range(num_chunks):
                chunk, _ = librosa.core.load(audio_path, sr=SR, mono=True, offset=i * chunk_sec, duration=chunk_sec)
                if len(chunk) < SR * 20:  # 不足 20 秒捨棄，避免 CNN pooling 崩掉
                    break
                with self._inference_sem:
                    (clipwise_output, _) = self.audio_tagging_model.inference(chunk[None, :])
                all_clipwise.append(clipwise_output[0])
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            if not all_clipwise:
                logger.warning(f"Audio too short to classify (< 2s): {audio_path}")
                return []

            mean_output = np.mean(all_clipwise, axis=0)
            sorted_indexes = np.argsort(mean_output)[::-1]
            top_results = [
                {"label": self.labels[i], "probability": round(float(mean_output[i]), 4)}
                for i in sorted_indexes[:top_k]
            ]
            logger.info(f"Classification completed ({num_chunks} chunks). Top label: {top_results[0]['label']}")
            return top_results

        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise
    
    @traceable(run_type="tool", name="segmented_classify", project_name=os.getenv("LANGSMITH_PROJECT", "audioprocess"))
    def segmented_classify(self, audio_path: str, audio_duration_sec: float, segment_length_sec: float = 30.0) -> List[Dict[str, Any]]:
        """
        每 segment_length_sec 秒為一段逐段載入並推理，儲存每段前五大類別與平均機率。
        逐段處理避免整部影片一次送進 VRAM。
        """
        SR = 32000
        try:
            logger.info(f"Performing segmented classification (segment={segment_length_sec}s): {audio_path}")
            segments = []
            chunk_idx = 0

            while True:
                offset = chunk_idx * segment_length_sec
                if offset >= audio_duration_sec:
                    break

                chunk, _ = librosa.core.load(audio_path, sr=SR, mono=True, offset=offset, duration=segment_length_sec)
                if len(chunk) < SR * 20:  # 不足 20 秒捨棄，避免 CNN pooling 崩掉
                    break

                with self._inference_sem:
                    framewise_output = self.sound_event_detection_model.inference(chunk[None, :])[0]
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                mean_probs = np.mean(framewise_output, axis=0)
                top_indices = np.argsort(mean_probs)[::-1][:5]
                top_classes = [
                    {"label": labels[j], "probability": round(float(mean_probs[j]), 4)}
                    for j in top_indices
                ]
                segments.append({
                    "segment_start": round(offset, 2),
                    "segment_end": round(min(offset + segment_length_sec, audio_duration_sec), 2),
                    "top_classes": top_classes
                })
                chunk_idx += 1

            logger.info(f"Segmented classification completed. {len(segments)} segments processed")
            return segments

        except Exception as e:
            logger.error(f"Segmented classification failed: {e}")
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
        audio_path = f"{base_name}_classifier.wav"
        
        try:
            probe = ffmpeg.probe(file_path)
            has_video = any(stream['codec_type'] == 'video' for stream in probe['streams'])
            
            if has_video:
                logger.info(f"Extracting audio from video for classification: {file_path}")
                ffmpeg.input(file_path).output(
                    audio_path, 
                    ac=1,  # 單聲道
                    ar=32000  # 32kHz 採樣率 (PANNs 需要)
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
        
        Args:
            audio_path (str): 音頻檔案路徑
            
        Returns:
            float: 音頻時長（秒）
        """
        try:
            probe = ffmpeg.probe(audio_path)
            duration = float(probe['streams'][0]['duration'])
            return duration
        except Exception as e:
            logger.error(f"Failed to get audio duration: {e}")
            # 備用方案：使用 librosa
            try:
                y, sr = librosa.load(audio_path, sr=None)
                return len(y) / sr
            except Exception as fallback_error:
                logger.error(f"Fallback duration calculation failed: {fallback_error}")
                raise RuntimeError(f"Cannot determine audio duration: {e}")

    def process_audio_file(self, file_path: str, request_id: str, classification_type: str = "segmented", **kwargs) -> Dict[str, Any]:
        """
        處理音頻檔案並返回分類結果
        
        Args:
            file_path (str): 輸入檔案路徑（可能是音頻或視頻）
            request_id (str): 請求 ID
            classification_type (str): 分類類型 ("basic" 或 "segmented")
            **kwargs: 額外參數
            
        Returns:
            dict: 分類結果
        """
        extracted_audio_path = None
        
        try:
            # 檢查是否為視頻檔案，如果是則提取音頻
            audio_path = self.extract_audio_if_video(file_path)
            
            # 記錄是否創建了新的音頻檔案
            if audio_path != file_path:
                extracted_audio_path = audio_path
                logger.info(f"Audio extracted to: {audio_path}")
            
            # 根據分類類型執行不同的分類方法
            if classification_type == "basic":
                top_k = kwargs.get("top_k", 5)
                classifications = self.classify(audio_path, top_k)
                    
                result = {
                    "top_classes": classifications,
                    "classification_type": "basic",
                    "top_k": top_k
                }

                
            elif classification_type == "segmented":
                # 獲取音頻時長
                duration = self.get_audio_duration(audio_path)
                segment_length = kwargs.get("segment_length_sec", 30.0)
                
                segments = self.segmented_classify(audio_path, duration, segment_length)
                
                result = {
                    "segments": segments,
                    "audio_duration": duration,
                    "segment_length": segment_length,
                    "classification_type": "segmented",
                    "segments_count": len(segments)
                }
                
                
            else:
                raise ValueError(f"Unknown classification type: {classification_type}")
            
            return result
            
        except Exception as e:
            logger.error(f"Failed to process audio file {file_path}: {e}")
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
            if hasattr(self, 'audio_tagging_model'):
                del self.audio_tagging_model
            if hasattr(self, 'sound_event_detection_model'):
                del self.sound_event_detection_model
            
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("AudioClassifier resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")