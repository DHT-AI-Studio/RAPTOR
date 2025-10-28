# services/audio_classifier_service/processors/classifier.py

import torch
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

logger = logging.getLogger(__name__)

class AudioClassifier:
    def __init__(self):
        # 動態檢測 CUDA 是否可用，與 recognizer.py 保持一致
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Loading PANNs models on {self.device}")
        
        try:
            self.audio_tagging_model = AudioTagging(checkpoint_path=None, device=self.device)
            self.sound_event_detection_model = SoundEventDetection(checkpoint_path=None, device=self.device)
            
            # 將 PANNs 提供的標籤列表儲存起來，方便後續查找
            self.labels = labels
            
            logger.info("PANNs models loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PANNs models: {e}")
            raise

    @traceable(run_type="tool", name="classify", project_name=os.getenv("LANGSMITH_PROJECT", "audioprocess"))
    def classify(self, audio_path: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        對音訊檔案進行分類，並返回可能性最高的幾個標籤。

        Args:
          audio_path (str): 音訊檔案的路徑。
          top_k (int): 要返回的標籤數量。

        Returns:
          list: 一個包含 top_k 個最可能標籤的列表，每個元素包含 label 和 probability。
        """
        try:
            logger.info(f"Classifying audio: {audio_path}")
            
            # 使用 librosa 載入音訊
            # 關鍵：sr=32000 指定了 PANNs 模型需要的取樣率，librosa 會自動進行重採樣
            # mono=True 將音訊轉為單聲道
            (waveform, _) = librosa.core.load(audio_path, sr=32000, mono=True)

            # 為模型推理增加一個批次維度 (batch_size, segment_samples)
            waveform = waveform[None, :]  

            # 執行模型推理，返回的是 (clipwise_output, embedding)
            (clipwise_output, _) = self.audio_tagging_model.inference(waveform)
            
            # clipwise_output 的形狀是 (batch_size, classes_num)，我們取第一筆資料
            clipwise_output = clipwise_output[0]

            # 使用 np.argsort 找出機率值從高到低排序後的索引
            sorted_indexes = np.argsort(clipwise_output)[::-1]

            # 根據排序後的索引，從 labels 列表中取得對應的標籤名稱和機率
            top_results = []
            for i in sorted_indexes[:top_k]:
                top_results.append({
                    "label": self.labels[i],
                    "probability": round(float(clipwise_output[i]), 4)
                })
            
            logger.info(f"Classification completed. Top label: {top_results[0]['label']}")
            return top_results
            
        except Exception as e:
            logger.error(f"Classification failed: {e}")
            raise
    
    @traceable(run_type="tool", name="segmented_classify", project_name=os.getenv("LANGSMITH_PROJECT", "audioprocess"))
    def segmented_classify(self, audio_path: str, audio_duration_sec: float, segment_length_sec: float = 30.0) -> List[Dict[str, Any]]:
        """
        將 framewise_output 合併為每 segment_length_sec 秒為一段，儲存每段的前三大類別與平均機率

        Args:
            audio_path (str): 音訊檔案的路徑
            audio_duration_sec (float): 音訊的實際長度（秒）
            segment_length_sec (float): 每個段落的秒數（預設 30 秒）

        Returns:
            list: 每段的開始時間、結束時間、前三大類別與平均機率
        """
        try:
            logger.info(f"Performing segmented classification: {audio_path}")
            
            # 使用 librosa 載入音訊
            # 關鍵：sr=32000 指定了 PANNs 模型需要的取樣率，librosa 會自動進行重採樣
            # mono=True 將音訊轉為單聲道
            (audio, _) = librosa.core.load(audio_path, sr=32000, mono=True)
            
            audio = audio[None, :]  # (batch_size, segment_samples)
            
            framewise_output = self.sound_event_detection_model.inference(audio)[0]  # 取第一筆資料的 framewise_output

            time_steps, num_classes = framewise_output.shape

            frame_duration = audio_duration_sec / time_steps
            frames_per_segment = int(segment_length_sec / frame_duration)

            segments = []

            for seg_start_frame in range(0, time_steps, frames_per_segment):
                seg_end_frame = min(seg_start_frame + frames_per_segment, time_steps)
                segment = framewise_output[seg_start_frame:seg_end_frame, :]  # 這段的所有 frames

                # 對這段中每個類別做平均（代表此段的整體機率）
                mean_probs = np.mean(segment, axis=0)

                # 找出前5大類別
                top_indices = np.argsort(mean_probs)[::-1][:5]
                top_classes = [
                    {
                        "label": labels[j],
                        "probability": round(float(mean_probs[j]), 4)
                    }
                    for j in top_indices
                ]

                segments.append({
                    "segment_start": round(seg_start_frame * frame_duration, 2),
                    "segment_end": round(seg_end_frame * frame_duration, 2),
                    "top_classes": top_classes
                })

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
