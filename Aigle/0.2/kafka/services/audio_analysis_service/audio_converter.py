# services/audio_analysis_service/audio_converter.py

import os
import logging
import ffmpeg
from typing import Optional

logger = logging.getLogger(__name__)

class AudioConverter:
    def __init__(self):
        pass
    
    def extract_audio_if_video(self, file_path: str) -> str:
        """
        如果是影片檔案則提取音頻，否則直接返回檔案路徑
        返回音頻檔案路徑
        """
        base_name = os.path.splitext(file_path)[0]
        audio_path = f"{base_name}.wav"
        
        try:
            # 使用 ffmpeg 探測檔案
            probe = ffmpeg.probe(file_path)
            has_video = any(stream['codec_type'] == 'video' for stream in probe['streams'])
            
            if has_video:
                logger.info(f"Extracting audio from video: {file_path}")
                # 提取音頻並轉換為 16kHz 單聲道 WAV
                (
                    ffmpeg
                    .input(file_path)
                    .output(audio_path, ac=1, ar=16000)
                    .run(overwrite_output=True, quiet=True)
                )
                logger.info(f"Audio extracted successfully: {audio_path}")
                return audio_path
            else:
                logger.info(f"File is already audio format: {file_path}")
                # 如果已經是音頻檔案，檢查是否需要轉換格式
                return self._ensure_wav_format(file_path)
                
        except Exception as e:
            logger.error(f"ffmpeg probe/conversion failed: {e}")
            raise RuntimeError(f"Audio conversion failed: {e}")
    
    def _ensure_wav_format(self, audio_path: str) -> str:
        """
        確保音頻檔案是 WAV 格式且符合要求的規格
        """
        if audio_path.lower().endswith('.wav'):
            # 檢查音頻規格是否符合要求
            try:
                probe = ffmpeg.probe(audio_path)
                audio_stream = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
                
                sample_rate = int(audio_stream.get('sample_rate', 0))
                channels = int(audio_stream.get('channels', 0))
                
                if sample_rate == 16000 and channels == 1:
                    logger.info(f"Audio file already in correct format: {audio_path}")
                    return audio_path
                    
            except Exception as e:
                logger.warning(f"Could not probe audio specs: {e}")
        
        # 需要轉換格式
        base_name = os.path.splitext(audio_path)[0]
        converted_path = f"{base_name}_converted.wav"
        
        try:
            logger.info(f"Converting audio to WAV format: {audio_path} -> {converted_path}")
            (
                ffmpeg
                .input(audio_path)
                .output(converted_path, ac=1, ar=16000)
                .run(overwrite_output=True, quiet=True)
            )
            return converted_path
            
        except Exception as e:
            logger.error(f"Audio format conversion failed: {e}")
            raise RuntimeError(f"Audio format conversion failed: {e}")
    
    def get_audio_info(self, audio_path: str) -> dict:
        """
        獲取音頻檔案資訊
        """
        try:
            probe = ffmpeg.probe(audio_path)
            audio_stream = next(s for s in probe['streams'] if s['codec_type'] == 'audio')
            
            return {
                "duration": float(probe['format'].get('duration', 0)),
                "sample_rate": int(audio_stream.get('sample_rate', 0)),
                "channels": int(audio_stream.get('channels', 0)),
                "codec": audio_stream.get('codec_name', 'unknown'),
                "bitrate": int(audio_stream.get('bit_rate', 0)) if audio_stream.get('bit_rate') else 0
            }
            
        except Exception as e:
            logger.error(f"Failed to get audio info: {e}")
            return {}
