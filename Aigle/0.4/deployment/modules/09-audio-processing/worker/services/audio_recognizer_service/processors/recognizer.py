import whisperx
import torch
import gc
import re
import threading
from dotenv import load_dotenv
import os
load_dotenv('.env')
from langsmith import traceable

import ffmpeg
import logging
from typing import Dict, Any, Tuple
from config import ASYNC_PROCESSING_CONFIG

logger = logging.getLogger(__name__)

# Known lowerCamelCase-style compound names that the fused-boundary regex below
# would otherwise incorrectly split (e.g. "OpenAI" -> "Open AI"). Re-glued after
# the blind regex runs. Extend as new false positives turn up in real audio.
_KNOWN_COMPOUNDS = [
    "OpenAI", "YouTube", "GitHub", "GitLab", "iPhone", "iPad", "iOS", "macOS",
    "eBay", "PayPal", "WeChat", "TikTok", "LinkedIn", "PowerPoint", "MacBook",
    "JavaScript", "TypeScript", "WordPress", "FaceTime", "AirPods", "WhatsApp",
    "DeepSeek", "DeepMind",
]


def _fix_fused_word_boundaries(text: str) -> str:
    """Best-effort safety net for whisperx dropping the word-separating space
    (and/or punctuation) at a sentence boundary inside a single decoded chunk.
    Not a guarantee -- can misfire on other compound proper nouns not in the
    list above, but the current unconditional break is worse."""
    text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
    text = re.sub(r'([.!?,;:])([A-Z])', r'\1 \2', text)
    for compound in _KNOWN_COMPOUNDS:
        spaced = re.sub(r'([a-z])([A-Z])', r'\1 \2', compound)
        if spaced != compound:
            text = text.replace(spaced, compound)
    return text


class SpeechRecognizer:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float16" if self.device == "cuda" else "int8"

        self.model_name = "large-v3"
        self.batch_size = 16
        self._inference_sem = threading.Semaphore(ASYNC_PROCESSING_CONFIG["max_inference_concurrency"])
        self._align_model_cache: Dict[str, Any] = {}

        logger.info(f"Loading WhisperX model '{self.model_name}' on {self.device}")

        try:
            self.model = whisperx.load_model(self.model_name, self.device, compute_type=self.compute_type)
            logger.info("WhisperX model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load WhisperX model: {e}")
            raise

    @traceable(run_type="llm", name="SpeechRecognizer", project_name=os.getenv("LANGSMITH_PROJECT", "audioprocess"))
    def transcribe(self, audio_path: str) -> Tuple[Dict[str, Any], Any]:
        """ Transcribe audio to text using WhisperX model, then align for word-level timestamps.
        Args:
            audio_path (str): Path to the audio file.
        Returns:
            tuple: (transcription_result, audio_data)
        """
        try:
            logger.info(f"Loading audio file: {audio_path}")
            audio = whisperx.load_audio(audio_path)

            logger.info("Starting transcription...")
            for batch_size in [self.batch_size, self.batch_size // 2, 1]:
                try:
                    with self._inference_sem:
                        result = self.model.transcribe(audio, batch_size=batch_size)
                    break
                except torch.cuda.OutOfMemoryError:
                    logger.warning(f"OOM at batch_size={batch_size}, retrying smaller")
                    torch.cuda.empty_cache()
            else:
                raise RuntimeError("Transcription OOM at all batch sizes")

            language = result.get("language", "zh")
            logger.info(f"Transcription completed. Language detected: {language}")

            # Align for word-level timestamps
            align_model = metadata = None
            try:
                with self._inference_sem:
                    if language not in self._align_model_cache:
                        logger.info(f"Loading align model for language: {language}")
                        self._align_model_cache[language] = whisperx.load_align_model(
                            language_code=language, device=self.device
                        )
                    align_model, metadata = self._align_model_cache[language]
                    result = whisperx.align(
                        result["segments"], align_model, metadata, audio, self.device,
                        return_char_alignments=False,
                    )
                logger.info("Word-level alignment completed")
            except Exception as align_err:
                logger.warning(f"Alignment failed (word timestamps unavailable): {align_err}")
            finally:
                del align_model, metadata
                self._align_model_cache.pop(language, None)
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            # whisperx.align() drops language and has no top-level text/duration;
            # restore them so callers don't need to reconstruct from segments.
            result["language"] = language

            # whisperx can drop the word-separating space itself at a sentence
            # boundary INSIDE a single ~30s chunk (not just the punctuation), e.g.
            # "...replacedThe AI...", and even when punctuation does come through
            # it can still be glued to the next word, e.g. "...finds out.So
            # what...". This has to run per-segment, before any joining: downstream
            # consumers (e.g. audio_analysis_service/result_merger.py::
            # merge_all_data) read each segment's own `text` directly, they never
            # see the joined string built below. Best-effort regex, not a
            # guarantee -- can rarely misfire on compound proper nouns (e.g.
            # "McDonald"), but the current unconditional break is worse.
            for seg in result.get("segments", []):
                seg["text"] = _fix_fused_word_boundaries(seg.get("text", "").strip())

            result["text"] = " ".join(
                seg.get("text", "") for seg in result.get("segments", [])
            )
            result["audio_info"] = {"duration": round(len(audio) / 16000, 3)}

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
            del audio_data
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return result
        except Exception as e:
            logger.error(f"Failed to process audio file {file_path}: {e}")
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
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
            self._align_model_cache.clear()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            logger.info("SpeechRecognizer resources cleaned up")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
