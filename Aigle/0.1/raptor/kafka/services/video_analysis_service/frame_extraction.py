# services/video_analysis_service/frame_extraction.py

import asyncio
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any
import ffmpeg
import os
from PIL import Image
import concurrent.futures
import cv2
from scipy.stats import entropy
import torch
import torchvision.transforms as transforms
import aiofiles
from io import BytesIO
import json
from datetime import datetime
import glob
import psutil
import time
import sys
import re
import tempfile
from config import (
    FRAME_EXTRACTION_BASE_DIR,
    FRAME_EXTRACTION_FPS,
    FRAME_FORMAT,
    FRAME_QUALITY
)

logger = logging.getLogger(__name__)

def initialize_output_directories(directories: List[str], clear_existing: bool = True) -> None:
    """
    統一初始化所有輸出目錄，避免各模組重複創建目錄。
    
    參數:
        directories: 要初始化的目錄列表
        clear_existing: 是否清除現有目錄（預設為 True）
    
    異常:
        拋出 PermissionError 如果目錄不可寫
        拋出其他異常如果目錄創建失敗
    """
    for directory in directories:
        try:
            if os.path.exists(directory) and clear_existing:
                import shutil
                shutil.rmtree(directory)  # 清除現有目錄
                logger.info(f"Cleared and recreated directory: {directory}")
            os.makedirs(directory, exist_ok=True)  # 創建目錄，忽略已存在的情況
            if not os.access(directory, os.W_OK):
                raise PermissionError(f"Directory {directory} is not writable")  # 檢查目錄寫入權限
            logger.debug(f"Initialized directory: {directory}")
        except Exception as e:
            logger.error(f"Failed to initialize directory {directory}: {e}")
            raise

def calculate_batch_size(vram_per_frame_mb: int, safety_margin: float = 0.8):
    """
    根據可用記憶體動態計算批次大小，確保記憶體使用安全。
    
    參數:
        vram_per_frame_mb: 每幀估計的記憶體使用量（MB）
        safety_margin: 記憶體安全餘量（預設 0.8）
    
    返回:
        計算出的批次大小（整數）
    """
    try:
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
            total_vram = torch.cuda.get_device_properties(device).total_memory / (1024 ** 2)  # 總顯存
            used_vram = torch.cuda.memory_reserved(device) / (1024 ** 2)  # 已用顯存
            available_vram = total_vram - used_vram
            batch_size = max(1, int(available_vram * safety_margin / vram_per_frame_mb))  # 計算批次大小
            return batch_size
        else:
            available_memory = psutil.virtual_memory().available / (1024 ** 2)  # 可用系統記憶體
            batch_size = max(1, int(available_memory * safety_margin / vram_per_frame_mb))
            return batch_size
    except Exception as e:
        logger.error(f"Error calculating batch size: {e}")
        return 1  # 發生錯誤時返回最小批次大小

class FrameExtractor:
    """
    負責從影片中提取幀，執行預處理和篩選。
    使用進階並行處理和品質篩選。
    """
    
    def __init__(self, max_size: int = 640, min_mean: float = 5, min_entropy: float = 0.5):
        """
        初始化 FrameExtractor。
        參數:
            max_size: 幀的最大尺寸
            min_mean: 幀平均亮度的最小值
            min_entropy: 幀熵的最小值
        """
        self.max_size = max_size
        self.min_mean = min_mean
        self.min_entropy = min_entropy
        self.batch_size = min(32, calculate_batch_size(vram_per_frame_mb=100))
        logger.info(f"Initialized FrameExtractor with batch_size={self.batch_size}")

    def preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """
        預處理幀，包括顏色轉換、增強和縮放。
        參數:
            frame: 輸入幀（RGB 格式）
        返回:
            預處理後的幀（BGR 格式用於保存）
        """
        # 如果是 RGB，轉為 BGR
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # 轉為 BGR 以進行處理
        
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 創建 CLAHE 增強器
        channels = cv2.split(frame)
        enhanced_channels = [clahe.apply(ch) for ch in channels]  # 增強每個通道
        frame = cv2.merge(enhanced_channels)
        gamma = 1.5 if np.mean(frame) < 50 else 1.0  # 根據平均亮度調整伽馬值
        frame = np.array(255 * (frame / 255) ** (1 / gamma), dtype=np.uint8)
        blurred = cv2.GaussianBlur(frame, (5, 5), 0)
        frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)  # 銳化
        h, w = frame.shape[:2]
        scale = min(self.max_size / w, self.max_size / h)
        if scale < 1:
            frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)  # 縮放
        return frame
    
    def filter_frame(self, frame: np.ndarray) -> bool:
        """
        根據亮度和熵篩選幀，過濾低品質幀（如全黑或模糊）。
        參數:
            frame: 輸入幀
        返回:
            是否保留該幀（布林值）
        """
        mean_val = np.mean(frame)
        hist, _ = np.histogram(frame.flatten(), bins=256, range=(0, 256))
        frame_entropy = entropy(hist + 1e-9)
        return mean_val > self.min_mean and frame_entropy > self.min_entropy
    
    async def save_frame_batch(self, batch: List[Tuple[int, np.ndarray, str]]):
        """
        非同步批次儲存幀到檔案。
        參數:
            batch: 包含幀索引、圖像和檔案路徑的列表
        """
        async def save_single(i, frame, path):
            try:
                encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), FRAME_QUALITY] if FRAME_FORMAT.lower() == 'jpg' else []
                success, encoded_img = await asyncio.get_event_loop().run_in_executor(
                    None, lambda: cv2.imencode(f'.{FRAME_FORMAT}', frame, encode_params)
                )
                if success:
                    async with aiofiles.open(path, 'wb') as f:
                        await f.write(encoded_img.tobytes())
                    logger.debug(f"Saved frame {i} to {path}")
                    return True
                else:
                    logger.warning(f"Failed to encode frame {i}")
                    return False
            except Exception as e:
                logger.error(f"Failed to save frame {i}: {e}")
                return False
        
        tasks = []
        for i, frame, path in batch:
            tasks.append(save_single(i, frame, path))
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        successful_saves = sum(1 for result in results if result is True)
        logger.info(f"Saved {successful_saves}/{len(batch)} frames in batch")
    
    @staticmethod
    def _decode_chunk_static(
        video_path: str, 
        start_time: float, 
        end_time: float, 
        target_fps: float,
        max_size: int,
        min_mean: float,
        min_entropy: float
    ) -> List[Tuple[int, np.ndarray]]:
        """
        靜態方法，用於在子進程中解碼影片的一個時間段。
        必須是靜態方法或模組頂層函數才能被 multiprocessing 序列化。
        """
        try:
            import ffmpeg
            import numpy as np
            from scipy.stats import entropy
            import cv2
            # 獲取影片資訊
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if not video_stream:
                raise ValueError("No video stream found in the chunk")
            width, height = int(video_stream['width']), int(video_stream['height'])
            # 設定 ffmpeg 輸入，從 start_time 開始，到 end_time 結束
            stream = (
                ffmpeg
                .input(video_path, ss=start_time, to=end_time)
                .filter('fps', fps=target_fps)
                .output('pipe:', format='rawvideo', pix_fmt='rgb24')
            )
            process = stream.run_async(pipe_stdout=True)
            frames = []
            frame_idx = int(start_time * target_fps)  # 根據開始時間計算起始幀索引
            # 定義篩選和預處理函數（在子進程中重新定義）
            def filter_frame(frame):
                mean_val = np.mean(frame)
                hist, _ = np.histogram(frame.flatten(), bins=256, range=(0, 256))
                frame_entropy = entropy(hist + 1e-9)
                return mean_val > min_mean and frame_entropy > min_entropy
            def preprocess_frame(frame):
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                channels = cv2.split(frame)
                enhanced_channels = [clahe.apply(ch) for ch in channels]
                frame = cv2.merge(enhanced_channels)
                gamma = 1.5 if np.mean(frame) < 50 else 1.0
                frame = np.array(255 * (frame / 255) ** (1 / gamma), dtype=np.uint8)
                blurred = cv2.GaussianBlur(frame, (5, 5), 0)
                frame = cv2.addWeighted(frame, 1.5, blurred, -0.5, 0)
                h, w = frame.shape[:2]
                scale = min(max_size / w, max_size / h)
                if scale < 1:
                    frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LANCZOS4)
                return frame
            # 讀取並處理這個時間段內的所有幀
            while True:
                raw_frame = process.stdout.read(width * height * 3)
                if not raw_frame:
                    break
                frame = np.frombuffer(raw_frame, np.uint8).reshape((height, width, 3))
                # 在子進程中執行篩選和預處理
                if filter_frame(frame):
                    processed_frame = preprocess_frame(frame)
                    frames.append((frame_idx, processed_frame))
                frame_idx += 1
            process.wait()
            return frames
        except Exception as e:
            logger.error(f"Error in _decode_chunk_static: {e}")
            return []

    async def extract_frames(
        self, 
        video_path: str, 
        request_id: str, 
        filename: str
    ) -> Dict[str, Any]:
        """
        從影片中提取幀並保存到指定目錄（異步版本）。
        
        參數:
            video_path: 影片檔案路徑
            request_id: 請求 ID
            filename: 檔案名稱
        
        返回:
            包含提取結果的字典
        """
        # 創建輸出目錄
        output_dir = os.path.join(
            FRAME_EXTRACTION_BASE_DIR, 
            f"{request_id}_{filename}"
        )
        
        # 初始化目錄
        initialize_output_directories([output_dir], clear_existing=False)
        
        try:
            return await self._extract_with_metadata(video_path, output_dir, request_id, filename)
                
        except Exception as e:
            logger.error(f"Frame extraction failed: {e}")
            # 清理可能創建的目錄
            if os.path.exists(output_dir):
                try:
                    import shutil
                    shutil.rmtree(output_dir)
                except Exception as cleanup_error:
                    logger.error(f"Failed to cleanup directory {output_dir}: {cleanup_error}")
            raise

    async def _extract_with_metadata(self, video_path: str, output_dir: str, request_id: str, filename: str) -> Dict[str, Any]:
        """進階提取方法，返回完整的元數據"""
        try:
            # 獲取影片資訊
            probe = ffmpeg.probe(video_path)
            video_stream = next((stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
            if not video_stream:
                raise ValueError("No video stream found")
            
            fps = float(video_stream.get('r_frame_rate', '30/1').split('/')[0]) / float(video_stream.get('r_frame_rate', '30/1').split('/')[1])
            duration = float(probe['format']['duration'])
            total_frames = int(fps * duration)
            
            logger.info(f"Video info - FPS: {fps}, Duration: {duration:.2f}s, Total frames: {total_frames}")
            
            # 並行解碼提取幀
            num_workers = min(os.cpu_count(), 4)  # 限制最大工作進程數
            chunk_duration = duration / num_workers
            logger.info(f"Using parallel decoding with {num_workers} workers")

            # 使用 run_in_executor 來執行並行處理，避免事件循環衝突
            loop = asyncio.get_event_loop()
            all_frames = await loop.run_in_executor(
                None, 
                self._parallel_decode_chunks,
                video_path,
                duration,
                num_workers,
                FRAME_EXTRACTION_FPS,
                self.max_size,
                self.min_mean,
                self.min_entropy
            )

            all_frames.sort(key=lambda x: x[0])
            logger.info(f"Parallel decoding completed. Extracted {len(all_frames)} frames in total.")

            # 準備保存批次和生成元數據
            save_batches = []
            extracted_frames = []
            
            for i, (frame_idx, frame) in enumerate(all_frames):
                timestamp = frame_idx / FRAME_EXTRACTION_FPS
                frame_filename = f"frame_{i:06d}_{timestamp:.2f}s.{FRAME_FORMAT}"
                frame_path = os.path.join(output_dir, frame_filename)
                
                save_batches.append((i, frame, frame_path))
                extracted_frames.append({
                    "frame_index": i,
                    "original_frame_index": frame_idx,
                    "timestamp": timestamp,
                    "filename": frame_filename,
                    "file_path": frame_path
                })
                
                # 批次保存
                if len(save_batches) >= self.batch_size:
                    await self.save_frame_batch(save_batches)
                    save_batches = []
            
            # 保存剩餘的幀
            if save_batches:
                await self.save_frame_batch(save_batches)

            result = {
                "output_directory": output_dir,
                "total_frames_extracted": len(extracted_frames),
                "extracted_frames": extracted_frames,
                "video_info": {
                    "fps": fps,
                    "total_frames": total_frames,
                    "duration": duration,
                    "extraction_fps": FRAME_EXTRACTION_FPS
                },
                "extraction_method": "advanced_parallel"
            }
            
            logger.info(f"Frame extraction completed: {len(extracted_frames)} frames saved to {output_dir}")
            return result

        except Exception as e:
            logger.error(f"Frame extraction with metadata failed: {e}")
            raise

    def _parallel_decode_chunks(
        self,
        video_path: str,
        duration: float,
        num_workers: int,
        target_fps: float,
        max_size: int,
        min_mean: float,
        min_entropy: float
    ) -> List[Tuple[int, np.ndarray]]:
        """
        在線程池中執行並行解碼，避免事件循環衝突
        """
        chunk_duration = duration / num_workers
        all_frames = []
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                start_time = i * chunk_duration
                end_time = min((i + 1) * chunk_duration, duration)
                future = executor.submit(
                    FrameExtractor._decode_chunk_static,
                    video_path,
                    start_time,
                    end_time,
                    target_fps,
                    max_size,
                    min_mean,
                    min_entropy
                )
                futures.append(future)

            for future in concurrent.futures.as_completed(futures):
                chunk_frames = future.result()
                all_frames.extend(chunk_frames)
        
        return all_frames
    
    def cleanup_frames(self, output_dir: str):
        """清理提取的幀檔案"""
        try:
            if os.path.exists(output_dir):
                import shutil
                shutil.rmtree(output_dir)
                logger.info(f"Frame directory cleaned up: {output_dir}")
        except Exception as e:
            logger.error(f"Failed to cleanup frame directory {output_dir}: {e}")
