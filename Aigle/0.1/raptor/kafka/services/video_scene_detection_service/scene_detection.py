import asyncio
import logging
import numpy as np
from typing import List, Dict
import cv2
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
import os
import time
from langsmith import traceable

# 設定日誌
logger = logging.getLogger(__name__)

def skip_frames_input_serialization(inputs):
    """
    序列化輸入框架資訊，用於日誌或追蹤。
    """
    return {
        "frames_info": {
            "count": len(inputs["frames"]),
            "sample_shape": inputs["frames"][0].shape if inputs["frames"] else None
        }
    }

class SceneDetector:
    """
    優化後的場景檢測器，包含多項性能改進
    """
    def __init__(self, output_dir: str = "scene_frames", diff_plot_path: str = "scene_diff_plot.jpg"):
        """
        初始化場景檢測器
        
        參數:
            output_dir: 場景幀輸出目錄
            diff_plot_path: 差異圖表檔案名稱
        """
        self.output_dir = output_dir
        self.diff_plot_path = diff_plot_path
        self.max_comparison_size = 320  # 最大比較尺寸
        self.threshold = None  # 將在檢測過程中動態計算
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
    def _resize_for_comparison(self, frame: np.ndarray) -> np.ndarray:
        """縮小圖像用於快速比較"""
        h, w = frame.shape[:2]
        scale = min(self.max_comparison_size / max(h, w), 1.0)
        
        if scale < 1.0:
            return cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        return frame
    
    def _calculate_ssim_gray(self, frame: np.ndarray, prev_frame: np.ndarray) -> float:
        """使用灰度圖像計算SSIM"""
        gray1 = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        return ssim(gray1, gray2, win_size=3, k1=0.01, k2=0.03)
    
    def _is_frames_similar_fast(self, frame1: np.ndarray, frame2: np.ndarray) -> bool:
        """快速相似性檢查"""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        gray1 = cv2.resize(gray1, (32, 32), interpolation=cv2.INTER_AREA)
        gray2 = cv2.resize(gray2, (32, 32), interpolation=cv2.INTER_AREA)
        diff = np.mean(np.abs(gray1.astype(np.float32) - gray2.astype(np.float32)))
        return diff < 5.0
    
    @traceable(run_type="tool", name="SceneDetector.detect", project_name="video_analysis_service", 
               process_inputs=skip_frames_input_serialization)
    async def detect(self, frames: List[np.ndarray], target_fps: float, threshold: float = None) -> List[Dict]:
        """
        優化後的場景檢測，結合多項加速技術
        
        參數:
            frames: 已提取的幀列表（主要輸入）
            target_fps: 目標幀率，用於計算時間戳
            threshold: 場景變化閾值，若為None則自動計算
            
        返回:
            場景變化點列表，每個包含 frame_index 和 timestamp
        """
        start_time = time.time()
        fps_for_calculation = target_fps
        logger.info(f"Starting optimized scene detection with target_fps={target_fps}")
        
        if len(frames) < 2:
            logger.warning("Not enough frames for scene detection")
            return []
        
        # 1. 預先縮小所有幀
        resized_frames = [self._resize_for_comparison(frame) for frame in frames]
        
        # 2. 準備GPU（如果可用）
        use_gpu = False
        frames_gpu = None
        
        try:
            import torch
            if torch.cuda.is_available():
                try:
                    import cupy as cp
                    frames_gpu = [cp.asarray(frame.astype(cp.float32)) for frame in resized_frames]
                    use_gpu = True
                    logger.info(f"Loaded {len(frames)} frames to GPU")
                except ImportError:
                    logger.info("CuPy not available, using CPU for scene detection")
                    use_gpu = False
        except ImportError:
            logger.info("PyTorch not available, using CPU for scene detection")
        
        # 3. 計算場景差異
        combined_diffs = []
        
        for i in range(1, len(resized_frames)):
            # 快速相似性檢查
            if self._is_frames_similar_fast(resized_frames[i], resized_frames[i-1]):
                combined_diffs.append(0.1)
                continue
            
            # 計算像素差異
            if use_gpu and frames_gpu:
                pixel_diff = float(cp.mean(cp.abs(frames_gpu[i] - frames_gpu[i-1])).get())
            else:
                pixel_diff = float(np.mean(np.abs(
                    resized_frames[i].astype(np.float32) - 
                    resized_frames[i-1].astype(np.float32)
                )))
            
            # 計算SSIM
            ssim_score = self._calculate_ssim_gray(resized_frames[i], resized_frames[i-1])
            combined_diff = pixel_diff * (1 - ssim_score)
            combined_diffs.append(combined_diff)
        
        # 4. 應用閾值
        scenes = []
        if threshold is None and combined_diffs:
            median_diff = np.median(combined_diffs)
            std_diff = np.std(combined_diffs)
            threshold = median_diff + 0.5 * std_diff
            self.threshold = threshold  # 儲存計算出的閾值
        elif threshold is not None:
            self.threshold = threshold
        
        for idx, diff in enumerate(combined_diffs):
            frame_idx = idx + 1
            if diff > threshold:
                timestamp = frame_idx / fps_for_calculation
                scenes.append({
                    "frame_index": frame_idx, 
                    "timestamp": timestamp,
                    "combined_diff": diff  # 添加差異值用於後續分析
                })
        
        # 5. 保存結果
        if scenes:
            scene_frames_to_save = [(frames[s["frame_index"]], s["frame_index"], s["timestamp"]) for s in scenes]
            for frame, idx, timestamp in scene_frames_to_save:
                img_path = os.path.join(self.output_dir, f"scene_frame_{idx:04d}.jpg")
                # 確保frame是BGR格式用於cv2.imwrite
                if len(frame.shape) == 3 and frame.shape[2] == 3:
                    # 假設輸入是RGB，轉換為BGR
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                else:
                    frame_bgr = frame
                cv2.imwrite(img_path, frame_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        
        # 6. 保存差異圖
        if combined_diffs:
            self._save_diff_plot(combined_diffs, threshold, fps_for_calculation)
        
        logger.info(f"Scene detection completed in {time.time() - start_time:.2f}s, detected {len(scenes)} scenes")
        return scenes
    
    def _save_diff_plot(self, diffs: List[float], threshold: float, analysis_fps: float) -> None:
        """
        儲存場景差異圖表，顯示幀間差異隨時間變化的趨勢。
        """
        plt.figure(figsize=(12, 6))
        # X軸是時間（秒）
        timestamps = np.arange(1, len(diffs) + 1) / analysis_fps
        # 繪製差異圖
        plt.plot(timestamps, diffs, linewidth=1.2, label="Frame Difference")
        # 繪製閾值線
        plt.axhline(y=threshold, color='r', linestyle='--', linewidth=1, label=f"Threshold ({threshold:.2f})")
        # 設定標籤和標題
        plt.xlabel("Time (seconds)")
        plt.ylabel("Combined Difference")
        plt.title("Scene Change Detection Over Time")
        plt.legend()
        plt.grid(True, alpha=0.3)
        # 保存圖表
        plot_path = os.path.join(self.output_dir, self.diff_plot_path)
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        logger.debug(f"Saved diff plot to {plot_path}")