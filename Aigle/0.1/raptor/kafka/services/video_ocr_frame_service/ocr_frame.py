# services/video_ocr_frame_service/ocr_frame.py

import asyncio
import json
import logging
import os
import glob
import time
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import torch
import aiofiles
from paddleocr import PaddleOCR
from zhconv import convert
from datetime import datetime, timezone
from langsmith import traceable
from config import (
    OCR_BATCH_SIZE,
    OCR_CONFIDENCE_THRESHOLD
)

logger = logging.getLogger(__name__)

def skip_frames_input_serialization(inputs):
    """序列化輸入框架資訊，用於日誌或追蹤"""
    return {
        "scene_frames_info": {
            "count": len(inputs.get("scene_frames", [])),
            "scene_directory": inputs.get("scene_output_directory", "")
        }
    }

class VideoOCRFrameProcessor:
    """處理場景幀的 OCR 識別"""
    
    def __init__(self, output_dir: str = "ocr_frames"):
        self.output_dir = output_dir
        self.ocr = None
        self.ocr_initialized = False
        
        # 確保輸出目錄存在
        os.makedirs(output_dir, exist_ok=True)
        
        logger.info(f"Initialized VideoOCRFrameProcessor with output_dir={output_dir}")
        
    async def initialize_ocr(self):
        """初始化 OCR 引擎"""
        if self.ocr_initialized:
            return
            
        try:
            loop = asyncio.get_event_loop()
            self.ocr = await loop.run_in_executor(
                None,
                lambda: PaddleOCR(
                    lang="chinese_cht",
                    ocr_version="PP-OCRv5",
                    use_doc_unwarping=False,
                    text_detection_model_name="PP-OCRv5_mobile_det",
                    text_recognition_model_name="PP-OCRv5_mobile_rec",
                    text_recognition_batch_size=self._calculate_batch_size(),
                    textline_orientation_batch_size=self._calculate_batch_size(),
                    device="gpu" if torch.cuda.is_available() else "cpu",
                    precision="fp16",
                )
            )
            self.ocr_initialized = True
            logger.info("OCR engine initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OCR engine: {e}")
            raise
    
    def _calculate_batch_size(self) -> int:
        """計算 OCR 批次大小"""
        if torch.cuda.is_available():
            try:
                total_vram = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
                used_vram = torch.cuda.memory_reserved(0) / (1024 ** 2)
                available_vram = total_vram - used_vram
                return max(8, min(32, int(available_vram * 0.7 / 20)))
            except:
                pass
        return OCR_BATCH_SIZE
    
    @traceable(run_type="tool", name="VideoOCRFrameProcessor.process_scene_frames", 
               project_name="video_analysis_service", 
               process_inputs=skip_frames_input_serialization)
    async def process_scene_frames(
        self,
        request_id: str,
        primary_filename: str,
        scene_output_directory: str
    ) -> Dict[str, Any]:
        """
        處理場景幀的 OCR
        
        Args:
            request_id: 請求 ID
            primary_filename: 原始檔案名
            scene_output_directory: 場景圖片目錄
            
        Returns:
            處理結果字典
        """
        start_time = time.time()
        
        try:
            # 初始化 OCR
            await self.initialize_ocr()
            
            # 獲取場景圖片檔案列表
            scene_frames = self._get_scene_frame_files(scene_output_directory)
            if not scene_frames:
                raise ValueError("No scene frames found")
            
            logger.info(f"Processing {len(scene_frames)} scene frames for OCR")
            
            # 執行 OCR 處理
            ocr_results = await self._process_frames_ocr(scene_frames)
            
            processing_time = time.time() - start_time
            
            result = {
                "ocr_results": ocr_results,
            }
            
            logger.info(f"OCR processing completed in {processing_time:.2f}s for {len(scene_frames)} scenes")
            return result
            
        except Exception as e:
            logger.error(f"Error processing scene frames: {e}")
            raise
        finally:
            await self._cleanup_ocr()
    
    def _get_scene_frame_files(self, scene_output_directory: str) -> List[str]:
        """獲取場景圖片檔案列表"""
        try:
            if not os.path.exists(scene_output_directory):
                logger.error(f"Scene output directory not found: {scene_output_directory}")
                return []
            
            # 查找場景幀檔案（排除 diff plot）
            pattern = os.path.join(scene_output_directory, "scene_frame_*.jpg")
            scene_files = glob.glob(pattern)
            scene_files.sort()  # 按檔名排序
            
            logger.info(f"Found {len(scene_files)} scene frame files")
            return scene_files
            
        except Exception as e:
            logger.error(f"Error getting scene frame files: {e}")
            return []
    
    async def _process_frames_ocr(self, scene_frames: List[str]) -> Dict[str, Dict]:
        """批次處理場景幀的 OCR"""
        ocr_results = {}
        
        try:
            # 批次載入圖片
            batch_size = self._calculate_batch_size()
            
            for i in range(0, len(scene_frames), batch_size):
                batch_files = scene_frames[i:i + batch_size]
                batch_images = []
                batch_keys = []
                
                # 載入批次圖片
                for frame_path in batch_files:
                    try:
                        img = cv2.imread(frame_path)
                        if img is not None:
                            batch_images.append(img)
                            frame_name = os.path.basename(frame_path)
                            batch_keys.append(frame_name)
                    except Exception as e:
                        logger.warning(f"Failed to load image {frame_path}: {e}")
                
                if not batch_images:
                    continue
                
                # 執行批次 OCR
                try:
                    ocr_batch_results = await asyncio.get_event_loop().run_in_executor(
                        None, self.ocr.predict, batch_images
                    )
                    
                    # 處理批次結果並保存帶框的圖片
                    for j, (frame_key, ocr_result) in enumerate(zip(batch_keys, ocr_batch_results)):
                        ocr_text_result = self._extract_text_from_ocr_result(
                            ocr_result, 
                            batch_images[j].shape[0]
                        )
                        ocr_results[frame_key] = ocr_text_result
                        
                        # 保存帶 OCR 框的圖片（僅當有文字時）
                        if ocr_text_result["text"]:
                            await self._save_ocr_frame_with_boxes(
                                batch_images[j], frame_key, ocr_result
                            )
                        
                except Exception as e:
                    logger.error(f"OCR batch processing error: {e}")
                    # 為失敗的批次設置空結果
                    for frame_key in batch_keys:
                        ocr_results[frame_key] = {"text": "", "regions": {}}
            
            logger.info(f"OCR processing completed for {len(ocr_results)} frames")
            return ocr_results
            
        except Exception as e:
            logger.error(f"Error in OCR processing: {e}")
            return {}
    
    def _extract_text_from_ocr_result(self, ocr_result, frame_height: int) -> Dict:
        """從 OCR 結果提取文字和位置資訊"""
        regions = {'upper': [], 'middle': [], 'lower': [], 'full': []}
        
        try:
            if not ocr_result or not hasattr(ocr_result, 'json') or 'res' not in ocr_result.json:
                return {"text": "", "regions": regions}
            
            res_data = ocr_result.json['res']
            texts = res_data.get('rec_texts', [])
            boxes = res_data.get('rec_boxes', [])
            scores = res_data.get('rec_scores', [])
            
            # 處理多邊形框
            if not boxes and 'rec_polys' in res_data:
                boxes = []
                for poly in res_data['rec_polys']:
                    xs = [point[0] for point in poly]
                    ys = [point[1] for point in poly]
                    boxes.append([min(xs), min(ys), max(xs), max(ys)])
            
            # 提取文字並分區域
            for text, box, score in zip(texts, boxes, scores):
                if score < OCR_CONFIDENCE_THRESHOLD:
                    continue
                
                # 轉換為繁體中文
                text = convert(text, 'zh-hant')
                
                # 根據位置分類
                center_y = (box[1] + box[3]) / 2
                if center_y < frame_height * 0.25:
                    regions['upper'].append(text)
                elif center_y < frame_height * 0.75:
                    regions['middle'].append(text)
                else:
                    regions['lower'].append(text)
                regions['full'].append(f"{int(center_y)}:{text}")
            
            # 合併文字
            combined_text = " | ".join([
                f"{k}: {' '.join(v)}" 
                for k, v in regions.items() 
                if v and k != 'full'
            ])
            
            return {
                "text": combined_text,
                "regions": regions
            }
            
        except Exception as e:
            logger.error(f"Error extracting OCR text: {e}")
            return {"text": "", "regions": regions}
    
    async def _save_ocr_frame_with_boxes(self, frame: np.ndarray, frame_key: str, ocr_result):
        """保存帶有 OCR 框的圖片"""
        try:
            if not ocr_result or not hasattr(ocr_result, 'json') or 'res' not in ocr_result.json:
                return
            
            res_data = ocr_result.json['res']
            boxes = res_data.get('rec_boxes', [])
            scores = res_data.get('rec_scores', [])
            
            frame_with_boxes = frame.copy()
            
            # 處理多邊形框
            if not boxes and 'rec_polys' in res_data:
                boxes = []
                for poly in res_data['rec_polys']:
                    xs = [point[0] for point in poly]
                    ys = [point[1] for point in poly]
                    boxes.append([min(xs), min(ys), max(xs), max(ys)])
            
            # 繪製框
            for box, score in zip(boxes, scores):
                if score >= OCR_CONFIDENCE_THRESHOLD:
                    x1, y1, x2, y2 = map(int, box)
                    h_img, w_img = frame_with_boxes.shape[:2]
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_img, x2), min(h_img, y2)
                    cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 保存圖片
            img_path = os.path.join(self.output_dir, f"ocr_{frame_key}")
            success = cv2.imwrite(img_path, frame_with_boxes, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
            
            if success:
                logger.debug(f"Saved OCR frame with boxes: {img_path}")
            else:
                logger.warning(f"Failed to save OCR frame: {img_path}")
                
        except Exception as e:
            logger.error(f"Error saving OCR frame with boxes for {frame_key}: {e}")
    
    async def _cleanup_ocr(self):
        """清理 OCR 資源"""
        try:
            if self.ocr_initialized and self.ocr:
                del self.ocr
                self.ocr = None
                self.ocr_initialized = False
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info("OCR resources cleaned up")
        except Exception as e:
            logger.error(f"Error cleaning up OCR resources: {e}")