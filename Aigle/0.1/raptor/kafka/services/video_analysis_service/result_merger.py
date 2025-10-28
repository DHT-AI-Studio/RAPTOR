# services/video_analysis_service/result_merger.py

import json
import os
import logging
import aiofiles
from typing import Dict, Any, List, Union
from config import MERGED_RESULTS_BASE_DIR

logger = logging.getLogger(__name__)

class ResultMerger:
    def __init__(self):
        # 確保輸出目錄存在
        os.makedirs(MERGED_RESULTS_BASE_DIR, exist_ok=True)
    
    async def merge_all_video_results(
        self, 
        scene_json_path: str, 
        ocr_json_path: str, 
        frame_description_json_path: str,
        request_id: str,
        primary_filename: str
    ) -> str:
        """
        合併場景檢測、OCR 和 Frame Description 結果
        返回合併後的 JSON 檔案路徑
        """
        try:
            # 讀取三個結果檔案
            scene_data = await self._read_json_file(scene_json_path)
            ocr_data = await self._read_json_file(ocr_json_path)
            frame_description_data = await self._read_json_file(frame_description_json_path)
            
            logger.info(f"Scene data type: {type(scene_data)}, length: {len(scene_data) if isinstance(scene_data, list) else 'N/A'}")
            logger.info(f"OCR data type: {type(ocr_data)}")
            logger.info(f"Frame description data type: {type(frame_description_data)}")
            
            # 合併結果
            merged_results = await self._merge_all_results(scene_data, ocr_data, frame_description_data)
            
            # 保存合併結果
            merged_filename = f"merged_video_analysis_{request_id}_{primary_filename.split('.')[0]}.json"
            merged_file_path = os.path.join(MERGED_RESULTS_BASE_DIR, merged_filename)
            
            await self._save_json_file(merged_results, merged_file_path)
            
            logger.info(f"Video results merged successfully: {merged_file_path}")
            logger.info(f"Total frames with analysis: {len(merged_results)}")
            
            return merged_file_path
            
        except Exception as e:
            logger.error(f"Failed to merge video results: {e}")
            raise
    
    async def _read_json_file(self, file_path: str) -> Union[Dict[str, Any], List[Any]]:
        """異步讀取 JSON 檔案"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to read JSON file {file_path}: {e}")
            raise
    
    async def _save_json_file(self, data: List[Dict[str, Any]], file_path: str):
        """異步保存 JSON 檔案"""
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to save JSON file {file_path}: {e}")
            raise
    
    async def _merge_all_results(
        self, 
        scene_data: List[Dict[str, Any]], 
        ocr_data: Dict[str, Any],
        frame_description_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        合併場景檢測、OCR 和 Frame Description 結果
        
        場景檢測格式: [{"frame_index": 11, "timestamp": 22.0, "combined_diff": 70.09}, ...]
        OCR 結果格式: {"ocr_results": {"scene_frame_0011.jpg": {"text": "...", "regions": {...}}, ...}}
        Frame Description 格式: {
            "event_summary": "事件摘要...",
            "scene_frames_data": [{"frame_index": 1, "timestamp": 2.0, "combined_diff": 54.26}, ...]
        }
        
        返回格式: {
            "event_summary": "...",
            "frames": [{"frame_index": int, "timestamp": float, "text": str, "combined_diff": float}, ...]
        }
        """
        merged_results = []
        
        # 確保 scene_data 是列表格式
        if not isinstance(scene_data, list):
            logger.error(f"Scene data should be a list, got {type(scene_data)}")
            raise ValueError(f"Scene data should be a list, got {type(scene_data)}")
        
        # 從結果中獲取相應的數據
        ocr_results = ocr_data.get("ocr_results", {}) if isinstance(ocr_data, dict) else {}
        event_summary = frame_description_data.get("event_summary", "") if isinstance(frame_description_data, dict) else ""
        
        logger.info(f"OCR results contains {len(ocr_results)} frames")
        logger.info(f"Event summary: {event_summary[:100]}..." if len(event_summary) > 100 else f"Event summary: {event_summary}")

        merged_frames = []
        
        # 遍歷每個場景
        for scene in scene_data:
            frame_index = scene.get("frame_index", 0)
            timestamp = scene.get("timestamp", 0.0)
            
            # 構建對應的場景幀檔名 (格式: scene_frame_0011.jpg)
            frame_filename = f"scene_frame_{frame_index:04d}.jpg"
            
            # 從 OCR 結果中獲取對應的文字資訊
            ocr_info = ocr_results.get(frame_filename, {})
            text = ocr_info.get("text", "")
            
            
            # 創建合併結果
            merged_result = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "text": text,
            }
            
            merged_frames.append(merged_result)
            
            # 記錄處理狀態
            has_text = bool(text.strip())
            
            if has_text:
                logger.debug(f"Frame {frame_index}: Found OCR text")
            else:
                logger.debug(f"Frame {frame_index}: No OCR text found for {frame_filename}")
        
        # 按 frame_index 排序
        merged_frames.sort(key=lambda x: x["frame_index"])
        # 創建最終的合併結果，包含 event_summary
        merged_results = {
            "event_summary": event_summary,
            "frames": merged_frames
        }
        
        logger.info(f"Merged {len(merged_frames)} scenes with OCR data and event summary")
        
        return merged_results

    # 保留原有的方法以保持向後兼容性
    async def merge_scene_and_ocr_results(
        self, 
        scene_json_path: str, 
        ocr_json_path: str, 
        request_id: str,
        primary_filename: str
    ) -> str:
        """
        舊版本的合併方法（僅合併場景檢測和 OCR 結果）
        保留以維持向後兼容性
        """
        try:
            # 讀取場景檢測結果
            scene_data = await self._read_json_file(scene_json_path)
            logger.info(f"Scene data type: {type(scene_data)}, length: {len(scene_data) if isinstance(scene_data, list) else 'N/A'}")
            
            # 讀取 OCR 結果
            ocr_data = await self._read_json_file(ocr_json_path)
            logger.info(f"OCR data type: {type(ocr_data)}")
            
            # 合併結果（不包含 frame_description）
            merged_results = await self._merge_scene_and_ocr_only(scene_data, ocr_data)
            
            # 保存合併結果
            merged_filename = f"merged_analysis_result_{request_id}_{primary_filename}.json"
            merged_file_path = os.path.join(MERGED_RESULTS_BASE_DIR, merged_filename)
            
            await self._save_json_file_list(merged_results, merged_file_path)
            
            logger.info(f"Results merged successfully: {merged_file_path}")
            logger.info(f"Total frames with analysis: {len(merged_results)}")
            
            return merged_file_path
            
        except Exception as e:
            logger.error(f"Failed to merge results: {e}")
            raise
    
    async def _merge_scene_and_ocr_only(
        self, 
        scene_data: List[Dict[str, Any]], 
        ocr_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """僅合併場景檢測和 OCR 結果（不包含 frame_description）"""
        merged_results = []
        
        # 確保 scene_data 是列表格式
        if not isinstance(scene_data, list):
            logger.error(f"Scene data should be a list, got {type(scene_data)}")
            raise ValueError(f"Scene data should be a list, got {type(scene_data)}")
        
        # 從 OCR 結果中獲取 ocr_results
        ocr_results = ocr_data.get("ocr_results", {}) if isinstance(ocr_data, dict) else {}
        logger.info(f"OCR results contains {len(ocr_results)} frames")
        
        # 遍歷每個場景
        for scene in scene_data:
            frame_index = scene.get("frame_index", 0)
            timestamp = scene.get("timestamp", 0.0)
            combined_diff = scene.get("combined_diff", 0.0)
            
            # 構建對應的場景幀檔名 (格式: scene_frame_0011.jpg)
            frame_filename = f"scene_frame_{frame_index:04d}.jpg"
            
            # 從 OCR 結果中獲取對應的文字資訊
            ocr_info = ocr_results.get(frame_filename, {})
            text = ocr_info.get("text", "")
            regions = ocr_info.get("regions", {})
            
            # 創建合併結果（舊格式，不包含 frame_description）
            merged_result = {
                "frame_index": frame_index,
                "timestamp": timestamp,
                "text": text,
            }
            
            merged_results.append(merged_result)
            
            if text:
                logger.debug(f"Frame {frame_index}: Found OCR text")
            else:
                logger.debug(f"Frame {frame_index}: No OCR text found for {frame_filename}")
        
        # 按 frame_index 排序
        merged_results.sort(key=lambda x: x["frame_index"])
        
        logger.info(f"Merged {len(merged_results)} scenes with OCR data")
        
        return merged_results
