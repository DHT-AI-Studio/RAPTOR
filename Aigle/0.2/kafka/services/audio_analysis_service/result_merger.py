# services/audio_analysis_service/result_merger.py

import json
import os
import logging
import uuid
import aiofiles
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from config import AUDIO_MERGED_RESULTS_DIR
import opencc


logger = logging.getLogger(__name__)

class AudioResultMerger:
    def __init__(self):
        # 確保輸出目錄存在
        os.makedirs(AUDIO_MERGED_RESULTS_DIR, exist_ok=True)
        self.cc = opencc.OpenCC('s2t')
    
    def time_overlap(self, start1, end1, start2, end2):
        """計算時間重疊"""
        return max(0, min(end1, end2) - max(start1, start2))
    
    def get_audio_labels(self, start, end, classification_data, threshold=0.4):
        """
        根據時間範圍獲取音頻標籤
        Args:
            start: 開始時間
            end: 結束時間
            classification_data: 分類數據
            threshold: 概率閾值
        Returns:
            list: 符合條件的標籤列表（去重）
        """
        labels = set()  # 使用set來避免重複
        
        for segment in classification_data:
            # 檢查是否有時間重疊
            overlap = self.time_overlap(start, end, segment['segment_start'], segment['segment_end'])
            if overlap > 0:
                # 添加概率大於閾值的標籤
                for class_info in segment['top_classes']:
                    if class_info['probability'] > threshold:
                        labels.add(class_info['label'])
        
        return list(labels)
    
    def merge_all_data(self, segments, diarization, classification, filename, asset_path, version_id):
        """
        整合語音識別、說話人分離和音頻分類數據
        Args:
            segments: 語音識別結果 (list)
            diarization: 說話人分離結果 (list)
            classification: 音頻分類結果 (list)
            filename: 檔案名稱
            asset_path: 資產路徑
            version_id: 版本ID
        Returns:
            list: 整合後的結果
        """
        merged_all = []
        
        for index, seg in enumerate(segments):
            start = seg['start']
            end = seg['end']
            text = seg['text'].strip()
            text = self.cc.convert(text)
            # 計算與每個 speaker segment 的重疊時間
            overlaps = []
            for d in diarization:
                overlap = self.time_overlap(start, end, d['start'], d['end'])
                if overlap > 0:
                    overlaps.append((overlap, d['speaker']))
            
            # 取重疊最多的 speaker（如果有）
            if overlaps:
                speaker = max(overlaps, key=lambda x: x[0])[1]
            else:
                speaker = None
            
            # 獲取音頻標籤
            audio_labels = self.get_audio_labels(start, end, classification)
            
            merged_item = {
                "start_time": start,
                "end_time": end,
                "speaker": speaker,
                "text": text,
                "audio_labels": audio_labels,
            }
            
            merged_all.append(merged_item)
        
        return merged_all
    
    async def merge_audio_results(
        self,
        classifier_result_path: str,
        recognizer_result_path: str,
        diarization_result_path: str,
        request_id: str,
        primary_filename: str,
        asset_path: Optional[str] = None,
        version_id: Optional[str] = None
    ) -> str:
        """
        合併音頻分析的三個結果
        返回合併後的 JSON 檔案路徑
        """
        try:
            # 讀取三個結果檔案
            classifier_data = await self._read_json_file(classifier_result_path)
            recognizer_data = await self._read_json_file(recognizer_result_path)
            diarization_data = await self._read_json_file(diarization_result_path)
            
            logger.info(f"Loaded classifier data: {len(classifier_data) if isinstance(classifier_data, list) else 'dict'}")
            logger.info(f"Loaded recognizer data: {len(recognizer_data) if isinstance(recognizer_data, list) else 'dict'}")
            logger.info(f"Loaded diarization data: {len(diarization_data) if isinstance(diarization_data, list) else 'dict'}")
            
            # 驗證數據格式
            if not isinstance(recognizer_data, list):
                raise ValueError(f"Recognition data should be a list, got {type(recognizer_data)}")
            if not isinstance(diarization_data, list):
                raise ValueError(f"Diarization data should be a list, got {type(diarization_data)}")
            if not isinstance(classifier_data, list):
                raise ValueError(f"Classification data should be a list, got {type(classifier_data)}")
            
            # 使用參考代碼的合併邏輯
            merged_results = self.merge_all_data(
                segments=recognizer_data,
                diarization=diarization_data,
                classification=classifier_data,
                filename=primary_filename,
                asset_path=asset_path or "",
                version_id=version_id or ""
            )
            
            # 保存合併結果
            merged_filename = f"{request_id}_{primary_filename.split('.')[0]}_merged.json"
            merged_file_path = os.path.join(AUDIO_MERGED_RESULTS_DIR, merged_filename)
            
            await self._save_json_file(merged_results, merged_file_path)
            
            logger.info(f"Audio results merged successfully: {merged_file_path}")
            logger.info(f"Merged {len(merged_results)} segments")
            
            return merged_file_path
            
        except Exception as e:
            logger.error(f"Failed to merge audio results: {e}")
            raise
    
    async def _read_json_file(self, file_path: str) -> Any:
        """異步讀取 JSON 檔案"""
        try:
            async with aiofiles.open(file_path, 'r', encoding='utf-8') as f:
                content = await f.read()
                return json.loads(content)
        except Exception as e:
            logger.error(f"Failed to read JSON file {file_path}: {e}")
            raise
    
    async def _save_json_file(self, data: Any, file_path: str):
        """異步保存 JSON 檔案"""
        try:
            async with aiofiles.open(file_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2, ensure_ascii=False))
        except Exception as e:
            logger.error(f"Failed to save JSON file {file_path}: {e}")
            raise
    
