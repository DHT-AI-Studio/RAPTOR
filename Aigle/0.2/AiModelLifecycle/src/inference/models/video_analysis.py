# src/inference/models/video_analysis.py
"""
視頻分析模型處理器
"""

import logging
from typing import Any, Dict
from .base import BaseModelHandler

logger = logging.getLogger(__name__)

class VideoAnalysisHandler(BaseModelHandler):
    """視頻分析處理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.description = "視頻分析任務處理器"
        self.supported_formats = ['video/mp4', 'video/avi']
    
    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """預處理視頻數據"""
        self.validate_input(data, ['video'])
        return {'video': data['video']}
    
    def postprocess(self, result: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """後處理分析結果"""
        if isinstance(result, dict):
            return {'analysis': result}
        else:
            return {'analysis': str(result)}