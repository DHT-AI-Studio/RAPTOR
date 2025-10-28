# src/inference/models/audio_classification.py
"""
音頻分類模型處理器
"""

import logging
from typing import Any, Dict
from .base import BaseModelHandler

logger = logging.getLogger(__name__)

class AudioClassificationHandler(BaseModelHandler):
    """音頻分類處理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.description = "音頻分類任務處理器"
        self.supported_formats = ['audio/wav', 'audio/mp3']
    
    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """預處理音頻數據"""
        self.validate_input(data, ['audio'])
        return {'audio': data['audio']}
    
    def postprocess(self, result: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """後處理分類結果"""
        if isinstance(result, list) and len(result) > 0:
            return {'classification': result[0], 'all_results': result}
        else:
            return {'classification': str(result), 'all_results': [result]}