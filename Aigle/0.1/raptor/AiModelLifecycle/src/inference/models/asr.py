# src/inference/models/asr.py
"""
自動語音識別模型處理器
"""

import logging
from typing import Any, Dict
from .base import BaseModelHandler

logger = logging.getLogger(__name__)

class ASRHandler(BaseModelHandler):
    """ASR 模型處理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.description = "自動語音識別任務處理器"
        self.supported_formats = ['audio/wav', 'audio/mp3', 'audio/flac']
    
    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """預處理音頻數據"""
        self.validate_input(data, ['audio'])
        return {'audio': data['audio']}
    
    def postprocess(self, result: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """後處理 ASR 結果"""
        if isinstance(result, dict) and 'text' in result:
            return {'transcription': result['text'], 'confidence': result.get('confidence', 1.0)}
        elif isinstance(result, str):
            return {'transcription': result, 'confidence': 1.0}
        else:
            return {'transcription': str(result), 'confidence': 1.0}