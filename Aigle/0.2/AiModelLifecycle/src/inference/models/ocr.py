# src/inference/models/ocr.py
"""
光學字符識別模型處理器
"""

import logging
from typing import Any, Dict
from .base import BaseModelHandler

logger = logging.getLogger(__name__)

class OCRHandler(BaseModelHandler):
    """OCR 模型處理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.description = "光學字符識別任務處理器"
        self.supported_formats = ['image/jpeg', 'image/png', 'image/bmp']
    
    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """預處理圖像數據"""
        self.validate_input(data, ['image'])
        return {'image': data['image']}
    
    def postprocess(self, result: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """後處理 OCR 結果"""
        if isinstance(result, dict) and 'text' in result:
            return {'text': result['text'], 'confidence': result.get('confidence', 1.0)}
        elif isinstance(result, str):
            return {'text': result, 'confidence': 1.0}
        else:
            return {'text': str(result), 'confidence': 1.0}