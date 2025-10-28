# src/inference/models/document_analysis.py
"""
文檔分析模型處理器
"""

import logging
from typing import Any, Dict
from .base import BaseModelHandler

logger = logging.getLogger(__name__)

class DocumentAnalysisHandler(BaseModelHandler):
    """文檔分析處理器"""
    
    def __init__(self, config: Dict[str, Any] = None):
        super().__init__(config)
        self.description = "文檔分析任務處理器"
        self.supported_formats = ['application/pdf', 'image/jpeg', 'image/png']
    
    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """預處理文檔數據"""
        self.validate_input(data, ['document'])
        return {'document': data['document']}
    
    def postprocess(self, result: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """後處理分析結果"""
        if isinstance(result, dict):
            return {'analysis': result}
        else:
            return {'analysis': str(result)}