# src/inference/models/text_generation.py
"""
文本生成模型處理器

處理文本生成任務的輸入預處理和輸出後處理。
支持 Ollama 和 Transformers 引擎的文本生成模型。
"""

import logging
from typing import Any, Dict
from .base import BaseModelHandler

logger = logging.getLogger(__name__)

class TextGenerationHandler(BaseModelHandler):
    """
    文本生成處理器
    
    處理文本生成任務，包括：
    - 輸入文本的清理和格式化
    - 輸出文本的後處理和格式化
    - 特殊字符和標記的處理
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化文本生成處理器
        
        Args:
            config (Dict[str, Any], optional): 處理器配置
        """
        super().__init__(config)
        self.description = "文本生成任務處理器，支持各種文本生成模型"
        self.supported_formats = ['text/plain', 'application/json']
        
        # 預設配置
        self.default_options = {
            'max_length': 512,
            'temperature': 0.7,
            'top_p': 0.9,
            'top_k': 50,
            'do_sample': True,
            'clean_input': True,
            'clean_output': True
        }
    
    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        預處理文本生成輸入
        
        Args:
            data (Dict[str, Any]): 包含 'inputs' 鍵的輸入數據
            options (Dict[str, Any]): 處理選項
            
        Returns:
            Dict[str, Any]: 預處理後的數據
            
        Raises:
            ValueError: 輸入數據無效
        """
        try:
            # 驗證輸入
            self.validate_input(data, ['inputs'])
            
            input_text = data['inputs']
            
            # 文本清理（如果啟用）
            if options.get('clean_input', self.default_options['clean_input']):
                input_text = self._clean_input_text(input_text)
            
            # 應用模板（如果提供）
            if 'template' in options:
                input_text = self._apply_template(input_text, options['template'])
            
            # 長度檢查
            max_input_length = options.get('max_input_length', 2048)
            if len(input_text) > max_input_length:
                logger.warning(f"輸入文本長度 {len(input_text)} 超過限制 {max_input_length}，將截斷")
                input_text = input_text[:max_input_length]
            
            processed_data = {
                'inputs': input_text,
                'original_length': len(data['inputs']),
                'processed_length': len(input_text)
            }
            
            logger.debug(f"文本生成預處理完成，輸入長度: {len(input_text)}")
            return processed_data
            
        except Exception as e:
            logger.error(f"文本生成預處理失敗: {e}")
            raise
    
    def postprocess(self, result: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        後處理文本生成結果
        
        Args:
            result (Any): 原始推理結果
            options (Dict[str, Any]): 處理選項
            
        Returns:
            Dict[str, Any]: 格式化的結果
        """
        try:
            # 處理不同引擎的結果格式
            if isinstance(result, dict):
                if 'response' in result:
                    # Ollama 格式
                    generated_text = result['response']
                    metadata = {
                        'model': result.get('model', ''),
                        'done': result.get('done', True),
                        'total_duration': result.get('total_duration', 0),
                        'eval_count': result.get('eval_count', 0)
                    }
                elif 'response' in result:
                    # Transformers 格式
                    generated_text = result['response']
                    metadata = {
                        'input_length': result.get('input_length', 0),
                        'output_length': result.get('output_length', 0)
                    }
                else:
                    # 通用格式
                    generated_text = str(result)
                    metadata = {}
            else:
                # 純文本結果
                generated_text = str(result)
                metadata = {}
            
            # 文本清理（如果啟用）
            if options.get('clean_output', self.default_options['clean_output']):
                generated_text = self._clean_output_text(generated_text)
            
            # 應用後處理過濾器
            if 'output_filters' in options:
                generated_text = self._apply_output_filters(generated_text, options['output_filters'])
            
            # 長度限制
            max_output_length = options.get('max_output_length')
            if max_output_length and len(generated_text) > max_output_length:
                generated_text = generated_text[:max_output_length] + "..."
                metadata['truncated'] = True
            
            processed_result = {
                'generated_text': generated_text,
                'length': len(generated_text),
                'metadata': metadata
            }
            
            logger.debug(f"文本生成後處理完成，輸出長度: {len(generated_text)}")
            return processed_result
            
        except Exception as e:
            logger.error(f"文本生成後處理失敗: {e}")
            raise
    
    def _clean_input_text(self, text: str) -> str:
        """
        清理輸入文本
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 清理後的文本
        """
        # 移除多餘的空白字符
        text = ' '.join(text.split())
        
        # 移除控制字符
        text = ''.join(char for char in text if ord(char) >= 32 or char in '\n\t')
        
        # 規範化換行符
        text = text.replace('\r\n', '\n').replace('\r', '\n')
        
        return text.strip()
    
    def _clean_output_text(self, text: str) -> str:
        """
        清理輸出文本
        
        Args:
            text (str): 原始文本
            
        Returns:
            str: 清理後的文本
        """
        # 移除多餘的空白
        text = text.strip()
        
        # 移除常見的生成標記
        unwanted_tokens = ['<|endoftext|>', '<|im_end|>', '<|im_start|>', '[INST]', '[/INST]']
        for token in unwanted_tokens:
            text = text.replace(token, '')
        
        # 規範化空白
        text = ' '.join(text.split())
        
        return text
    
    def _apply_template(self, text: str, template: str) -> str:
        """
        應用文本模板
        
        Args:
            text (str): 輸入文本
            template (str): 模板字符串，使用 {input} 作為佔位符
            
        Returns:
            str: 應用模板後的文本
        """
        try:
            return template.format(input=text)
        except KeyError:
            logger.warning("模板格式錯誤，使用原始輸入")
            return text
    
    def _apply_output_filters(self, text: str, filters: list) -> str:
        """
        應用輸出過濾器
        
        Args:
            text (str): 輸出文本
            filters (list): 過濾器列表
            
        Returns:
            str: 過濾後的文本
        """
        for filter_config in filters:
            if isinstance(filter_config, dict):
                filter_type = filter_config.get('type')
                
                if filter_type == 'remove_prefix':
                    prefix = filter_config.get('value', '')
                    if text.startswith(prefix):
                        text = text[len(prefix):].strip()
                
                elif filter_type == 'remove_suffix':
                    suffix = filter_config.get('value', '')
                    if text.endswith(suffix):
                        text = text[:-len(suffix)].strip()
                
                elif filter_type == 'replace':
                    old_value = filter_config.get('old', '')
                    new_value = filter_config.get('new', '')
                    text = text.replace(old_value, new_value)
        
        return text
    
    def get_info(self) -> Dict[str, Any]:
        """
        獲取處理器詳細信息
        
        Returns:
            Dict[str, Any]: 處理器信息
        """
        base_info = super().get_info()
        base_info.update({
            'default_options': self.default_options,
            'cleaning_features': [
                'input_text_cleaning',
                'output_text_cleaning', 
                'template_support',
                'output_filtering',
                'length_control'
            ]
        })
        return base_info