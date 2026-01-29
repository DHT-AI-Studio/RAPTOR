# src/inference/models/vlm.py
"""
視覺語言模型處理器

處理 VLM（Visual Language Model）任務的輸入預處理和輸出後處理。
支持圖像和文本的多模態輸入。
"""

import logging
import base64
import io
from typing import Any, Dict
from PIL import Image
from .base import BaseModelHandler

logger = logging.getLogger(__name__)

class VLMHandler(BaseModelHandler):
    """
    視覺語言模型處理器
    
    處理 VLM 任務，包括：
    - 圖像數據的解碼和預處理
    - 文本提示的格式化
    - 輸出結果的格式化
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        初始化 VLM 處理器
        
        Args:
            config (Dict[str, Any], optional): 處理器配置
        """
        super().__init__(config)
        self.description = "視覺語言模型任務處理器，支持圖像理解和描述生成"
        self.supported_formats = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp']
        
        # 預設配置
        self.default_options = {
            'max_image_size': (1024, 1024),
            'image_quality': 95,
            'prompt_template': None,
            'max_length': 256,
            'temperature': 0.7
        }
    
    def preprocess(self, data: Dict[str, Any], options: Dict[str, Any]) -> Dict[str, Any]:
        """
        預處理 VLM 輸入
        
        Args:
            data (Dict[str, Any]): 包含 'image' 和 'prompt' 的輸入數據
            options (Dict[str, Any]): 處理選項
            
        Returns:
            Dict[str, Any]: 預處理後的數據
            
        Raises:
            ValueError: 輸入數據無效
        """
        try:
            # 驗證輸入
            self.validate_input(data, ['image', 'prompt'])
            
            # 處理圖像
            image = self._process_image(data['image'], options)
            
            # 處理提示
            prompt = self._process_prompt(data['prompt'], options)
            
            processed_data = {
                'image': image,
                'prompt': prompt,
                'original_prompt': data['prompt']
            }
            
            logger.debug("VLM 預處理完成")
            return processed_data
            
        except Exception as e:
            logger.error(f"VLM 預處理失敗: {e}")
            raise
    
    def postprocess(self, result: Any, options: Dict[str, Any]) -> Dict[str, Any]:
        """
        後處理 VLM 結果
        
        Args:
            result (Any): 原始推理結果
            options (Dict[str, Any]): 處理選項
            
        Returns:
            Dict[str, Any]: 格式化的結果
        """
        try:
            # 處理不同格式的結果
            if isinstance(result, dict):
                if 'response' in result:
                    description = result['response']
                    metadata = {k: v for k, v in result.items() if k != 'response'}
                else:
                    description = str(result)
                    metadata = {}
            else:
                description = str(result)
                metadata = {}
            
            # 清理和格式化描述
            description = self._clean_description(description)
            
            # 應用後處理過濾器
            if 'output_filters' in options:
                description = self._apply_filters(description, options['output_filters'])
            
            processed_result = {
                'description': description,
                'confidence': metadata.get('confidence', 1.0),
                'length': len(description),
                'metadata': metadata
            }
            
            logger.debug(f"VLM 後處理完成，描述長度: {len(description)}")
            return processed_result
            
        except Exception as e:
            logger.error(f"VLM 後處理失敗: {e}")
            raise
    
    def _process_image(self, image_data: Any, options: Dict[str, Any]) -> Image.Image:
        """
        處理圖像數據
        
        Args:
            image_data (Any): 圖像數據（base64、路徑或PIL圖像）
            options (Dict[str, Any]): 處理選項
            
        Returns:
            Image.Image: 處理後的PIL圖像
        """
        try:
            if isinstance(image_data, str):
                if image_data.startswith('data:image'):
                    # 處理 data URL
                    header, encoded = image_data.split(',', 1)
                    image_bytes = base64.b64decode(encoded)
                    image = Image.open(io.BytesIO(image_bytes))
                elif image_data.startswith(('//', 'http')):
                    # 處理 URL（需要下載）
                    import requests
                    response = requests.get(image_data)
                    image = Image.open(io.BytesIO(response.content))
                else:
                    # 處理 base64 字符串或文件路徑
                    try:
                        # 嘗試作為 base64 解碼
                        image_bytes = base64.b64decode(image_data)
                        image = Image.open(io.BytesIO(image_bytes))
                    except:
                        # 嘗試作為文件路徑
                        image = Image.open(image_data)
            elif isinstance(image_data, Image.Image):
                image = image_data
            else:
                raise ValueError(f"不支持的圖像數據類型: {type(image_data)}")
            
            # 轉換為 RGB 模式
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # 調整大小（如果需要）
            max_size = options.get('max_image_size', self.default_options['max_image_size'])
            if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                image.thumbnail(max_size, Image.LANCZOS)
                logger.debug(f"圖像已調整大小至: {image.size}")
            
            return image
            
        except Exception as e:
            logger.error(f"圖像處理失敗: {e}")
            raise ValueError(f"無法處理圖像數據: {e}")
    
    def _process_prompt(self, prompt: str, options: Dict[str, Any]) -> str:
        """
        處理文本提示
        
        Args:
            prompt (str): 原始提示
            options (Dict[str, Any]): 處理選項
            
        Returns:
            str: 處理後的提示
        """
        # 清理提示文本
        prompt = prompt.strip()
        
        # 應用模板（如果提供）
        template = options.get('prompt_template', self.default_options['prompt_template'])
        if template:
            prompt = template.format(prompt=prompt)
        
        return prompt
    
    def _clean_description(self, description: str) -> str:
        """
        清理生成的描述文本
        
        Args:
            description (str): 原始描述
            
        Returns:
            str: 清理後的描述
        """
        # 移除多餘的空白
        description = ' '.join(description.split())
        
        # 移除常見的生成標記
        unwanted_tokens = [
            '<|endoftext|>', '<|im_end|>', '<|im_start|>',
            '[INST]', '[/INST]', '<image>', '</image>'
        ]
        for token in unwanted_tokens:
            description = description.replace(token, '')
        
        # 確保描述以適當的標點結尾
        description = description.strip()
        if description and not description[-1] in '.!?':
            description += '.'
        
        return description
    
    def _apply_filters(self, text: str, filters: list) -> str:
        """
        應用輸出過濾器
        
        Args:
            text (str): 輸入文本
            filters (list): 過濾器配置列表
            
        Returns:
            str: 過濾後的文本
        """
        for filter_config in filters:
            if isinstance(filter_config, dict):
                filter_type = filter_config.get('type')
                
                if filter_type == 'remove_redundant':
                    # 移除冗餘表達
                    redundant_phrases = [
                        'The image shows', 'This image depicts', 'In this image',
                        'The picture shows', 'This picture depicts'
                    ]
                    for phrase in redundant_phrases:
                        if text.startswith(phrase):
                            text = text[len(phrase):].strip()
                            break
                
                elif filter_type == 'capitalize_first':
                    # 首字母大寫
                    if text:
                        text = text[0].upper() + text[1:]
                
                elif filter_type == 'length_limit':
                    # 長度限制
                    max_len = filter_config.get('max_length', 200)
                    if len(text) > max_len:
                        text = text[:max_len] + '...'
        
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
            'image_processing_features': [
                'base64_decoding',
                'url_loading',
                'format_conversion',
                'size_adjustment',
                'mode_normalization'
            ],
            'text_processing_features': [
                'prompt_templating',
                'description_cleaning',
                'output_filtering',
                'punctuation_normalization'
            ]
        })
        return base_info